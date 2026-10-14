from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from source.training.data import (SELECTION_MANIFEST, MarketDataset, PKSampler, assert_disjoint, collate, list_split,
                   selection_identities,
                   train_transform)
from source.evaluation.metrics import all_pairs_verification
from source.training.losses import CosFaceHead, triplet_loss
from source.models.reid_resnet18.model import _extract_state_dict, _load_raw
from source.utils.device_utils import select_device
from source.logic.preprocessing import build_transform

log = logging.getLogger(__name__)

DEFAULTS = {
    "name": "experiment", "seed": 0, "epochs": 3, "lr": 1e-4, "weight_decay": 5e-4, "warmup_iters": 50,
    "p": 16, "k": 4, "num_workers": 4, "max_batches_per_epoch": 0,
    "ce_weight": 1.0, "head": "linear",
    "cosface_scale": 30.0, "cosface_margin": 0.25,
    "triplet_weight": 0.0, "triplet_margin": 0.3, "mining": "batch_hard",
    "selection_far": 0.01, "device": "auto",
}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class Backbone(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = torchvision.models.resnet18(weights=None)
        self.net.fc = nn.Linear(512, num_classes)

    def embed(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return torch.flatten(n.avgpool(x), 1)


@torch.inference_mode()
def embed_samples(model: Backbone, samples, pre_cfg: dict, device, batch_size: int = 64, workers: int = 0):
    ds = MarketDataset(samples, build_transform(pre_cfg))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate)
    was = model.training
    model.eval()
    out = [F.normalize(model.embed(x.to(device)), dim=1).cpu().numpy() for x, _, _ in dl]
    model.train(was)
    return np.concatenate(out).astype(np.float32)


def validate(model, val_samples, pre_cfg, device, far: float, workers: int) -> dict:
    f = embed_samples(model, val_samples, pre_cfg, device, workers=workers)
    pids = np.array([s.pid for s in val_samples]); cams = np.array([s.camid for s in val_samples])
    v = all_pairs_verification(f, pids, cams, fars=(1e-3, far, 1e-1))
    v.pop("_scores", None)
    return {"val_auc": v["roc_auc"], "val_tar_at_far": v["operating_points"][str(far)]["tar"],
            "val_tar_at_far_1e-3": v["operating_points"]["0.001"]["tar"],
            "val_genuine_pairs": v["genuine_pairs"], "val_impostor_pairs": v["impostor_pairs"]}


def export_state_dict(model: Backbone, head) -> dict:
    sd = {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
    if isinstance(head, CosFaceHead):
        sd["fc.weight"] = F.normalize(head.weight.detach().cpu(), dim=1)
        sd["fc.bias"] = torch.zeros(len(sd["fc.weight"]))
    return sd


RESUME_TOLERATED = {"num_workers", "name"}


def _check_resume_compatible(out_dir: Path, exp: dict, split: dict, n_train_images: int) -> None:
    old = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    diff = {k: (old.get(k), exp.get(k)) for k in set(old) | set(exp)
            if k not in RESUME_TOLERATED and old.get(k) != exp.get(k)}
    if diff:
        raise ValueError(f"cannot resume {out_dir.name}: experiment settings changed {diff}")
    old_split = json.loads((out_dir / "split.json").read_text(encoding="utf-8"))
    if sorted(old_split["selection_pids"]) != split["selection_pids"]:
        raise ValueError(f"cannot resume {out_dir.name}: selection identities differ from the manifest")
    if old_split["train_images"] != n_train_images:
        raise ValueError(f"cannot resume {out_dir.name}: training images {old_split['train_images']} -> {n_train_images}")


def train(cfg_all: dict, exp: dict, out_dir: Path, resume: bool = False) -> dict:
    exp = {**DEFAULTS, **exp}
    if exp["mining"] not in ("batch_hard", "batch_all"):
        raise ValueError("mining must be batch_hard or batch_all")
    if exp["ce_weight"] <= 0 and exp["triplet_weight"] <= 0:
        raise ValueError("at least one of ce_weight / triplet_weight must be > 0")
    out_dir = Path(out_dir)
    if out_dir.resolve() == Path(cfg_all["model"]["checkpoint"]).resolve().parent:
        raise ValueError("refusing to write experiments into the baseline artifact folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(exp["seed"])
    device = select_device(exp["device"])
    pre = cfg_all["preprocessing"]
    root = Path(cfg_all["data"]["market1501_root"])

    train_all = list_split(root / "bounding_box_train")
    all_pids = sorted({s.pid for s in train_all})
    n_cls = int(cfg_all["model"]["num_train_classes"])
    if len(all_pids) != n_cls:
        raise ValueError(f"{len(all_pids)} train identities != model.num_train_classes {n_cls}")
    label_map = {p: i for i, p in enumerate(all_pids)}
    test_pool = list_split(root / "query") + list_split(root / "bounding_box_test")
    test_ids = np.unique([s.pid for s in test_pool])
    val_ids = selection_identities()
    if len(np.setdiff1d(val_ids, test_ids)):
        raise ValueError("selection manifest contains ids that are not in the Market-1501 test pool")
    cal_test_ids = np.setdiff1d(test_ids, val_ids)
    assert_disjoint(train=all_pids, selection=val_ids, calibration_and_test=cal_test_ids)
    val_set = set(val_ids.tolist())
    tr_samples = train_all
    val_samples = [s for s in test_pool if s.pid in val_set]
    split = {"seed": exp["seed"], "train_pids": "all 751 bounding_box_train identities",
             "selection_pids": val_ids.tolist(), "selection_manifest": SELECTION_MANIFEST.name,
             "train_images": len(tr_samples), "selection_images": len(val_samples),
             "calibration_and_test_identities": int(len(cal_test_ids))}
    if resume and (out_dir / "last.pth").is_file():
        _check_resume_compatible(out_dir, exp, split, len(tr_samples))
    else:
        (out_dir / "split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
        (out_dir / "config.json").write_text(json.dumps(exp, indent=2), encoding="utf-8")

    model = Backbone(n_cls)
    sd = {k.removeprefix("module."): v for k, v in _extract_state_dict(_load_raw(cfg_all["model"]["checkpoint"])).items()}
    model.net.load_state_dict(sd, strict=True)
    if exp["head"] == "cosface":
        head = CosFaceHead(512, n_cls, exp["cosface_scale"], exp["cosface_margin"])
        head.weight.data.copy_(sd["fc.weight"])
    elif exp["head"] == "linear":
        head = model.net.fc
    else:
        raise ValueError("head must be linear or cosface")
    model.to(device); head.to(device)
    params = list(model.parameters()) + ([] if head is model.net.fc else list(head.parameters()))
    opt = torch.optim.AdamW(params, lr=exp["lr"], weight_decay=exp["weight_decay"])

    sampler = PKSampler([s.pid for s in tr_samples], exp["p"], exp["k"], exp["seed"])
    steps_per_epoch = len(sampler) if not exp["max_batches_per_epoch"] else min(len(sampler), exp["max_batches_per_epoch"])
    total = steps_per_epoch * exp["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / max(exp["warmup_iters"], 1))
                                              * 0.5 * (1 + np.cos(np.pi * min(it, total) / max(total, 1))))
    ds = MarketDataset(tr_samples, train_transform(pre), label_map)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=exp["num_workers"], collate_fn=collate,
                    persistent_workers=exp["num_workers"] > 0)

    history, start_epoch, best = [], 1, None
    last_path, best_path = out_dir / "last.pth", out_dir / "best.pth"
    if resume and last_path.is_file():
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        model.net.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"])
        if head is not model.net.fc:
            head.load_state_dict(st["head"])
        history, best, start_epoch = st["history"], st["best"], st["epoch"] + 1
        torch.set_rng_state(st["torch_rng"]); np.random.set_state(st["numpy_rng"]); random.setstate(st["python_rng"])
        log.info("resumed from epoch %d", st["epoch"])
    else:
        v0 = validate(model, val_samples, pre, device, exp["selection_far"], exp["num_workers"])
        history.append({"epoch": 0, "note": "baseline before fine-tuning", **v0})
        best = {"epoch": 0, "metric": v0["val_tar_at_far"]}
        log.info("epoch 0 (baseline): %s", v0)

    for epoch in range(start_epoch, exp["epochs"] + 1):
        sampler.set_epoch(epoch)
        model.train()
        t0, agg = time.time(), {"loss": 0.0, "ce": 0.0, "tri": 0.0, "active": 0.0, "acc": 0.0, "n": 0}
        for step, (x, y, _) in enumerate(dl):
            if step >= steps_per_epoch:
                break
            x, y = x.to(device), y.to(device)
            feat = model.embed(x)
            loss = feat.new_zeros(())
            if exp["ce_weight"] > 0:
                logits = head(feat, y) if isinstance(head, CosFaceHead) else head(feat)
                ce = F.cross_entropy(logits, y)
                loss = loss + exp["ce_weight"] * ce
                agg["ce"] += ce.item(); agg["acc"] += (logits.argmax(1) == y).float().mean().item()
            if exp["triplet_weight"] > 0:
                ts = triplet_loss(feat, y, exp["triplet_margin"], exp["mining"])
                loss = loss + exp["triplet_weight"] * ts.loss
                agg["tri"] += ts.loss.item(); agg["active"] += ts.active_fraction
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            agg["loss"] += loss.item(); agg["n"] += 1
            if step % 20 == 0:
                log.info("epoch %d step %d/%d loss %.4f lr %.2e", epoch, step, steps_per_epoch, loss.item(),
                         sched.get_last_lr()[0])
        n = max(agg.pop("n"), 1)
        row = {"epoch": epoch, **{f"train_{k}": v / n for k, v in agg.items()}, "seconds": round(time.time() - t0, 1)}
        row.update(validate(model, val_samples, pre, device, exp["selection_far"], exp["num_workers"]))
        history.append(row)
        log.info("epoch %d: %s", epoch, row)
        if row["val_tar_at_far"] > best["metric"]:
            best = {"epoch": epoch, "metric": row["val_tar_at_far"]}
            torch.save({"epoch": epoch, "model_state_dict": export_state_dict(model, head),
                        "val_tar_at_far": row["val_tar_at_far"], "experiment": exp}, best_path)
        torch.save({"epoch": epoch, "model": model.net.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "head": head.state_dict(), "history": history, "best": best,
                    "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate()}, last_path)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    result = {"name": exp["name"], "best_epoch": best["epoch"], "best_val_tar_at_far": best["metric"],
              "selection_metric": f"SELECTION-identity TAR @ FAR {exp['selection_far']} (cross-camera pairs)",
              "best_checkpoint": str(best_path) if best["epoch"] > 0 else None,
              "note": None if best["epoch"] > 0 else "no fine-tuned epoch beat the baseline on SELECTION ids; baseline kept"}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
