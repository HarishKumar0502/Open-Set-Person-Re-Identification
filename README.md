# PS1 — Open-Set Person Re-Identification

Given a person image, decide whether it matches an **enrolled identity** in a gallery or is an
**UNKNOWN** person. This project wraps the supplied trained ResNet18 checkpoint and embeddings
in a modular, tested inference package, a FastAPI web UI, and a video module: YOLO person detection +
ByteTrack tracking + open-set ReID on uploaded CCTV-style videos (§9).

> **Layout (since 2026-10-14, §17):** backend code lives in the `source/` package (`source/main.py` = FastAPI entry
> point, `source/models/` = one package per model, `source/logic/` = detection / ReID / video logic); the video job
> worker is the root `worker.py`. Run commands from the project root.
>
> **Tools outside the project (§17.5):** `scripts/`, `tests/`, `reports/`, `outputs/` and `pytest.ini` live in the
> sibling folder `D:\zed digital\ps1\PS1_extras\`. Commands below written as `python ..\PS1_extras\scripts\…` are
> run from the project folder; references to `reports\…` / `outputs\…` in this README mean `..\PS1_extras\reports\…`
> / `..\PS1_extras\outputs\…`. It applies an open-set threshold that has been **re-validated on held-out identities**.

> The open-set algorithm, data, models and evaluation are **PS1's own**. Since 2026-10-13 the project follows the
> **PS2 project layout** and reuses PS2's inference-backend code (automatic TensorRT / ONNX Runtime / PyTorch selection
> with verification and runtime fallback). PS2's tracking algorithm is **not** used. See **§16** for the mapping, what was
> reused, what was excluded, and the verified compatibility of the PS2 model files.

---

## 1. Quick start (Windows, existing Python, no venv)

All commands run from the project folder in **PowerShell** or **cmd**, using the existing Python install
(tested: `C:\Python314\python.exe`, Python 3.14.7, CPU only).

```powershell
cd "D:\zed digital\ps1\PS1_Open_Set_Person_ReID"

# 1) dependencies -> installed into the existing Python (no venv)
python -m pip install -r requirements.txt

# 2) check artifacts, checkpoint, and that preprocessing reproduces the supplied embeddings
python ..\PS1_extras\scripts\verify_artifacts.py

# 3) web interface -> open http://127.0.0.1:8000   (API docs: http://127.0.0.1:8000/docs)
python worker.py               # starts the backend; options: --port 8080 --host 127.0.0.1 --config configs\default.yaml
# (python -m source.main starts the same server)

# 4) tests (real backends that are not installed are skipped with the reason, never mocked)
cd ..\PS1_extras; python -m pytest -rs
```

The inference backend is chosen automatically at start-up (TensorRT -> ONNX Runtime CUDA/DirectML -> PyTorch CUDA ->
ONNX Runtime CPU -> PyTorch CPU) and verified against the PyTorch reference; the log line `ReID backend selected: ...`
and `GET /api/health` show the choice. Nothing about backends, devices or precision is asked of the user (§16.4).

If `python` opens the Microsoft Store, call the interpreter explicitly (`C:\Python314\python.exe ...`)
or disable the *App execution alias* for python in Windows settings.
Scripts installed by pip (`uvicorn.exe`, `yolo.exe`) go to
`%APPDATA%\Python\Python314\Scripts`, which is not on PATH. Nothing here needs them;
always use `python -m ...` or the `.py` launchers.

### Other commands

```powershell
# identify one person crop (whole image = one person)
python ..\PS1_extras\scripts\query_image.py --image data\Market-1501-v15.09.15\query\0001_c1s1_001051_00.jpg

# full-scene photo: YOLO person detection first, then ReID per person
python ..\PS1_extras\scripts\query_image.py --image C:\path\street.jpg --detect

# override the operating point / get JSON
python ..\PS1_extras\scripts\query_image.py --image crop.jpg --threshold 0.88 --margin 0 --json

# one video job through the worker (same job manager, cleanup and outputs as a web upload, no server needed)
python worker.py --video C:\path\cctv.mp4 --max-frames 300        # -> outputs\video_jobs\<job id>\

# video WITH tracking (recommended): YOLO + ByteTrack + sampled open-set ReID per track
#   -> H.264 annotated mp4, per-frame track CSV, per-sample ReID CSV, summary JSON   (see §9)
python ..\PS1_extras\scripts\track_video.py --video C:\path\cctv.mp4 --out outputs\video_tracked
python ..\PS1_extras\scripts\track_video.py --video C:\path\cctv.avi --stride 2 --max-frames 500 --threshold 0.88

# video: older per-frame detection + ReID (NO tracking), kept for comparison
python ..\PS1_extras\scripts\process_video.py --video C:\path\clip.mp4 --out outputs\video --stride 5

# tracked-video evaluation against ground truth (TAR / FAR / ROC / TAR@FAR, ID switches), see §9.5
python ..\PS1_extras\scripts\make_synthetic_video.py --crossing --known 4 --unknown 2 --frames 200 --seed 1 --out outputs\video_eval\cross_s1
python ..\PS1_extras\scripts\track_video.py --video outputs\video_eval\cross_s1\synthetic_market_people.mp4 --out outputs\video_eval\cross_s1
python ..\PS1_extras\scripts\evaluate_video_tracking.py --run outputs\video_eval\cross_s1 --out reports\video_evaluation.json

# re-run threshold validation / calibration on the supplied embeddings (~15 s)
python ..\PS1_extras\scripts\evaluate_open_set.py --seeds 10

# latency on this machine
python ..\PS1_extras\scripts\benchmark_latency.py

# enrol your own people (folder per identity), no retraining needed
python ..\PS1_extras\scripts\build_gallery.py --images C:\path\people --out outputs\galleries\mine.npz

# synthetic video functional test (Market crops pasted on a background, with ground truth)
python ..\PS1_extras\scripts\make_synthetic_video.py
python ..\PS1_extras\scripts\process_video.py --video outputs\synthetic_video\synthetic_market_people.mp4 --out outputs\synthetic_video
python ..\PS1_extras\scripts\score_synthetic_video.py --dir outputs\synthetic_video
```

Research / deployment commands added in this revision (§10–§13):

```powershell
# metric-learning fine-tuning (triplet, batch-hard mining, CosFace); ~35 min per run on this CPU
python ..\PS1_extras\scripts\train_metric.py --experiment C_ce_triplet_hard --smoke     # 3-batch setup check (~1 min)
python ..\PS1_extras\scripts\train_metric.py --experiment C_ce_triplet_hard             # -> artifacts\experiments\C_ce_triplet_hard\
python ..\PS1_extras\scripts\train_metric.py --experiment C_ce_triplet_hard --resume    # continue from last.pth
python ..\PS1_extras\scripts\extract_embeddings.py --checkpoint artifacts\experiments\C_ce_triplet_hard\best.pth --out artifacts\experiments\C_ce_triplet_hard\embeddings

# ablation: every model, same protocol -> reports\ablation\ (JSON, CSV, ROC / TAR-FAR / CMC / reliability plots)
python ..\PS1_extras\scripts\run_ablation.py

# low inter-class-variance (look-alike) evaluation -> reports\low_variance\
python ..\PS1_extras\scripts\evaluate_low_variance.py

# ONNX export + verification against PyTorch -> models\onnx\, reports\onnx\onnx_verification.json
python ..\PS1_extras\scripts\export_onnx.py
python ..\PS1_extras\scripts\benchmark_latency.py --backend onnx --out reports\benchmark\latency_onnx_cpu.json
```

Web server options: `python -m source.main --port 8080 --host 127.0.0.1 --config configs\default.yaml`
(`python source\main.py ...` works too), or `PS1_PORT` / `PS1_HOST` / `PS1_CONFIG` (e.g. `configs\profiles\cosface_d.yaml`).
The server binds to localhost only by default; do not expose it publicly (see §7).

---

## 2. Folder structure

Backend-only layout (§17): one `source/` package with the entry point, one package per model, the reusable logic,
and training/evaluation kept apart from runtime inference. The worker stays at the project root. Old-to-new mapping:
§17.2 (this layout) and §16.2 (the earlier PS2-template step).

```
D:\zed digital\ps1\
├── PS1_Download_Package.zip  archive.zip  Zed_Digital_PS1_..._Plan.pdf     original inputs (untouched)
├── PS1_source_backup_pre_restructure_20261013_1405.zip                     source backup before §16
└── PS1_Open_Set_Person_ReID\
    ├── worker.py            video job manager (one job at a time, cleanup, stale-upload sweep, shutdown) + CLI
    ├── source\
    │   ├── main.py               FastAPI entry point: start-up, /api/health, video job API, web page
    │   ├── config.py             YAML loading (+ `extends:` profiles), PS1_* env overrides, typed ReIDSettings, paths
    │   ├── dependencies.py       app state: config, the ReID engine (loaded once), the shared VideoJobs instance
    │   ├── errors.py             typed exceptions (PS1Error -> ConfigError, ArtifactError, ImageError, BackendError, ...)
    │   ├── models\               one package per model / model format
    │   │   ├── base.py               ReIDBackend interface (preprocessed batch -> L2-normalised 512-d embeddings)
    │   │   ├── reid_resnet18\        model.py (strict checkpoint loader) · backend.py (PyTorch reference backend)
    │   │   ├── onnx_runtime\         export.py (checkpoint -> ONNX) · backend.py (ONNX Runtime CUDA/DirectML/CPU)
    │   │   ├── tensorrt\             backend.py (native TensorRT) · engine_builder.py (ONNX -> engine) · gpu_runtime.py
    │   │   └── yolo_detector\        detector.py (YOLOv8 person detector)
    │   ├── logic\
    │   │   ├── model_manager.py      backend selection + verification vs PyTorch, ResilientBackend runtime fallback
    │   │   ├── preprocessing.py      verified 256x128 transform + robust image decoding
    │   │   ├── reid\                 engine.py · gallery.py · open_set.py · calibration.py · pipeline.py
    │   │   └── video\                video_tracking.py · tracker.py (ByteTrack) · track_identity.py · annotate.py
    │   │                             · frame_pipeline.py (older per-frame pipeline without tracking)
    │   ├── schemas\requests.py   validation of the video-job parameters
    │   ├── utils\                logging_config.py · device_utils.py
    │   ├── training\             data.py (Market parsing, identity roles, leakage guard) · losses.py · trainer.py
    │   └── evaluation\           metrics.py (rank-k, mAP, ROC, open-set TAR/FAR, calibration) · low_variance.py
    ├── configs\             default.yaml · profiles\cosface_d.yaml · experiments\ablation.yaml · splits\
    ├── models\              onnx\ · tensorrt\ (built per machine) · detector\yolov8n.pt · reid\ (PS2 checkpoint goes here)
    ├── webapp\              templates\ + static\ (page served by source/main.py at `/`; kept unchanged)
    ├── artifacts\           original\ (supplied, unmodified) · experiments*\ (fine-tuned checkpoints + embeddings)
    ├── data\                Market-1501-v15.09.15\ (training images; also read at start-up for the gallery check)
    └── outputs\             created at run time: video_jobs\<job id>\ (web uploads and worker.py jobs)
└── PS1_extras\              development tools, outside the backend project (§17.5)
    ├── scripts\             CLIs (§1, §16.6); _bootstrap.py finds the project (or $PS1_PROJECT_ROOT)
    ├── tests\  pytest.ini   pytest suite (§17.4)
    ├── reports\             evaluation JSON/CSV/figures, calibration\, benchmark\, test logs
    └── outputs\             earlier generated outputs (video jobs, logs, demos, synthetic test video)
```

---

## 3. What was supplied, and what was verified

| Artifact | Finding |
|---|---|
| `ps1_resnet18_3epoch_best.pth` | dict with `epoch`=1, `model_state_dict` (122 tensors), `optimizer_state_dict`, `val_accuracy`=0.9536. Standard **torchvision ResNet18** with `fc: Linear(512 → 751)`, where 751 is the number of Market-1501 training identities. Loads with `strict=True` and `weights_only=True`. |
| `query_embeddings.npz` | `features (3368, 512) float32`, `pids`, `camids`, `filenames`. L2-normalised, non-negative (post-ReLU GAP features). 750 identities, cameras 1–6. |
| `gallery_embeddings.npz` | `(13115, 512)`, same keys. This is Market-1501 `bounding_box_test` **without** junk (−1) and distractor (0000) images: 19 732 − 3 819 − 2 798 = 13 115. All 750 query identities are present. |
| Preprocessing | Not documented in the ZIP. Determined empirically: **Resize(256×128, bilinear) → ToTensor → ImageNet mean/std**, then GAP feature and L2 norm. This reproduces the stored embeddings at cosine **1.000000** (400 random images, min 0.99999988). Other candidates scored ≤ 0.9955. |
| `ps1_final_results.json` | rank-1 0.7767, mAP 0.566. **Reproduced exactly** from the embeddings. |
| `ps1_verification_results.json` | AUC 0.9901, threshold 0.6848 at FAR 0.01, TAR 0.8701. **Reproduced** (AUC 0.9900, threshold 0.6855, TAR 0.8709; pair sampling differs slightly). |
| `ps1_evaluation_results.json` | rank-1 0.8121, mAP 0.6012, threshold 0.6940. **Not reproducible** from the supplied embeddings. The file calls itself an "initial diagnostic"; it probably came from a different run or protocol. Treated as unverified. |

No retraining was done; there was no technical reason to. The checkpoint and the embeddings are
consistent with each other.

---

## 4. Method

```
image ─► [optional YOLO person detection] ─► crop ─► Resize 256×128 + ImageNet norm
      ─► ResNet18 (fc removed) ─► 512-d GAP feature ─► L2 normalise
      ─► cosine similarity to the gallery ─► per-identity score ─► open-set decision
```

* **Identity score (aggregation).** Default `mean_prototype`: cosine similarity to the L2-normalised mean
  of that identity's gallery embeddings (cached; ~4 ms search over 13 115 images). The
  alternative `max` uses the best single image. Prototype aggregation gave a higher DIR at equal
  FAR in every protocol (§5).
* **Decision rule.** Return KNOWN (top-1 identity) only if
  `top1 ≥ threshold` **and** `top1 − top2 ≥ margin`; otherwise return UNKNOWN. The reason is reported.
* **Variable gallery.** Identities can be added or removed at runtime (`Gallery.add` / `remove_identity`),
  and `build_gallery.py` builds a new gallery from folders. No classifier head is involved, so no retraining.

## 5. Threshold validation — the supplied threshold is **not** reliable for open-set use

The supplied thresholds (0.6848 / 0.6940) were chosen at 1% FAR on **independent impostor pairs**.
Open-set identification instead takes the *maximum* similarity over hundreds of identities, so an
unknown person's best score is almost always far above 0.69.

Two identity-disjoint protocols were built from the supplied embeddings (`scripts/evaluate_open_set.py`).
The model never saw any of these 750 test identities during training.
* **Protocol A:** 10 random splits into a 375-identity CALIBRATION half and a 375-identity TEST half,
  each with 50% enrolled and 50% unknown identities. Unknowns have all their gallery images removed.
* **Protocol B (deployment-sized):** 5 folds. Each has 600 enrolled and 150 unknown identities (about 10.5k
  gallery images). The threshold is calibrated on another fold's unknown identities and tested on this fold's.

Results on the held-out side, mean ± std. DIR = correct identity accepted; uFAR = unknown person accepted;
misID = enrolled person accepted as the wrong identity.

| Operating point | Protocol | threshold | DIR | uFAR | misID |
|---|---|---|---|---|---|
| **supplied 0.6848** (max) | B | 0.6848 | 0.800 | **1.000** | 0.200 |
| **supplied 0.6848** (max) | A | 0.6848 | 0.890 | **0.996** | 0.109 |
| max, @5% FAR | B | 0.884 ± 0.004 | 0.300 ± 0.033 | 0.051 ± 0.009 | 0.021 |
| prototype, @5% FAR | B | 0.884 ± 0.005 | 0.381 ± 0.045 | 0.051 ± 0.012 | 0.020 |
| **prototype + margin 0.05, @5% FAR (default)** | B | **0.856 ± 0.013** | **0.428 ± 0.044** | **0.052 ± 0.018** | **0.004** |
| prototype + margin 0.05, @1% FAR | B | 0.899 ± 0.011 | 0.210 ± 0.067 | 0.011 ± 0.009 | 0.001 |
| prototype + margin 0.05, @10% FAR | B | 0.812 ± 0.022 | 0.508 ± 0.015 | 0.100 ± 0.028 | 0.006 |
| prototype + margin 0.05, @5% FAR | A (187 IDs) | 0.848 ± 0.006 | 0.576 ± 0.037 | 0.054 ± 0.015 | 0.003 |

What the numbers show:
* With the supplied threshold, practically every unknown person is accepted as someone, so it does no
  open-set rejection at all. It is kept only for reference.
* At ≈5% unknown-FAR, the system correctly identifies only ~43% of enrolled queries (B) and rejects the rest.
  That is the honest capability of a 3-epoch ResNet18 with a classification-only loss.
* The margin rule cuts wrong-identity accepts by ~5× (0.020 → 0.004) at the same FAR.
* **The threshold depends on gallery size.** With a fixed threshold, uFAR rose from 3.1% to 8.7% as enrolled
  identities grew from 94 to 281 (Protocol A robustness check). The web UI uses all 750 identities,
  more than the 600 that were calibrated, so expect its uFAR to be **somewhat above 5%**. Re-calibrate for
  any other gallery.

Figures: `reports/figures/open_set_dir_vs_far.png`, `open_set_score_hist.png`, `pair_roc_reproduced.png`.

## 6. Web interface (FastAPI)

`python -m source.main` → http://127.0.0.1:8000

The web UI is video-only: upload a CCTV-style video, adjust the settings (each one has an explanation),
press **Process Video**, follow the progress bar, then watch the annotated video (boxes, track IDs, identity or
UNKNOWN), read the detection / tracking / timing statistics and download the processed video. See §9.3.
Single-image identification remains available from the command line (`scripts/query_image.py`).

* API: `GET /api/health` (settings, selected backend + every selection attempt with its reason, gallery consistency,
  GPU/TensorRT detection, installed backends, model files present). Interactive docs: http://127.0.0.1:8000/docs.
* Errors: unhandled server errors return `500 {"detail": "Internal server error."}` without stack traces or paths.
  An empty upload returns 400; disk errors 507. A failed job's upload and partial outputs are deleted.
* Video API: `POST /api/video/jobs` (multipart `file` + `threshold`, `margin`, `det_confidence`, `frame_stride`,
  `reid_interval`, `min_votes`, `imgsz`, `max_frames`) → job; `GET /api/video/jobs/{id}` → state/progress/summary;
  `POST /api/video/jobs/{id}/cancel`; `GET /api/video/jobs/{id}/files/{name}[?download=1]` (only that job's outputs).
  Wrong file type → 415, bad setting → 422, over `webapp.max_video_upload_mb` → 413.

## 7. Limitations (please read)

1. **Accuracy is modest.** The supplied baseline is a 3-epoch ResNet18 with a classification loss only. Closed-set
   rank-1 is 0.777 and mAP 0.566 (state-of-the-art Market-1501 models exceed 0.94 rank-1). At 5% unknown-FAR,
   DIR is ~0.43. Metric-learning fine-tuning (triplet, batch-hard mining, CosFace) is evaluated in §10–§11.
2. **Scores are cosine similarities, not probabilities.** The UI and video labels show cosine scores. Platt /
   isotonic probability calibration is evaluated in §11.3 (raw cosine read as a probability is badly
   miscalibrated), but it is not wired into the UI.
3. **Thresholds are dataset- and gallery-size-specific.** They were calibrated on Market-1501 test identities with a
   600-identity gallery. Different cameras, resolutions or gallery sizes need re-calibration with your own
   known and unknown queries (`evaluation.calibrate_threshold`).
4. **Calibration identities = Market-1501 test identities.** No separate validation identity set exists in the
   supplied artifacts. Protocols A and B keep calibration and test identities disjoint, but both come from
   the same pool, cameras and time period.
5. **No manually verified same-clothing labels exist** for Market-1501 here. §12 builds a look-alike subset with a
   model-independent clothing-colour descriptor. It is a proxy: it ignores garment type, body shape and
   accessories, and nobody verified the pairs by hand.
6. **Video has been tested only on synthetic clips** (Market crops pasted on a flat background, §9.5), not on real
   CCTV footage, and those crops come from the same cameras as the gallery. Real cameras, viewpoints, lighting
   and resolutions will score lower and need their own threshold calibration. ByteTrack uses motion/IoU only, so
   people who cross or are fully occluded can get new track IDs (fragmentation) or swap tracks. The occlusion
   reset (§9.2) stops a swapped track from carrying the old identity, but cannot prevent the swap itself.
   `scripts/process_video.py` (per-frame, no tracking) is kept unchanged for comparison.
7. **Detection** uses the generic COCO YOLOv8n (person class). It is not tuned for surveillance viewpoints, and small
   people (< 48 px tall) are ignored. Crops from a detector differ from Market's DPM crops, which may shift scores.
8. **Custom-gallery demo** used Market training identities. Their scores are optimistic because the model was trained on them.
9. **Privacy / ethics.** Person ReID is sensitive biometric-adjacent processing. Use only permitted data,
   keep the server on localhost, and do not deploy on real people without a lawful basis and notice/consent.
   Market-1501 is for research use and must not be redistributed.

## 8. Configuration reference (`configs/default.yaml`)

| key | default | meaning |
|---|---|---|
| `model.checkpoint` | `artifacts/original/ps1_resnet18_3epoch_best.pth` | supplied weights |
| `model.device` | `auto` | `auto` / `cpu` / `cuda` (env `PS1_DEVICE`) |
| `model.backend` | `auto` | automatic verified selection (§16.4). Internal (env `PS1_REID_BACKEND`): `tensorrt`, `onnx-cuda`, `onnx-dml`, `pytorch-cuda`, `onnx-cpu`, `pytorch-cpu`, or legacy `torch` / `onnx` (onnx falls back to PyTorch) |
| `model.onnx_path` | `models/onnx/ps1_resnet18_embedding.onnx` | ONNX export of `model.checkpoint`; source of the TensorRT engine |
| `model.tensorrt_dir` / `tensorrt_precision` / `tensorrt_auto_build` | `models/tensorrt` / `fp32` / `true` | engine `<onnx stem>_<precision>.engine`, built on first start (env `PS1_TENSORRT_PRECISION`) |
| `model.verify_min_cosine` | `0.9995` | a backend must reproduce the PyTorch reference to this cosine |
| `gallery.consistency_check` | `strict` | refuse to start if the gallery was not produced by `model.checkpoint` (`warn` / `off`) |
| `open_set.calibrated_for_sha256` | SHA-256 of the baseline checkpoint | the threshold is refused with any other checkpoint |
| `extends` (top level) | — | a profile YAML inherits another file and overrides only what differs (`configs/profiles/`) |
| `preprocessing.*` | 256×128, bilinear, ImageNet mean/std | verified against the stored embeddings |
| `data.gallery_embeddings` | supplied gallery `.npz` | swap for a `build_gallery.py` output |
| `gallery.aggregation` | `mean_prototype` | or `max` |
| `open_set.threshold` / `margin` | `0.856` / `0.05` | Protocol B, 5% unknown-FAR |
| `detection.yolo_weights` | `models/detector/yolov8n.pt` | auto-downloaded on first use if missing |
| `video.frame_stride` | `5` | per-frame (untracked) script: process every Nth frame |
| `video_tracking.frame_stride` | `1` | tracked pipeline: process every Nth frame (keep 1–2 for ByteTrack) |
| `video_tracking.det_confidence` / `det_low_confidence` | `0.25` / `0.1` | start/confirm a track / only extend one |
| `video_tracking.imgsz` / `output_max_side` | `640` / `1280` | YOLO input size / annotated video size |
| `video_tracking.tracker.*` | `track_buffer: 30`, `match_thresh: 0.8` | ByteTrack overrides |
| `video_tracking.identity.*` | see §9.2 | sampling interval, vote rule, crop gates, occlusion reset |
| `webapp.host/port/max_upload_mb` | `127.0.0.1` / `8000` / `15` | |
| `webapp.max_video_upload_mb` / `video_jobs_dir` | `1024` / `outputs/video_jobs` | video uploads |

ultralytics normally tries to `pip install` missing extras at runtime. `ps1_reid` disables that
(`YOLO_AUTOINSTALL=False`), so the app never modifies your Python installation.

See `reports/TEST_REPORT.md` for the full test log and measured results.

## 9. Video: detection + tracking + open-set re-identification

### 9.1 Pipeline

```
video file ─► OpenCV decode (one frame at a time, every Nth frame)
           ─► YOLOv8n person boxes (score ≥ 0.1)                       detection
           ─► ByteTrack: boxes ≥ 0.25 start/confirm tracks,           tracking  (temporary track IDs)
              0.1–0.25 boxes only extend existing tracks
           ─► for tracks that are DUE and whose crop passes the gates:  re-identification
              crop ─► same 256×128 preprocessing + ResNet18 checkpoint ─► 512-d L2 embedding (one batch per frame)
              ─► cosine to gallery prototypes ─► calibrated threshold + margin rule (unchanged) ─► one vote
           ─► track label = vote over the last 8 votes                 temporal decision
           ─► draw boxes/labels ─► H.264 mp4 (bundled ffmpeg) + CSV + JSON
```

Detection, tracking and identity are separate modules (`detector.py`, `tracker.py`, `track_identity.py`).
The track ID is **never used as an identity**. A person who leaves and returns gets a new track ID, and
is re-identified from scratch against the gallery, which is the point of ReID.

### 9.2 Temporal open-set rule (`video_tracking.identity` in the config)

* **Sampling.** A track is embedded on its first usable frame, then every `reid_interval` (3) processed
  frames while its label is unsettled, and every `stable_interval` (15) once its 8-vote window is full and
  the label has not changed for 4 votes. Stable tracks therefore cost ~5× less ReID.
* **Crop gates.** A crop is embedded only if its box is ≥ 64×24 px and has a detector score ≥ 0.4, it
  doesn't touch the frame border (person cut off), and no other person box covers more than 30% of it.
  A failed crop is retried on the next frame. Skip counts per reason are in the diagnostics.
* **Per-crop decision** = the existing `OpenSetMatcher` (threshold 0.856, margin 0.05), unchanged.
* **Track label** from the last 8 per-crop decisions:
  `PENDING` (fewer than 2 crops; drawn grey as `UNKNOWN (pending)`);
  `KNOWN P` if ≥ 2 crops were accepted as P, P has ≥ 60% of all accept votes and ≥ 25% of all votes;
  otherwise `UNKNOWN`. The label holds between samples and changes only when votes change, so it does not flicker.
* **Occlusion reset.** If another person box covers a track by > 50% (people crossing), ByteTrack may hand
  the track to the other person. When the people separate, that track's votes are discarded and it re-verifies.
  On the evaluation set this cut frames showing a *wrong enrolled identity* from 56 to 15 (2.1% → 0.56%),
  at a cost of 1.4 points of frame-level TAR (`reports/video_evaluation_ablation_no_reset.json`).
* **No gallery updates.** Unknown or unverified people are never enrolled. To enrol people, use reference
  images and `scripts/build_gallery.py` (folder per identity), then point `data.gallery_embeddings` at it.
  Recalibrate the threshold for that gallery (§5, limitation 3).

### 9.3 Using the web UI

1. `python -m source.main`, then open http://127.0.0.1:8000.
2. Choose a file (mp4/avi/mov/mkv/…). The *Original* player previews it locally. Browsers can't play AVI,
   but AVI is still processed.
3. Adjust the settings if needed, then press **Process Video**. A progress bar shows frames, fps and tracks
   so far. You can *Cancel*, which keeps partial results. Processing runs in a background thread, one video at
   a time, so the page stays usable.
4. When it finishes, the *Processed* player shows the annotated video. Below it are frames processed, detections,
   person tracks, known tracks (with identities), unknown tracks, tracks with insufficient evidence, the measured
   processing fps / real-time factor / total time, a **Download processed video** button, further download buttons (mp4, tracks CSV, ReID observations CSV,
   summary JSON), *Diagnostics*, and a per-track table.

**Reading the video.**
* Green, thick: `Track 12 | 0004 | Score: 0.91`, an accepted enrolled identity. The score is the mean
  **cosine similarity** of the supporting crops; it is not a probability or a confidence percentage.
* Red: `Track 12 | UNKNOWN`, rejected (no identity passed the threshold, margin and vote).
* Grey: `Track 12 | UNKNOWN (pending)`, not enough usable crops yet (new, small, cut off or occluded person).
* The HUD shows the frame number, time and active threshold/margin.

The page reports counts. **Counts are not accuracy.** Whether the identities are right can only be measured
against ground truth (§9.5).

### 9.4 Output files (CLI `--out` dir, or `outputs/video_jobs/<job id>/` for web uploads)

| file | content |
|---|---|
| `*_tracked.mp4` | annotated video, H.264 yuv420p, fps = source fps / stride, longer side ≤ 1280 |
| `*_tracks.csv` | one row per processed frame × visible track: box, det score, displayed status/identity/score, votes |
| `*_reid_observations.csv` | one row per embedded crop: single-crop decision, top-1 candidate, top-1/top-2 cosine, margin, reason |
| `*_tracking_summary.json` | counts, timing (fps, per stage), diagnostics, per-track summary |

Diagnostics include checkpoint, embedding dim vs gallery dim, crop size range, crop skips by reason,
*adjacent-frame track continuity* (share of tracks in frame t that still have the same ID in t+1), track lengths,
label changes, and occlusion resets. The web upload itself is deleted after processing.

### 9.5 Evaluation on labelled (synthetic) video — `reports/video_evaluation.json`

There is no labelled real CCTV footage, so ground truth comes from `make_synthetic_video.py`, which pastes Market-1501
crops onto a background: enrolled = query images, unknown = training identities, which are not in the gallery. Five clips:
the original clip (3 enrolled + 1 unknown, 60 frames, no overlap) and 4 `--crossing` clips (seeds 1–4, 4 enrolled +
2 unknown each, 200 frames, two lanes walking towards each other with occlusions). In total, 28 people (19 enrolled,
9 unknown) and 4,320
ground-truth person-frames. Metrics use IoU ≥ 0.5 matching. TAR = enrolled person shown/accepted as the correct
identity; FAR = unknown person accepted as anyone. 95% Wilson intervals are shown.

| level | TAR | FAR | wrong-identity accept |
|---|---|---|---|
| per ReID crop, at threshold 0.856 / margin 0.05 (382 known / 174 unknown crops) | 0.531 [0.481, 0.581] | 0.121 [0.080, 0.178] | 0.003 |
| displayed label per person-frame (2,663 known / 1,054 unknown, after detection) | 0.527 [0.508, 0.546] | 0.104 [0.087, 0.124] | 0.006 |
| final label per track (39 known / 18 unknown tracks) | 0.410 [0.271, 0.566] | 0.056 [0.010, 0.258] | 1 track |

Per-crop TAR at FAR (ROC: `reports/video_evaluation.roc.png`):
TAR@FAR 1% = 0.479 (threshold 0.880) · TAR@FAR 5% = 0.497 (0.874) · TAR@FAR 10% = 0.508 (0.866).
Closed-set top-1 of enrolled crops is 0.945, so most rejections of enrolled people are threshold or margin rejections,
not wrong matches.

What this shows, and what it doesn't:
* The video FAR here (≈10–12%) is **higher than the 5% target** of the image calibration. 109 of the 110
  false-accept frames, and all 21 false-accepted crops, come from **one unknown person (Market train pid 1473)
  whom the model consistently matched to gallery identity 1156**. Voting can't remove errors that repeat on every
  crop of the same person. This is the low inter-class-variance failure the problem statement is about. The other
  8 unknown people were never accepted. With only 9 unknown people, the FAR estimate is dominated by individuals:
  treat it as "unknown look-alikes do get through", not as a precise rate. For a stricter operating point use a higher
  threshold (e.g. 0.88 ≈ 1% per-crop FAR here, at TAR 0.48).
* Tracking: missed person-frames 14% (half of them in heavy occlusion), adjacent-frame continuity 0.96–1.0, and
  **55 ground-truth ID switches** for 28 people. Fully occluded synthetic people (rigid sliding crops) often come
  back with a new track ID. That costs identity evidence (more PENDING frames), but the occlusion reset keeps it
  from producing wrong labels.
* These are synthetic, same-camera-domain clips. They verify that the pipeline is functionally correct and
  measure its behaviour; they are not a CCTV accuracy benchmark. To evaluate your own footage, write a
  ground-truth JSON in the format in the `evaluate_video_tracking.py` docstring (per-frame boxes, pid,
  enrolled or not) and run the script.

**Speed (measured, CPU only, Intel 16 logical cores, no CUDA on this machine):** 4.7–7.2 processed fps at
1280×600 and 960×540 with `imgsz 640`, stride 1. That is about 0.2–0.3× real time for 25 fps video. YOLO accounts
for ~65% of the time and ReID for ~25%. Faster options: `frame_stride 2`, `imgsz 480`, a larger `reid_interval`,
or a CUDA GPU (`model.device: auto` uses it automatically; install a CUDA build of torch). GPU speed was not
measured here.

> **Superseded results (§10–§12): identity leakage, found in the 2026-10-12 audit.** The fine-tuned rows A1/B in
> `reports/ablation/` and `reports/low_variance/` were evaluated on 650 identities of which **26 had been used to
> select training epochs**. The filename parser dropped test identity 1488 (`*.jpg.jpg`), so training drew its
> selection set from 749 identities while the evaluation drew it from 750. Those folders are kept unchanged as a
> record. The leakage-free protocol and its results are in **§15** (`reports/ablation_clean_v2/`,
> `reports/low_variance_clean_v2/`). The baseline (A0) rows were never affected.

## 10. Metric learning and hard-negative mining (fine-tuning)

Code: `src/ps1_reid/losses.py`, `data.py`, `training.py`, `scripts/train_metric.py`, rows in `configs/experiments/ablation.yaml`.

* **Objective.** `ce_weight · identity loss + triplet_weight · triplet loss`, both on the unchanged ResNet18 512-d GAP
  embedding (no BNNeck or other architecture change, so the checkpoints load with the existing loader and
  preprocessing). The identity head starts from the supplied `fc` (sorted-pid label order; verified 99.5% top-1 on
  train images).
  * Triplet (anchor / positive = same identity / negative = different identity), margin 0.3 on L2-normalised
    embeddings (squared distance = 2 − 2·cos).
  * **Hard-negative mining = batch-hard:** for each anchor, the farthest positive and the *closest different identity*
    in a P×K batch (16 identities × 4 images), i.e. the look-alike. Negatives never share the anchor's identity.
    Anchors without a positive or a negative are skipped, and an all-degenerate batch gives a zero loss (unit-tested).
  * **Margin-based head (row D):** CosFace / AM-Softmax, s = 30, m = 0.25.
* **Reproducibility.** Fixed seed (Python / NumPy / torch, P×K sampler seeded per epoch), AdamW lr 1e-4, wd 5e-4,
  50-iteration warm-up and cosine decay, 15 P×K epochs (1 epoch = 46 batches), `last.pth` every epoch with optimizer,
  scheduler and RNG state (`--resume` verified), `split.json` / `config.json` / `history.json` per run.
* **Identity roles (no leakage, `assert_disjoint` is checked at start):** TRAIN = all 751 Market train identities.
  SELECTION = 100 fixed test-pool identities (`data.selection_identities`, seed 2026), used only to pick the best
  epoch, by TAR at FAR 1% on cross-camera pairs. Held-out *train* identities cannot do this: the baseline already
  trained on them and scores TAR@FAR1% = 1.0 there. The other **650** identities are calibration/test and are
  never seen by training or selection. Epoch 0 (the untouched baseline) is evaluated as well, so "no improvement" is a
  possible outcome.
* New checkpoints go to `artifacts/experiments/<row>/`; `artifacts/original` is never written.

| row | change | status | best epoch | SELECTION TAR @ FAR 1% |
|---|---|---|---|---|
| A0 | supplied baseline | — | — | 0.847 |
| A1 | + 15 epochs CE only (control for "more training") | done | 7 | 0.851 |
| B | + triplet, all valid triplets (no hard mining) | done | 12 | 0.872 |
| C | + triplet, **batch-hard** mining | **interrupted at epoch 8 of 15** (stopped for low system memory) | (7 so far) | (0.858 so far) |
| D | CosFace margin head + batch-hard triplet | **not run** | — | — |

C and D are incomplete and are **not** in the ablation tables. To finish them (about 45–55 min each on this CPU;
use `--set num_workers=2` if memory is tight):

```powershell
python ..\PS1_extras\scripts\train_metric.py --experiment C_ce_triplet_hard --resume
python ..\PS1_extras\scripts\extract_embeddings.py --checkpoint artifacts\experiments\C_ce_triplet_hard\best.pth --out artifacts\experiments\C_ce_triplet_hard\embeddings
python ..\PS1_extras\scripts\train_metric.py --experiment D_cosface_triplet_hard
python ..\PS1_extras\scripts\extract_embeddings.py --checkpoint artifacts\experiments\D_cosface_triplet_hard\best.pth --out artifacts\experiments\D_cosface_triplet_hard\embeddings
python ..\PS1_extras\scripts\run_ablation.py; python ..\PS1_extras\scripts\evaluate_low_variance.py
```

Loss weights were not swept (CE 1 : triplet 1). `--set triplet_weight=0.5 --name C_w0.5` runs a sweep point.

## 11. Ablation, open-set TAR/FAR, fixed-recall operating points and calibration — `reports/ablation/`

`python ../PS1_extras/scripts/run_ablation.py` evaluates every model with one identical protocol:
* **Identities:** the 650 evaluation ids, i.e. 750 Market test ids minus the 100 SELECTION ids. This applies to
  *every* row, including the baseline, so these numbers differ slightly from §5, which used all 750.
* **Per seed (5 seeds):** 650 ids are split into a CALIBRATION half (325) and a TEST half (325), which are
  identity-disjoint. In each half 50% of the identities are enrolled (gallery kept) and 50% are UNKNOWN (all
  their gallery images removed). Seed 0, TEST half: 162 enrolled ids, 2,954 gallery images, 744 genuine trials
  (enrolled-identity queries) and 716 impostor trials (unknown-identity queries).
* **Definitions.** TAR = genuine query accepted *with the correct identity* (open-set DIR). FAR = unknown query
  accepted as anyone. misID = enrolled query accepted as the wrong identity. Thresholds are chosen on CALIBRATION only;
  all numbers below are on TEST, mean ± std over 5 seeds. Market rule: gallery images with the query's
  pid **and** camera are ignored.
* **Fixed-TAR (target recall) point:** the highest threshold whose CALIBRATION TAR still reaches the target, which is
  the lowest FAR for that TAR (`evaluation.calibrate_threshold_for_tar`). Ties are accepted together (`>=`). If the
  target is unattainable (wrong top-1 / margin failures), the result is flagged and the threshold falls back to −1.
  Pair FAR targets with fewer than 1/FAR impostors are flagged "not resolvable".

### 11.1 Model rows (matching rule = prototype + margin 0.05)

| model | rank-1 | mAP | pair AUC | pair TAR @ FAR 1e-3 / 1e-2 | target FAR 5%: test TAR / FAR | target TAR 0.7: test TAR / FAR |
|---|---|---|---|---|---|---|
| A0 baseline | 0.791 | 0.585 | 0.9898 | 0.575 / 0.869 | 0.579 ± 0.017 / 0.038 ± 0.007 | 0.694 ± 0.022 / 0.095 ± 0.018 |
| A1 CE fine-tune | 0.795 | 0.588 | 0.9900 | 0.577 / 0.872 | 0.592 ± 0.015 / 0.039 ± 0.009 | 0.697 ± 0.031 / 0.084 ± 0.008 |
| B + triplet (all) | 0.796 | 0.599 | 0.9911 | 0.544 / 0.878 | 0.601 ± 0.025 / 0.046 ± 0.008 | 0.704 ± 0.021 / 0.092 ± 0.012 |

**Conclusion.** B improves closed-set mAP (+1.4 points), pair AUC and pair TAR at 1e-2 over both the baseline and the
CE-only control, so the triplet term does change the embedding. At the open-set operating points, though, the
differences are within seed noise: at the 5%-FAR target B's TAR is higher but so is its achieved FAR, and at TAR 0.7
the FARs overlap. Pair TAR at FAR 1e-3 got *worse* (0.575 → 0.544). **The baseline stays the deployed model**: there is
no measured evidence that justifies replacing it. Batch-hard mining (C) and CosFace (D) are still untested (§10).

### 11.2 Open-set components (rows E, F), baseline model

| rule | threshold | test TAR | test FAR | misID |
|---|---|---|---|---|
| **E: uncalibrated** supplied pair threshold 0.6848 (max) | 0.685 | 0.909 | **0.986** | 0.091 |
| calibrated @1% FAR, prototype + margin | 0.884 | 0.352 ± 0.046 | 0.013 ± 0.006 | 0.001 |
| calibrated @5% FAR, max over images | 0.858 | 0.533 ± 0.015 | 0.044 ± 0.013 | 0.013 |
| calibrated @5% FAR, mean prototype (F: aggregation) | 0.856 | 0.609 ± 0.037 | 0.051 ± 0.016 | 0.012 |
| calibrated @5% FAR, prototype + top1−top2 margin 0.05 (F: gap) | 0.850 | 0.579 ± 0.017 | 0.038 ± 0.007 | **0.001** |
| calibrated @10% FAR, prototype + margin | 0.825 | 0.675 ± 0.013 | 0.076 ± 0.018 | 0.003 |
| target TAR 0.5, prototype + margin | 0.865 | 0.494 ± 0.035 | 0.024 ± 0.004 | 0.001 |
| target TAR 0.7, prototype + margin | 0.815 | 0.694 ± 0.022 | 0.095 ± 0.018 | 0.004 |

* Calibration is what provides rejection. Without it, 98.6% of unknown people are accepted.
* Prototype aggregation beats max-over-images (TAR 0.609 vs 0.533 at about the same FAR).
* The margin (gap) rule mainly removes **wrong-identity** accepts (misID 0.012 → 0.0014). It costs a little TAR and
  lowers achieved FAR. It is kept as the default. Track-level evidence (voting across video frames) is evaluated in §9.5.
* Open-set AUROC (correct enrolled vs unknown, top-1 score): 0.845 for prototype vs 0.831 for max.
* Plots: `open_set_tar_far.png` (TAR vs FAR, all models), `matching_criteria_baseline.png`, `pair_roc.png`, `cmc.png`.

### 11.3 Probability calibration (`src/ps1_reid/calibration.py`)

Event: "the query is enrolled and the top-1 identity is correct". Features: [top-1, top-1 − top-2]. Fitted on the
CALIBRATION half and scored on the TEST half (baseline, mean of 5 seeds):

| predictor | Brier ↓ | ECE ↓ |
|---|---|---|
| raw cosine read as a probability | 0.338 | 0.378 |
| constant base rate | 0.248 | — |
| isotonic (top-1 score) | 0.110 | 0.033 |
| **Platt / logistic (top-1, gap)** | **0.104** | **0.029** |

Raw cosine is worse than a constant guess, so it must not be shown as a confidence. Platt scaling gives usable
probabilities (`reliability.png`), but they apply only to this gallery size and domain. The UI still shows cosine scores.

## 12. Low inter-class variance (look-alike) evaluation — `reports/low_variance/`

Market-1501 as supplied has **no** clothing, attribute or "same uniform" annotations, and none were invented.
`scripts/evaluate_low_variance.py` selects look-alikes with a **model-independent** clothing-colour descriptor
(`src/ps1_reid/low_variance.py`: HSV histograms of the upper and lower body, Bhattacharyya similarity, identity =
mean over its gallery images). Because the descriptor does not use the ReID model, the subset is not biased
towards or against any model.

* **Selection rule:** different-identity pairs in the top 2% of colour similarity among the 650 evaluation ids
  (similarity ≥ 0.948) → **4,219 look-alike identity pairs**; 574 of 650 ids have at least one. The most similar pairs
  are in `lookalike_examples.jpg` (e.g. white shirt + black trousers, yellow tops). Nobody checked them by hand.
* **Splits:** the same CALIBRATION / TEST identity halves and 5 seeds as §11. Look-alike pairs are formed within a half.
* **Trials per seed (TEST half, mean):** 21,568 genuine pairs (same id, different camera), 190,365 look-alike impostor
  pairs, 8,076,987 other impostor pairs.

| model | pair, threshold @ FAR 1% (all impostors): TAR | FAR look-alike | FAR other | strict threshold (FAR 1% on look-alikes): TAR | open-set @5%: FAR of unknowns *with* / *without* an enrolled look-alike |
|---|---|---|---|---|---|
| A0 baseline | 0.857 | **0.153** | 0.0065 | 0.366 | 0.049 / 0.012 |
| A1 CE fine-tune | 0.862 | 0.152 | 0.0067 | 0.383 | 0.050 / 0.011 |
| B + triplet (all) | 0.872 | 0.161 | 0.0064 | 0.378 | 0.058 / 0.015 |

* At an operating point that looks like "1% FAR", a look-alike impostor is accepted **~24× more often** (15.3%) than
  an ordinary one (0.65%). This is the failure the problem statement describes.
* Pushing the look-alike FAR down to 1% costs most of the TAR (0.857 → 0.366).
* In the open-set protocol, nearly all false accepts come from unknown people who have an enrolled look-alike. Enrolled
  people with an enrolled look-alike are accepted less often (TAR 0.53 vs 0.69).
* No model trained so far improves the look-alike FAR. Batch-hard mining (C), which targets exactly these pairs, is
  the experiment still to finish.

**Limits of Market-1501 for same-uniform surveillance:** 1,501 people in everyday clothing, filmed on one campus over a
short period. There are no uniforms and no clothing changes. Colour similarity is only a proxy for "dressed alike"
(garment type, logos, body shape and accessories are ignored). The low-variance numbers therefore describe *colour
look-alikes in Market-1501*, not a staff population in identical uniforms. That case needs its own labelled data.

## 13. Deployment: profiling, ONNX, TensorRT

Measured on this machine (Intel CPU, 16 logical cores, 12 torch threads; torch 2.14.1 **CPU build**):

| measurement (embedding model, 256×128) | PyTorch | ONNX Runtime 1.31 (CPU EP) |
|---|---|---|
| forward, batch 1, median / p95 | 10.9 / 16.2 ms | **4.9 / 7.3 ms** |
| forward, batch 32 throughput | 113 img/s | **255 img/s** |
| identify one decoded crop (preprocess + embed + 13,115-image search), median | 17.0 ms | 11.4 ms |
| batch-32 embedding incl. preprocessing | 104 img/s | 173 img/s |
| gallery search + decision, median | 3.7 ms | 3.7 ms |

Sources: `reports/onnx/onnx_verification.json`, `reports/benchmark/latency_torch_cpu.json`, `latency_onnx_cpu.json`.
Model: 11.18 M parameters, 44.7 MB fp32 (ONNX file 44.7 MB; the supplied `.pth` is 138.9 MB because it also stores
optimizer state). CUDA memory: not measurable, because the installed torch is a CPU build.

**ONNX verification (`python ../PS1_extras/scripts/export_onnx.py`) — passed:** TorchScript exporter, opset 17, dynamic batch. It
passes `onnx.checker` and loads in ONNX Runtime. Output shape is (N, 512) for N = 1 and 32. Both backends get the
same tensor from the verified transform. On 256 Market query images: max |Δ| 7.3e-6, minimum cosine between
normalised embeddings 0.99999982. Top-1 identity agreement against the gallery is **100%**, and rank-1 is
identical (0.785) for both backends.

**Using it:** set `model.backend: onnx` in `configs/default.yaml`. The web app, video pipeline and scripts then embed
through ONNX Runtime. If onnxruntime or the `.onnx` file is missing, the engine logs a warning and uses PyTorch
(unit-tested). The default stays `torch` (the verified reference).

**TensorRT — implemented and verified (2026-10-13, §16).** PS2's native TensorRT runner and engine builder are reused
(`inference/backends.py`, `inference/tensorrt_engine.py`). On this machine (NVIDIA GeForce RTX 2050, driver CUDA 12.5,
TensorRT 11.4.0.106 + cuda-python from PS2's project-local `.gpu_runtime`, CPU-only torch unchanged):

| backend (embedding model, 256×128) | forward batch 1, median / p95 | batch-32 embedding incl. preprocessing | min cosine vs PyTorch |
|---|---|---|---|
| ONNX Runtime CPU (earlier run, `latency_onnx_cpu.json`) | 5.3 / 12.5 ms | 173 img/s | 0.9999998 |
| **TensorRT FP32** (`reports/benchmark/latency_tensorrt_fp32.json`) | **3.5 / 4.2 ms** | **414 img/s** | 0.9999999 (random batches 1–32, probe crops, 32 Market crops) |
| TensorRT FP16 (built in tests) | — | — | ≥ 0.999; same top-1 identity and open-set decision on 48 Market crops |

The FP32 engine for PS1's ONNX model built in 22 s (`models/tensorrt/ps1_resnet18_embedding_fp32.engine` + `.json`).
TensorRT is used automatically when importable; without it the app runs on ONNX Runtime CPU. To enable it, either
point `PS1_GPU_RUNTIME` at an existing TensorRT install (e.g. `D:\zed digital\ps2\.gpu_runtime`) or install it into
`<project>\.gpu_runtime` as in PS2's README §5.3. `python ..\PS1_extras\scripts\build_reid_tensorrt.py` builds/validates explicitly.

**Video throughput** (§9.5) is dominated by YOLO (~65%). ReID is already batched per frame, sampled per track
(`reid_interval` / `stable_interval`) and run under `torch.inference_mode()`.

## 14. Implementation checklist (status as of 2026-10-12)

| requirement | status | where |
|---|---|---|
| Baseline ResNet-18 preserved; architecture, preprocessing and normalisation verified | **done** | §3, `verify_artifacts.py`, tests |
| Variable gallery, add/remove identities, multiple images per id, no classifier at inference | **done** | `gallery.py`, tests |
| Open-set UNKNOWN rejection: threshold + top1−top2 gap, prototype aggregation | **done** | `open_set.py`, §4–5, §11.2 |
| Threshold calibration on held-out identities, target FAR **and** target TAR; ties, boundaries, unattainable targets handled | **done** | `evaluation.py`, `open_set.py`, §11 |
| Fixed float32 tie bug in `threshold_for_far` (could overshoot the target FAR) | **done** | `open_set.py`, regression test |
| Score → probability calibration with Brier / ECE / reliability | **done** | `calibration.py`, §11.3 |
| Market-1501 rank-k / CMC / mAP / pair ROC / AUC / TAR@FAR | **done** | `evaluation.py`, `run_ablation.py` |
| Leakage separation: train / selection / calibration / test identities | **done after fix** (26-id leak found by the audit; fixed and re-evaluated; zero overlap checked in code and tests) | §15, `data.identity_roles`, `tests/test_identity_leakage.py` |
| Triplet loss, batch-hard mining, CosFace margin head, reproducible training + resume | **done** (code, unit tests, smoke run and resume verified) | `losses.py`, `training.py` |
| Ablation A0, A1, B, E, F | **done** (clean re-evaluation in §15; §11 numbers superseded for A1/B) | §15 |
| Ablation C (batch-hard) | **done**: resumed from epoch 8 after compatibility checks, 15/15 epochs | §15 |
| Ablation D (CosFace) | **done**: 15/15 epochs | §15 |
| Loss-weight sweep | **not run** (single 1:1 setting) | §10 |
| Low-variance (look-alike) protocol | **done** (colour proxy; no manual labels exist) | §12 |
| Video UI: upload, Process Video, progress, original + processed preview, MP4 download, counts, FPS | **done** (pre-existing; verified on the live server) | §6, §9 |
| Image upload / Market sample-query UI removed, backend kept | **done** (pre-existing; test `test_image_and_sample_ui_removed`) | `webapp/` |
| YOLO detection + ByteTrack + sampled track-level ReID + voting + occlusion reset | **done** (pre-existing) | §9 |
| Video ground-truth metrics | **partial**: synthetic clips only (TAR/FAR, ID switches, fragmentation); no labelled real CCTV; IDF1/HOTA not computed | §9.5 |
| Latency p50/p95/p99, throughput, model size | **done** | §13 |
| ONNX export + numerical/retrieval verification + ONNX Runtime backend with PyTorch fallback | **done** | §13 |
| TensorRT / GPU measurements | **done** (2026-10-13): TensorRT FP32/FP16 on the RTX 2050, verified vs PyTorch; ONNX Runtime CUDA / DirectML and PyTorch CUDA not installed here (skipped, not claimed) | §13, §16 |
| Restructure to the PS2 template, verified backend selection + fallback | **done** (2026-10-13) | §16 |
| Tests | **done**: 143 passed, 7 skipped (CPU-only); 147 passed, 3 skipped with TensorRT (`reports/pytest_output_2026-10-13_restructure_*.txt`) | `tests/` |

## 15. Identity-leakage fix and leakage-free re-evaluation (2026-10-12)

### 15.1 What was wrong

* **Parser.** `data.parse_market_name` only accepted `*.jpg`. Market-1501 stores every image of test identity
  **1488** (6 query + 13 gallery), plus 2 junk and 3 distractor gallery images, as `*.jpg.jpg`. These were silently
  dropped. `bounding_box_train` has no such files, so training-set membership was never affected (12,936 images,
  751 ids).
* **Selection.** Training drew its 100 SELECTION ids with `default_rng(2026).permutation(pool)[:100]` over the
  *folder* pool (749 ids, no 1488). `run_ablation.py` and `evaluate_low_variance.py` used the same draw over the
  *embedding* pool (750 ids). Different input lengths give different sets: 74/100 overlap. **26 identities used to
  choose the best epoch of A1/B/C were inside the 650 calibration/test identities** of `reports/ablation/` and
  `reports/low_variance/`. The baseline A0 has no epoch selection and was not affected.

### 15.2 Fix (changed files)

* `src/ps1_reid/data.py`
  * regex accepts one or more `.jpg` extensions (`.db`, `.mat`, `.png` and other strays are still rejected)
  * `selection_identities()` now **reads** the frozen manifest and checks its SHA-256
  * `derive_selection_identities()` keeps the original draw for provenance only
  * `calibration_test_halves()` is the single per-seed split used by every protocol
  * `identity_roles()` raises on any forbidden overlap
* `configs/splits/selection_identities_v1.json` (new) holds the exact 100 ids that A1, B and C used. It is frozen
  because those checkpoints selected epochs on exactly these ids; re-deriving over 750 would leak 26 of them. Its
  provenance block records the derivation, which a test re-runs.
  * **Dataset membership change:** pid 1488 is now parsed. It is not in SELECTION, so it is a calibration/test
    identity, as it already was in every evaluation (the supplied embeddings always contained it).
* `src/ps1_reid/training.py`
  * selection is read from the manifest and recorded in `split.json`
  * `--resume` refuses on any changed setting except `num_workers`/`name`, on different selection ids, or on a
    different training-image count
* `scripts/train_metric.py`: `--out-root` (new runs go to `artifacts/experiments_v2/`).
* `scripts/run_ablation.py` / `scripts/evaluate_low_variance.py`
  * use the manifest and the shared split
  * **refuse to run** if any model's recorded `split.json` selection differs from the manifest
  * write `identity_splits.json` (every role per seed)
  * the ablation reports `sd_sample` and a 95% t-interval (`ci95_mean`) over seeds
  * the default outputs are now `reports/ablation_clean_v2/` and `reports/low_variance_clean_v2/`
* `tests/test_identity_leakage.py` (new, 18 tests):
  * parser cases
  * folder pool == the 750 embedding ids
  * manifest provenance and tamper detection
  * every run on disk used the manifest
  * zero overlap between train / selection / calibration / test for every seed
  * thresholds unchanged when the whole TEST half is scrambled (they only use CALIBRATION)
  * resume guards

Kept unchanged:
* `artifacts/original/` (8/8 SHA-256 OK)
* `artifacts/experiments/{A1,B,C}` (the original C is hash-verified against `outputs/training_logs_v2/C_original_sha256.txt`)
* `reports/ablation/`, `reports/low_variance/` (pre-fix, leaky; kept as a record)

### 15.3 Runs

| row | run folder | status |
|---|---|---|
| A1 CE fine-tune | `artifacts/experiments/A1_ce_finetune` | unchanged checkpoint; selection == manifest |
| B + triplet (all) | `artifacts/experiments/B_ce_triplet_all` | unchanged checkpoint; selection == manifest |
| C + batch-hard | `artifacts/experiments_v2/C_ce_triplet_hard` | **resumed** from the original epoch-8 `last.pth` (copied). Config, strict model load, optimizer groups, scheduler step 368 = 8×46, train images and selection all verified first. Epochs 0–8 identical to the original history. Best epoch 12 (selection TAR@FAR1% 0.859). Epochs 9–15 ran with `num_workers=0` because of low free RAM (`resume_note.json`) |
| D CosFace + batch-hard | `artifacts/experiments_v2/D_cosface_triplet_hard` | new run, `num_workers=0`; best epoch 11 (0.855). The CosFace head (s=30, m=0.25) trained slowly from the linear head (train accuracy 35% at epoch 15) |

### 15.4 Leakage-free results — `reports/ablation_clean_v2/final/`

* **Protocol:** as in §11, but with the frozen SELECTION ids excluded for every model: 650 evaluation ids, 5 seeds of
  325 CALIBRATION / 325 TEST ids. Thresholds come from CALIBRATION only.
* **Uncertainty:** values are mean over seeds with a [95% t-interval]. The seeds re-split the same 650 ids, so the
  intervals are approximate and probably narrow.
* **Trials per seed (TEST):** 715–746 genuine (enrolled-identity) queries, 723–760 impostor (unknown-identity)
  queries, 162 enrolled ids. Calibration halves are similar (718–738 genuine / 712–738 impostor).
* **Pair verification:** 50,000 genuine + 50,000 impostor sampled pairs.
* Matching rule: prototype + margin 0.05.

| model | rank-1 / mAP | pair TAR @ FAR 1e-3 / 1e-2 | target FAR 5%: threshold, TAR, FAR | target TAR 0.7: threshold, TAR, FAR | target FAR 1%: TAR, FAR |
|---|---|---|---|---|---|
| A0 baseline | 0.790 / 0.584 | 0.559 / 0.865 | 0.848, 0.580 [0.536, 0.624], 0.043 [0.029, 0.057] | 0.811, 0.688 [0.665, 0.711], 0.101 [0.049, 0.153] | 0.333, 0.011 |
| A1 CE fine-tune | 0.793 / 0.588 | 0.571 / 0.872 | 0.843, 0.598 [0.565, 0.631], 0.043 [0.029, 0.057] | 0.811, 0.692 [0.659, 0.724], 0.093 [0.059, 0.127] | 0.355, 0.011 |
| B + triplet (all) | 0.797 / 0.598 | 0.548 / 0.878 | 0.848, 0.600 [0.575, 0.624], 0.044 [0.033, 0.055] | 0.817, 0.695 [0.668, 0.722], 0.091 [0.062, 0.119] | 0.353, 0.011 |
| C + batch-hard | 0.801 / 0.600 | 0.572 / 0.877 | 0.844, 0.598 [0.559, 0.638], 0.039 [0.028, 0.051] | 0.816, 0.694 [0.671, 0.717], 0.081 [0.054, 0.108] | 0.344, 0.008 |
| D CosFace + batch-hard | 0.802 / 0.596 | 0.576 / 0.873 | 0.832, 0.623 [0.601, 0.646], 0.046 [0.030, 0.061] | 0.804, 0.700 [0.672, 0.728], 0.089 [0.053, 0.126] | 0.337, 0.008 |

Low-variance (`reports/low_variance_clean_v2/`, threshold for FAR 1% on all impostors): the look-alike FAR is
0.147 (A0), 0.147 (A1), 0.155 (B), 0.146 (C) and 0.145 (D), against an ordinary-impostor FAR of 0.0064–0.0069.

**Conclusion (leakage-free).** The comparison is now leakage-free: zero overlap checked in code, in tests and in
`identity_splits.json`, and every result reproduces exactly from the saved artifacts. On this protocol, **no
fine-tuning variant gives a clear improvement at the open-set operating points.**
* All 95% intervals overlap the baseline's.
* At the 0.7-TAR target, FAR is 0.081–0.093 for C, D, B and A1 vs 0.101 for A0. The direction is favourable, but the
  baseline interval [0.049, 0.153] covers all of them.
* Closed-set rank-1/mAP rise by at most 0.012/0.016.
* Look-alike false accepts stay at ~15% for every model.

The baseline remains the deployed model. Caveats: one hyperparameter setting per row (no loss-weight or margin
sweep), and 15 short PK epochs from an already-trained checkpoint. D's CosFace head was still far from converged.

## 16. Restructuring to the PS2 template (2026-10-13)

> File paths in §10–§16 are historical. Current locations: §2 and the mapping in §17.2.

PS1 now follows the architecture of PS2 (*Identity-Preserving Multi-Object Tracking through Dense Occlusion*):
one entry point (`main.py`), a job manager (`worker.py`), one configuration module (`config.py`), an `inference/`
package with interchangeable, automatically selected and verified backends, and the project's own algorithm package.
**PS1's purpose and algorithm are unchanged**: L2-normalised ResNet-18 embeddings, cosine search against a variable
gallery with prototype aggregation, and the calibrated threshold + top1−top2 margin rule that returns an identity
or UNKNOWN. Safety: `git` branch `restructure-ps2-template` (baseline commit `45262bf`) and
`..\PS1_source_backup_pre_restructure_20261013_1405.zip`. Large data/artifacts were not moved or modified.

### 16.1 Inspection findings that shaped the design

| finding (verified, `reports/model_integration.json`) | consequence |
|---|---|
| PS2 `models/reid/best.pth` is **byte-identical** (SHA-256 `86859329…`) to PS1's own `artifacts/experiments_v2/D_cosface_triplet_hard/best.pth` (CosFace + batch-hard, epoch 11) | PS2's ReID model *is* a PS1 model; it loads with PS1's strict loader |
| PS2 `models/onnx/reid_resnet18.onnx` (`input`→`embedding`, opset 17) reproduces D (min cosine 0.9999999) but **not** PS1's deployed baseline A0 (mean cosine 0.9745) | different embedding space: the A0 gallery and threshold 0.856 are invalid with it |
| D queries against the A0 gallery: rank-1 0.766 vs 0.828 (A0/A0), scores shifted by ~0.03 around a 0.856 threshold | mixing silently degrades decisions → start-up guards (§16.4) |
| PS2 preprocessing (OpenCV `INTER_LINEAR` on BGR) vs PS1's verified PIL/torchvision transform: min cosine 0.99929 (64×128 crops upscaled), 0.99857 (large crops downscaled) | PS1 keeps its own transform; backends consume preprocessed tensors |
| Leakage-free calibration (`scripts/calibrate_threshold.py`, 650 ids): A0 0.861 → DIR 0.427 @ uFAR 0.053; D 0.855 → DIR 0.425 @ uFAR 0.052 | **A0 stays the default**; D (PS2's model) is an opt-in profile |
| PS2 TensorRT engines were built from PS2's ONNX (metadata `onnx_sha256 688a6dbc…`) for TensorRT 11.4 / RTX 2050 | usable only with that ONNX (D profile); PS1 builds its own engine from its ONNX |

### 16.2 Old → new file mapping

| old (PS1) | new | change |
|---|---|---|
| `webapp/app.py` + `run_webapp.py` server start | `main.py` | PS2 entry pattern: lifespan, Windows Selector loop, generic 500 handler, extended `/api/health`; video routes moved here (URLs unchanged) |
| `webapp/video_jobs.py` | `worker.py` | job manager; + stale-upload sweep, `shutdown()`, partial outputs removed on failure |
| `run_webapp.py` | `run_webapp.py` | thin launcher → `main.run()` (same CLI flags) |
| `src/ps1_reid/config.py` | `config.py` | + `extends:` profiles, `PS1_*` env overrides, `ReIDSettings`, `resolve_device`, `file_sha256` |
| `src/ps1_reid/errors.py` | `errors.py` | + `BackendError`, `NoBackendAvailable` |
| `src/ps1_reid/model_loader.py` | `inference/model_loader.py` | unchanged |
| `src/ps1_reid/preprocessing.py` | `inference/preprocessing.py` | unchanged |
| `src/ps1_reid/onnx_backend.py` | `inference/onnx_export.py` | export only; `OnnxEmbedder` replaced by `inference/backends.py::OnnxBackend` |
| `src/ps1_reid/detector.py` | `inference/detector.py` | unchanged |
| — | `inference/backends.py`, `backend_selection.py`, `tensorrt_engine.py`, `gpu_runtime.py` | from PS2 (§16.3) |
| `src/ps1_reid/reid_engine.py` | `reid/engine.py` | embeds through the selected backend; threshold/gallery guards; `preprocess()`, `backend_info`, `close()` |
| `src/ps1_reid/gallery.py` | `reid/gallery.py` | + optional provenance (`model_sha256`, `image_root`) in the `.npz` |
| `src/ps1_reid/open_set.py`, `calibration.py`, `pipeline.py` | `reid/…` | unchanged |
| `src/ps1_reid/tracker.py`, `track_identity.py`, `video_tracking.py`, `annotate.py` | `video/…` | unchanged (summary reports the ReID backend) |
| `src/ps1_reid/video.py` | `video/frame_pipeline.py` | renamed only |
| `src/ps1_reid/data.py`, `losses.py`, `training.py` | `training/data.py`, `losses.py`, `trainer.py` | manifest path depth only |
| `src/ps1_reid/evaluation.py`, `low_variance.py` | `evaluation/metrics.py`, `low_variance.py` | unchanged |
| `src/ps1_reid/__init__.py` | `inference/__init__.py`, `video/__init__.py` | `YOLO_AUTOINSTALL=False` guard kept where ultralytics is imported |
| `models/yolov8n.pt` | `models/detector/yolov8n.pt` | moved (same SHA-256 `f59b3d83…`) |
| `scripts/*.py`, `tests/*.py` | same place | imports updated; `--backend` options use the new names |

### 16.3 PS2 components reused, adapted, excluded

**Reused** (minimal diffs, PS2 attribution in the docstrings): `inference/gpu_runtime.py` (`PS1_GPU_RUNTIME`),
`inference/tensorrt_engine.py` (engine named after the ONNX file), the backend classes and the selection/verification/
`ResilientBackend` logic, the Windows event-loop guard and error handler of `main.py`, the job-lifecycle duties of
`worker.py`, `scripts/build_reid_tensorrt.py`, the mocked selection tests and the `hardware` test marker.

**Adapted for PS1**: backends take PS1-preprocessed tensors (no `preprocess_crops`); the probe batch goes through PS1's
transform; legacy names `torch`/`onnx` keep PS1's documented behaviour (ONNX falls back to PyTorch); verification
failures name the likely cause (model file exported from another checkpoint); reference embeddings are cached per
checkpoint file.

**Excluded (MOT-specific, no demonstrated need in PS1)**: `tracker/kalman_filter.py`, `association.py` (Hungarian),
`track_manager.py` (track lifecycle, occlusion handling, lost-track re-association), `tracker/reid_extractor.py`
(crop/occlusion heuristics for MOT), `inference/video_processor.py` / `video_io.py` (MOT video pipeline and MOTChallenge
output), `evaluation/evaluate_mot.py`, `trackeval_runner.py`, `simulator.py`, `stress_test.py` (MOTA/HOTA/IDF1),
`yolo11n.pt` (PS1's evaluated results use YOLOv8n; YOLO11n works via `detection.yolo_weights` but is not re-evaluated),
and the `/track` API. PS1's video path keeps its own ByteTrack + per-track open-set voting (§9); its primary metrics stay
TAR/FAR, DIR, ROC and fixed-recall thresholds (§5, §11, §12), not MOTA/HOTA.

### 16.4 Inference backends, automatic selection and calibration guards

* **Order** (first that loads *and* verifies wins): `tensorrt` → `onnx-cuda` → `onnx-dml` → `pytorch-cuda` →
  `onnx-cpu` → `pytorch-cpu`. **Verification**: a fixed probe batch (PS1 transform) at batch 1 and 4 must give finite,
  unit-norm (B, 512) embeddings, batch-consistent and within cosine 0.9995 of the PyTorch-CPU reference built from
  `model.checkpoint`. An ONNX/engine from a different checkpoint fails this (test
  `test_onnx_exported_from_another_checkpoint_is_rejected`, real models).
* **Runtime fallback**: if the active backend raises mid-job, the next candidate is verified and the batch retried;
  the event is recorded in the backend info and the video summary.
* **Calibration guards** (`gallery.consistency_check: strict`): the threshold carries
  `open_set.calibrated_for_sha256`; the gallery must carry a matching `model_sha256` (new galleries from
  `build_gallery.py` / `extract_embeddings.py`) or reproduce 8 of its stored rows when re-embedded (cosine ≥ 0.999).
  Measured: same model 1.000000 (ONNX CPU and TensorRT FP32), PS2/D model on the A0 gallery min 0.9529 → start-up refused
  with an explanation.
* **On this machine**: CPU-only environment → `onnx-cpu` (GPU candidates listed with reasons in the log and
  `/api/health`); with `PS1_GPU_RUNTIME` pointing at a TensorRT install → `tensorrt` on the RTX 2050.

### 16.5 Model files from PS2: status and verified compatibility

Expected locations (PS2 layout). **None had been copied into PS1 at the time of writing**; compatibility was verified
by reading the PS2 files in place (read-only): `python ..\PS1_extras\scripts\verify_model_integration.py --ps2 "D:\zed digital\ps2"`.

| PS2 file | copy to | actual format / content | compatible with | status in PS1 |
|---|---|---|---|---|
| `models/reid/best.pth` | `models/reid/best.pth` | PyTorch zip checkpoint; ResNet-18, fc 751×512 (CosFace), epoch 11, strict-loads | **D profile** (`configs/profiles/cosface_d.yaml`). Not the default model | not copied; byte-identical file already at `artifacts/experiments_v2/D_cosface_triplet_hard/best.pth`, which the tests use |
| `models/onnx/reid_resnet18.onnx` | `models/onnx/reid_resnet18.onnx` | ONNX opset 17, `input [batch,3,256,128]` → `embedding [batch,512]`, raw GAP features | D only (rejected with the A0 checkpoint) | not copied; optional (D profile then runs on PyTorch) |
| `models/tensorrt/reid_resnet18_fp32.engine` (+ `.json`) | `models/tensorrt/` | TensorRT 11.4 engine built from the ONNX above for the RTX 2050 | D only, same GPU/TensorRT; reused only if its metadata matches | not needed: engines are built automatically from the configured ONNX (PS1's A0 engine was built here in 22 s) |
| `models/tensorrt/reid_resnet18_fp16.*` | `models/tensorrt/` | FP16 engine + derived FP16 graph | D only | optional |
| `models/detector/yolo11n.pt` | `models/detector/` | ultralytics YOLO11n pickle | detector option | not used by default (§16.3) |

After copying: run `python ..\PS1_extras\scripts\verify_model_integration.py`. It identifies each file by content, checks every
profile and writes `reports/model_integration.json`. Use the D profile with `$env:PS1_CONFIG="configs\profiles\cosface_d.yaml"`.

### 16.6 Commands added

```powershell
python main.py                                         # server (auto backend); PS1_CONFIG / PS1_HOST / PS1_PORT
python ..\PS1_extras\scripts\verify_model_integration.py [--ps2 <PS2 folder>]   # identify + verify model files, start every profile
python ..\PS1_extras\scripts\calibrate_threshold.py [--config <profile>] [--embeddings <dir>]   # leakage-free threshold for a model
python ..\PS1_extras\scripts\build_reid_tensorrt.py [--precision fp16] [--config <profile>]     # explicit TensorRT build + validation
python ..\PS1_extras\scripts\benchmark_latency.py --backend tensorrt --out reports\benchmark\latency_tensorrt_fp32.json
$env:PS1_GPU_RUNTIME = "D:\zed digital\ps2\.gpu_runtime"   # optional: use an existing TensorRT install
```

### 16.7 Tests (`reports/pytest_output_2026-10-13_restructure_cpu.txt`, `…_tensorrt.txt`)

| environment | result |
|---|---|
| default (CPU-only torch, ONNX Runtime CPU) | **143 passed, 7 skipped** (tensorrt ×4, onnx-cuda, onnx-dml, pytorch-cuda: runtimes not installed) |
| `PS1_GPU_RUNTIME` = PS2's TensorRT 11.4 install, RTX 2050 | **147 passed, 3 skipped** (onnx-cuda, onnx-dml, pytorch-cuda) |

All 79 pre-existing tests pass unchanged in substance (imports updated; the ONNX test uses `OnnxBackend`; the web
end-to-end test now writes to a temporary folder instead of `outputs/video_jobs/`, so nothing is deselected).
New: `test_backend_selection.py` (21, **mocked**: order, fallback, rejection of wrong numerics/NaN, runtime switch,
dependency probes), `test_backend_hardware.py` (11, **real**, skipped per missing runtime: each backend vs PyTorch on
48 Market crops with identical open-set decisions, TensorRT FP16, corrupted/stale engines, mismatched ONNX rejected),
`test_model_integration.py` (11, real: discovery, profiles/env, threshold and gallery guards, D/PS2 model end-to-end,
missing files), `test_open_set_identity.py` (11: look-alike ambiguity, unknown rejection, variable gallery size,
runtime enrol/remove; data-backed bounds on the deployed operating point), `test_api.py` (14: start-up/health,
OpenAPI has no backend/device fields, routes unchanged, 400/404/415/422/500, failed-job cleanup, stale-upload sweep,
shutdown, start-up error exit code), plus 2 input-validation tests in `test_model_and_engine.py`.

Look-alike evidence (deployed A0, threshold 0.856, 600 enrolled / 150 unknown, 3 seeds): unknown people whose colour
look-alike is enrolled are accepted 15.2–16.0% of the time without the margin rule and **4.5–6.2% with margin 0.05**;
58–85% of the remaining false accepts go to that look-alike. Known DIR 0.40–0.44, misID ≤ 0.0075.

### 16.8 Remaining limitations of this step

* ONNX Runtime CUDA/DirectML and PyTorch CUDA paths are implemented and unit-tested with mocks but **not run** here
  (only CPU onnxruntime and CPU torch are installed). TensorRT was run with PS2's runtime folder; PS1 has no TensorRT
  install of its own (≈3.3 GB; install into `.gpu_runtime` or keep `PS1_GPU_RUNTIME`).
* The D-profile threshold (0.855) is calibrated on Market-1501 like the baseline; like every threshold here it is
  gallery-size and domain specific (§7).
* Engine start-up now verifies the backend and gallery (~1–3 s on CPU; the first TensorRT start builds the engine, ~22 s).
* Test-suite time: 48.6 s for the final CPU-only run (150 tests, vs 19.9 s for the 79 tests of 2026-10-12); earlier
  runs today took 66–124 s under heavier machine load, including the pre-existing evaluation tests.

---

## 17. Backend-only `source/` layout (2026-10-14)

### 17.1 What changed and why

The §16 layout spread the backend over seven top-level packages next to the entry point. It is now one `source/`
package with one sub-package per model, so each model can be loaded and tested on its own, and with the reusable
detection / ReID / video logic in `source/logic/`. No algorithm, threshold, class mapping, preprocessing, tracking
setting or API contract changed; files were moved with `git mv` (history kept) and only imports, paths and docstrings
were edited, apart from the splits listed below.

There is no "OwnerX" or "BTH" model in this project. The real models are: the ResNet-18 ReID checkpoint (PyTorch),
the same model exported to ONNX, its TensorRT engine, and the YOLOv8n person detector. Each has its own package under
`source/models/`, named after what it is.

| Model | Code | Weights / files | Configuration |
|---|---|---|---|
| ResNet-18 ReID (PyTorch, reference) | `source/models/reid_resnet18/` | `artifacts/original/ps1_resnet18_3epoch_best.pth` (default); `artifacts/experiments*/…/best.pth`; `models/reid/best.pth` (D profile, not present on this machine) | `model.*` + `preprocessing.*` in `configs/default.yaml`; `configs/profiles/cosface_d.yaml` |
| Same model, ONNX Runtime | `source/models/onnx_runtime/` | `models/onnx/ps1_resnet18_embedding.onnx` | `model.onnx_path` |
| Same model, TensorRT | `source/models/tensorrt/` | `models/tensorrt/ps1_resnet18_embedding_fp32.engine` + `.engine.json` (built per GPU / TensorRT version) | `model.tensorrt_dir`, `tensorrt_precision`, `tensorrt_auto_build` |
| YOLOv8n person detector | `source/models/yolo_detector/` | `models/detector/yolov8n.pt` | `detection.*`, `video_tracking.*` |
| Gallery + open-set threshold | `source/logic/reid/` | `artifacts/original/{query,gallery}_embeddings.npz` | `gallery.*`, `open_set.*` (threshold 0.856, margin 0.05, bound to the checkpoint's SHA-256) |

Responsibilities:

* `source/main.py`: HTTP layer only. Sets up logging, initialises the app state (config + ReID engine) once,
  validates uploads and parameters, submits jobs, serves job status/files, `/api/health`, and the page at `/`.
* `worker.py` (root): the video job manager used by the API (one job at a time on a worker thread, progress,
  cancellation, cleanup of inputs and failed outputs, stale-upload sweep, shutdown). It also runs one job standalone:
  `python worker.py --video clip.mp4`. It calls the pipeline in `source/logic/video/`; nothing is duplicated.
* `source/models/`: loading and running each model format behind one `ReIDBackend` interface (plus the detector).
* `source/logic/`: everything reusable: backend selection (`model_manager.py`, the model is loaded once at start-up
  and reused for every frame), preprocessing, the ReID engine / gallery / open-set rule, and the video pipeline
  (detection -> ByteTrack -> sampled ReID -> temporal vote -> annotated video, CSV, JSON).

### 17.2 Old → new file mapping

| Before (§16) | Now | Change |
|---|---|---|
| `main.py` | `source/main.py` | app state moved to `source/dependencies.py`, parameter checks to `source/schemas/requests.py`; `--config/--host/--port` CLI added |
| `run_webapp.py` | — (deleted) | duplicate launcher; its flags are now on `python -m source.main` |
| `config.py`, `errors.py` | `source/config.py`, `source/errors.py` | `PROJECT_ROOT` = parent of `source/`; `resolve_device` moved to `source/utils/device_utils.py` |
| `worker.py` | `worker.py` | imports updated; standalone CLI added |
| `inference/backends.py` | `source/models/base.py` + `reid_resnet18/backend.py` + `onnx_runtime/backend.py` + `tensorrt/backend.py` | split, one class per file, code unchanged; `backend_factories()` moved to `source/logic/model_manager.py` |
| `inference/backend_selection.py` | `source/logic/model_manager.py` | unchanged apart from imports |
| `inference/model_loader.py` | `source/models/reid_resnet18/model.py` | `select_device` moved to `source/utils/device_utils.py` (re-imported) |
| `inference/onnx_export.py` | `source/models/onnx_runtime/export.py` | unchanged |
| `inference/tensorrt_engine.py`, `gpu_runtime.py` | `source/models/tensorrt/engine_builder.py`, `gpu_runtime.py` | `.gpu_runtime` still resolved at the project root |
| `inference/detector.py` | `source/models/yolo_detector/detector.py` | unchanged |
| `inference/preprocessing.py` | `source/logic/preprocessing.py` | unchanged |
| `reid/*`, `video/*` | `source/logic/reid/*`, `source/logic/video/*` | unchanged apart from imports |
| `training/*`, `evaluation/*` | `source/training/*`, `source/evaluation/*` | manifest path now from `CONFIG_DIR` |
| `inference/__init__.py`, `video/__init__.py` | `source/__init__.py` | `YOLO_AUTOINSTALL=False` guard now set once for the whole package |
| — | `source/utils/logging_config.py`, `tests/test_worker_cli.py` | new |

Deleted: `run_webapp.py` and the generated `__pycache__/` and `.pytest_cache/` folders. Nothing under `data/`,
`artifacts/`, `models/`, `outputs/`, `reports/`, `configs/` or `webapp/` was deleted. Old video-job results in
`outputs/video_jobs/` and logs in `outputs/` were left in place, since they are your results rather than build output.

### 17.3 Run

```powershell
python -m pip install -r requirements.txt
python worker.py                                         # backend: http://127.0.0.1:8000, docs at /docs
python worker.py --video C:\path\clip.mp4               # one job without the server
cd ..\PS1_extras; python -m pytest -rs                    # tests (in the tools folder, §17.5)
```

### 17.4 Verification (this machine: Python 3.14.7, torch 2.14.1+cpu, no CUDA/TensorRT runtime installed)

* Test suite before the move: 143 passed, 7 skipped. After the move: 151 passed, 7 skipped
  (`reports/pytest_output_2026-10-14_source_layout.txt`; 8 new tests for the request schema and the worker CLI).
  The 7 skips are the real TensorRT / ONNX-CUDA / DirectML / PyTorch-CUDA backends, which are not installed here.
* Every `source.*` module imports; every script in `scripts/` starts (`--help`).
* `python -m source.main` started; `/api/health` returned `ok` with the `onnx-cpu` backend, 13115 gallery images and
  threshold 0.856; `/` and `/docs` returned 200; a non-video upload returned 415 and an unknown job id 404.
* `python worker.py` processed 10 frames of the synthetic video to `done`, deleted its input copy, and left
  the original video untouched (test). `scripts/query_image.py` and `scripts/verify_artifacts.py` ran
  (preprocessing still reproduces the supplied embeddings, cosine 1.000000).
* **Not verified here:** TensorRT and any GPU inference. The engine file `models/tensorrt/ps1_resnet18_embedding_fp32.engine`
  was built on 2026-10-13 for an RTX 2050 with TensorRT 11.4 (§16). It loads only on a matching GPU / TensorRT
  version; otherwise the model manager rejects it with a clear reason and uses the next verified backend.

### 17.5 Tools moved out of the project

The project folder now holds only what the backend needs at run time: `source/`, `worker.py`, `configs/`, `models/`,
`artifacts/`, `data/`, `webapp/`, `requirements.txt` and this README. Everything else was moved, not deleted, to the
sibling folder `D:\zed digital\ps1\PS1_extras\`:

| Moved | Why it is not needed at run time |
|---|---|
| `scripts/` | command-line tools for evaluation, training, export and benchmarks; the server and worker never import them |
| `tests/`, `pytest.ini` | test suite; run with `cd ..\PS1_extras; python -m pytest -rs` |
| `reports/` | evaluation results, figures and test logs |
| `outputs/` | earlier outputs: old web video jobs, logs, demo galleries, the synthetic test video |

These stay in the project because the backend reads them:

* `artifacts/`: the default checkpoint, the gallery and query embeddings, and the `cosface_d` profile's embeddings.
* `data/`: at start-up the gallery check re-embeds 8 Market-1501 gallery images. Without them the gallery is
  reported as `unverified`. These are also the training images.
* `webapp/`: the page served at `/`.

New video jobs are written to `outputs/video_jobs/` inside the project (created on first use).

`PS1_extras/scripts/_bootstrap.py` finds the project at `..\PS1_Open_Set_Person_ReID`, or at `$PS1_PROJECT_ROOT`.
The scripts still read configs, models, artifacts and data from the project. They write reports and outputs to
`PS1_extras\reports\` and `PS1_extras\outputs\`. Verified after the move, running from `PS1_extras`:

* the test suite: 151 passed, 7 skipped, the same as before;
* every script starts (`--help`), and `query_image.py` ran.

The backend was also started on its own (`python -m source.main` in the project folder): `/api/health` returned
`ok`, with the `onnx-cpu` backend and gallery check `verified`.

`PS1_extras` is outside this git repository, so git records `scripts/`, `tests/` and `reports/` as deleted here. Their
history stays in the commits before this change.
