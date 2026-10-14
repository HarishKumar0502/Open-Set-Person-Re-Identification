from __future__ import annotations

import logging
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

from source.config import PROJECT_ROOT

log = logging.getLogger("ps1.gpu")


def gpu_runtime_dir() -> Path:
    return Path(os.environ.get("PS1_GPU_RUNTIME", PROJECT_ROOT / ".gpu_runtime"))


def ensure_gpu_runtime_on_path() -> Optional[Path]:
    d = gpu_runtime_dir()
    if d.is_dir() and str(d) not in sys.path:
        sys.path.append(str(d))
    return d if d.is_dir() else None


class CudaError(RuntimeError):
    pass


def _driver():
    ensure_gpu_runtime_on_path()
    try:
        from cuda.bindings import driver as cu
    except ImportError as exc:
        raise CudaError("cuda-python (cuda.bindings) is not installed.") from exc
    return cu


def check(result):
    cu = _driver()
    err, *rest = result if isinstance(result, tuple) else (result,)
    if err != cu.CUresult.CUDA_SUCCESS:
        _, msg = cu.cuGetErrorString(err)
        raise CudaError(f"{err.name}: {msg.decode() if isinstance(msg, bytes) else msg}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else tuple(rest)


@lru_cache(maxsize=1)
def probe_cuda() -> dict:
    try:
        cu = _driver()
        check(cu.cuInit(0))
        if check(cu.cuDeviceGetCount()) < 1:
            return {"available": False, "reason": "No CUDA device found."}
        dev = check(cu.cuDeviceGet(0))
        name = check(cu.cuDeviceGetName(128, dev)).split(b"\0")[0].decode()
        attr = cu.CUdevice_attribute
        major = check(cu.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev))
        minor = check(cu.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev))
        ver = check(cu.cuDriverGetVersion())
        return {"available": True, "name": name, "compute_capability": f"{major}.{minor}",
                "total_mem_mb": int(check(cu.cuDeviceTotalMem(dev)) // 2**20),
                "driver_cuda": f"{ver // 1000}.{(ver % 1000) // 10}"}
    except CudaError as exc:
        return {"available": False, "reason": str(exc)}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def tensorrt_version() -> Optional[str]:
    ensure_gpu_runtime_on_path()
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return None


class CudaDevice:
    _lock = threading.Lock()
    _ctx = None

    def __init__(self):
        cu = _driver()
        with CudaDevice._lock:
            if CudaDevice._ctx is None:
                check(cu.cuInit(0))
                dev = check(cu.cuDeviceGet(0))
                CudaDevice._ctx = check(cu.cuDevicePrimaryCtxRetain(dev))
        self.cu = cu
        self.ctx = CudaDevice._ctx

    def activate(self):
        check(self.cu.cuCtxSetCurrent(self.ctx))

    def malloc(self, nbytes: int) -> int:
        self.activate()
        return int(check(self.cu.cuMemAlloc(nbytes)))

    def free(self, ptr: int):
        if ptr:
            self.activate()
            check(self.cu.cuMemFree(ptr))

    def stream(self) -> int:
        self.activate()
        return int(check(self.cu.cuStreamCreate(0)))

    def destroy_stream(self, stream: int):
        if stream:
            self.activate()
            check(self.cu.cuStreamDestroy(stream))

    def htod(self, dst: int, host, stream: int):
        check(self.cu.cuMemcpyHtoDAsync(dst, host.ctypes.data, host.nbytes, stream))

    def dtoh(self, host, src: int, stream: int):
        check(self.cu.cuMemcpyDtoHAsync(host.ctypes.data, src, host.nbytes, stream))

    def sync(self, stream: int):
        check(self.cu.cuStreamSynchronize(stream))

    def mem_info(self) -> dict:
        self.activate()
        free, total = check(self.cu.cuMemGetInfo())
        return {"free_mb": int(free // 2**20), "total_mb": int(total // 2**20)}
