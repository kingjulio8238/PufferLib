"""ctypes bindings for libpuffer_verify."""

import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path

_VERIFY_DIR = Path(__file__).parent

def _lib_name():
    if platform.system() == "Darwin":
        return "libpuffer_verify.dylib"
    return "libpuffer_verify.so"

def _build_lib():
    """Build the shared library if it doesn't exist."""
    lib_path = _VERIFY_DIR / _lib_name()
    if lib_path.exists():
        return lib_path
    # Auto-build
    try:
        result = subprocess.run(["make", "all"], cwd=str(_VERIFY_DIR),
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Build failed (exit {result.returncode}).\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}\n"
                f"Run 'make' in {_VERIFY_DIR} manually."
            )
    except FileNotFoundError:
        raise RuntimeError(
            f"'make' not found. Build the library manually:\n"
            f"  cd {_VERIFY_DIR} && cc -O2 -shared -fPIC -o {_lib_name()} math_verify.c -lm"
        )
    if not lib_path.exists():
        raise RuntimeError(
            f"Build succeeded but {lib_path} not found. Check Makefile output."
        )
    return lib_path

class _VerifyResult(ctypes.Structure):
    _fields_ = [
        ("response", ctypes.c_char_p),
        ("ground_truth", ctypes.c_char_p),
        ("reward", ctypes.c_float),
        ("parsed_value", ctypes.c_double),
        ("expected_value", ctypes.c_double),
        ("parse_success", ctypes.c_int),
    ]

_lib = None

def _get_lib():
    global _lib
    if _lib is not None:
        return _lib
    lib_path = _build_lib()
    _lib = ctypes.CDLL(str(lib_path))

    # verify_math
    _lib.verify_math.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]
    _lib.verify_math.restype = _VerifyResult

    # verify_math_batch
    _lib.verify_math_batch.argtypes = [
        ctypes.POINTER(_VerifyResult), ctypes.c_int, ctypes.c_double
    ]
    _lib.verify_math_batch.restype = None

    # extract_number
    _lib.extract_number.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
    _lib.extract_number.restype = ctypes.c_int

    # verify_math_bulk
    _lib.verify_math_bulk.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_double
    ]
    _lib.verify_math_bulk.restype = None

    return _lib

def verify_single(response: str, ground_truth: str, tolerance: float = 1e-5) -> _VerifyResult:
    if not isinstance(response, str) or not isinstance(ground_truth, str):
        raise TypeError("response and ground_truth must be strings")
    lib = _get_lib()
    return lib.verify_math(
        response.encode("utf-8"),
        ground_truth.encode("utf-8"),
        ctypes.c_double(tolerance),
    )

def verify_batch(responses: list[str], ground_truths: list[str],
                 tolerance: float = 1e-5) -> list[_VerifyResult]:
    n = len(responses)
    if len(ground_truths) != n:
        raise ValueError("responses and ground_truths must have the same length")

    lib = _get_lib()
    ResultArray = _VerifyResult * n
    results = ResultArray()

    # Encode strings and keep references alive
    resp_bufs = [r.encode("utf-8") for r in responses]
    gt_bufs = [g.encode("utf-8") for g in ground_truths]

    for i in range(n):
        results[i].response = resp_bufs[i]
        results[i].ground_truth = gt_bufs[i]

    lib.verify_math_batch(results, n, ctypes.c_double(tolerance))
    return list(results)

def extract_number(text: str) -> tuple[bool, float]:
    lib = _get_lib()
    out = ctypes.c_double()
    ok = lib.extract_number(text.encode("utf-8"), ctypes.byref(out))
    return bool(ok), out.value


def verify_bulk(responses: list[str], ground_truths: list[str],
                tolerance: float = 1e-5) -> list[float]:
    """Bulk verification with minimal Python↔C overhead.
    Packs all strings into contiguous buffers before a single C call."""
    n = len(responses)
    if len(ground_truths) != n:
        raise ValueError("responses and ground_truths must have the same length")
    if n == 0:
        return []

    lib = _get_lib()

    # Pack responses into one buffer with null terminators
    resp_parts = [r.encode("utf-8") + b"\x00" for r in responses]
    gt_parts = [g.encode("utf-8") + b"\x00" for g in ground_truths]

    resp_buf = b"".join(resp_parts)
    gt_buf = b"".join(gt_parts)

    # Compute offsets
    IntArray = ctypes.c_int * n
    resp_offsets = IntArray()
    gt_offsets = IntArray()

    off = 0
    for i, part in enumerate(resp_parts):
        resp_offsets[i] = off
        off += len(part)

    off = 0
    for i, part in enumerate(gt_parts):
        gt_offsets[i] = off
        off += len(part)

    # Output array
    FloatArray = ctypes.c_float * n
    rewards = FloatArray()

    lib.verify_math_bulk(resp_buf, gt_buf, resp_offsets, gt_offsets,
                         rewards, n, ctypes.c_double(tolerance))

    return list(rewards)
