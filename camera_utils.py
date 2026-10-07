"""Helper kamera: buka kamera dengan backend yang stabil (Windows: DirectShow)
dan pindah ke kamera berikutnya saat program berjalan."""
import sys
import cv2

_API = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "default": cv2.CAP_ANY}


def default_backend():
    return "dshow" if sys.platform.startswith("win") else "default"


def open_camera(index, backend="auto", size=None):
    """Return cv2.VideoCapture yang benar-benar bisa memberi frame, atau None."""
    if backend == "auto":
        backend = default_backend()
    cap = cv2.VideoCapture(index, _API[backend])
    if size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if not cap.isOpened():
        cap.release()
        return None
    for _ in range(10):              # beberapa frame awal sering kosong
        ok, _f = cap.read()
        if ok:
            return cap
    cap.release()
    return None


def next_camera(current, backend="auto", size=None, max_index=5):
    """Coba index berikutnya (melingkar 0..max_index-1). Return (cap, index) atau (None, current)."""
    for step in range(1, max_index):
        idx = (current + step) % max_index
        cap = open_camera(idx, backend, size)
        if cap is not None:
            return cap, idx
    return None, current


def list_cameras(max_index=5, backend="auto"):
    return [i for i in range(max_index) if (c := open_camera(i, backend)) and not c.release()]
