"""Ambil citra dari webcam (opsional undistort pakai calib.npz).

Pemakaian:
    python capture.py arduino_uno terang
    python capture.py arduino_uno terang --cam 1 --mirror

Tombol:  SPASI = simpan | M = mirror on/off | C = ganti kamera | Q = keluar
Catatan: mirror HANYA mengubah tampilan di layar; file yang disimpan tetap
         gambar asli (tidak dibalik).
Hasil :  dataset_raw/<kelas>/<kelas>_<tanggal>_lab_<kondisi>_<nnn>.png
         + baris baru di dataset_raw/metadata.csv
"""
import argparse, csv, os
from datetime import datetime
import cv2
import numpy as np
from camera_utils import open_camera, next_camera

ap = argparse.ArgumentParser()
ap.add_argument("kelas")
ap.add_argument("kondisi", help="mis. terang, redup, jendela, bayangan")
ap.add_argument("--cam", type=int, default=0, help="nomor kamera (0,1,2,...)")
ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "default"])
ap.add_argument("--mirror", action="store_true", help="tampilan preview dibalik (seperti cermin)")
ap.add_argument("--calib", default="calib.npz")
ap.add_argument("--out", default="dataset_raw")
a = ap.parse_args()

# --- kalibrasi (opsional) ---
K = D = None
calib_size = None
if os.path.exists(a.calib):
    z = np.load(a.calib)
    for kk, dd in (("mtx", "dist"), ("K", "D"), ("camera_matrix", "dist_coeffs")):
        if kk in z and dd in z:
            K, D = z[kk], z[dd]
            if "image_size" in z:
                calib_size = tuple(int(v) for v in z["image_size"])
            break
print("Undistort:", "aktif" if K is not None else "TIDAK (calib.npz tidak ditemukan)")

os.makedirs(f"{a.out}/{a.kelas}", exist_ok=True)
meta = f"{a.out}/metadata.csv"
new = not os.path.exists(meta)
tanggal = datetime.now().strftime("%Y%m%d")
# lanjutkan dari nomor terbesar yang ada (aman walau ada foto yang pernah dihapus)
import re
_nums = [int(m.group(1)) for f in os.listdir(f"{a.out}/{a.kelas}")
         if f"_{tanggal}_lab_{a.kondisi}_" in f
         and (m := re.search(r"_(\d+)\.png$", f))]
n = max(_nums, default=0)

cam_idx, mirror = a.cam, a.mirror
cap = open_camera(cam_idx, a.backend, calib_size)
if cap is None:
    print(f"Kamera {cam_idx} tidak bisa dibuka, mencari kamera lain...")
    cap, cam_idx = next_camera(cam_idx, a.backend, calib_size)
if cap is None:
    raise SystemExit("Tidak ada kamera yang bisa dibuka. Tutup aplikasi lain yang memakai "
                     "webcam (Zoom, Teams, Camera, browser) lalu coba lagi. "
                     "Cek juga Settings > Privacy > Camera.")
print(f"Memakai kamera {cam_idx}. Tombol: SPASI simpan | M mirror | C ganti kamera | Q keluar")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Gagal membaca frame, mencoba kamera lain...")
        cap.release()
        cap, cam_idx = next_camera(cam_idx, a.backend, calib_size)
        if cap is None:
            break
        continue
    if calib_size and (frame.shape[1], frame.shape[0]) != calib_size:
        raise SystemExit(f"Resolusi kamera {frame.shape[1]}x{frame.shape[0]} != kalibrasi "
                         f"{calib_size}. Kalibrasi ulang atau samakan resolusi.")
    if K is not None:
        frame = cv2.undistort(frame, K, D)

    view = cv2.flip(frame, 1) if mirror else frame.copy()   # mirror hanya untuk tampilan
    cv2.putText(view, f"{a.kelas} | {a.kondisi} | tersimpan: {n}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(view, f"cam {cam_idx} | mirror {'ON' if mirror else 'OFF'}  "
                      "[SPASI]simpan [M]mirror [C]kamera [Q]keluar", (10, view.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imshow("capture", view)

    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        n += 1
        nama = f"{a.kelas}_{tanggal}_lab_{a.kondisi}_{n:03d}.png"
        cv2.imwrite(f"{a.out}/{a.kelas}/{nama}", frame)        # gambar asli, tidak di-mirror
        with open(meta, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["nama_file", "kelas", "tanggal", "kondisi_cahaya"])
                new = False
            w.writerow([nama, a.kelas, tanggal, a.kondisi])
        print("simpan", nama)
    elif k in (ord("m"), ord("M")):
        mirror = not mirror
    elif k in (ord("c"), ord("C")):
        new_cap, new_idx = next_camera(cam_idx, a.backend, calib_size)
        if new_cap is not None:
            cap.release()
            cap, cam_idx = new_cap, new_idx
            print("Pindah ke kamera", cam_idx)
        else:
            print("Tidak ada kamera lain yang tersedia")
    elif k in (ord("q"), ord("Q"), 27):
        break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()