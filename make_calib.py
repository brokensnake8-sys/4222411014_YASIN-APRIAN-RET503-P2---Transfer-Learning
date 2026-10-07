"""Hitung kalibrasi kamera MEMAKAI kode dosen (calibration_utils.py) lalu simpan ke calib.npz
supaya bisa dipakai capture.py / predict.py.

    python make_calib.py --images "calib_images/*.jpg"

Syarat: foto checkerboard diambil dengan KAMERA ROBOT/KAMERA PROYEKMU sendiri,
resolusi sama dengan resolusi pengambilan dataset. calibration_utils.py harus
satu folder dengan file ini (CHECKERBOARD dosen = 6x9 sudut dalam; sesuaikan di
file itu kalau papanmu beda).
"""
import argparse
import numpy as np
from calibration_utils import calibrate_images, print_calibration

ap = argparse.ArgumentParser()
ap.add_argument("--images", default="calib_images/*.jpg")
ap.add_argument("--out", default="calib.npz")
ap.add_argument("--show", action="store_true", help="tampilkan deteksi tiap gambar")
a = ap.parse_args()

r = calibrate_images(a.images, display=a.show)
print_calibration(r)
np.savez(a.out, mtx=r.camera_matrix, dist=r.distortion_coefficients,
         image_size=np.array(r.image_size), rmse=r.reprojection_rmse)
print(f"\nTersimpan: {a.out}  (resolusi {r.image_size}, RMSE {r.reprojection_rmse:.3f}px)")
if r.reprojection_rmse > 1.0:
    print("PERINGATAN: RMSE > 1 px, ambil ulang foto checkerboard yang lebih beragam.")
