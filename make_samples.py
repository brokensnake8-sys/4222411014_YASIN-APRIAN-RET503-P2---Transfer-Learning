"""Buat satu gambar kolase berisi 1 foto contoh per kelas (untuk dokumen desain).

    python make_samples.py
Keluaran: docs/contoh_kelas.png
"""
import glob, os, random
import cv2
import numpy as np

random.seed(1)
RAW, W, H, COLS = "dataset_raw", 320, 240, 3

classes = sorted(d for d in os.listdir(RAW) if os.path.isdir(f"{RAW}/{d}"))
if not classes:
    raise SystemExit("Folder dataset_raw/<kelas> tidak ditemukan.")

tiles = []
for c in classes:
    files = sorted(glob.glob(f"{RAW}/{c}/*terang*.png")) or sorted(glob.glob(f"{RAW}/{c}/*.png"))
    if not files:
        continue
    img = cv2.resize(cv2.imread(random.choice(files)), (W, H))
    cv2.rectangle(img, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(img, c, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    tiles.append(img)

while len(tiles) % COLS:
    tiles.append(np.full((H, W, 3), 255, np.uint8))
sheet = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
os.makedirs("docs", exist_ok=True)
cv2.imwrite("docs/contoh_kelas.png", sheet)
print("Tersimpan: docs/contoh_kelas.png  (kelas:", ", ".join(classes) + ")")
