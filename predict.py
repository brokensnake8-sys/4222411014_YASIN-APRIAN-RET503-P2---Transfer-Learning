"""Uji real-time dari webcam dengan penolakan benda asing (BUKAN).

    python predict.py --weights runs/feature/best.pt --mirror
Jalankan build_gallery.py dulu agar penolakan 'BUKAN' aktif.

Tombol: M mirror | C ganti kamera | [ ] ambang lebih ketat/longgar | Q keluar
"""
import argparse, json, os
import numpy as np
import cv2, torch
import torch.nn.functional as F
from common import build_model, embed, MEAN, STD, IMG_SIZE
from camera_utils import open_camera, next_camera

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="runs/feature/best.pt")
ap.add_argument("--classes", default="runs/classes.json")
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "default"])
ap.add_argument("--mirror", action="store_true")
ap.add_argument("--thr", type=float, default=0.6, help="softmax di bawah ini -> 'tidak yakin'")
ap.add_argument("--neg", nargs="*", default=["kosong", "bukan_komponen"],
                help="nama kelas yang dianggap BUKAN komponen")
ap.add_argument("--no-open-set", action="store_true")
a = ap.parse_args()

classes = json.load(open(a.classes))
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = build_model(len(classes), "scratch")
m.load_state_dict(torch.load(a.weights, map_location=dev)); m.to(dev).eval()
mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(dev)
std = torch.tensor(STD).view(1, 3, 1, 1).to(dev)

protos = thr = None
gp = os.path.join(os.path.dirname(a.weights), "gallery.npz")
if not a.no_open_set and os.path.exists(gp):
    z = np.load(gp, allow_pickle=True)
    protos = torch.tensor(z["protos"]).to(dev); thr = z["thr"]
    print("Penolakan BUKAN aktif (gallery.npz dimuat).")
else:
    print("PERINGATAN: gallery.npz tidak ada -> jalankan build_gallery.py agar benda asing ditolak.")

offset = 0.0
cam_idx, mirror = a.cam, a.mirror
cap = open_camera(cam_idx, a.backend)
if cap is None:
    cap, cam_idx = next_camera(cam_idx, a.backend)
if cap is None:
    raise SystemExit("Tidak ada kamera yang bisa dibuka (tutup aplikasi lain yang memakai webcam).")
print(f"Kamera {cam_idx}. Tombol: M mirror | C kamera | [ ] ambang | Q keluar")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)  # OpenCV = BGR!
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    with torch.no_grad():
        f = embed(m, (x - mean) / std)
        p = torch.softmax(m.classifier(f), 1)[0]
    conf, i = p.max(0); i = int(i)
    name = classes[i]
    sim = None
    if protos is not None:
        sim = float(F.normalize(f, dim=1)[0] @ protos[i])

    if name in a.neg:
        label, ok_det = f"BUKAN komponen ({name})", False
    elif sim is not None and sim < thr[i] + offset:
        label, ok_det = "BUKAN", False
    elif conf < a.thr:
        label, ok_det = "tidak yakin", False
    else:
        label, ok_det = name, True

    color = (0, 255, 0) if ok_det else (0, 0, 255)
    info = f"conf {conf:.2f}"
    if sim is not None:
        info += f" | sim {sim:.2f} (min {thr[i] + offset:.2f})"
    view = cv2.flip(frame, 1) if mirror else frame
    cv2.putText(view, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(view, info, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(view, f"cam {cam_idx} | mirror {'ON' if mirror else 'OFF'} | offset {offset:+.2f}  [M][C][ ][ ][Q]",
                (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imshow("predict", view)

    k = cv2.waitKey(1) & 0xFF
    if k in (ord("m"), ord("M")):
        mirror = not mirror
    elif k in (ord("c"), ord("C")):
        new_cap, new_idx = next_camera(cam_idx, a.backend)
        if new_cap is not None:
            cap.release(); cap, cam_idx = new_cap, new_idx
    elif k == ord("]"):
        offset -= 0.02          # lebih longgar (lebih mudah menerima)
    elif k == ord("["):
        offset += 0.02          # lebih ketat (lebih mudah menolak)
    elif k in (ord("q"), ord("Q"), 27):
        break
cap.release(); cv2.destroyAllWindows()