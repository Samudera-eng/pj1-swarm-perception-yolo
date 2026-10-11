"""Ubah dataset Bird vs Drone (Kaggle) menjadi dataset_yolo/ yang dipahami program 01-06 dosen.

Kenapa perlu: label asli semuanya id 0 (kelas ada di AWALAN NAMA FILE: B* = bird, D* = drone),
sebagian baris berupa polygon (>5 angka), dan nama file memakai spasi/kurung. Program 01 hanya
menerima baris 5 angka (class_id cx cy w h), dan classes.txt harus ada.

Keluaran : dataset_yolo/{classes.txt, images/, labels/}   (urutan kelas: 0 = drone, 1 = bird)
Pemakaian: python siapkan_bird_drone.py --src ~/Downloads/archive/Dataset --per-kelas 150
Lanjutan : python 01_cek_anotasi.py dataset_yolo

CATATAN: ini untuk UJI ALUR (smoke test) dan data tambahan. Gambarnya frame video yang saling mirip,
jadi split acak 05 akan bocor (val/test terlalu bagus). Data uji utama harus dari kamera drone sendiri.
"""
import argparse
import random
import re
import shutil
from pathlib import Path

from PIL import Image

KELAS = ["drone", "bird"]                    # class_id 0, 1
AWALAN = {"D": 0, "B": 1}
MIN_PX = 8                                   # sama dengan MIN_SISI_PX di 01_cek_anotasi.py

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="folder berisi train/valid/test (mis. ~/Downloads/archive/Dataset)")
ap.add_argument("--out", default="dataset_yolo")
ap.add_argument("--per-kelas", type=int, default=150, help="jumlah gambar per kelas")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed)

src, out = Path(args.src).expanduser(), Path(args.out)
if not (src / "train" / "images").exists():
    raise SystemExit(f"{src}/train/images tidak ada. Arahkan --src ke folder yang berisi train/valid/test.")


def baca_kotak(lbl: Path, W, H):
    """Baris bbox (5 angka) atau polygon (>5 angka) -> list (cx, cy, w, h) ternormalisasi, terpotong 0-1."""
    kotak = []
    for baris in lbl.read_text().splitlines():
        p = baris.split()
        try:
            v = [float(t) for t in p[1:]]
        except ValueError:
            continue
        if len(v) == 4:
            cx, cy, w, h = v
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        elif len(v) >= 6:
            v = v[: len(v) // 2 * 2]
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        x1, y1, x2, y2 = (min(max(t, 0.0), 1.0) for t in (x1, y1, x2, y2))
        w, h = x2 - x1, y2 - y1
        if w * W < MIN_PX or h * H < MIN_PX:
            continue
        kotak.append(((x1 + x2) / 2, (y1 + y2) / 2, w, h))
    return kotak


# kumpulkan kandidat dari split train asli saja (mengurangi kembaran antar split asli)
kandidat = {0: [], 1: []}
for img in sorted((src / "train" / "images").iterdir()):
    if img.suffix.lower() in (".jpg", ".jpeg", ".png") and img.stem[:1].upper() in AWALAN:
        kandidat[AWALAN[img.stem[:1].upper()]].append(img)

if out.exists():
    shutil.rmtree(out)
(out / "images").mkdir(parents=True)
(out / "labels").mkdir(parents=True)
(out / "classes.txt").write_text("\n".join(KELAS) + "\n")

hitung_gambar, hitung_kotak = [0, 0], [0, 0]
for cid, daftar in kandidat.items():
    random.shuffle(daftar)
    for img in daftar:
        if hitung_gambar[cid] >= args.per_kelas:
            break
        lbl = src / "train" / "labels" / (img.stem + ".txt")
        if not lbl.exists():
            continue
        W, H = Image.open(img).size
        kotak = baca_kotak(lbl, W, H)
        if not kotak:
            continue
        nama = re.sub(r"[^A-Za-z0-9]+", "_", img.stem).strip("_")      # "BTR (1000)" -> "BTR_1000"
        shutil.copy(img, out / "images" / f"{nama}.jpg")
        (out / "labels" / f"{nama}.txt").write_text(
            "\n".join(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in kotak) + "\n")
        hitung_gambar[cid] += 1
        hitung_kotak[cid] += len(kotak)

for i, k in enumerate(KELAS):
    print(f"{i} = {k:6s}: {hitung_gambar[i]} gambar, {hitung_kotak[i]} kotak")
print(f"Siap di {out}/ -> lanjutkan: python 01_cek_anotasi.py {out}")
