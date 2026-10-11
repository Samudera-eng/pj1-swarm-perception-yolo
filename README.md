# PJ1 — Sistem Persepsi Swarm Drone: Deteksi Drone dan Burung dengan YOLO

RET503 · Praktik terpadu minggu 4-7 · NIM 4222411047 · Rianda Galang Samudera

## 1. Lingkup persepsi
Modul persepsi swarm drone: mendeteksi **drone lain** dan membedakannya dari **burung** pada gambar kamera onboard.
Keluaran (kotak, kelas, confidence) menjadi masukan modul penjaga jarak/penghindar tabrakan, bukan kontrol langsung.
Kelas: `0 = drone`, `1 = bird`.

## 2. Target kinerja
Target di bawah adalah usulan dan perlu dikonfirmasi kelompok.

| Metrik | Target | Hasil val (E3) | Hasil test (E3) |
|---|---|---|---|
| mAP50 | ≥ 0,85 | 0,929 | 0,927 |
| Recall (penghindar tabrakan butuh recall tinggi) | ≥ 0,80 | 0,843 | tidak dilaporkan program 06 |
| Latensi inferensi | ≤ 10 ms/gambar di GPU uji | 4,5 ms | - |
| FPS di komputer onboard | belum ditetapkan | belum diukur | - |

Latensi diukur di GPU laptop (RTX 5050), bukan di komputer onboard drone.

## 3. Rancangan pipeline
Kamera onboard (RGB) → resize (416 atau 640) → YOLO26n → kotak + kelas + confidence → saring confidence → modul swarm.
Posisi kamera diasumsikan menghadap depan. Komputer onboard belum ditentukan.

## 4. Dataset (Tahap 1-2)
- Sumber: Bird vs Drone (Kaggle `stealthknight/bird-vs-drone`), gambar frame video 640×640.
- Label asli semuanya id 0 dan sebagian berupa polygon. Kelas diambil dari awalan nama file (`D*` = drone, `B*` = bird),
  polygon diubah ke kotak, kotak < 8 px dibuang (`siapkan_bird_drone.py`).
- Diambil 300 gambar acak per kelas dari split latih asli, lalu dibagi 70/20/10 oleh `05_split_dataset_yolo.py`:
  **420 train / 120 val / 60 test** (val: 384 kotak). Sidik jari label: `02E19071FB74`.
- Anotasi diperiksa dengan `01_cek_anotasi.py` (gambar berkotak di `dataset_yolo_bvd/cek_visual/`).

## 5. Eksperimen (Tahap 3)
Pengaturan sama untuk semua: `yolo26n`, 30 epoch, batch 16, seed 0, optimizer otomatis Ultralytics.

| | Model | Perubahan |
|---|---|---|
| E1 | `yolo26n.pt` pretrained COCO | baseline, imgsz 416 |
| E2 | `yolo26n.yaml` bobot acak | pembanding tanpa transfer |
| E3 | `yolo26n.pt` | satu perubahan: imgsz 640 |

## 6. Hasil (Tahap 4-5)
Model terbaik dipilih dengan mAP50-95 pada val, lalu diuji sekali pada test.

| Eksperimen | mAP50 | mAP50-95 | P | R | ms/gambar | menit latih |
|---|---|---|---|---|---|---|
| E1 pretrained | 0,911 | 0,610 | 0,873 | 0,842 | 2,7 | 1,8 |
| E2 scratch | 0,315 | 0,124 | 0,401 | 0,407 | 1,8 | 2,0 |
| E3 imgsz=640 | 0,929 | 0,624 | 0,903 | 0,843 | 4,5 | 2,8 |

Model terpilih: **E3**. Hasil test (sekali): **mAP50 0,927, mAP50-95 0,593**.
Berkas: `hasil_praktik_4222411047.txt/.csv`, `E1_results.png`, `E3_results.png`.

## 7. Analisis
- **Transfer learning:** E1 mencapai mAP50-95 0,610 dibanding 0,124 pada E2 (sekitar 5 kali), dengan waktu latih sama.
  Dengan 420 gambar latih, bobot pretrained COCO sangat menentukan.
- **Resolusi:** E3 (640) sedikit lebih tinggi dari E1 (+0,014 mAP50-95), dengan inferensi 4,5 vs 2,7 ms dan latih 2,8 vs 1,8 menit.
  Selisih itu kecil dibanding ukuran val (120 gambar, satu seed), jadi hanya disimpulkan sebagai kecenderungan.
- **Val dan test:** nilai test mendekati val, tetapi test hanya 60 gambar dan pembagian acak per gambar dapat membuat nilai terlalu optimis.
- **Latensi E2** yang lebih rendah kemungkinan karena sedikit deteksi, bukan model yang lebih efisien.
- **Waktu tidak stabil antar run:** pada dua kali run, semua metrik akurasi sama persis (seed tetap), tetapi ms/gambar bergeser
  (E1 3,0→2,7; E2 1,2→1,8; E3 4,1→4,5) dan waktu latih ikut berubah. Selisih latensi kecil tidak dijadikan kesimpulan.

## 8. Batasan
- Dataset publik, belum ada data dari kamera swarm sendiri. Domain: gambar siang hari dari video publik.
- Program 05 membagi acak per gambar dari frame video yang saling mirip, sehingga ada risiko kebocoran antar split.
- Satu seed per eksperimen; val 120 gambar dan test 60 gambar membuat selisih kecil tidak dapat diandalkan.
- Latensi diukur di GPU laptop, bukan di komputer onboard.

## 9. Reproduksi
```bash
python siapkan_bird_drone.py --src <folder Dataset Kaggle> --out dataset_yolo_bvd --per-kelas 300
python 01_cek_anotasi.py dataset_yolo_bvd
python 05_split_dataset_yolo.py dataset_yolo_bvd dataset_v1_bvd
python 06_latih_bandingkan.py --nim 4222411047 --nama "Rianda Galang Samudera" --data dataset_v1_bvd/data.yaml
```
