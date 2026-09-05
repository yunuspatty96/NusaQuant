# DEPLOY.md — NusaQuant © 2026 Patty Kyoudai

Ikuti berurutan untuk proses deployment.

**Total biaya API: ~495 kredit secara default (akan bertambah tergantung penyesuaian), biaya API di-charge sekali saja.** Setelah itu semuanya gratis.

---

## Ringkasan biaya

| Langkah | Kredit |
|---|---:|
| 1. Install | 0 |
| 2. Set API key | 0 |
| 3. Cek rencana (`--dry-run`) | 0 |
| 4. **Training** | **~495** |
| 5. Tes lokal | 0 |
| 6. Push ke GitHub | 0 |
| 7. Deploy Streamlit | 0 |
| Training ulang kapan pun (`--offline`) | 0 |

Kredit hanya terpakai lagi kalau pengguna memilih **Live Sectors API** di aplikasi (~9 kredit per saham).

---

## 1. Install

```bash
cd NusaQuant
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Jangan lewatkan. Model harus dilatih dengan versi library yang sama seperti saat disajikan.

---

## 2. Set API key di terminal

```bash
export SECTORS_API_KEY=kunci-anda-di-sini        # macOS / Linux
```
```powershell
$env:SECTORS_API_KEY='kunci-anda-di-sini'        # Windows PowerShell
```

PENTING: Jangan memasukan API Key di file kode!

---

## 3. Cek rencana dulu — gratis

```bash
python train.py --dry-run
```

Contoh keluaran:

```
Budget              : 600 credits (100 reserved for the dashboard)
Cost per company    : 16 quarters + 22 price = 38 credits
Universe size       : 13 companies
ESTIMATED SPEND     : ~495 credits
Expected dataset    : ~117 rows
```

---

## 4. Training — ini satu-satunya yang memakan kredit

```bash
python train.py
```

Sekitar 5–10 menit. Tunggu sampai muncul:

```
Exported: model_6m_xgb.joblib, model_12m_xgb.joblib + metadata.json
Re-running costs NOTHING — every company is cached.
```

**Kalau gagal di tengah jalan:** jalankan lagi perintah yang sama. Setiap saham disimpan begitu tiba, jadi yang sudah terbeli tidak dibeli ulang.

**Kalau kredit habis:** program berhenti sebelum melewati batas, bukan setelahnya. Yang sudah terkumpul tetap tersimpan.

Kalau angkanya tidak cocok dan anda memiliki budget lebih, sesuaikan:

```bash
python train.py --budget 3000 --quarters 24 --companies 50
```

**Melatih ulang tanpa kredit dan tanpa API key.** Setelah `data/cache/` terisi,
model bisa dilatih ulang sepenuhnya dari cache — tidak ada permintaan yang
keluar dari mesin Anda:

```bash
python train.py --offline
```

Gunakan ini setiap kali Anda mengubah fitur, kandidat model, atau protokol
validasi. Tidak perlu `SECTORS_API_KEY`.

---

## 5. Tes lokal — gratis

```bash
streamlit run app.py
```

Aplikasi terbuka di mode **Cached snapshot**. Tanpa API key, tanpa kredit. Pastikan **6M probability** dan **12M probability** menampilkan angka persen, bukan `—`.

Angkanya akan berkumpul rapat di sekitar base rate historis (sekitar 50% untuk
6M, 58% untuk 12M). Itu memang disengaja — lihat bagian terakhir dokumen ini.

---

## 6. Push ke GitHub

**Ini langkah yang paling sering gagal.** `models/` dan `data/cache/` wajib ikut ter-commit — `train.py` tidak bisa jalan di Streamlit Cloud.

```bash
git add -f models/ data/cache/
git add app.py nusaquant.py train.py requirements.txt .gitignore .streamlit/
git commit -m "NusaQuant: models and snapshot"
git push
```

Lalu **buka repo Anda di browser** dan pastikan:

- `models/` berisi 3 file (`model_6m_xgb.joblib`, `model_12m_xgb.joblib`, `metadata.json`)
- `data/cache/` berisi file `.parquet`

Kalau folder kosong di GitHub, aplikasinya akan kosong juga. Ulangi langkah ini.

---

## 7. Deploy

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. **New app** → pilih repo Anda
3. Main file: `app.py`
4. **Deploy**

Tunggu 2–5 menit. **Jangan** isi apa pun di menu Secrets — tidak diperlukan.

---

## Selesai

Aplikasi jalan di mode cache. Bisa juga menjalankan mode live Sectors API.

---

## Kalau probabilitas menampilkan "—"

Buka panel **Diagnostics** di aplikasi. Ia menyebutkan penyebabnya:

| Diagnosis | Artinya | Perbaikan |
|---|---|---|
| `absent` | Model tidak ada | Ulangi langkah 4 dan 6 |
| `unreadable` | Ada, tapi gagal dibaca | Versi library beda — ulangi langkah 1 lalu 4 |

---

## Yang perlu Anda tahu sebelum demo

Dengan snapshot 15 saham saat ini, **kedua horizon melaporkan "No measurable
edge"** — bukan sekadar "Weak".

| | 6M | 12M |
|---|---:|---:|
| Fold walk-forward (di-purge) | 9 | 5 |
| Baris out-of-sample | 135 | 75 |
| ROC-AUC (rata-rata di dalam fold) | 0.470 | 0.478 |
| Baseline (selalu menebak prior) | 0.500 | 0.500 |
| Mengalahkan baseline? | tidak | tidak |

Itu bukan bug, dan bukan pula sesuatu yang disembunyikan. Aplikasi:

- memberi label **No measurable edge** pada kedua horizon,
- menyusutkan probabilitas ke arah base rate historis, sehingga sebarannya
  hanya sekitar 2 poin persen (bukan 0.66 yang terdengar meyakinkan padahal
  tidak tervalidasi),
- dan memberi peringatan di tab **Best 10** bahwa urutannya bukan bukti.

**Kenapa, dan apa yang akan mengubahnya.** Target model adalah tanda dari
*absolute return*, dan dalam 6–12 bulan tanda itu sebagian besar ditentukan
arah pasar, bukan perusahaannya — base rate per kuartal di panel ini berkisar
dari 0.00 sampai 1.00. Selain itu, 15 saham berarti setiap cross-section
kuartalan hanya selebar 15 titik. Yang membatasi adalah **lebar universe**,
bukan algoritmanya. Kalau Anda punya kredit lebih, tambah jumlah saham lebih
dulu — itu jauh lebih berpengaruh daripada menambah riwayat untuk 15 nama yang
sama:

```bash
python train.py --budget 3000 --companies 50
```

Metodologi: validasi walk-forward yang di-purge per tanggal rebalance, audit
kebocoran data 9 poin, penyelarasan point-in-time 90 hari, pemilihan model
berdasarkan log loss out-of-sample, penyusutan probabilitas ke base rate yang
di-fit leave-one-fold-out, dan skor reliability yang menolak memberi nilai pada
model yang tidak bisa memeringkat.

Ketika melakukan training kembali, cache yang lama tetap dipakai — Anda hanya membayar API credit untuk saham yang baru.
