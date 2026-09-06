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
| 4b. Screen sektor + dividen (`--screen`) | 1 |
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
Exported: model_6m_xgb.joblib, model_12m_xgb.joblib, model_risk_6m_xgb.joblib, model_risk_12m_xgb.joblib + metadata.json
Re-running costs NOTHING — every company is cached.
```

**Kalau gagal di tengah jalan:** jalankan lagi perintah yang sama. Setiap saham disimpan begitu tiba, jadi yang sudah terbeli tidak dibeli ulang.

**Kalau kredit habis:** program berhenti sebelum melewati batas, bukan setelahnya. Yang sudah terkumpul tetap tersimpan.

Kalau angkanya tidak cocok dan anda memiliki budget lebih, sesuaikan:

```bash
python train.py --budget 3000 --quarters 24 --companies 50
```

**Klasifikasi sektor dan dividen — 1 kredit.** Satu panggilan screener mengambil
sector, sub-sector, industry dan dividen trailing untuk seluruh universe (200
perusahaan sekaligus):

```bash
python train.py --screen
```

Jalankan ulang kapan pun angka dividennya perlu disegarkan; klasifikasinya tidak
basi. Tanpa ini, chip sektor dan rincian sektor pada Portfolio Analysis
tidak akan muncul.

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

Aplikasi terbuka di mode **Cached snapshot**. Tanpa API key, tanpa kredit. Pastikan **6M probability of positive return** dan **12M probability of positive return** menampilkan angka persen, bukan `—`.

Narasi di aplikasi ditulis untuk pembaca — investor dan peneliti dari yang sangat awam sampai mahir — bukan untuk pengembang. Istilah internal seperti nama endpoint, jumlah kredit di mode cached, atau nama fungsi tidak ditampilkan. Kalau Anda menambahkan teks baru, ikuti aturan yang sama.

Tiga tampilan tersedia di sidebar: **Single Stock Analysis**, **Machine Learning Screening**, dan **Portfolio Analysis**.

Di Single Stock Analysis, angka utama sekarang adalah kelas risiko dari prakiraan volatilitas — satu-satunya model yang lolos ujinya, dan tersedia untuk 6 dan 12 bulan. Probabilitas return turun ke bawah dengan label jujurnya. Machine Learning Screening memeringkat seluruh universe berdasarkan prakiraan volatilitas 6 bulan, dari yang paling tenang.

Angkanya akan berkumpul rapat di sekitar base rate historis (sekitar 50% untuk
6M, 58% untuk 12M). Itu memang disengaja — lihat bagian terakhir dokumen ini.

---

## 6. Push ke GitHub

**Ini langkah yang paling sering gagal.** `models/` dan `data/cache/` wajib ikut ter-commit — `train.py` tidak bisa jalan di Streamlit Cloud. Keduanya sengaja **tidak** masuk `.gitignore`, jadi `git add -A` sudah cukup dan `-f` tidak diperlukan.

Pertama kali saja — sambungkan ke repo GitHub yang sudah Anda buat:

```bash
git remote add origin https://github.com/USERNAME/NusaQuant.git
git push -u origin main
```

Selanjutnya cukup:

```bash
git add -A
git commit -m "pesan yang menjelaskan perubahannya"
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
| `absent` | Machine learning model tidak ada | Ulangi langkah 4 dan 6 |
| `unreadable` | Ada, tapi gagal dibaca | Versi library beda — ulangi langkah 1 lalu 4 |

---

## Yang perlu Anda tahu sebelum demo

Dengan snapshot 31 saham saat ini, **kedua horizon melaporkan "No measurable
edge"** — bukan sekadar "Weak".

| | 6M | 12M |
|---|---:|---:|
| Fold walk-forward (di-purge) | 9 | 5 |
| Baris out-of-sample | 266 | 147 |
| ROC-AUC (rata-rata di dalam fold) | 0.515 | 0.487 |
| Baseline (selalu menebak prior) | 0.500 | 0.500 |
| Mengalahkan baseline (log loss)? | ya, selisih 0.0019 | tidak |

Selisih 0.0019 itu sengaja tidak dirayakan: angkanya lebih kecil daripada
`LOG_LOSS_TIE`, ambang yang di tempat lain sudah dianggap "tidak berbeda".

Itu bukan bug, dan bukan pula sesuatu yang disembunyikan. Aplikasi:

- memberi label **No measurable edge** pada kedua horizon,
- menyusutkan probabilitas ke arah base rate historis, sehingga sebarannya
  hanya sekitar 2 poin persen (bukan 0.66 yang terdengar meyakinkan padahal
  tidak tervalidasi),
- dan pada **Machine Learning Screening** menyebut jelas kolom mana yang
  lolos uji dan kolom mana yang tidak.

**Kenapa, dan apa yang akan mengubahnya.** Target model adalah tanda dari
*absolute return*, dan dalam 6–12 bulan tanda itu sebagian besar ditentukan
arah pasar, bukan perusahaannya — base rate per kuartal di panel ini berkisar
dari 0.00 sampai 1.00.

Yang membatasi sekarang adalah **presisi pengukuran, bukan algoritmanya**.
Dengan 9 fold, standard error ROC-AUC 6M sekitar 0.043, sehingga selang 95%
di sekitar 0.515 membentang kira-kira 0.43–0.60 dan masih melewati ambang
0.55. Artinya eksperimennya belum mampu menjawab apakah edge itu ada.

Sebagai gambaran betapa tipisnya semua ini: menaikkan ambang IC dari 0.05 ke
0.06 saja menggeser skor 6M dari 0.542 ke 0.515.

Menambah jumlah saham sudah dicoba dan tidak menolong: panel tumbuh
15 → 19 → 22 → 25 → 31 perusahaan, dan ROC-AUC 6M out-of-sample bergerak
0.470, 0.483, 0.516, 0.521, 0.499. Yang menambah jumlah fold — dan karena itu
menambah presisi — adalah **riwayat kuartal yang lebih panjang**, bukan nama
yang lebih banyak:

```bash
python train.py --budget 3000 --quarters 32
```

Metodologi: validasi walk-forward yang di-purge per tanggal rebalance, audit
kebocoran data 9 poin, penyelarasan point-in-time 90 hari, pemilihan machine
learning model berdasarkan log loss out-of-sample, penyusutan probabilitas ke
base rate yang di-fit leave-one-fold-out, dan skor reliability yang menolak
memberi nilai pada model yang tidak bisa memeringkat.

Fitur: 27 metrik dalam 7 kategori. 24 dihitung point-in-time dari cache tanpa
kredit; 3 metrik dividen berasal dari screener dan **tidak pernah masuk model**
karena bukan data point-in-time. Hanya 11 rasio bebas-skala yang boleh menjadi
input model, gate missingness menyisakan 6 di antaranya, lalu **screening
information coefficient** membuang rasio yang tidak punya daya memeringkat.

Screening itu di-fit ulang di dalam setiap fold, memakai baris training saja.
Melakukan screening sekali di seluruh panel lalu menuliskan pemenangnya secara
hardcode memang menghasilkan angka yang terlihat lebih baik, tapi tidak ada
artinya: rasionya dipilih memakai return yang nanti dipakai untuk menilai model
itu sendiri. NPL, LDR dan NIM
tidak ada karena endpoint quarterly tidak mengembalikan field yang dibutuhkan.

Ketika melakukan training kembali, cache yang lama tetap dipakai — Anda hanya membayar API credit untuk saham yang baru.
