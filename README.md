# Topsus
Projek topik khusus robotika dan AI Kelompok 11 dengan tema SDG 4 : Quality Education

## Deskripsi

Aplikasi ini membaca teks dari kamera, memperbaiki kesalahan
hasil OCR secara otomatis, lalu mengubah teks tersebut menjadi suara (audio) menggunakan Piper TTS. 
Tujuannya adalah membantu proses belajar/membaca untuk siswa tunanetra, misalnya untuk membacakan teks dari buku. 
Mendukung akses pendidikan yang lebih inklusif (SDG 4).

## Fitur

- Live preview kamera dengan kotak deteksi teks (bounding box) dan hasil
OCR yang ditampilkan langsung di atas video.
- OCR menggunakan PaddleOCR,
dengan penyaringan berdasarkan skor keyakinan (confidence threshold) dan
penyusunan ulang teks sesuai urutan baris/baca.
- Koreksi teks otomatis (TextCorrector) menggunakan Rapidfuzz.
- Text-to-Speech menggunakan Pipertts. 

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/Egghead27/Topik-Khusus-Robotika-dan-Kecerdasan-Buatan.git
   cd Topik-Khusus-Robotika-dan-Kecerdasan-Buatan
```

### 2. Install Dependency

```bash
pip install -r requirements.txt
```

## Cara Menjalankan 

```bash
python main.py
```
Pastikan jendela preview kamera aktif.

### Kontrol
| Key | Fungsi |
|-----|--------|
|**SPACE** / **ENTER**| Membaca teks yang berada dikamera (capture)|
|**C**|Toggle Text Correction|
|**Q**| Keluar |
