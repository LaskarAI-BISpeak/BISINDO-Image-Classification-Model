# BISINDO Image Classification Model 🇮🇩🤟

Proyek ini bertujuan untuk mengembangkan sistem klasifikasi gambar menggunakan Machine Learning untuk mendeteksi isyarat Bahasa Isyarat Indonesia (BISINDO). Sistem ini dibangun dengan Python dan Streamlit, dan mampu mengenali isyarat satu tangan maupun dua tangan berdasarkan dataset gambar yang dikumpulkan secara manual.

## 📁 Struktur Proyek

```
├── app.py                      # Aplikasi utama Streamlit
├── collect_images.py           # Script untuk mengumpulkan data gambar
├── create_dataset_1_hand.py    # Buat dataset untuk isyarat satu tangan
├── create_dataset_2_hand.py    # Buat dataset untuk isyarat dua tangan
├── inference.py                # Proses inferensi model
├── model_1_hand.p              # Model hasil training untuk satu tangan
├── model_2_hand.p              # Model hasil training untuk dua tangan
├── train_classifier.py         # Script pelatihan model klasifikasi
├── requirements.txt            # Daftar dependensi
├── packages.txt                # Alternatif list packages
└── streamlit/                  # Folder tambahan untuk file Streamlit
```

## 🚀 Cara Menjalankan

1. **Clone repository**:
   ```bash
   git clone https://github.com/LaskarAI-BISpeak/BISINDO-Image-Classification-Model.git
   cd BISINDO-Image-Classification-Model
   ```

2. **Buat virtual environment dan aktifkan**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # untuk Mac/Linux
   venv\Scripts\activate      # untuk Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi Streamlit**:
   ```bash
   streamlit run app.py
   ```

## 🧠 Teknologi yang Digunakan

- Python
- OpenCV
- Scikit-learn
- Streamlit
- Pickle

## 📸 Cara Menggunakan

1. Jalankan `collect_images.py` untuk mengumpulkan gambar tangan.
2. Gunakan `create_dataset_1_hand.py` atau `create_dataset_2_hand.py` untuk membuat dataset dari gambar.
3. Latih model menggunakan `train_classifier.py`.
4. Jalankan `app.py` untuk mengakses aplikasi klasifikasi melalui browser.

## 💡 Tujuan

Meningkatkan aksesibilitas komunikasi untuk teman tuli dengan menyediakan sistem pendeteksi BISINDO berbasis AI yang ringan dan mudah digunakan.

## 🤝 Kontribusi

Pull request sangat terbuka! Untuk perubahan besar, silakan buka issue terlebih dahulu untuk mendiskusikan apa yang ingin Anda ubah.
