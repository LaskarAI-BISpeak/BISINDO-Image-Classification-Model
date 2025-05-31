# BISINDO-Image-Classification-Model


Repositori ini berisi pengembangan model klasifikasi gambar untuk Bahasa Isyarat Indonesia (BISINDO). Tujuan utama dari proyek ini adalah untuk membangun sebuah model yang akurat dan efisien dalam mengenali berbagai isyarat BISINDO dari input gambar.

## Status Saat Ini

Perlu diketahui bahwa model-model yang terdapat dalam repositori ini **masih bersifat sementara dan dalam tahap pengembangan awal**. Beberapa arsitektur seperti MobileNetV2, dan EfficientNet sedang dieksplorasi, sebagaimana terlihat dari beberapa file berikut:

* **`BiSpeak Model CNN MobileNetv2`**: Direktori atau kumpulan file yang berkaitan dengan eksperimen awal menggunakan arsitektur CNN dasar dan MobileNetV2.
* **`EfficientNet.ipynb`**: Notebook Jupyter yang berisi proses pengembangan, pelatihan, dan evaluasi model menggunakan arsitektur EfficientNet.
* **`bisindo_efficientnet_initial_best.weights.h5`**: File ini menyimpan bobot (weights) dari model EfficientNet yang menunjukkan performa terbaik pada tahap pelatihan awal. Ini adalah titik awal yang akan kami gunakan untuk iterasi dan perbaikan selanjutnya.

## Rencana Pengembangan Selanjutnya

Model-model yang ada saat ini, termasuk bobot `bisindo_efficientnet_initial_best.weights.h5`, **akan melalui sesi pelatihan lebih lanjut**. Kami berencana untuk:

* Melakukan augmentasi data yang lebih ekstensif.
* Melakukan fine-tuning pada hyperparameter.
* Mengeksplorasi teknik regularisasi untuk mencegah overfitting.
* Melakukan evaluasi yang lebih komprehensif pada dataset yang lebih beragam.

Tujuannya adalah untuk secara signifikan meningkatkan akurasi, robustisitas, dan kemampuan generalisasi model.


---
