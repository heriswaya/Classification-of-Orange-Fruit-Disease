# Citrus Leaf Disease Classification using Deep Learning

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk melakukan **klasifikasi gambar daun/buah jeruk** ke dalam 3 kategori penyakit:
- **Citrus Canker**
- **Healthy**
- **Melanose**

Model dibangun menggunakan pendekatan **Transfer Learning dengan InceptionV4** dan diimplementasikan menggunakan **TensorFlow dan Keras**. Augmentasi gambar juga diterapkan untuk meningkatkan jumlah data dan performa model.

---

## 🗂️ Struktur Dataset
Dataset terdiri dari 3 folder utama untuk masing-masing kelas:
- `dataset/citrus_canker/`
- `dataset/healthy/`
- `dataset/melanose/`

Masing-masing folder berisi gambar asli dan gambar hasil augmentasi, dengan total data setimbang antar kelas.

---

## ⚙️ Data Augmentasi
Dilakukan augmentasi terhadap gambar menggunakan transformasi acak seperti:
- Rotasi
- Flip horizontal/vertikal
- Zoom
- Brightness dan Contrast adjustment
- Blur
- Sheared
- Warp Shift

Jumlah augmentasi:
- **Melanose**: 900 gambar
- **Healthy**: 500 gambar
- **Citrus Canker**: 500 gambar

---

## 🧠 Model Arsitektur

Model menggunakan arsitektur **Sequential** dengan basis **InceptionV3**:

```python
model = Sequential([
    tf.keras.layers.Input(shape=(224,224,3)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),  # Now shape=(112,112,64)
    tf.keras.layers.Conv2D(3, (1,1)),  # Reduce to 3 channels (shape=112,112,3)
    tf.keras.layers.UpSampling2D(size=(2,2)),  # Resize to (224,224,3)
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])
```

Model dikompilasi dengan:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

---

## 📊 Hasil Evaluasi Model

Model diuji pada 1.575 gambar (525 per kelas) dan menghasilkan performa sebagai berikut:

### 📈 Classification Report:

| Class           | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Citrus Canker  | 0.9483    | 0.9429 | 0.9456   | 525     |
| Healthy        | 0.9544    | 0.9562 | 0.9553   | 525     |
| Melanose       | 0.9564    | 0.9600 | 0.9582   | 525     |

**Akurasi total**: **95.30%**  
**Macro Average**: Precision/Recall/F1 = **0.9530**  
**Weighted Average**: **0.9530**

**Akurasi Test: 96.00%**
**Akurasi Train: 95.04%**
---

## 🖼️ Contoh Hasil Inferensi

Berikut adalah salah satu contoh hasil prediksi:

![Prediction](ae939b5a-203c-48ae-b178-d75444da2c2c.png)

> **Prediction: Melanose (90.84%)**

Gambar diunggah dan diproses dengan model SavedModel (`.pb` format) menggunakan Google Colab.

---

## 💻 Teknologi yang Digunakan
- Python
- TensorFlow + Keras
- OpenCV & skimage
- Google Colab

---

## ✅ Kriteria Pemenuhan Proyek
- ✅ Model menggunakan arsitektur **Sequential**
- ✅ Menggunakan layer `Conv2D` dan pooling melalui **base InceptionV4**
- ✅ Data augmentasi dilakukan
- ✅ SavedModel digunakan untuk **inference**
- ✅ Terdapat **classification report** dan **screenshot hasil prediksi**

---

## 📁 File Terkait
- `model/saved_model/`: folder berisi model dengan tiga format yang ditentukan
- `dataset-final/`: folder berisi gambar asli dan augmentasi

---

## 🙌 Kontributor
Heriswaya – Mahasiswa Matematika UIN Sunan Kalijaga

---
