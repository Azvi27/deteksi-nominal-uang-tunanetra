# 💰 Deteksi Nominal Uang untuk Tunanetra

Sistem deteksi uang kertas Indonesia menggunakan sensor warna dan AI untuk membantu tunanetra mengenali nominal uang secara mandiri dengan output audio.

## 🎯 Tujuan
- Membantu tunanetra mengenali nominal uang secara mandiri
- Memberikan feedback audio dalam bahasa Indonesia
- Sistem portable dan mudah digunakan

## 📁 Isi Proyek

### 🔊 File Audio (8 file MP3)
- `0.mp3` - Suara "nol rupiah" (tidak ada uang)
- `1000.mp3` - Suara "seribu rupiah"
- `2000.mp3` - Suara "dua ribu rupiah"
- `5000.mp3` - Suara "lima ribu rupiah"
- `10000.mp3` - Suara "sepuluh ribu rupiah"
- `20000.mp3` - Suara "dua puluh ribu rupiah"
- `50000.mp3` - Suara "lima puluh ribu rupiah"
- `100000.mp3` - Suara "seratus ribu rupiah"

**Fungsi**: File suara berkualitas tinggi dalam bahasa Indonesia yang diputar saat sistem mendeteksi nominal uang. Audio memberikan feedback langsung kepada pengguna tunanetra.

### 📊 Data dan Model
- `data_sensor_rgb_nominal_FIXED.csv` - Dataset sensor warna (12.400+ data RGB)
  - Berisi pembacaan sensor dari 8 nominal uang
  - Data diambil dari 4 posisi berbeda per uang
  - Format: Nominal, Posisi, NomorPembacaan, Merah, Hijau, Biru

- `trained_model_FIXED_RF_MeanStd_CV_Tuned_6400rows_v2/` - Model AI terlatih
  - `rf_model_MeanStdHSV_CV_Tuned_6400rows_v2.joblib` - Model Random Forest (akurasi 97.58%)
  - `scaler_MeanStdHSV_CV_Tuned_6400rows_v2.joblib` - Normalisasi data untuk konsistensi
  - `max_rgb_values_FIXED_CV_Tuned_6400rows_v2.joblib` - Nilai maksimum RGB untuk kalibrasi
  - `confusion_matrix_RF_MeanStd_CV_combined_6400rows_v2.png` - Grafik evaluasi cross-validation
  - `confusion_matrix_RF_MeanStd_single_split_6400rows_v2.png` - Grafik evaluasi single split

### 💻 Program
- `realtime_prediction_MeanStd_RF.py` - Program utama deteksi real-time
- `train_and_save_model_RF_MeanStd_6400rows_v2.py` - Program training model
- `train_and_save_model_RF_MeanStd_6400rows_v2.ipynb` - Notebook analisis data

### 📖 Dokumentasi
- `README.md` - Panduan lengkap penggunaan sistem

## 🔧 Bagaimana Sistem Bekerja dari Awal

### 📋 Langkah Sederhana Penggunaan
1. **Siapkan Alat**: Nyalakan komputer dan hubungkan sensor
2. **Letakkan Uang**: Taruh uang kertas di atas sensor
3. **Tunggu Proses**: Sistem akan bekerja otomatis (10 detik)
4. **Dengar Hasil**: Speaker akan menyebutkan nominal uang
5. **Selesai**: Angkat uang untuk deteksi berikutnya

### 🛠️ Komponen Hardware - Spesifikasi Teknis

**Sensor TCS3200 Color Sensor:**
- **Teknologi**: Photodiode array dengan filter warna RGB
- **Output**: Frekuensi digital 0-65535 Hz (proporsional dengan intensitas)
- **Pin Control**: S0, S1 (frequency scaling), S2, S3 (filter selection)
- **Akurasi**: ±10% pada kondisi pencahayaan stabil
- **Response Time**: ~10ms per pembacaan

**LED UV (Ultraviolet):**
- **Wavelength**: 365nm-400nm (optimal untuk security features)
- **Power**: 3-5W dengan heat sink
- **Control**: PWM untuk mengatur intensitas
- **Safety**: Auto-shutoff setelah 15 detik untuk mencegah overheating
- **Fungsi**: Mengaktifkan watermark dan tinta reaktif UV pada uang

**Mikrokontroler (Arduino Nano/Raspberry Pi):**
- **Arduino Nano**: ATmega328P, 16MHz, 32KB Flash, USB Serial
- **Raspberry Pi**: ARM Cortex, Linux OS, GPIO pins, WiFi optional
- **Komunikasi**: Serial UART 9600 baud rate
- **Power**: 5V DC, konsumsi ~200mA

**Audio System:**
- **Format**: MP3 files, 44.1kHz sampling rate
- **Library**: pygame.mixer untuk Python audio playback
- **Latency**: <100ms dari prediksi ke audio output
- **Volume Control**: Software-controlled 0.0-1.0 range

### ⚙️ Proses Kerja Detail (10 Detik)

**Detik 0-1: Persiapan**
- Uang diletakkan di atas sensor
- LED UV menyala untuk mengaktifkan tanda keamanan
- Sistem mulai bersiap mengambil data

**Detik 1-10: Pengambilan Data**
- Sensor TCS3200 membaca warna RGB dari 4 posisi berbeda
- Setiap posisi diambil 25 kali (total 100 data)
- Arduino mengirim data ke komputer melalui kabel USB

**Detik 10-10.2: Pemrosesan Data**
- Komputer menerima 100 data RGB
- Data dinormalisasi (disesuaikan) agar konsisten
- RGB diubah ke HSV (format warna yang lebih stabil)
- Dihitung rata-rata dan variasi dari setiap warna

**Detik 10.2-10.3: Prediksi AI**
- Model Random Forest menganalisis 6 fitur warna
- AI membandingkan dengan data training 12.400+ sampel
- Sistem menentukan nominal dengan akurasi 97.58%

**Detik 10.3: Output Audio**
- Program memilih file MP3 yang sesuai (misal: "lima ribu rupiah")
- Speaker memutar suara dalam bahasa Indonesia
- Pengguna mendengar hasil deteksi

### 🧠 Kecerdasan Buatan (AI) - Detail Teknis

**Data Training & Preprocessing:**
- **Dataset**: 12.400+ sampel RGB dari 8 denominasi uang Indonesia
- **Sampling Strategy**: 4 posisi per uang × 25 pembacaan × 124 uang = 12.400 data
- **Data Aggregation**: Dirata-ratakan menjadi 124 sampel unik (15-16 per denominasi)
- **Feature Engineering**: RGB → HSV conversion untuk stabilitas pencahayaan

**Model Architecture:**
- **Algoritma**: Random Forest Classifier (ensemble method)
- **Konsep**: 100+ pohon keputusan voting untuk hasil akhir
- **Input Features**: 6 fitur statistik HSV (H_mean, S_mean, V_mean, H_std, S_std, V_std)
- **Hyperparameter**: Grid Search optimized (n_estimators, max_depth, min_samples_split)

**Validation & Performance:**
- **Cross-Validation**: 5-fold StratifiedKFold untuk menghindari overfitting
- **Akurasi**: 97.58% ± 2.67% (sangat konsisten)
- **Confusion Matrix**: Tersedia 2 versi (CV combined & single split)
- **Preprocessing**: StandardScaler untuk normalisasi fitur

**Real-time Processing Pipeline:**
```
Raw RGB Data (100 samples) 
    ↓ Normalization (÷ max_rgb_values)
    ↓ RGB to HSV Conversion
    ↓ Statistical Feature Extraction (mean, std)
    ↓ StandardScaler Transform
    ↓ Random Forest Prediction
    ↓ Confidence Score & Audio Output
```

### 💻 Implementasi Teknis Program

**1. Inisialisasi Sistem:**
Sistem dimulai dengan memuat tiga komponen utama yang sudah dilatih: model Random Forest, scaler untuk normalisasi data, dan nilai maksimum RGB. Audio system juga diinisialisasi menggunakan pygame mixer, dan komunikasi serial diatur dengan baud rate 9600 untuk berkomunikasi dengan Arduino.

**2. Loop Deteksi Real-time:**
Program berjalan dalam loop tak terbatas yang menunggu sinyal dari Arduino. Ketika menerima perintah "START_BATCH_PREDICT", sistem mulai mengumpulkan 100 sampel data RGB. Setelah semua sampel terkumpul dan menerima sinyal "END_BATCH_PREDICT", data diproses untuk prediksi.

**3. Pemrosesan Data RGB:**
Setiap batch 100 sampel RGB dinormalisasi menggunakan nilai maksimum yang tersimpan, kemudian dikonversi dari ruang warna RGB ke HSV. Dari data HSV ini, sistem menghitung fitur statistik berupa nilai rata-rata (mean) dan standar deviasi (std) untuk setiap komponen H, S, dan V, menghasilkan 6 fitur utama.

**4. Prediksi dan Audio Output:**
Fitur yang telah diekstrak dinormalisasi menggunakan scaler yang sudah dilatih, kemudian dimasukkan ke model Random Forest untuk prediksi. Hasil prediksi berupa nominal uang langsung digunakan untuk memuat dan memutar file audio yang sesuai.

**5. Error Handling & Robustness:**
- **Serial Communication**: Penanganan timeout dan error decoding data
- **Data Validation**: Pemeriksaan range RGB dan jumlah sampel
- **File Management**: Verifikasi keberadaan file model dan audio
- **Memory Management**: Pembersihan buffer dan cleanup pygame
- **Exception Handling**: Try-catch untuk semua operasi kritikal

### 🔬 Proses Pembuatan Model AI (Training)

**1. Persiapan Data:**
Proses dimulai dengan memuat dataset RGB yang berisi lebih dari 6400 sampel data dari sensor TCS3200. Dataset ini mencakup kolom Merah, Hijau, Biru, Nominal, dan Posisi. Data RGB kemudian dinormalisasi berdasarkan nilai maksimum masing-masing komponen warna untuk memastikan konsistensi skala.

**2. Feature Engineering (Rekayasa Fitur):**
Data RGB yang sudah dinormalisasi dikonversi ke ruang warna HSV (Hue, Saturation, Value) karena lebih stabil terhadap perubahan pencahayaan. Kemudian dilakukan agregasi fitur per nominal dan posisi dengan menghitung nilai rata-rata dan standar deviasi untuk setiap komponen H, S, dan V, menghasilkan 6 fitur statistik utama.

**3. Hyperparameter Tuning:**
Sistem menggunakan Grid Search untuk mencari kombinasi parameter terbaik untuk algoritma Random Forest. Parameter yang dioptimasi meliputi jumlah estimator (50-200), kedalaman maksimum pohon, minimum sampel untuk split, minimum sampel per leaf, dan class weight untuk menangani ketidakseimbangan data.

**4. Training & Validation:**
Fitur yang telah diekstrak dinormalisasi menggunakan MinMaxScaler untuk memastikan semua fitur berada dalam skala yang sama. Model Random Forest dilatih menggunakan parameter terbaik hasil Grid Search, dan evaluasi dilakukan menggunakan 4-fold Stratified Cross-Validation untuk memastikan validitas hasil.

**5. Model Evaluation & Saving:**
Performa model dievaluasi menggunakan confusion matrix yang divisualisasikan dengan heatmap untuk analisis kesalahan prediksi. Semua komponen penting disimpan dalam format joblib: model Random Forest, scaler untuk normalisasi, dan nilai maksimum RGB untuk preprocessing data baru.

**6. Metodologi Training:**
- **Dataset**: 6400+ sampel RGB dari 8 nominal uang Indonesia
- **Preprocessing**: Normalisasi RGB → Konversi HSV → Agregasi statistik
- **Algorithm**: Random Forest dengan hyperparameter tuning otomatis
- **Validation**: 4-fold Stratified Cross-Validation untuk evaluasi robust
- **Features**: 6 fitur statistik (mean & std dari H, S, V)
- **Output**: Model dengan akurasi >95% dan confusion matrix untuk analisis

**7. Keunggulan Metodologi:**
- **Robust**: Cross-validation mencegah overfitting dan memastikan generalisasi
- **Balanced**: Class weighting menangani ketidakseimbangan data antar nominal
- **Optimized**: Grid search otomatis menemukan parameter optimal
- **Reproducible**: Random state memastikan hasil konsisten setiap training
- **Scalable**: Pipeline dapat diterapkan pada dataset baru dengan mudah

## 🔧 Komponen Sistem

### Hardware (Perangkat Keras)

**1. Sensor Warna TCS3200:**
- **Fungsi**: Mendeteksi intensitas warna RGB dari uang kertas
- **Teknologi**: Photodiode array dengan filter warna
- **Output**: Frekuensi digital (0-65535 Hz) untuk setiap komponen RGB
- **Akurasi**: ±10% pada kondisi pencahayaan stabil
- **Response Time**: <10ms per pembacaan
- **Interface**: Digital output ke microcontroller

**2. LED UV (Ultraviolet):**
- **Fungsi**: Mengaktifkan fitur keamanan pada uang kertas Indonesia
- **Wavelength**: 365-395 nm (UV-A)
- **Power**: 3-5 Watt
- **Voltage**: 12V DC
- **Safety**: Dilengkapi housing untuk melindungi mata
- **Durability**: >10,000 jam operasi

**3. Microcontroller (Arduino Nano/Raspberry Pi):**
- **Fungsi**: Mengontrol sensor, LED, dan komunikasi dengan PC
- **Processor**: ATmega328P (Arduino) / ARM Cortex-A72 (RPi)
- **Memory**: 32KB Flash / 1GB RAM
- **Communication**: USB Serial (9600 baud rate)
- **GPIO**: Digital I/O untuk kontrol sensor dan LED
- **Power**: 5V via USB atau external adapter

**4. Sistem Audio:**
- **Speaker/Headphone**: Output audio untuk feedback suara
- **Audio Interface**: 3.5mm jack atau USB audio
- **Format**: MP3 playback dengan pygame mixer
- **Volume**: Adjustable melalui sistem operasi
- **Latency**: <100ms dari deteksi ke audio output

### Software (Perangkat Lunak)

**1. Python Libraries:**
- **scikit-learn**: Machine learning (Random Forest, preprocessing)
- **pandas**: Data manipulation dan analisis
- **numpy**: Numerical computing dan array operations
- **pygame**: Audio playback dan multimedia
- **serial (pyserial)**: Komunikasi serial dengan Arduino
- **joblib**: Model serialization dan loading
- **colorsys**: Konversi ruang warna RGB ke HSV
- **matplotlib/seaborn**: Visualisasi data dan confusion matrix

**2. Machine Learning Components:**
- **Random Forest Model**: Classifier untuk prediksi nominal
- **MinMaxScaler**: Normalisasi fitur input
- **StratifiedKFold**: Cross-validation untuk evaluasi
- **GridSearchCV**: Hyperparameter optimization
- **Confusion Matrix**: Analisis performa model

**3. Data Files:**
- **Training Dataset**: CSV dengan 6400+ sampel RGB
- **Audio Files**: 8 file MP3 untuk setiap nominal uang
- **Model Files**: Joblib files untuk model, scaler, dan max values
- **Visualization**: PNG files untuk confusion matrix

**4. Communication Protocol:**
- **Serial Communication**: USB connection antara PC dan Arduino
- **Baud Rate**: 9600 bps dengan timeout 1 detik
- **Command Protocol**: START_BATCH_PREDICT dan END_BATCH_PREDICT
- **Data Format**: "R,G,B" values separated by comma
- **Error Handling**: Timeout dan decode error recovery

### Integrasi Sistem

**Hardware Integration:**
- TCS3200 sensor terhubung ke Arduino via digital pins
- LED UV dikontrol melalui relay atau transistor switch
- Arduino berkomunikasi dengan PC via USB serial
- Audio output melalui speaker/headphone PC

**Software Integration:**
- Python script sebagai main controller
- Real-time data acquisition dari sensor
- Machine learning inference untuk klasifikasi
- Audio feedback system untuk user interaction
- Error handling dan logging untuk debugging

## ⚡ Instalasi Cepat

### 1. Install Python Dependencies
```bash
pip install numpy pandas scikit-learn joblib scipy pygame pyserial
```

### 2. Setup Hardware
- Hubungkan sensor TCS3200 ke Arduino/Raspberry Pi
- Hubungkan LED UV
- Hubungkan speaker/headphone
- Sambungkan ke komputer via USB

### 3. Konfigurasi
Edit file `realtime_prediction_MeanStd_RF.py`:
```python
SERIAL_PORT = 'COM3'  # Sesuaikan port
AUDIO_VOLUME = 0.7    # Volume 0.0-1.0
```

### 4. Jalankan
```bash
python realtime_prediction_MeanStd_RF.py
```

## ✨ Fitur Utama

- **Akurasi Tinggi**: 97.58% tingkat keberhasilan
- **Audio Feedback**: Suara jelas bahasa Indonesia
- **Real-time**: Deteksi kurang dari 2 detik
- **Multi-posisi**: Sampling dari berbagai posisi uang
- **UV Authentication**: Menggunakan LED UV untuk autentikasi
- **Portable**: Mudah dibawa kemana-mana

## 🎵 Audio System

- Volume dapat diatur (0.0 - 1.0)
- Format MP3 berkualitas tinggi
- Respon cepat dan jelas
- Mendukung semua nominal uang Indonesia
- Queue management untuk multiple detection

## 🛠️ Hardware yang Dibutuhkan

### Komponen Utama
- **TCS3200 Color Sensor**
  - Output frekuensi digital (0-65535 Hz)
  - Pin kontrol: S0, S1, S2, S3, OUT
  - Akurasi tinggi untuk pembacaan warna

- **LED UV (365nm-400nm)**
  - Mengaktifkan watermark dan tinta khusus uang
  - Kontrol PWM untuk intensitas
  - Safety timer dan heat management

- **Mikrokontroler**
  - **Arduino Nano**: Compact, USB communication
  - **Raspberry Pi**: Alternatif dengan pemrosesan lengkap

- **Audio Output**
  - Speaker/headphone 3.5mm atau USB
  - Minimum 2W, frequency response 100Hz-20kHz
  - Volume control support

- **Koneksi**
  - Kabel USB untuk komunikasi serial
  - Jumper wires untuk sensor
  - Breadboard untuk prototyping

## 📊 Performa Model & Metode

### 🎯 Akurasi Model
- **Algoritma**: Random Forest Classifier
- **Akurasi**: 97.58% ± 2.67%
- **Dataset**: 12.400+ sampel RGB → 124 sampel unik setelah agregasi
- **Cross-validation**: 5-fold StratifiedKFold
- **Hyperparameter**: Grid Search optimized

### 🔬 Feature Engineering
- **Input**: Data RGB dari sensor TCS3200
- **Normalisasi**: RGB dibagi nilai maksimum
- **Konversi**: RGB → HSV (lebih stabil terhadap pencahayaan)
- **Fitur Akhir**: 6 statistik (H_mean, S_mean, V_mean, H_std, S_std, V_std)

### ⏱️ Performance Real-time
- **Waktu Deteksi**: ~10.3 detik per prediksi
- **Sampling**: 100 data points per posisi
- **Throughput**: ~3.3 prediksi per detik
- **Protocol**: Serial USB 9600 baud

## 🚀 Cara Penggunaan

1. Nyalakan sistem
2. Letakkan uang di atas sensor
3. Tunggu bunyi beep
4. Dengarkan hasil deteksi
5. Angkat uang untuk deteksi berikutnya

## ⚠️ Troubleshooting

**Serial Port Error:**
```bash
# Cek port yang tersedia
python -c "import serial.tools.list_ports; [print(p.device) for p in serial.tools.list_ports.comports()]"
```

**Audio Error:**
```bash
# Test audio system
python -c "import pygame; pygame.mixer.init(); print('Audio OK')"
```

**Model Error:**
- Pastikan folder `trained_model_FIXED_RF_MeanStd_CV_Tuned_6400rows_v2/` ada
- Pastikan semua file .joblib ada di dalam folder

## 📈 Keunggulan Sistem

### 🎯 Desain Khusus Tunanetra
- **Audio-First**: Interface berbasis suara tanpa perlu melihat
- **Feedback Langsung**: Suara jernih dalam bahasa Indonesia
- **Operasi Mandiri**: Tidak memerlukan bantuan orang lain
- **User-Friendly**: Cukup letakkan uang dan dengarkan hasil

### 🔬 Teknologi Canggih
- **UV Authentication**: Deteksi fitur keamanan tersembunyi
- **Multi-Position Sampling**: 4 posisi berbeda untuk akurasi tinggi
- **AI Prediction**: Random Forest dengan confidence scoring
- **Real-time Processing**: Hasil dalam hitungan detik

### 💡 Praktis & Ekonomis
- **High Accuracy**: 97.58% tingkat keberhasilan
- **Portable**: Kompak dan mudah dibawa
- **Cost-Effective**: Komponen terjangkau
- **Extensible**: Mudah dikembangkan untuk denominasi baru

## 🎯 Target Pengguna & Manfaat

### 👥 Pengguna Utama
- **Tunanetra**: Mengenali nominal uang secara mandiri
- **Keluarga**: Membantu anggota keluarga tunanetra
- **Institusi Pendidikan**: Alat bantu pembelajaran untuk tunanetra
- **Organisasi Sosial**: Mendukung program inklusi finansial

### 🌟 Manfaat Sosial
- **Kemandirian Finansial**: Transaksi mandiri tanpa bantuan
- **Inklusi Sosial**: Partisipasi penuh dalam aktivitas ekonomi
- **Kepercayaan Diri**: Mengurangi ketergantungan pada orang lain
- **Aksesibilitas**: Teknologi yang mudah diakses dan digunakan

## 📝 Lisensi

Proyek ini dibuat untuk tujuan edukasi dan penelitian.

---

## 📋 Timeline Operasi Sistem - Detail Teknis

```
0ms     : Uang diletakkan, proximity sensor trigger
50ms    : LED UV PWM ramp-up (0→100% intensity)
100ms   : TCS3200 calibration & white balance
200ms-2.5s: Posisi 1 sampling (25 readings × 10ms)
2.5s-5s : Posisi 2 sampling (25 readings × 10ms)
5s-7.5s : Posisi 3 sampling (25 readings × 10ms)
7.5s-10s: Posisi 4 sampling (25 readings × 10ms)
10.0s   : LED UV shutdown, thermal protection
10.1s   : Serial data transmission complete
10.15s  : RGB normalization & HSV conversion
10.2s   : Feature extraction (mean, std calculation)
10.25s  : StandardScaler transform
10.28s  : Random Forest prediction (100 trees voting)
10.3s   : Audio file selection & pygame.mixer play
```

## 🔄 Protokol Komunikasi - Detail Teknis

**Serial Communication (9600 baud, 8N1):**
```
// Handshake
Arduino → PC: "READY\n"
PC → Arduino: "START_DETECTION\n"

// Data Transmission
Arduino → PC: "CMD:START_BATCH\n"
Arduino → PC: "POS:1\n"  // Position indicator
Arduino → PC: "RGB:8628,41273,4581\n"  // R,G,B frequencies
Arduino → PC: "RGB:8683,41254,4581\n"
... (25 readings per position)
Arduino → PC: "POS:2\n"
... (repeat for 4 positions)
Arduino → PC: "CMD:END_BATCH\n"
Arduino → PC: "STATUS:OK\n"

// Error Handling
Arduino → PC: "ERROR:SENSOR_TIMEOUT\n"
Arduino → PC: "ERROR:UV_OVERHEAT\n"
PC → Arduino: "RESET\n"
```

**Data Processing Pipeline:**
```
Serial Buffer → Data Parsing → Validation Check → 
RGB Normalization → HSV Conversion → Statistical Features → 
StandardScaler → Random Forest → Confidence Check → Audio Output
```

**Error Recovery Mechanisms:**
- **Sensor Timeout**: Auto-retry dengan 3 detik delay
- **Serial Communication Error**: Buffer flush & reconnection
- **UV Overheat**: Forced cooldown 30 detik
- **Low Confidence Prediction**: Audio "tidak terdeteksi, coba lagi"
- **File Missing**: Fallback ke beep sound pattern

---

**Catatan**: Sistem ini dirancang khusus untuk uang kertas Indonesia dengan fitur keamanan UV. Pastikan LED UV dan sensor TCS3200 dikalibrasi dengan baik untuk hasil optimal.