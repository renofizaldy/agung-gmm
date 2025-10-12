Install library untuk raw dan gui:
`pip3 install opencv-python-headless scikit-learn matplotlib`

Install library untuk extract_fitur:
`pip3 install pandas scikit-image opencv-python-headless scikit-learn matplotlib`

Jalankan aplikasi:
`python3 gui.py`

---

Skrip GMM-EM tersebut sangat sesuai sebagai langkah awal yang fundamental dalam proyek Anda untuk mengidentifikasi tingkat kepadatan tulang.

Namun, penting untuk memahami perannya. Skrip yang kita buat BUKAN alat diagnosis akhir, melainkan alat analisis citra yang kuat untuk mengekstrak informasi penting dari gambar.

## Peran Skrip GMM-EM dalam Tujuan Anda

Pikirkan skrip GMM-EM ini sebagai "mata" dari sistem Anda. Tugasnya adalah melakukan segmentasi, yaitu memisahkan dan memberi label pada area-area berbeda di dalam citra X-ray berdasarkan intensitas pikselnya.

Secara spesifik, skrip ini akan mengelompokkan piksel menjadi beberapa cluster, yang bisa kita interpretasikan sebagai:

Cluster 1 (Paling Gelap): Latar belakang atau jaringan lunak.

Cluster 2 (Abu-abu): Area tulang dengan kepadatan rendah atau berpori (mungkin indikasi osteopenia/osteoporosis).

Cluster 3 (Paling Terang): Area tulang yang sangat padat (indikasi tulang normal).

Jadi, output dari skrip ini adalah sebuah gambar tersegmentasi yang secara visual menunjukkan distribusi area padat dan tidak padat.

## Alur Kerja untuk Klasifikasi (Normal, Osteopenia, Osteoporosis)

Untuk mencapai tujuan akhir Anda (memberi label "Normal", "Osteopenia", atau "Osteoporosis"), Anda perlu menambahkan beberapa langkah setelah menggunakan skrip GMM-EM.

Berikut adalah alur kerja penelitian yang umum:

### **Tahap 1: Segmentasi dengan GMM-EM (✅ Sudah Selesai)**
Gunakan skrip `gui.py` untuk mengubah setiap gambar X-ray menjadi gambar tersegmentasi. Ini adalah langkah pertama dan paling penting.

### **Tahap 2: Ekstraksi Fitur (Langkah Selanjutnya)**
Dari gambar tersegmentasi, Anda perlu menghitung **data kuantitatif** (fitur). Ini mengubah gambar visual menjadi angka yang bisa dianalisis. Contoh fitur yang bisa diekstrak:
-   **Rasio Kepadatan**: Hitung persentase piksel yang termasuk dalam cluster "tulang sangat padat" dibandingkan dengan total piksel tulang.
  -   _Logikanya_: Pasien **osteoporosis** akan memiliki persentase cluster padat yang **jauh lebih rendah**.
-   **Intensitas Rata-rata**: Hitung nilai intensitas rata-rata dari cluster tulang padat.
-   **Fitur Tekstur**: Analisis statistik yang lebih canggih untuk melihat seberapa "kasar" atau "halus" tekstur tulang pada gambar.

### **Tahap 3: Klasifikasi dengan Machine Learning (Tujuan Akhir)**
Setelah Anda memiliki sekumpulan data fitur dari banyak gambar, Anda bisa melatih model machine learning (seperti SVM, Random Forest, atau Neural Network) untuk:
1.  Mempelajari pola fitur dari gambar yang sudah diketahui labelnya (Normal, Osteopenia, Osteoporosis).
2.  Membuat model yang dapat **memprediksi** label untuk gambar baru berdasarkan fitur yang diekstrak.

---

## Referensi Jurnal Ilmiah

Saya telah memilihkan beberapa contoh yang menggunakan pendekatan berbeda namun tetap relevan dengan tujuan Anda.

### **1. Referensi Fokus pada Analisis Tekstur (Fitur Klasik)**

-   **Judul**: _An Automated System for the Detection of Osteoporosis using GLCM features and Neural Network_.
    
-   **Penulis**: M. S. Kavitha et al.
    
-   **Jurnal/Publikasi**: _International Journal of Computer Applications_.
    
-   **Fokus Utama**: Penelitian ini adalah contoh klasik yang sangat cocok untuk Anda. Mereka menggunakan radiograf (citra X-ray) dan melakukan **ekstraksi fitur tekstur** menggunakan metode _Gray-Level Co-occurrence Matrix_ (GLCM). Kemudian, mereka menggunakan **Jaringan Saraf Tiruan (Neural Network)** sebagai classifier untuk membedakan antara tulang normal dan osteoporosis. Ini adalah contoh langsung dari Tahap 2 dan 3.


### **2. Referensi Menggunakan Radiografi Gigi (Aplikasi Umum)**

-   **Judul**: _A Computer-Aided Diagnosis System for Osteoporosis Screening on Dental Panoramic Radiographs_.
    
-   **Penulis**: F. S. C. Leite et al.
    
-   **Jurnal/Publikasi**: _Journal of Clinical and Experimental Dentistry_.
    
-   **Fokus Utama**: Jurnal ini menunjukkan aplikasi yang sangat umum, yaitu mendeteksi risiko osteoporosis dari X-ray gigi panoramik. Mereka mengekstrak **fitur morfometri** (berdasarkan bentuk) dan **tekstur**. Untuk klasifikasi, mereka menggunakan **Support Vector Machine (SVM)**. Ini memberi Anda contoh variasi fitur dan model klasifikasi yang bisa digunakan.


### **3. Referensi dengan Pendekatan Deep Learning (Modern)**

-   **Judul**: _Deep Learning Approach for Osteoporosis Classification Based on Hip X-Ray Images_.
    
-   **Penulis**: M. Unal et al.
    
-   **Jurnal/Publikasi**: _Applied Sciences_.
    
-   **Fokus Utama**: Ini adalah contoh pendekatan yang lebih modern. Mereka menggunakan _Convolutional Neural Networks_ (CNN), sebuah arsitektur _Deep Learning_. Kelebihan metode ini adalah **ekstraksi fitur dan klasifikasi terjadi secara otomatis** di dalam satu model. Meskipun lebih kompleks, ini menunjukkan ke mana arah penelitian saat ini. Anda bisa melihat bagaimana mereka membandingkan performa model otomatis ini dengan metode klasik.


### Poin Kunci dari Referensi di Atas

-   **Ekstraksi Fitur itu Wajib**: Semua penelitian (yang tidak menggunakan _deep learning end-to-end_) selalu memiliki tahap ekstraksi fitur setelah segmentasi. Fitur yang paling umum adalah **fitur tekstur** (GLCM, LBP) dan **morfometri** (bentuk dan ukuran).
    
-   **Model Klasifikasi Bervariasi**: Tidak ada satu model terbaik. **SVM** dan **Jaringan Saraf Tiruan (ANN)** adalah pilihan yang sangat populer dan terbukti efektif untuk masalah klasifikasi medis seperti ini.
    
-   **Validasi Pendekatan Anda**: Jurnal-jurnal ini secara kolektif memvalidasi bahwa alur kerja 3 tahap (Segmentasi ⟶ Ekstraksi Fitur ⟶ Klasifikasi) adalah metodologi standar dan diterima secara ilmiah untuk masalah diagnosis berbantuan komputer (_Computer-Aided Diagnosis_).

---

## Alur Kerja Tahap 2: Ekstraksi Fitur

Berikut adalah langkah-langkah konkret yang bisa Anda lakukan:

### **1. Input**

Input untuk tahap ini adalah **gambar yang sudah tersegmentasi** dari Tahap 1. Ini adalah sebuah array 2D di mana setiap nilainya adalah label cluster (misalnya, 0 untuk background, 1 untuk tulang berpori, 2 untuk tulang padat).

### **2. Pilih dan Hitung Fitur**

Anda akan menulis kode untuk menghitung beberapa angka statistik dari setiap gambar tersegmentasi. Mari kita mulai dari yang paling sederhana hingga yang lebih canggih.

**A. Fitur Paling Sederhana: Rasio Proporsi Cluster** Ini adalah fitur yang paling intuitif dan kuat untuk kasus Anda.

-   **Apa yang dihitung?** Berapa persen dari total area tulang yang merupakan "tulang padat"?
    
-   **Cara menghitung:**
    
    1.  Hitung jumlah piksel untuk cluster tulang padat (misal, cluster label `2`).
        
    2.  Hitung jumlah piksel untuk semua area tulang (cluster `1` + cluster `2`).
        
    3.  Bagi keduanya: `Rasio = (Jumlah Piksel Padat) / (Total Piksel Tulang)`.
        
-   **Hasil:** Anda akan mendapatkan **satu angka** (misalnya `0.75`) untuk setiap gambar. Angka ini merepresentasikan kepadatan tulang secara keseluruhan.
    

**B. Fitur Statistik Intensitas** Anda bisa menggunakan segmentasi sebagai "topeng" untuk menganalisis gambar asli (grayscale).

-   **Apa yang dihitung?** Seberapa terang rata-rata area tulang padat?
    
-   **Cara menghitung:**
    
    1.  Ambil gambar X-ray asli (sebelum segmentasi).
        
    2.  Hitung nilai intensitas rata-rata (`mean`) dan standar deviasi (`std`) dari piksel-piksel yang _hanya_ termasuk dalam cluster tulang padat.
        
-   **Hasil:** Anda mendapatkan **dua angka lagi** untuk setiap gambar: `rata_rata_intensitas` dan `std_dev_intensitas`.
    

**C. Fitur Tekstur (Lebih Lanjut)** Seperti yang disebutkan di jurnal, fitur tekstur dari **Gray-Level Co-occurrence Matrix (GLCM)** sangat populer.

-   **Apa yang dihitung?** Karakteristik tekstur seperti **kontras**, **homogenitas**, **energi**, dan **korelasi** pada area tulang.
    
-   **Bagaimana caranya?** Anda tidak perlu menghitung manual. Pustaka `scikit-image` di Python memiliki fungsi untuk menghitung ini dengan mudah.
    
-   **Hasil:** Anda mendapatkan **beberapa angka lagi** yang mendeskripsikan tekstur tulang.
    

### **3. Struktur Data Output**

Setelah Anda menghitung semua fitur ini untuk **satu gambar**, Anda akan memiliki sebuah **array 1D** atau list, contohnya:

`[0.75, 185.5, 12.3, 0.92, 0.88, ...]` `[rasio, mean_intensitas, std_dev, kontras, homogenitas, ...]`

Anda akan melakukan proses ini untuk **semua gambar** dalam dataset Anda. Hasil akhirnya akan menjadi sebuah **tabel besar (array 2D)**, di mana:

-   Setiap **baris** mewakili **satu gambar (satu pasien)**.
    
-   Setiap **kolom** mewakili **satu fitur** yang Anda hitung.
    
-   Anda juga akan menambahkan satu kolom terakhir, yaitu **label/target** (Normal, Osteopenia, Osteoporosis) yang sudah Anda ketahui.
    

Struktur data ini biasanya disimpan dalam **file CSV** menggunakan pustaka **Pandas**

---

## Daftar Kolom (Fitur) untuk Tabel Data Anda
Berikut adalah rincian kolom yang akan Anda buat.

Kolom Identifikasi
Ini adalah kolom dasar untuk melacak data Anda.

image_id: Nama file atau ID unik dari setiap gambar.

Contoh: xray_001.jpg

diagnosis: Label target yang akan diprediksi. Ini adalah data "jawaban" yang sudah Anda ketahui dari ahli medis.

Contoh: Normal, Osteopenia, Osteoporosis

### A. Fitur Rasio Proporsi Cluster
Fitur ini mengukur distribusi area berdasarkan hasil segmentasi. Sangat intuitif untuk masalah kepadatan.

rasio_tulang_padat: (Jumlah piksel cluster tulang padat) / (Total piksel semua cluster tulang).

Deskripsi: Seberapa besar proporsi tulang yang paling padat. Diharapkan nilai ini tinggi untuk tulang normal dan rendah untuk osteoporosis.

rasio_tulang_berpori: (Jumlah piksel cluster tulang berpori) / (Total piksel semua cluster tulang).

Deskripsi: Seberapa besar proporsi tulang yang kurang padat.

### B. Fitur Statistik Intensitas
Fitur ini mengukur karakteristik kecerahan piksel pada gambar X-ray asli, menggunakan hasil segmentasi sebagai panduan.

intensitas_mean_padat: Nilai rata-rata (mean) intensitas piksel di area tulang padat.

Deskripsi: Seberapa cerah rata-rata area tulang yang paling padat.

intensitas_std_padat: Standar deviasi intensitas piksel di area tulang padat.

Deskripsi: Seberapa seragam tingkat kecerahan di area padat. Nilai yang rendah menandakan area yang homogen.

intensitas_mean_berpori: Nilai rata-rata (mean) intensitas piksel di area tulang berpori.

intensitas_std_berpori: Standar deviasi intensitas piksel di area tulang berpori.

### C. Fitur Tekstur (dari GLCM)
Fitur-fitur ini menangkap karakteristik tekstural dari area tulang pada gambar grayscale. Ini adalah fitur yang sangat kuat untuk membedakan pola.

glcm_contrast: Kontras. Mengukur variasi lokal dalam gambar. Tekstur yang kasar memiliki kontras tinggi.

glcm_dissimilarity: Disimilaritas. Mirip dengan kontras, mengukur seberapa berbeda piksel yang berdekatan.

glcm_homogeneity: Homogenitas. Mengukur seberapa mirip piksel yang berdekatan. Nilainya tinggi jika teksturnya seragam.

glcm_energy: Energi. Ukuran keseragaman tekstur. Nilainya tinggi jika gambar memiliki sedikit transisi warna abu-abu.

glcm_correlation: Korelasi. Mengukur keteraturan pola piksel.

### Hasil Akhir Tabel Data (Contoh File CSV)
Jadi, file data_fitur.csv Anda akan memiliki header kolom seperti ini:

image_id,diagnosis,rasio_tulang_padat,rasio_tulang_berpori,intensitas_mean_padat,intensitas_std_padat,intensitas_mean_berpori,intensitas_std_berpori,glcm_contrast,glcm_dissimilarity,glcm_homogeneity,glcm_energy,glcm_correlation