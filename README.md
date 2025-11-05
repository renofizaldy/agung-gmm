Install library untuk raw dan gui:
`pip3 install opencv-python-headless scikit-learn matplotlib`

Install library untuk extract_fitur:
`pip3 install pandas scikit-image opencv-python-headless scikit-learn matplotlib`

Jalankan aplikasi:
`python3 gui.py`

---

## Penentuan Fitur atau Klasifikasi

### 1. rasio_total_tulang_terhadap_gambar

* **Nama Ilmiah:** Bone Area / Total Area (B.Ar/T.Ar) atau Bone Area Fraction.
* **Validasi Konsep:** Ini adalah parameter histomorfometri paling dasar. Fitur ini mengukur berapa persen dari total area sampel (ROI - *Region of Interest*) yang benar-benar terisi oleh materi tulang (baik padat maupun berpori). Ini adalah ukuran langsung dari massa tulang dalam 2D.
* **Rumus Perhitungan:**
    ```
    (pixels_padat + pixels_berpori) / pixels_total_gambar
    ```
* **Referensi Jurnal:**
    * **Judul:** *Bone histomorphometry revisited* (Tinjauan ulang histomorfometri tulang)
    * **Penjelasan Relevan:** Artikel ini secara eksplisit mendefinisikan parameter-parameter standar. Dinyatakan: *"Bone area (2D, volume in 3D) is the percentage of occupied area by calcified bone in relation to the total area."* (Area tulang (2D, volume dalam 3D) adalah persentase area yang ditempati oleh tulang terkalsifikasi dalam kaitannya dengan total area).
    * **Tautan:** https://www.researchgate.net/publication/257814134_Bone_histomorphometry_revisited

### 2. rasio_tulang_padat_terhadap_total_tulang

* **Nama Ilmiah:** Percent Cortical Area (%Ct.Ar) atau Cortical Bone Fraction.
* **Validasi Konsep:** Setelah mengetahui total area tulang (dari fitur #1), langkah logis berikutnya adalah membedah komposisinya. Fitur ini mengukur seberapa besar bagian dari total tulang tersebut yang merupakan tulang kortikal (padat). Banyak penelitian menggunakan ketebalan kortikal (*Cortical Thickness*) sebagai proxy, namun mengukur area adalah analisis 2D yang lebih lengkap.
* **Rumus Perhitungan:**
    ```
    pixels_padat / (pixels_padat + pixels_berpori)
    ```
* **Referensi Jurnal:**
    * **Judul:** *Panoramic Measures for Oral Bone Mass in Detecting Osteoporosis: A Systematic Review and Meta-Analysis* (Pengukuran Panoramik untuk Massa Tulang Oral dalam Mendeteksi Osteoporosis: Tinjauan Sistematis dan Meta-Analisis)
    * **Penjelasan Relevan:** Tinjauan sistematis ini membahas berbagai "indeks radiomorfometrik" yang dihitung dari X-ray panoramik (gigi) untuk mendeteksi kepadatan tulang yang rendah (BMD). Banyak dari indeks ini, seperti *Panoramic Mandibular Index (PMI)*, didasarkan pada pengukuran rasio ketebalan kortikal terhadap dimensi tulang lainnya. Ini memvalidasi konsep bahwa rasio yang melibatkan tulang kortikal (padat) adalah alat skrining yang efektif untuk osteoporosis.
    * **Tautan:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4541087/

### 3. rasio_padat_vs_berpori

* **Nama Ilmiah:** Cortical-to-Trabecular Ratio (Rasio Kortikal-terhadap-Trabekular) atau Cortical/Cancellous Ratio.
* **Validasi Konsep:** Ini adalah fitur yang sangat kuat secara diagnostik. Penelitian menunjukkan bahwa osteoporosis seringkali memengaruhi tulang trabekular (berpori) terlebih dahulu dan lebih agresif daripada tulang kortikal (padat). Oleh karena itu, rasio antara keduanya adalah indikator yang sangat sensitif terhadap perubahan penyakit.
* **Rumus Perhitungan:**
    ```
    pixels_padat / pixels_berpori
    ```
* **Referensi Jurnal:**
    * **Judul:** *Bone mechanical properties and changes with osteoporosis* (Sifat mekanis tulang dan perubahannya dengan osteoporosis)
    * **Penjelasan Relevan:** Artikel ini menjelaskan perbedaan antara kedua jenis tulang. Ia menyebutkan: *"Thus, the bone loss in early osteoporosis is mainly a trabecular bone loss."* (Dengan demikian, kehilangan tulang pada osteoporosis dini sebagian besar adalah kehilangan tulang trabekular). Hal ini secara langsung memvalidasi mengapa fitur `(Piksel Padat) / (Piksel Berpori)` sangat penting. Ketika tulang berpori (penyebut) hilang, rasio ini akan berubah secara dramatis.
    * **Tautan:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4955555/

---

## Metodologi Penetapan Nilai Patokan (Baseline) untuk Analisis Fitur

Untuk melakukan analisis kuantitatif yang bermakna terhadap fitur rasio, langkah metodologis pertama adalah menetapkan **rentang nilai referensi (reference range)** atau **garis dasar (baseline)**. Hal ini didasarkan pada prinsip bahwa sebuah nilai fitur (misalnya, rasio 4.5) tidak dapat diinterpretasikan sebagai "tinggi" atau "rendah" tanpa perbandingan terhadap nilai "normal" yang telah ditetapkan untuk dataset yang sama.

Proses untuk menetapkan baseline ini adalah sebagai berikut:

1.  **Penetapan Garis Dasar "Normal"**: Sejumlah sampel citra (misalnya, 5-10) yang secara visual diidentifikasi sebagai "normal" atau "sehat" diproses. Fitur rasio yang diekstraksi dari sampel-sampel ini (misalnya, `[1.9, 2.1, 2.0, 1.8, 2.2]`) dikumpulkan.
2.  **Penentuan Rentang Referensi**: Dari kumpulan data tersebut, sebuah rentang referensi awal dapat disimpulkan (misalnya, ~1.8 - 2.2). Rentang ini bersifat spesifik untuk dataset yang digunakan dan dipengaruhi oleh parameter akuisisi citra (misalnya, model mesin X-ray).
3.  **Identifikasi Pola Patologis**: Selanjutnya, kumpulan sampel pembanding yang secara visual tampak "berpori" atau patologis dianalisis. Kumpulan fitur rasio yang dihasilkan (misalnya, `[4.5, 5.1, 4.8, 5.3, 4.9]`) akan dibandingkan dengan rentang referensi.
4.  **Kesimpulan Analisis Awal**: Analisis komparatif ini menghasilkan patokan kuantitatif kasar. Nilai yang berada dalam rentang referensi (~2.0) dapat dianggap normal, sedangkan nilai yang secara signifikan melebihi rentang tersebut (>4.5) merupakan indikator potensi patologi.

Oleh karena itu, agregasi nilai fitur (misalnya, dalam bentuk array atau tabel) dari kelompok sampel yang diketahui kondisinya adalah langkah penting untuk mendefinisikan rentang patokan yang unik dan spesifik untuk dataset tersebut.

### Validasi Metodologi dalam Literatur Ilmiah

Pendekatan untuk menetapkan baseline ini didukung oleh praktik metodologis standar dalam penelitian diagnosis berbantuan komputer (Computer-Aided Diagnosis - CAD).

* **Referensi:** *A Computer-Aided Diagnosis System for Osteoporosis Screening on Dental Panoramic Radiographs*
* **Tautan:** [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5303323/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5303323/)

#### Dukungan Metodologis:

Penelitian ini memvalidasi pendekatan tersebut. Alih-alih menggunakan ambang batas universal yang telah ditentukan sebelumnya, penelitian tersebut menerapkan protokol berikut:

1.  **Pengumpulan Sampel:** Mengumpulkan data dari 140 pasien.
2.  **Stratifikasi Grup:** Pasien dibagi menjadi dua kelompok berdasarkan hasil tes standar (DEXA scan):
    * **Kelompok Kontrol (Control Group):** 70 pasien dengan kepadatan tulang **Normal**.
    * **Kelompok Tes (Test Group):** 70 pasien dengan kepadatan tulang rendah (**Osteopenia/Osteoporosis**).
3.  **Ekstraksi Fitur:** Berbagai fitur, termasuk indeks morfometri dan tekstur (analog dengan fitur rasio dan GLCM yang digunakan), dihitung dari citra X-ray.
4.  **Analisis Statistik (Metodologi Inti):** Nilai fitur dari **Kelompok Kontrol (Normal)** diisolasi. **Nilai rata-rata (mean)** dan **standar deviasi** dari kelompok ini kemudian dihitung untuk menetapkan **garis dasar referensi statistik** mereka.
5.  **Kesimpulan:** Nilai fitur dari Kelompok Tes kemudian dibandingkan secara statistik terhadap baseline normal tersebut, yang menunjukkan adanya perbedaan signifikan.

### Kesimpulan

Jurnal tersebut memvalidasi metodologi yang diusulkan. Proses "mengumpulkan nilai dalam bentuk array" adalah implementasi praktis dari langkah ilmiah krusial yang dikenal sebagai **"menganalisis distribusi fitur dari kelompok kontrol"**. Langkah ini esensial untuk membangun rentang patokan kuantitatif yang unik dan spesifik untuk sebuah dataset penelitian.

---

## Analisis Statistik Deskriptif untuk Penentuan Rentang Fitur

Setelah agregasi data fitur dari sekumpulan sampel (misalnya, N=10), langkah metodologis selanjutnya adalah melakukan analisis statistik deskriptif. Tujuan dari analisis ini adalah untuk mengkuantifikasi distribusi nilai fitur dan menetapkan rentang referensi (baseline) untuk setiap kelompok yang dianalisis.

### Parameter Statistik Kunci

Karakterisasi kuantitatif dari distribusi data ini dicapai dengan menghitung parameter statistik deskriptif dasar berikut:

1.  **Nilai Minimum (Min):** Merepresentasikan batas bawah dari rentang data yang diamati.
2.  **Nilai Maksimum (Max):** Merepresentasikan batas atas dari rentang data yang diamati.
3.  **Nilai Rata-rata (Mean):** Ukuran tendensi sentral (*central tendency*), yang mengindikasikan nilai "tipikal" atau rata-rata dalam kelompok sampel.
4.  **Standar Deviasi (Standard Deviation):** Ukuran dispersi statistik, yang mengkuantifikasi jumlah variasi atau sebaran data dari nilai rata-ratanya.

### Aplikasi pada Penentuan Rentang

Secara teknis, "rentang nilai" (range) untuk suatu kelompok sampel didefinisikan oleh nilai minimum dan maksimumnya. Nilai rata-rata menyediakan titik pusat dari distribusi tersebut.

#### Studi Kasus Hipotetis:

Sebagai ilustrasi, dua kelompok sampel (Normal dan Berpori) dianalisis:

**1. Kelompok Kontrol (Sampel 'Normal')**
* **Kumpulan Data (N=10):** `[1.9, 2.1, 2.0, 1.8, 2.2, 1.9, 2.0, 2.1, 2.2, 1.8]`
* **Analisis Statistik:**
    * Nilai Minimum: **1.80**
    * Nilai Maksimum: **2.20**
    * Nilai Rata-rata (Mean): **2.00**
    * Standar Deviasi: 0.16
* **Interpretasi:** Rentang referensi "Normal" untuk dataset ini dapat ditetapkan antara **1.80 - 2.20**, dengan tendensi sentral di **2.00**.

**2. Kelompok Tes (Sampel 'Berpori')**
* **Kumpulan Data (N=10):** `[4.5, 5.1, 4.8, 5.3, 4.9, 4.7, 5.0, 5.2, 4.6, 5.3]`
* **Analisis Statistik:**
    * Nilai Minimum: **4.50**
    * Nilai Maksimum: **5.30**
    * Nilai Rata-rata (Mean): **4.94**
    * Standar Deviasi: 0.29
* **Interpretasi:** Rentang patologis "Berpori" untuk dataset ini teramati antara **4.50 - 5.30**, dengan tendensi sentral di **4.94**.

### Kesimpulan

Perhitungan statistik deskriptif (terutama Min, Max, dan Mean) adalah alat fundamental untuk menetapkan rentang patokan secara empiris dari data sampel. Perhitungan ini dapat difasilitasi menggunakan pustaka komputasi ilmiah seperti **NumPy** (misalnya, `np.min()`, `np.max()`, `np.mean()`).

Tentu. Pendekatan metodologis untuk menetapkan rentang patokan (baseline) dengan menganalisis statistik deskriptif (Min, Max, Mean, Std Dev) dari sekelompok sampel "normal" (kelompok kontrol) adalah praktik standar dalam penelitian biomedis.

Berikut adalah beberapa referensi ilmiah yang mendukung dan menjelaskan metodologi ini:

### 1. Jurnal tentang Metodologi Statistik di Laboratorium
* **Judul:** *Defining, Establishing, and Verifying Reference Intervals in the Clinical Laboratory* (Mendefinisikan, Menetapkan, dan Memverifikasi Rentang Referensi di Laboratorium Klinis)
* **Tautan:** [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4042858/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4042858/)
* **Penjelasan Relevan:** Ini adalah artikel ulasan (review paper) yang sangat mendasar. Artikel ini menjelaskan secara rinci **bagaimana para ilmuwan menentukan rentang "normal"** untuk segala hal (termasuk tes darah, dll.). Metodologi standarnya adalah mengambil sampel dari populasi sehat, kemudian menggunakan statistik deskriptif untuk menentukan rentang referensi (seringkali didefinisikan sebagai **Mean ± 2 Standar Deviasi**). Ini adalah validasi langsung dari proses yang Anda lakukan: mengambil sampel "normal" untuk menemukan rentangnya.

### 2. Jurnal tentang Analisis Khusus Fitur Tulang
* **Judul:** *Comparison of trabecular bone structure parameters of the mandible between a control group and an osteoporosis risk group: a cone-beam computed tomography study* (Perbandingan parameter struktur tulang trabekular... antara kelompok kontrol dan kelompok risiko osteoporosis...)
* **Tautan:** [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6132924/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6132924/)
* **Penjelasan Relevan:** Ini adalah contoh sempurna dari metodologi Anda dalam praktik. Para peneliti melakukan persis seperti yang Anda rencanakan:
    1.  Mereka membagi pasien menjadi **"Kelompok Kontrol" (sehat)** dan **"Kelompok Risiko Osteoporosis"**.
    2.  Mereka menghitung fitur-fitur dari gambar (mirip dengan fitur rasio Anda).
    3.  Mereka kemudian menyajikan temuan mereka sebagai **Statistik Deskriptif (Mean dan Standar Deviasi)** untuk *setiap kelompok*. Mereka tidak menggunakan "angka ajaib", tetapi membandingkan rentang yang mereka temukan di Kelompok Kontrol dengan rentang di Kelompok Tes.

### 3. Jurnal tentang Pentingnya "Tabel 1" (Statistik Deskriptif)
* **Judul:** *How to Read "Table 1" in a Research Paper* (Cara Membaca "Tabel 1" dalam Makalah Penelitian)
* **Tautan:** [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6482813/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6482813/)
* **Penjelasan Relevan:** Artikel ini menjelaskan fungsi dari "Tabel 1", yang merupakan tabel paling penting dalam sebagian besar penelitian klinis. "Tabel 1" adalah tempat di mana peneliti menyajikan **statistik deskriptif (Mean, SD, Min, Max, N)** untuk fitur-fitur kunci dari populasi penelitian mereka, yang hampir selalu dibagi menjadi **"Kelompok Kontrol"** dan **"Kelompok Perlakuan/Penyakit"**. Ini memvalidasi bahwa langkah pertama Anda—mengumpulkan angka dalam array untuk menemukan Min, Max, dan Mean—adalah praktik standar untuk menyajikan temuan penelitian.

---

## Penentuan Jumlah Cluster dalam Segmentasi GMM

Tujuan dari penentuan jumlah cluster (misalnya, 3 atau 4) dalam Gaussian Mixture Models (GMM) adalah untuk menginstruksikan algoritma mengenai jumlah kelompok kepadatan berbeda yang harus diidentifikasi di dalam citra X-ray. Pengaturan parameter ini bersifat krusial karena akan secara langsung memengaruhi hasil segmentasi citra.

### Skenario 3 Cluster

Skenario 3 cluster merupakan pendekatan awal yang paling logis dan sederhana. Dalam konfigurasi ini, GMM membagi semua piksel citra menjadi tiga kelompok utama berdasarkan intensitasnya:

* **Cluster 1 (Kepadatan Paling Rendah / Hitam):** Merepresentasikan **latar belakang** (background) atau udara. Piksel-piksel ini tidak memiliki informasi kepadatan tulang dan harus diisolasi agar tidak mengganggu kalkulasi fitur.
* **Cluster 2 (Kepadatan Menengah / Abu-abu):** Merepresentasikan **tulang berpori (trabekular)**. Area ini juga dapat mencakup jaringan lunak (otot/lemak) yang memiliki atenuasi sinar-X serupa.
* **Cluster 3 (Kepadatan Paling Tinggi / Putih):** Merepresentasikan **tulang padat (kortikal)**. Ini adalah area dengan kepadatan radiografi tertinggi.

**Keuntungan:** Model ini secara efisien menyediakan dua komponen tulang esensial (`Piksel Padat` dan `Piksel Berpori`) yang diperlukan untuk perhitungan fitur rasio.

#### Justifikasi Logis untuk 3 Cluster

Alasan paling logis untuk menggunakan 3 cluster adalah karena citra X-ray tulang secara fisik memiliki tiga "area" kepadatan utama yang berbeda dan relevan untuk analisis. Pemisahan ini mutlak diperlukan untuk mengekstrak fitur rasio kunci, seperti `rasio_padat_vs_berpori`, yang formulanya adalah:

`(Jumlah Piksel Padat) / (Jumlah Piksel Berpori)`

Untuk menghitung rasio ini, diperlukan tiga kelompok data:

1.  Satu kelompok untuk **Piksel Padat** (dari Cluster 3 / Putih).
2.  Satu kelompok untuk **Piksel Berpori** (dari Cluster 2 / Abu-abu).
3.  Satu kelompok untuk **Latar Belakang** (dari Cluster 1 / Hitam) yang harus dieliminasi dari perhitungan.

Dengan demikian, penggunaan 3 cluster adalah jumlah minimum yang logis dan diperlukan untuk mengekstrak dua komponen utama dalam formula fitur rasio tersebut.

### Skenario 4 Cluster (Alternatif)

Skenario 4 cluster dapat dipertimbangkan jika skenario 3 cluster dinilai terlalu sederhana, terutama jika terjadi percampuran signifikan antara jaringan lunak dan tulang berpori dalam satu cluster (Cluster 2).

Dengan memilih 4 cluster, GMM dapat memisahkan komponen-komponen ini dengan lebih detail:

* **Cluster 1 (Paling Gelap):** Latar Belakang / Udara.
* **Cluster 2 (Abu-abu Gelap):** Jaringan Lunak (Otot, Lemak).
* **Cluster 3 (Abu-abu Terang):** Tulang Berpori (Trabekular).
* **Cluster 4 (Paling Terang):** Tulang Padat (Kortikal).

**Keuntungan:** Pendekatan ini berpotensi menghasilkan segmentasi tulang yang lebih "bersih" dan akurat dengan mengisolasi jaringan lunak yang dapat mengganggu.

### Pertimbangan Metodologis

Rekomendasi umumnya adalah memulai analisis dengan **3 cluster** sebagai asumsi dasar. Jika hasil segmentasi visual menunjukkan bahwa cluster tulang berpori (trabekular) terlihat jelas tercampur dengan jaringan lunak di sekitarnya, eksperimen dapat dilanjutkan dengan meningkatkan jumlah cluster menjadi **4** untuk mengevaluasi apakah pemisahan yang lebih baik dapat dicapai.
