Install library untuk raw dan gui:
`pip3 install opencv-python-headless scikit-learn matplotlib`

Install library untuk extract_fitur:
`pip3 install pandas scikit-image opencv-python-headless scikit-learn matplotlib`

Jalankan aplikasi:
`python3 4-gui-fitur.py`

---

## Membedah Fungsi GMM-EM dalam Skrip Segmentasi Gambar

Dalam analisis gambar digital, khususnya pada citra medis seperti X-ray, segmentasi adalah proses fundamental. Segmentasi gambar merujuk pada proses partisi atau pemisahan gambar digital menjadi beberapa segmen atau himpunan piksel (juga dikenal sebagai super-piksel). Tujuannya adalah untuk menyederhanakan atau mengubah representasi gambar menjadi sesuatu yang lebih bermakna dan lebih mudah untuk dianalisis.

Dalam skrip yang dibahas, metode **Gaussian Mixture Model (GMM)** yang dilatih menggunakan algoritma **Expectation-Maximization (EM)** digunakan sebagai mesin utama untuk melakukan segmentasi gambar ini.

### Apa Itu Gaussian Mixture Model (GMM)?

GMM adalah model probabilistik yang mengasumsikan bahwa semua titik data (dalam hal ini, **semua piksel**) dihasilkan dari campuran (mixture) sejumlah distribusi Gaussian (distribusi normal atau kurva lonceng) yang berbeda, di mana setiap distribusi mewakili satu cluster.

Dalam konteks gambar X-ray:
* Model ini beranggapan bahwa intensitas piksel dalam gambar tidak berasal dari satu sumber tunggal, melainkan dari beberapa kelompok yang berbeda.
* Misalnya, dalam sebuah X-ray tulang, mungkin terdapat tiga kelompok utama: piksel yang mewakili **tulang padat** (cenderung paling terang), piksel untuk **jaringan lunak** (sedang), dan piksel untuk **latar belakang/udara** (gelap).
* GMM mengasumsikan bahwa distribusi intensitas piksel untuk masing-masing kelompok ini dapat direpresentasikan oleh sebuah kurva lonceng (Gaussian) yang unik.

### Peran Algoritma Expectation-Maximization (EM)

GMM adalah modelnya, sedangkan EM adalah algoritma yang digunakan untuk *melatih* atau *mencocokkan* (fit) model GMM tersebut dengan data piksel yang ada.

Di sinilah baris kode `gmm.fit(pixel_values)` mengambil peran sentral. Algoritma EM bekerja secara iteratif untuk menemukan parameter terbaik (seperti nilai rata-rata kecerahan dan variansi/sebaran) untuk setiap kurva lonceng (cluster) agar paling sesuai dengan distribusi data piksel yang sebenarnya. Proses ini secara cerdas mengoptimalkan parameter model GMM sehingga paling *mungkin* (maximum likelihood) menjelaskan data yang diamati.

### Proses dan Hasil Akhir: Segmentasi

Setelah model GMM berhasil dilatih oleh algoritma EM, langkah selanjutnya adalah melakukan prediksi.

1.  **Prediksi Cluster:** Baris kode `gmm.predict(pixel_values)` dieksekusi. Pada tahap ini, model GMM akan mengevaluasi setiap piksel dalam gambar.
2.  **Pemberian Label:** Untuk setiap piksel, model akan menentukan, "Berdasarkan parameter yang telah dipelajari, piksel dengan intensitas ini paling mungkin termasuk dalam kelompok (cluster) 0, 1, atau 2?"
3.  **Pembentukan Gambar Baru:** Hasil dari prediksi ini adalah sebuah array `labels`, di mana setiap piksel kini memiliki label cluster. Array ini kemudian diubah bentuknya (`reshape`) kembali ke dimensi gambar asli.

Hasil akhirnya adalah `segmented_image`—sebuah gambar baru di mana setiap piksel tidak lagi menampilkan nilai kecerahan aslinya, melainkan sebuah label (yang divisualisasikan dengan warna berbeda) yang menunjukkan keanggotaan clusternya.

### Kesimpulan

Secara singkat, fungsi **GMM-EM** dalam skrip ini adalah untuk **mengelompokkan (clustering) semua piksel** dalam gambar X-ray ke dalam beberapa kategori yang berbeda (misalnya, 3 cluster). Pengelompokan ini dilakukan secara otomatis berdasarkan properti statistik dari intensitas piksel, sehingga memungkinkan pemisahan wilayah-wilayah yang berbeda (seperti tulang dan jaringan) untuk analisis lebih lanjut.

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

---

## Flowchart Implementasi Gaussian Mixture Models (GMM) - Expectation-Maximization (EM)

Untuk menganalisis dan mengekstrak area kepadatan tulang dari citra digital, metodologi 8 langkah berikut telah dirumuskan. Proses ini menggunakan algoritma Gaussian Mixture Models (GMM) untuk mengelompokkan piksel berdasarkan intensitasnya.

![image](https://res.cloudinary.com/enterz/image/upload/v1762502851/SlimeNews/GMM-EM.png)

**1. Akuisisi Data (Input Citra)**

Langkah paling awal dalam alur kerja ini adalah akuisisi data. Dalam konteks ini, data merupakan citra digital mentah (misalnya, format .jpg atau .png) yang akan menjadi subjek analisis.

**2. Pra-pemrosesan: Konversi Grayscale**

Model GMM memerlukan fitur tunggal untuk setiap titik data (piksel). Oleh karena itu, citra input harus melalui pra-pemrosesan, di mana ia dikonversi dari ruang warna RGB (3 saluran) menjadi grayscale (1 saluran). Saluran tunggal ini merepresentasikan fitur intensitas (kecerahan), yang menjadi dasar untuk pengelompokan.

**3. Inisialisasi Model GMM**

Pada tahap ini, model probabilistik GMM disiapkan. Sebuah hyperparameter kunci, n_components (jumlah cluster), diatur ke 3. Pengaturan ini didasarkan pada asumsi apriori bahwa citra terdiri dari tiga komponen utama yang dapat dipisahkan secara statistik: tulang (intensitas tinggi), jaringan lunak (intensitas sedang), dan latar belakang (intensitas rendah).

**4. Pelatihan Model (Algoritma EM)**

Proses fitting model (gmm.fit()) dieksekusi menggunakan seluruh set data piksel grayscale. Di balik layar, Algoritma Expectation-Maximization (EM) bekerja secara iteratif. Tujuannya adalah untuk menemukan parameter optimal (khususnya gmm.means_ atau rata-rata intensitas) untuk setiap 3 cluster Gaussian agar paling sesuai dengan distribusi data.

**5. Segmentasi Awal (Prediksi Cluster)**

Setelah model dilatih, model tersebut digunakan untuk mengklasifikasikan setiap piksel dalam citra. Fungsi gmm.predict() menetapkan setiap piksel ke cluster yang paling mungkin (memberi label 0, 1, atau 2). Penting untuk dicatat bahwa GMM tidak menjamin urutan label ini; label '0' mungkin tidak selalu mewakili cluster tergelap.

**6. Standardisasi Label: Analisis Mean**

Untuk memastikan konsistensi hasil dan interpretabilitas, label cluster perlu distandarisasi. Rata-rata intensitas (gmm.means_) dari setiap cluster yang ditemukan diekstrak. Dengan menggunakan fungsi np.argsort(), sebuah "peta" pengurutan dibuat untuk mengidentifikasi indeks cluster dari yang tergelap (intensitas terendah) hingga yang terterang (intensitas tertinggi).

**7. Pemetaan Ulang (Re-mapping) Label**

"Peta" dari langkah 6 kini diterapkan. Sebuah array baru (sorted_labels) dibuat. Dengan melakukan iterasi pada peta, setiap piksel di gambar diberi label baru yang konsisten (misal, 0 untuk latar belakang/gelap, 1 untuk jaringan/sedang, 2 untuk tulang/terang). Proses ini memastikan bahwa label '2', misalnya, secara konsisten mewakili cluster dengan intensitas tertinggi di setiap gambar yang diuji.

**8. Hasil Segmentasi Gambar**

Sebagai langkah akhir, array label 1D (sorted_labels) dibentuk kembali (reshape) ke dimensi spasial 2D citra asli. Hasilnya kemudian divisualisasikan. Output ini berfungsi sebagai validasi visual dari keberhasilan proses segmentasi, yang menampilkan gambar di mana setiap komponen (tulang, jaringan, dan latar belakang) telah dipisahkan ke dalam kelasnya masing-masing.

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

### Referensi

Pendekatan metodologis untuk menetapkan rentang patokan (baseline) melalui analisis statistik deskriptif (Min, Max, Mean, Std Dev) dari kelompok sampel normal (kelompok kontrol) merupakan praktik standar dalam penelitian biomedis. Referensi ilmiah berikut mendukung dan menjelaskan metodologi ini:

#### 1. Jurnal tentang Metodologi Statistik di Laboratorium
* **Judul:** *Defining, Establishing, and Verifying Reference Intervals in the Clinical Laboratory* (Mendefinisikan, Menetapkan, dan Memverifikasi Rentang Referensi di Laboratorium Klinis)
* **Tautan:** [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4042858/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4042858/)
* **Penjelasan Relevan:** Artikel ulasan (review paper) ini memaparkan secara rinci **metodologi untuk menentukan rentang "normal"** dalam konteks klinis (misalnya, tes darah). Metodologi standar yang dijelaskan adalah pengambilan sampel dari populasi sehat, diikuti dengan analisis statistik deskriptif untuk menetapkan rentang referensi (sering didefinisikan sebagai **Mean ± 2 Standar Deviasi**). Hal ini secara langsung memvalidasi proses penetapan rentang berdasarkan analisis sampel "normal".

#### 2. Jurnal tentang Analisis Khusus Fitur Tulang
* **Judul:** *Comparison of trabecular bone structure parameters of the mandible between a control group and an osteoporosis risk group: a cone-beam computed tomography study* (Perbandingan parameter struktur tulang trabekular... antara kelompok kontrol dan kelompok risiko osteoporosis...)
* **Tautan:** [https.ncbi.nlm.nih.gov/pmc/articles/PMC6132924/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6132924/)
* **Penjelasan Relevan:** Penelitian ini merupakan contoh aplikasi praktis dari metodologi tersebut. Peneliti mengimplementasikan langkah-langkah berikut: 1. Stratifikasi pasien ke dalam **"Kelompok Kontrol" (sehat)** dan **"Kelompok Risiko Osteoporosis"**. 2. Ekstraksi fitur kuantitatif dari citra (analog dengan fitur rasio yang digunakan). 3. Penyajian temuan sebagai **Statistik Deskriptif (Mean dan Standar Deviasi)** untuk *setiap kelompok*. Pendekatan ini tidak bergantung pada nilai ambang batas universal, melainkan pada perbandingan statistik antara rentang yang ditemukan pada Kelompok Kontrol dan Kelompok Tes.

#### 3. Jurnal tentang Pentingnya "Tabel 1" (Statistik Deskriptif)
* **Judul:** *How to Read "Table 1" in a Research Paper* (Cara Membaca "Tabel 1" dalam Makalah Penelitian)
* **Tautan:** [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6482813/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6482813/)
* **Penjelasan Relevan:** Artikel ini menguraikan fungsi "Tabel 1", sebuah komponen standar dalam publikasi penelitian klinis. "Tabel 1" umumnya menyajikan **statistik deskriptif (Mean, SD, Min, Max, N)** untuk fitur-fitur kunci dari populasi penelitian, yang disajikan secara terpisah untuk **"Kelompok Kontrol"** dan **"Kelompok Perlakuan/Penyakit"**. Praktik ini memvalidasi langkah pengumpulan data fitur ke dalam array untuk analisis Min, Max, dan Mean sebagai prosedur standar dalam penyajian temuan penelitian.
