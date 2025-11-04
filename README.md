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