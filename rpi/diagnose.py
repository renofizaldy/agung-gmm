import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import graycomatrix, graycoprops
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

# ==========================================================
# FUNGSI PRE-PROCESSING: CLAHE
# ==========================================================
def preprocess_image(image_path):
    # Membaca file gambar dari jalur (path) yang diberikan dan langsung mengubahnya 
    # ke dalam format Grayscale (hitam putih) agar lebih mudah diproses secara matematis.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Validasi untuk memastikan gambar berhasil dibaca; jika file tidak ditemukan 
    # atau rusak, fungsi akan berhenti dan mengembalikan nilai None.
    if img is None: return None

    # Membuat objek CLAHE (Contrast Limited Adaptive Histogram Equalization). 
    # clipLimit=2.0 membatasi kontras agar tidak berlebihan (mencegah noise meningkat), 
    # dan tileGridSize=(8,8) membagi gambar menjadi kotak-kotak kecil untuk pemerataan kontras lokal.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Menerapkan algoritma CLAHE pada gambar. Tahap ini untuk memperjelas 
    # detail serat tulang yang mungkin tidak terlihat pada gambar asli yang terlalu gelap/terang.
    improved_img = clahe.apply(img)

    # Menerapkan Gaussian Blur dengan ukuran kernel 3x3. Fungsi ini bertujuan untuk 
    # sedikit menghaluskan gambar guna mengurangi gangguan (noise) berupa bintik-bintik kecil 
    # tanpa menghilangkan detail struktur utama tulang.
    improved_img = cv2.GaussianBlur(improved_img, (3, 3), 0)

    # Mengembalikan gambar yang telah "dibersihkan" dan diperbaiki kontrasnya 
    # untuk diproses lebih lanjut oleh tahap segmentasi GMM.
    return improved_img

# ==========================================================
# FUNGSI EKSTRAKSI FITUR
# ==========================================================
def extract_features_complete(img, segmented_image):
    # --- Bagian 1: Fitur Rasio dari GMM (Informasi Kepadatan/Makro) ---
    
    # Menghitung jumlah piksel yang dikategorikan sebagai "Tulang Padat" 
    # (Label 2 pada hasil segmentasi GMM).
    pixels_padat = np.count_nonzero(segmented_image == 2)
    
    # Menghitung jumlah piksel yang dikategorikan sebagai "Tulang Berpori" 
    # (Label 1 pada hasil segmentasi GMM).
    pixels_berpori = np.count_nonzero(segmented_image == 1)
    
    # Menjumlahkan kedua jenis piksel di atas untuk mendapatkan total luas area tulang.
    pixels_total = pixels_padat + pixels_berpori
    
    # Menghitung Rasio Padat vs Berpori. Jika nilai ini tinggi, berarti tulang masih padat. 
    # Logika 'if' digunakan untuk mencegah error pembagian dengan nol.
    rasio_pvb = pixels_padat / pixels_berpori if pixels_berpori > 0 else 0.0
    
    # Menghitung persentase tulang padat terhadap seluruh area tulang. 
    # Ini memberikan informasi seberapa dominan bagian padat dalam struktur tulang tersebut.
    rasio_pvt = pixels_padat / pixels_total if pixels_total > 0 else 0.0

    # --- Bagian 2: Statistik & GLCM (Informasi Tekstur/Mikro) ---
    
    # Menghitung rata-rata tingkat kecerahan seluruh piksel (Mean). 
    # Memberikan gambaran umum densitas pada citra tersebut.
    mean_val = np.mean(img)
    
    # Menghitung variansi (sebaran kontras). Semakin tinggi nilainya, 
    # berarti perbedaan antara area gelap dan terang semakin bervariasi.
    var_val = np.var(img)
    
    # Membangun matriks korelasi piksel (GLCM) dengan jarak 1 piksel dan sudut 0 derajat.
    # 'normed=True' digunakan agar nilai matriks menjadi probabilitas (rentang 0-1).
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Mengembalikan daftar (list) berisi 8 "identitas" angka dari gambar tersebut.
    # yang nantinya akan menjadi bahan bagi Random Forest.
    return [
        rasio_pvb,                              # 1. Rasio Kepadatan vs Pori
        rasio_pvt,                              # 2. Rasio Kepadatan vs Total
        graycoprops(glcm, 'contrast')[0, 0],    # 3. Kekasaran tekstur
        graycoprops(glcm, 'homogeneity')[0, 0], # 4. Keseragaman pola
        graycoprops(glcm, 'energy')[0, 0],      # 5. Keteraturan tekstur
        graycoprops(glcm, 'correlation')[0, 0], # 6. Hubungan serat tulang
        mean_val,                               # 7. Statistik Rata-rata
        var_val                                 # 8. Statistik Kontras
    ]

# ==========================================================
# BAGIAN 1: TRAINING MODEL
# ==========================================================
def train_ai_model():
    filename = "database_fitur.csv"
    if not os.path.exists(filename):
        return None, "Database (CSV) tidak ditemukan. Harap Training data dulu."

    try:
        # Membaca file CSV menggunakan library Pandas dan mengubahnya menjadi 
        # tabel (DataFrame) agar mudah diolah oleh algoritma Machine Learning.
        df = pd.read_csv(filename)
        if len(df) < 5:
            return None, "Data di database minimal 5 sampel untuk mulai belajar."

        # Memisahkan Fitur (X) dan Label Diagnosa (y)
        X = df[['rasio_p_v_b', 'rasio_p_v_t', 'glcm_contrast', 'glcm_homogeneity', 
                'glcm_energy', 'glcm_correlation', 'stat_mean', 'stat_variance']]
        y = df['diagnosa']

        # Membuat objek algoritma Random Forest. 'n_estimators=100' berarti akan 
        # membuat 100 "pohon keputusan" untuk mendapatkan hasil voting yang paling akurat.
        # 'random_state=42' memastikan hasil pembelajaran selalu konsisten setiap dijalankan.
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Proses 'FIT' (BELAJAR): Di sini mencari pola 
        # matematis yang memisahkan antara tulang Normal, Osteopenia, dan Osteoporosis.
        model.fit(X, y)
        
        return model, len(df)
    except Exception as e:
        return None, f"Error membaca database: {e}"

# ==========================================================
# BAGIAN 2: PROSES DIAGNOSA CITRA BARU
# ==========================================================
def start_diagnosis():
    # Memanggil fungsi train_ai_model() di atas untuk melatih model dari data 
    # yang ada di CSV sebelum mulai melakukan diagnosa pada citra baru.
    model, n_data = train_ai_model()
    if model is None:
        messagebox.showwarning("Peringatan", n_data)
        return

    # Membuka jendela dialog agar pengguna bisa memilih file gambar X-ray 
    # yang ingin didiagnosa.
    file_path = filedialog.askopenfilename(title="Pilih Citra X-ray")
    # Jika pengguna menutup jendela dialog tanpa memilih gambar, fungsi akan berhenti.
    if not file_path: return

    try:
        # --- LANGKAH 1: PRE-PROCESSING ---
        # Simpan gambar original asli (sebelum CLAHE) untuk histogram nanti
        img_original = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Membersihkan gambar pilihan pengguna menggunakan CLAHE & Gaussian Blur.
        img = preprocess_image(file_path)

        # --- LANGKAH 2: SEGMENTASI GMM ---
        # Mengubah gambar menjadi satu baris angka (vektor) dengan tipe data float64 
        # agar perhitungan matematis GMM lebih stabil dan tidak error.
        pixel_values = img.reshape(-1, 1).astype(np.float64)

        # Melatih GMM khusus untuk gambar ini guna memisahkan area tulang dan background.
        gmm = GaussianMixture(n_components=3, random_state=42).fit(pixel_values)
        labels = gmm.predict(pixel_values)

        # Mengambil nilai rata-rata kecerahan dari tiap cluster untuk menentukan 
        # mana yang area gelap (pori) dan mana yang area terang (tulang padat).
        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        sorted_labels = np.zeros_like(labels)

        # Mengurutkan ulang label: 0 untuk background, 1 untuk pori, dan 2 untuk padat.
        for i, idx in enumerate(sorted_indices):
            sorted_labels[labels == idx] = i

        # Mengembalikan bentuk data dari vektor menjadi bentuk gambar (2D).
        segmented_image = sorted_labels.reshape(img.shape)

        # --- LANGKAH 3: EKSTRAKSI FITUR ---
        # Daftar nama fitur harus sama persis dengan urutan kolom di database_fitur.csv.
        feature_names = [
            'rasio_p_v_b', 'rasio_p_v_t', 'glcm_contrast', 'glcm_homogeneity',
            'glcm_energy', 'glcm_correlation', 'stat_mean', 'stat_variance'
        ]

        # Menghitung 8 nilai fitur (Rasio & Tekstur) dari gambar yang sedang diperiksa.
        features_new = extract_features_complete(img, segmented_image)

        # Membungkus hasil fitur ke dalam format DataFrame (tabel) Pandas agar model 
        # mengenali nama fiturnya dan tidak memunculkan pesan peringatan (Warning).
        features_df = pd.DataFrame([features_new], columns=feature_names)

        # --- LANGKAH 4: PREDIKSI ---
        # Memasukkan data fitur ke model Random Forest untuk mendapatkan hasil diagnosa.
        diagnosa = model.predict(features_df)[0]

        # Menghitung seberapa besar tingkat keyakinan (persentase) terhadap diagnosa tersebut.
        probabilitas = np.max(model.predict_proba(features_df)) * 100

        # Menampilkan jendela hasil
        show_result(file_path, img_original, img, segmented_image, diagnosa, probabilitas, n_data)

    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan diagnosa:\n{e}")

def show_result(path, img_orig, img_clahe, seg, diagnosa, prob, n_data):
    report_text = (
        f"HASIL DIAGNOSA\n"
        f"----------------------------------\n"
        f"Diagnosa: {diagnosa.upper()}\n"
        f"Confidence Level: {prob:.2f}%\n"
        f"----------------------------------\n"
        f"Berdasarkan {n_data} Data Latih"
    )

    bg_color = "#ddffdd" # Normal
    if "Osteoporosis" in diagnosa: bg_color = "#ffdddd"
    elif "Osteopenia" in diagnosa: bg_color = "#fff4cc"

    plt.figure(figsize=(10, 7))
    plt.subplot(1, 2, 1)
    plt.title("Citra X-Ray")
    plt.imshow(img_clahe, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Hasil Prediksi: {diagnosa}")
    plt.imshow(seg, cmap='viridis')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.figtext(0.5, 0.88, report_text, ha='center', va='top', fontsize=11,
                bbox={"facecolor": bg_color, "alpha": 1, "pad": 10})

    # --- FUNGSI CALLBACK UNTUK TOMBOL HISTOGRAM ---
    def open_hist_orig(event):
        plt.figure("Histogram Original")
        plt.hist(img_orig.ravel(), 256, [0, 256], color='black')
        plt.title("Histogram Citra Original")
        plt.xlabel("Intensitas Piksel"); plt.ylabel("Frekuensi")
        plt.show()

    def open_hist_clahe(event):
        plt.figure("Histogram CLAHE")
        plt.hist(img_clahe.ravel(), 256, [0, 256], color='blue')
        plt.title("Histogram Citra Setelah CLAHE")
        plt.xlabel("Intensitas Piksel"); plt.ylabel("Frekuensi")
        plt.show()

    def open_hist_gmm(event):
        plt.figure("Histogram Segmentasi GMM")
        # Menggunakan 3 bins karena GMM hanya memiliki label 0, 1, 2
        plt.hist(seg.ravel(), bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8, color='green')
        plt.xticks([0, 1, 2], ['Background (0)', 'Pori (1)', 'Padat (2)'])
        plt.title("Distribusi Piksel Hasil Segmentasi GMM")
        plt.ylabel("Jumlah Piksel")
        plt.show()

    # --- MENAMBAHKAN TOMBOL ---
    ax_orig = plt.axes([0.15, 0.02, 0.2, 0.05]) # [left, bottom, width, height]
    ax_clahe = plt.axes([0.40, 0.02, 0.2, 0.05])
    ax_gmm = plt.axes([0.65, 0.02, 0.2, 0.05])

    btn_orig = Button(ax_orig, 'Hist. Original', color='#f0f0f0', hovercolor='lightblue')
    btn_clahe = Button(ax_clahe, 'Hist. CLAHE', color='#f0f0f0', hovercolor='lightblue')
    btn_gmm = Button(ax_gmm, 'Hist. GMM', color='#f0f0f0', hovercolor='lightblue')

    btn_orig.on_clicked(open_hist_orig)
    btn_clahe.on_clicked(open_hist_clahe)
    btn_gmm.on_clicked(open_hist_gmm)

    plt.show()

# ==========================================================
# GUI UTAMA
# ==========================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Diagnosis Citra")
    root.geometry("400x250")
    root.resizable(False, False)

    style = ttk.Style(root)
    style.theme_use('clam')

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")
    
    ttk.Label(main_frame, text="Diagnosis Citra", font=("Arial", 14, "bold")).pack(pady=10)

    btn_action = ttk.Button(main_frame, text="Mulai Pemeriksaan Citra", command=start_diagnosis)
    btn_action.pack(pady=20, ipady=10, fill='x')

    root.mainloop()