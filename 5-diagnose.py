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
import os

# ==========================================================
# FUNGSI PRE-PROCESSING: CLAHE
# ==========================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    improved_img = clahe.apply(img)
    improved_img = cv2.GaussianBlur(improved_img, (3, 3), 0)
    return improved_img

# ==========================================================
# FUNGSI EKSTRAKSI FITUR
# ==========================================================
def extract_features_complete(img, segmented_image):
    # 1. Fitur Rasio dari GMM
    pixels_padat = np.count_nonzero(segmented_image == 2)
    pixels_berpori = np.count_nonzero(segmented_image == 1)
    pixels_total = pixels_padat + pixels_berpori
    rasio_pvb = pixels_padat / pixels_berpori if pixels_berpori > 0 else 0.0
    rasio_pvt = pixels_padat / pixels_total if pixels_total > 0 else 0.0

    # 2. Statistik & GLCM
    mean_val = np.mean(img)
    var_val = np.var(img)
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    return [
        rasio_pvb, rasio_pvt,
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        mean_val, var_val
    ]

# ==========================================================
# BAGIAN 1: TRAINING MODEL
# ==========================================================
def train_ai_model():
    filename = "database_fitur.csv"
    if not os.path.exists(filename):
        return None, "Database (CSV) tidak ditemukan. Harap Training data dulu."

    try:
        df = pd.read_csv(filename)
        if len(df) < 5:
            return None, "Data di database minimal 5 sampel untuk mulai belajar."

        # Memisahkan Fitur (X) dan Label Diagnosa (y)
        X = df[['rasio_p_v_b', 'rasio_p_v_t', 'glcm_contrast', 'glcm_homogeneity', 
                'glcm_energy', 'glcm_correlation', 'stat_mean', 'stat_variance']]
        y = df['diagnosa']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, len(df)
    except Exception as e:
        return None, f"Error membaca database: {e}"

# ==========================================================
# BAGIAN 2: PROSES DIAGNOSA CITRA BARU
# ==========================================================
def start_diagnosis():
    model, n_data = train_ai_model()
    if model is None:
        messagebox.showwarning("Peringatan", n_data)
        return

    file_path = filedialog.askopenfilename(title="Pilih Citra X-ray")
    if not file_path: return

    try:
        img = preprocess_image(file_path)
        # Perbaikan: Tambahkan .astype(np.float64)
        pixel_values = img.reshape(-1, 1).astype(np.float64)
        gmm = GaussianMixture(n_components=3, random_state=42).fit(pixel_values)
        labels = gmm.predict(pixel_values)

        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        sorted_labels = np.zeros_like(labels)
        for i, idx in enumerate(sorted_indices):
            sorted_labels[labels == idx] = i
        segmented_image = sorted_labels.reshape(img.shape)

        # Urutan nama fitur untuk DataFrame
        feature_names = [
            'rasio_p_v_b', 'rasio_p_v_t', 'glcm_contrast', 'glcm_homogeneity',
            'glcm_energy', 'glcm_correlation', 'stat_mean', 'stat_variance'
        ]

        # Ekstrak fitur
        features_new = extract_features_complete(img, segmented_image)

        # Perbaikan: Gunakan DataFrame agar tidak muncul UserWarning tentang Feature Names
        features_df = pd.DataFrame([features_new], columns=feature_names)

        # 4. PREDIKSI MENGGUNAKAN AI
        diagnosa = model.predict(features_df)[0]
        probabilitas = np.max(model.predict_proba(features_df)) * 100

        show_result(file_path, img, segmented_image, diagnosa, probabilitas, n_data)

    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan diagnosa:\n{e}")

def show_result(path, img, seg, diagnosa, prob, n_data):
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

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.title("Citra X-Ray")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Hasil Prediksi: {diagnosa}")
    plt.imshow(seg, cmap='viridis')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.figtext(0.5, 0.88, report_text, ha='center', va='top', fontsize=11,
                bbox={"facecolor": bg_color, "alpha": 1, "pad": 10})
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

    ttk.Label(main_frame, text="Sistem akan membaca 'database_fitur.csv'\ndan mencocokkan citra baru.", justify="center").pack(pady=5)

    btn_action = ttk.Button(main_frame, text="Mulai Pemeriksaan Citra", command=start_diagnosis)
    btn_action.pack(pady=20, ipady=10, fill='x')

    root.mainloop()