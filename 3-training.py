import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from skimage.feature import graycomatrix, graycoprops # GLCM
import matplotlib
matplotlib.use('TkAgg') 

import matplotlib.pyplot as plt
import os
import csv 

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
# FUNGSI EKSTRAKSI FITUR TEKSTUR (GLCM & STATISTIK)
# ==========================================================
def extract_additional_features(img):
    """Menghitung fitur GLCM dan Statistik Tekstur"""
    # 1. Statistik Tekstur Dasar (Mean & Variance)
    mean_val = np.mean(img)
    var_val = np.var(img)

    # 2. GLCM (Gray-Level Co-occurrence Matrix)
    # Menggunakan jarak 1 piksel dan sudut 0 derajat untuk efisiensi
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    return {
        "mean": mean_val,
        "variance": var_val,
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation
    }

# ==========================================================
# FUNGSI INTI ANALISIS
# ==========================================================
def run_analysis(image_path, diagnosis_label, silent_mode=False, n_clusters=3):
    try:
        img = preprocess_image(image_path)
        if img is None: return False

        # --- GMM SEGMENTATION ---
        # Perbaikan: Tambahkan .astype(np.float64) untuk menghindari RuntimeWarning
        pixel_values = img.reshape(-1, 1).astype(np.float64)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(pixel_values) 
        labels = gmm.predict(pixel_values)

        # Mengurutkan label cluster berdasarkan kecerahan rata-rata
        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        sorted_labels = np.zeros_like(labels)
        for i, idx in enumerate(sorted_indices):
            sorted_labels[labels == idx] = i

        segmented_image = sorted_labels.reshape(img.shape)

        # --- FITUR RASIO ---
        pixels_padat = np.count_nonzero(segmented_image == 2)
        pixels_berpori = np.count_nonzero(segmented_image == 1)
        pixels_total_tulang = pixels_padat + pixels_berpori
        rasio_p_v_b = pixels_padat / pixels_berpori if pixels_berpori > 0 else 0.0
        rasio_p_v_t = pixels_padat / pixels_total_tulang if pixels_total_tulang > 0 else 0.0

        # --- FITUR TEKSTUR & STATISTIK ---
        extra = extract_additional_features(img)

        # --- SIMPAN KE CSV ---
        file_name = os.path.basename(image_path)
        csv_filename = "database_fitur.csv"
        file_exists = os.path.isfile(csv_filename)
        
        header = [
            'nama_file', 'rasio_p_v_b', 'rasio_p_v_t', 
            'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
            'stat_mean', 'stat_variance', 'diagnosa'
        ]

        with open(csv_filename, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(header)
            
            writer.writerow([
                file_name, rasio_p_v_b, rasio_p_v_t,
                extra['contrast'], extra['homogeneity'], extra['energy'], extra['correlation'],
                extra['mean'], extra['variance'], diagnosis_label
            ])

        print(f"Data Berhasil Disimpan: {file_name}")

        # Visualisasi (Jika bukan batch)
        if not silent_mode:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title("Original (CLAHE)")
            plt.subplot(1, 2, 2); plt.imshow(segmented_image, cmap='viridis'); plt.title("GMM Segmentation")
            plt.show()

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

# ==========================================================
# GUI CONTROL
# ==========================================================
def select_image_and_run():
    # 1. Ambil nilai dari Dropdown Diagnosa
    selected_diagnosis = diagnosis_combobox.get()
    
    # 2. Validasi: Pastikan user sudah memilih diagnosa
    if not selected_diagnosis:
        messagebox.showwarning("Peringatan", "Pilih jenis Diagnosa dulu!")
        return

    # 3. Buka Dialog File (BISA PILIH BANYAK / MULTIPLE)
    file_paths = filedialog.askopenfilenames(
        title=f"Pilih Gambar ({selected_diagnosis})",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif"), ("All Files", "*.*")]
    )

    # Jika tidak memilih file
    if not file_paths:
        status_label.config(text="Batal memilih file.", foreground="red")
        return

    # 4. Cek Jumlah File & Tentukan Mode
    total = len(file_paths)
    silent = total > 1
    success = 0
    
    for i, path in enumerate(file_paths):
        if silent:
            status_label.config(text=f"Memproses {i+1}/{total}...", foreground="blue")
            root.update() # Update UI agar tidak freeze

        # Jalankan Analisis
        if run_analysis(path, selected_diagnosis, silent_mode=silent):
            success += 1

    # 5. Laporan Selesai
    msg = f"Selesai! {success}/{total} data berhasil disimpan."
    status_label.config(text=msg, foreground="green")
    if silent: messagebox.showinfo("Sukses", msg)


# --- Setup Jendela Utama GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Training Data")
    root.geometry("450x250")
    root.resizable(False, False) 

    style = ttk.Style(root)
    style.theme_use('clam') 

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")

    ttk.Label(main_frame, text="Pilih Label Diagnosa", font=("Arial", 10, "bold")).pack(pady=5)
    diagnosis_combobox = ttk.Combobox(main_frame, state="readonly", values=("Normal", "Osteopenia", "Osteoporosis"))
    diagnosis_combobox.pack(pady=5); diagnosis_combobox.current(0)

    ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=15)

    status_label = ttk.Label(main_frame, text="Menunggu input...", justify="center", foreground="gray")
    status_label.pack(pady=(0, 5))

    ttk.Button(main_frame, text="Pilih Gambar & Proses", command=select_image_and_run).pack(pady=10, ipady=5)

    root.mainloop()