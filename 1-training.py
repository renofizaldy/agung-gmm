import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('TkAgg') 

import matplotlib.pyplot as plt
import os
import csv

# ==========================================================
# FUNGSI PRE-PROCESSING: CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ==========================================================
def preprocess_image(image_path):
    """
    Membaca gambar dan menerapkan CLAHE untuk menormalkan kontras.
    Agar gambar gelap/terang diperlakukan sama.
    """
    # 1. Baca gambar mode grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return None
    
    # 2. Terapkan CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clipLimit=2.0 (standar medis), tileGridSize=(8,8) (ukuran grid lokal)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    improved_img = clahe.apply(img)
    
    # 3. Gaussian Blur tipis untuk hilangkan noise bintik (opsional tapi bagus)
    improved_img = cv2.GaussianBlur(improved_img, (3, 3), 0)
    
    return improved_img

# ==========================================================
# FUNGSI INTI ANALISIS
# ==========================================================
def run_analysis(image_path, diagnosis_label, silent_mode=False, n_clusters=3):
    try:
        # --- LANGKAH 1: PRE-PROCESSING (DIGANTI) ---
        # Before: img = cv2.imread(...)
        # Sekarang: Panggil fungsi preprocess
        img = preprocess_image(image_path)
        
        if img is None:
            if not silent_mode: messagebox.showerror("Error", f"Gagal memuat:\n{image_path}")
            return False

        # --- LANGKAH 2: GMM SEGMENTATION ---
        pixel_values = img.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(pixel_values) 
        labels = gmm.predict(pixel_values)

        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        sorted_labels = np.zeros_like(labels)
        for i, idx in enumerate(sorted_indices):
            sorted_labels[labels == idx] = i

        segmented_image = sorted_labels.reshape(img.shape)

        # --- LANGKAH 3: EKSTRAKSI FITUR ---
        pixels_padat = np.count_nonzero(segmented_image == 2)
        pixels_berpori = np.count_nonzero(segmented_image == 1)
        pixels_total_tulang = pixels_padat + pixels_berpori
        
        # Fitur 1: Rasio Utama
        if pixels_berpori > 0:
            rasio_padat_vs_berpori = pixels_padat / pixels_berpori
        else:
            rasio_padat_vs_berpori = pixels_padat + 1.0 
            if pixels_total_tulang == 0: rasio_padat_vs_berpori = 0.0

        # Fitur 2: Rasio Validator
        if pixels_total_tulang > 0:
            rasio_tulang_padat_terhadap_total_tulang = pixels_padat / pixels_total_tulang
        else:
            rasio_tulang_padat_terhadap_total_tulang = 0.0

        # --- LANGKAH 4: SIMPAN KE CSV ---
        file_name = os.path.basename(image_path)
        csv_filename = "database_fitur.csv"
        file_exists = os.path.isfile(csv_filename)
        
        with open(csv_filename, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(['nama_file', 'rasio_padat_vs_berpori', 'rasio_tulang_padat_terhadap_total_tulang', 'diagnosa'])
            writer.writerow([file_name, rasio_padat_vs_berpori, rasio_tulang_padat_terhadap_total_tulang, diagnosis_label])

        print(f"Disimpan: {file_name} -> {diagnosis_label}")

        # --- LANGKAH 5: VISUALISASI (HANYA JIKA BUKAN SILENT MODE) ---
        if not silent_mode:
            fitur_text = (
                f"File: {file_name}\n"
                f"Label: {diagnosis_label.upper()}\n"
                f"---------------------------\n"
                f"1. Rasio Utama: {rasio_padat_vs_berpori:.4f}\n"
                f"2. Rasio Validasi: {rasio_tulang_padat_terhadap_total_tulang:.4f}\n\n"
                f"(CLAHE Applied & Disimpan)"
            )

            plt.figure(figsize=(12, 7)) 
            plt.subplot(1, 2, 1)
            plt.title("Gambar (Pre-processed CLAHE)")
            plt.imshow(img, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title(f"Segmentasi GMM ({diagnosis_label})")
            plt.imshow(segmented_image, cmap='viridis')
            plt.axis('off')

            plt.suptitle(f"Ekstraksi Data: {diagnosis_label}", fontsize=16)
            plt.tight_layout(rect=[0, 0.1, 1, 0.95]) 
            plt.figtext(0.5, 0.01, fitur_text, ha='center', va='bottom', fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
            plt.show()

        return True # Berhasil

    except Exception as e:
        print(f"Error pada {image_path}: {e}")
        return False

# ==========================================================
# GUI CONTROL
# ==========================================================
def select_image_and_run():
    # 1. Ambil Label
    selected_diagnosis = diagnosis_combobox.get()
    if not selected_diagnosis:
        messagebox.showwarning("Peringatan", "Pilih jenis Diagnosa dulu!")
        return

    # 2. Buka Dialog File (BISA PILIH BANYAK / MULTIPLE)
    file_paths = filedialog.askopenfilenames(
        title=f"Pilih Gambar ({selected_diagnosis})",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif"), ("All Files", "*.*")]
    )

    if not file_paths:
        status_label.config(text="Batal memilih file.", foreground="red")
        return

    # 3. Cek Jumlah File & Tentukan Mode
    total_files = len(file_paths)
    is_silent_mode = False
    
    if total_files > 1:
        is_silent_mode = True # Aktifkan mode batch (tanpa grafik)
        status_label.config(text=f"Mulai memproses {total_files} gambar...", foreground="blue")
    
    # 4. Looping Proses
    success_count = 0
    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        
        # Update status bar progress
        if is_silent_mode:
            status_label.config(text=f"Memproses {i+1}/{total_files}: {file_name}...", foreground="blue")
            root.update() # Wajib update UI agar tidak freeze
        else:
            status_label.config(text=f"Memproses: {file_name}", foreground="blue")

        # Jalankan Analisis
        if run_analysis(file_path, selected_diagnosis, silent_mode=is_silent_mode):
            success_count += 1

    # 5. Laporan Selesai
    final_msg = f"Selesai! {success_count} dari {total_files} data berhasil disimpan."
    status_label.config(text=final_msg, foreground="green")
    
    if is_silent_mode:
        messagebox.showinfo("Batch Processing Selesai", final_msg)

# --- Setup Jendela Utama GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Training Data (Support Batch & CLAHE)")
    root.geometry("450x250")
    root.resizable(False, False) 

    style = ttk.Style(root)
    style.theme_use('clam') 

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")

    # Dropdown
    lbl_instruction = ttk.Label(main_frame, text="Langkah 1: Pilih Label Diagnosa", font=("Arial", 10, "bold"))
    lbl_instruction.pack(pady=(0, 5))

    diagnosis_var = tk.StringVar()
    diagnosis_combobox = ttk.Combobox(main_frame, textvariable=diagnosis_var, state="readonly")
    diagnosis_combobox['values'] = ("Normal", "Osteopenia", "Osteoporosis")
    diagnosis_combobox.pack(pady=5)
    diagnosis_combobox.current(0)

    ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=15)

    # Tombol
    lbl_action = ttk.Label(main_frame, text="Langkah 2: Pilih File (Bisa Banyak Sekaligus)", font=("Arial", 10))
    lbl_action.pack(pady=(0, 5))

    status_label = ttk.Label(main_frame, text="Siap menerima input...", justify="center", foreground="gray")
    status_label.pack(pady=(0, 5))

    select_button = ttk.Button(
        main_frame,
        text="Pilih Gambar & Proses Batch",
        command=select_image_and_run
    )
    select_button.pack(pady=10, ipady=5) 

    root.mainloop()