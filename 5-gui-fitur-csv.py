import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('TkAgg') # Memaksa Matplotlib menggunakan backend Tkinter

import matplotlib.pyplot as plt
import os
import csv # Tambahan modul untuk menangani file CSV

# --- Fungsi Inti Analisis ---
def run_analysis(image_path, n_clusters=3):
    try:
        # Membaca gambar dalam mode grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", f"Gagal memuat gambar dari:\n{image_path}")
            return

        # Mengubah gambar menjadi array 1D piksel
        pixel_values = img.reshape(-1, 1)

        # Inisialisasi dan latih model GMM
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(pixel_values) # Algoritma EM
        labels = gmm.predict(pixel_values)

        # Mengurutkan label cluster berdasarkan kecerahan rata-rata
        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        sorted_labels = np.zeros_like(labels)
        for i, idx in enumerate(sorted_indices):
            sorted_labels[labels == idx] = i

        # Mengubah kembali array label menjadi bentuk gambar asli
        segmented_image = sorted_labels.reshape(img.shape)

        # ==========================================================
        # MENGHITUNG FITUR (Hanya 2 Fitur Sesuai Request)
        # ==========================================================
        
        # Asumsi label: 0=Background, 1=Berpori, 2=Padat
        pixels_padat = np.count_nonzero(segmented_image == 2)
        pixels_berpori = np.count_nonzero(segmented_image == 1)
        pixels_total_tulang = pixels_padat + pixels_berpori
        
        # --- Fitur 1: Rasio Padat vs Berpori (UTAMA) ---
        if pixels_berpori > 0:
            rasio_padat_vs_berpori = pixels_padat / pixels_berpori
        else:
            # Jika tidak ada piksel berpori, beri nilai tinggi sebagai penanda
            rasio_padat_vs_berpori = pixels_padat + 1.0 
            if pixels_total_tulang == 0: 
                rasio_padat_vs_berpori = 0.0

        # --- Fitur 2: Rasio Padat / Total Tulang (VALIDATOR) ---
        if pixels_total_tulang > 0:
            rasio_tulang_padat_terhadap_total_tulang = pixels_padat / pixels_total_tulang
        else:
            rasio_tulang_padat_terhadap_total_tulang = 0.0

        # ==========================================================
        # MENYIMPAN KE CSV
        # ==========================================================
        file_name = os.path.basename(image_path)
        csv_filename = "database_fitur.csv"
        
        # Cek apakah file sudah ada (untuk menentukan perlu tulis header atau tidak)
        file_exists = os.path.isfile(csv_filename)
        
        with open(csv_filename, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Tulis Header jika file baru dibuat
            if not file_exists:
                writer.writerow(['nama_file', 'rasio_padat_vs_berpori', 'rasio_tulang_padat_terhadap_total_tulang'])
            
            # Tulis Baris Data Baru
            writer.writerow([file_name, rasio_padat_vs_berpori, rasio_tulang_padat_terhadap_total_tulang])

        print(f"Data tersimpan ke {csv_filename}: {file_name}")

        # ==========================================================
        # MENAMPILKAN HASIL VISUAL
        # ==========================================================
        
        # String teks untuk ditampilkan (Hanya 2 fitur)
        fitur_text = (
            f"File: {file_name}\n"
            f"--- Hasil Analisis ---\n"
            f"1. Rasio Padat vs Berpori (Utama): {rasio_padat_vs_berpori:.4f}\n"
            f"2. Rasio Padat / Total Tulang (Validator): {rasio_tulang_padat_terhadap_total_tulang:.4f}\n\n"
            f"(Data telah disimpan ke {csv_filename})"
        )

        plt.figure(figsize=(12, 7)) 

        plt.subplot(1, 2, 1)
        plt.title("Gambar Asli")
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"Hasil Segmentasi GMM ({n_clusters} Cluster)")
        plt.imshow(segmented_image, cmap='viridis')
        plt.axis('off')

        plt.suptitle("Analisis & Ekstraksi Fitur Tulang", fontsize=16)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95]) 
        
        plt.figtext(0.5, 0.01, fitur_text, 
                    horizontalalignment='center', 
                    verticalalignment='bottom', 
                    fontsize=10, 
                    bbox={"facecolor":"white", "alpha":0.5, "pad":5})

        plt.show()

    except Exception as e:
        messagebox.showerror("Error Analisis", f"Terjadi kesalahan saat memproses gambar:\n{e}")

# --- Fungsi untuk GUI ---
def select_image_and_run():
    # Membuka dialog file dan memfilter hanya untuk file gambar
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar X-ray",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif"),
            ("All Files", "*.*")
        ]
    )

    # Jika memilih file (path tidak kosong)
    if file_path:
        file_name = os.path.basename(file_path)
        status_label.config(text=f"File Dipilih: {file_name}", foreground="blue")

        # Jalankan fungsi analisis
        run_analysis(file_path, n_clusters=3)
        
        status_label.config(text="Pilih gambar untuk dianalisis.", foreground="black")
    else:
        status_label.config(text="Tidak ada file yang dipilih.", foreground="red")


# --- Setup Jendela Utama GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Aplikasi Ekstraksi Fitur Tulang (GMM)")
    root.geometry("400x150") 
    root.resizable(False, False) 

    style = ttk.Style(root)
    style.theme_use('clam') 

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")

    status_label = ttk.Label(
        main_frame,
        text="Silakan pilih gambar untuk dianalisis.",
        justify="center"
    )
    status_label.pack(pady=(0, 10))

    select_button = ttk.Button(
        main_frame,
        text="📁 Pilih Gambar dan Jalankan Analisis",
        command=select_image_and_run
    )
    select_button.pack(pady=10, ipady=5) 

    root.mainloop()