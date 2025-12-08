import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import os

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
        # TAMBAHAN: Menghitung 3 Fitur Rasio
        # ==========================================================
        # Asumsi setelah diurutkan:
        # Label 0 = Background (Paling Gelap)
        # Label 1 = Tulang Berpori (Abu-abu)
        # Label 2 = Tulang Padat (Paling Terang)
        
        pixels_padat = np.count_nonzero(segmented_image == 2)
        pixels_berpori = np.count_nonzero(segmented_image == 1)
        pixels_total_tulang = pixels_padat + pixels_berpori
        pixels_total_gambar = img.size # Total piksel di seluruh gambar

        # 1. rasio_total_tulang_terhadap_gambar
        if pixels_total_gambar > 0:
            rasio_total_tulang_terhadap_gambar = pixels_total_tulang / pixels_total_gambar
        else:
            rasio_total_tulang_terhadap_gambar = 0.0

        # 2. rasio_tulang_padat_terhadap_total_tulang
        if pixels_total_tulang > 0:
            rasio_tulang_padat_terhadap_total_tulang = pixels_padat / pixels_total_tulang
        else:
            rasio_tulang_padat_terhadap_total_tulang = 0.0
            
        # 3. rasio_padat_vs_berpori
        if pixels_berpori > 0:
            rasio_padat_vs_berpori = pixels_padat / pixels_berpori
        else:
            # Jika tidak ada piksel berpori (mungkin sangat padat atau error segmentasi)
            # beri nilai tinggi (pixels_padat + 1) untuk penanda
            rasio_padat_vs_berpori = pixels_padat + 1.0 
            if pixels_total_tulang == 0: # Jika keduanya 0
                rasio_padat_vs_berpori = 0.0

        # 1. Dapatkan nama file
        file_name = os.path.basename(image_path)
        
        # 2. Tambahkan nama file ke string teks
        fitur_text = (
            f"File: {file_name}\n"
            f"1. Rasio Total Tulang: {rasio_total_tulang_terhadap_gambar:.4f}\n"
            f"2. Rasio Padat / Total Tulang: {rasio_tulang_padat_terhadap_total_tulang:.4f}\n"
            f"3. Rasio Padat vs Berpori: {rasio_padat_vs_berpori:.4f}"
        )

        # Menampilkan hasil menggunakan Matplotlib
        plt.figure(figsize=(12, 7)) # Tinggi

        plt.subplot(1, 2, 1)
        plt.title("Gambar Asli")
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"Hasil Segmentasi GMM ({n_clusters} Cluster)")
        plt.imshow(segmented_image, cmap='viridis')
        plt.axis('off')

        plt.suptitle("Hasil Analisis Segmentasi", fontsize=16)
        
        # Mengatur layout agar ada ruang di bawah untuk teks
        plt.tight_layout(rect=[0, 0.1, 1, 0.95]) 
        
        # Menampilkan teks fitur di bawah gambar
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
        # Menampilkan path file yang dipilih di label
        file_name = os.path.basename(file_path)
        status_label.config(text=f"File Dipilih: {file_name}", foreground="blue")

        # Jalankan fungsi analisis
        run_analysis(file_path, n_clusters=3)
        
        # Kembalikan teks label ke default setelah analisis selesai
        status_label.config(text="Pilih gambar untuk dianalisis.", foreground="black")
    else:
        # Jika membatalkan dialog
        status_label.config(text="Tidak ada file yang dipilih.", foreground="red")


# --- Setup Jendela Utama GUI ---
if __name__ == "__main__":
    # Membuat jendela utama
    root = tk.Tk()
    root.title("Segmentasi Gambar GMM-EM")
    root.geometry("400x150") # Mengatur ukuran jendela (lebar x tinggi)
    root.resizable(False, False) # Membuat jendela tidak bisa diubah ukurannya

    # Mengatur style untuk tampilan yang lebih modern
    style = ttk.Style(root)
    style.theme_use('clam') # 'alt', 'default', 'classic'

    # Membuat frame utama untuk menampung widget
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")

    # Membuat label status
    status_label = ttk.Label(
        main_frame,
        text="Silakan pilih gambar untuk dianalisis.",
        justify="center"
    )
    status_label.pack(pady=(0, 10))

    # Membuat tombol utama
    select_button = ttk.Button(
        main_frame,
        text="📁 Pilih Gambar dan Jalankan Analisis",
        command=select_image_and_run
    )
    select_button.pack(pady=10, ipady=5) # padding vertikal di dalam tombol

    # Menjalankan aplikasi
    root.mainloop()