# ==============================================================================
# Aplikasi GUI untuk Segmentasi Citra Kepadatan Tulang
# ==============================================================================
# Deskripsi:
# Aplikasi ini menyediakan antarmuka grafis sederhana untuk memilih gambar
# dan secara otomatis menjalankan analisis segmentasi menggunakan GMM-EM.
#
# Pustaka yang dibutuhkan:
# - opencv-python-headless
# - scikit-learn
# - matplotlib
# (Tkinter sudah termasuk dalam instalasi standar Python)
#
# Cara Menjalankan:
# 1. Pastikan semua pustaka di atas sudah terinstal.
# 2. Simpan skrip ini sebagai file Python (misal: app_gui.py).
# 3. Jalankan dari terminal: python app_gui.py
# 4. Sebuah jendela akan muncul, klik tombol untuk memilih gambar.
# ==============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

# Tambahkan dua baris ini di sini
import matplotlib
matplotlib.use('TkAgg') # Memaksa Matplotlib menggunakan backend Tkinter

import matplotlib.pyplot as plt
import os

# --- Fungsi Inti Analisis (diambil dari skrip sebelumnya) ---
def run_analysis(image_path, n_clusters=3):
    """
    Fungsi ini menjalankan proses segmentasi GMM-EM pada gambar yang diberikan.
    """
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

        # Menampilkan hasil menggunakan Matplotlib
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Gambar Asli")
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"Hasil Segmentasi GMM ({n_clusters} Cluster)")
        plt.imshow(segmented_image, cmap='viridis')
        plt.axis('off')

        plt.suptitle("Hasil Analisis Segmentasi", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        messagebox.showerror("Error Analisis", f"Terjadi kesalahan saat memproses gambar:\n{e}")

# --- Fungsi untuk GUI ---
def select_image_and_run():
    """
    Membuka dialog untuk memilih file gambar, lalu menjalankan analisis.
    """
    # Membuka dialog file dan memfilter hanya untuk file gambar
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar X-ray",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif"),
            ("All Files", "*.*")
        ]
    )

    # Jika pengguna memilih file (path tidak kosong)
    if file_path:
        # Menampilkan path file yang dipilih di label
        file_name = os.path.basename(file_path)
        status_label.config(text=f"File Dipilih: {file_name}", foreground="blue")

        # Jalankan fungsi analisis
        run_analysis(file_path, n_clusters=3)
        
        # Kembalikan teks label ke default setelah analisis selesai
        status_label.config(text="Pilih gambar untuk dianalisis.", foreground="black")
    else:
        # Jika pengguna membatalkan dialog
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
    style.theme_use('clam') # Anda bisa mencoba 'alt', 'default', 'classic'

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
    select_button.pack(pady=10, ipady=5) # ipady memberi padding vertikal di dalam tombol

    # Menjalankan aplikasi
    root.mainloop()