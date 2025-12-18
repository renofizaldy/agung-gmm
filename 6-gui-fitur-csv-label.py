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

# --- Fungsi Inti Analisis ---
def run_analysis(image_path, diagnosis_label, n_clusters=3):
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
        gmm.fit(pixel_values) 
        labels = gmm.predict(pixel_values)

        # Mengurutkan label cluster berdasarkan kecerahan rata-rata
        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        sorted_labels = np.zeros_like(labels)
        for i, idx in enumerate(sorted_indices):
            sorted_labels[labels == idx] = i

        segmented_image = sorted_labels.reshape(img.shape)

        # ==========================================================
        # MENGHITUNG FITUR 
        # ==========================================================
        
        # Asumsi label: 0=Background, 1=Berpori, 2=Padat
        pixels_padat = np.count_nonzero(segmented_image == 2)
        pixels_berpori = np.count_nonzero(segmented_image == 1)
        pixels_total_tulang = pixels_padat + pixels_berpori
        
        # --- Fitur 1: Rasio Padat vs Berpori (UTAMA) ---
        if pixels_berpori > 0:
            rasio_padat_vs_berpori = pixels_padat / pixels_berpori
        else:
            rasio_padat_vs_berpori = pixels_padat + 1.0 
            if pixels_total_tulang == 0: 
                rasio_padat_vs_berpori = 0.0

        # --- Fitur 2: Rasio Padat / Total Tulang (VALIDATOR) ---
        if pixels_total_tulang > 0:
            rasio_tulang_padat_terhadap_total_tulang = pixels_padat / pixels_total_tulang
        else:
            rasio_tulang_padat_terhadap_total_tulang = 0.0

        # ==========================================================
        # MENYIMPAN KE CSV DENGAN LABEL DIAGNOSA
        # ==========================================================
        file_name = os.path.basename(image_path)
        csv_filename = "database_fitur.csv"
        
        file_exists = os.path.isfile(csv_filename)
        
        with open(csv_filename, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Tulis Header jika file baru dibuat (Sekarang ada kolom 'diagnosa')
            if not file_exists:
                writer.writerow(['nama_file', 'rasio_padat_vs_berpori', 'rasio_tulang_padat_terhadap_total_tulang', 'diagnosa'])
            
            # Tulis Baris Data Baru beserta Label Diagnosa
            writer.writerow([file_name, rasio_padat_vs_berpori, rasio_tulang_padat_terhadap_total_tulang, diagnosis_label])

        print(f"Data tersimpan: {file_name} -> {diagnosis_label}")

        # ==========================================================
        # MENAMPILKAN HASIL VISUAL
        # ==========================================================
        
        fitur_text = (
            f"File: {file_name}\n"
            f"Label Diagnosa: {diagnosis_label.upper()}\n"
            f"---------------------------\n"
            f"1. Rasio Padat vs Berpori: {rasio_padat_vs_berpori:.4f}\n"
            f"2. Rasio Padat / Total: {rasio_tulang_padat_terhadap_total_tulang:.4f}\n\n"
            f"(Data tersimpan ke CSV)"
        )

        plt.figure(figsize=(12, 7)) 

        plt.subplot(1, 2, 1)
        plt.title("Gambar Asli")
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"Segmentasi GMM (Label: {diagnosis_label})")
        plt.imshow(segmented_image, cmap='viridis')
        plt.axis('off')

        plt.suptitle(f"Ekstraksi Data: {diagnosis_label}", fontsize=16)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95]) 
        
        plt.figtext(0.5, 0.01, fitur_text, ha='center', va='bottom', fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        plt.show()

    except Exception as e:
        messagebox.showerror("Error Analisis", f"Terjadi kesalahan:\n{e}")

# --- Fungsi untuk GUI ---
def select_image_and_run():
    # 1. Ambil nilai dari Dropdown Diagnosa
    selected_diagnosis = diagnosis_combobox.get()
    
    # 2. Validasi: Pastikan user sudah memilih diagnosa
    if not selected_diagnosis:
        messagebox.showwarning("Peringatan", "Harap pilih jenis Diagnosa (Label) terlebih dahulu!")
        return

    # 3. Buka dialog file
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar X-ray",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif"), ("All Files", "*.*")]
    )

    if file_path:
        file_name = os.path.basename(file_path)
        status_label.config(text=f"Memproses: {file_name} ({selected_diagnosis})", foreground="blue")

        # Jalankan fungsi analisis dengan membawa label diagnosa
        run_analysis(file_path, selected_diagnosis, n_clusters=3)
        
        status_label.config(text="Siap untuk data berikutnya.", foreground="black")
    else:
        status_label.config(text="Batal memilih file.", foreground="red")


# --- Setup Jendela Utama GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Training Data Latih")
    root.geometry("450x250") # Diperbesar sedikit
    root.resizable(False, False) 

    style = ttk.Style(root)
    style.theme_use('clam') 

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")

    # --- Bagian Input Diagnosa (Dropdown) ---
    lbl_instruction = ttk.Label(main_frame, text="Langkah 1: Pilih Label Diagnosa Gambar", font=("Arial", 10, "bold"))
    lbl_instruction.pack(pady=(0, 5))

    diagnosis_var = tk.StringVar()
    diagnosis_combobox = ttk.Combobox(main_frame, textvariable=diagnosis_var, state="readonly")
    diagnosis_combobox['values'] = ("Normal", "Osteopenia", "Osteoporosis")
    diagnosis_combobox.pack(pady=5)
    diagnosis_combobox.current(0) # Default pilih Normal

    ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=15)

    # --- Bagian Tombol ---
    lbl_action = ttk.Label(main_frame, text="Langkah 2: Pilih File Gambar", font=("Arial", 10, "bold"))
    lbl_action.pack(pady=(0, 5))

    status_label = ttk.Label(main_frame, text="Menunggu input...", justify="center", foreground="gray")
    status_label.pack(pady=(0, 5))

    select_button = ttk.Button(
        main_frame,
        text="📁 Buka Gambar & Simpan Data",
        command=select_image_and_run
    )
    select_button.pack(pady=10, ipady=5) 

    root.mainloop()