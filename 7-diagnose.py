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
# BAGIAN 1: OTAK DOKTER (MEMBACA DATASET)
# ==========================================================
def load_dataset_knowledge():
    filename = "database_fitur.csv"
    
    # 1. Cek apakah file database ada
    if not os.path.exists(filename):
        return False, "Database (CSV) tidak ditemukan.\nHarap kumpulkan data dulu menggunakan Script sebelumnya."

    # 2. Baca data dari CSV
    data_store = {"Normal": [], "Osteopenia": [], "Osteoporosis": []}
    total_data = 0
    
    try:
        with open(filename, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    # Ambil diagnosa dan nilai rasio utama
                    diag = row['diagnosa'] 
                    val = float(row['rasio_padat_vs_berpori'])
                    
                    # Masukkan ke list yang sesuai jika labelnya dikenali
                    if diag in data_store:
                        data_store[diag].append(val)
                        total_data += 1
                except ValueError:
                    continue # Lewati baris yang error/kosong
    except Exception as e:
        return False, f"Gagal membaca database: {e}"

    # 3. Validasi jumlah data (Minimal 5 data total)
    if total_data < 5:
        return False, f"Data belum cukup pintar!\nHanya ditemukan {total_data} data.\nKumpulkan minimal 5-10 data gambar dulu."

    # 4. Buat Rangkuman Statistik (Min, Max, Mean) untuk tiap kategori
    knowledge_base = {}
    print("\n--- INFO KECERDASAN DOKTER ---")
    for label, values in data_store.items():
        if len(values) > 0:
            knowledge_base[label] = {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "count": len(values)
            }
            print(f"[{label}] N={len(values)} | Range: {np.min(values):.2f} - {np.max(values):.2f} | Rata2: {np.mean(values):.2f}")
        else:
            knowledge_base[label] = None # Tidak ada data untuk kategori ini

    return True, knowledge_base

# ==========================================================
# BAGIAN 2: PROSES DIAGNOSA
# ==========================================================
def diagnose_image(image_path, knowledge_base):
    # --- Langkah A: Segmentasi & Hitung Fitur (Sama seperti sebelumnya) ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None, "Gagal baca gambar"

    pixel_values = img.reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(pixel_values)
    labels = gmm.predict(pixel_values)
    
    # Sorting label
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    sorted_labels = np.zeros_like(labels)
    for i, idx in enumerate(sorted_indices):
        sorted_labels[labels == idx] = i
    segmented_image = sorted_labels.reshape(img.shape)

    # Hitung Rasio Utama
    pixels_padat = np.count_nonzero(segmented_image == 2)
    pixels_berpori = np.count_nonzero(segmented_image == 1)
    
    if pixels_berpori > 0:
        nilai_pasien = pixels_padat / pixels_berpori
    else:
        nilai_pasien = pixels_padat + 1.0 # Fallback

    # --- Langkah B: ALGORITMA PENGAMBILAN KEPUTUSAN ---
    # Logika: Cek apakah nilai masuk range Min-Max salah satu kategori
    
    hasil_diagnosa = "TIDAK DIKETAHUI / SUSPECT"
    jarak_terdekat = float('inf')
    
    matches = []

    print(f"\nAnalisis Pasien Baru: Nilai = {nilai_pasien:.4f}")

    # Cek 1: Apakah masuk Range Pasti?
    for label, stats in knowledge_base.items():
        if stats is not None:
            # Cek apakah nilai ada di antara Min dan Max kategori ini
            if stats['min'] <= nilai_pasien <= stats['max']:
                matches.append(label)
    
    if len(matches) == 1:
        hasil_diagnosa = matches[0] # Cocok sempurna dengan satu kategori
    elif len(matches) > 1:
        # Jika overlap (masuk 2 kategori), pilih yang jarak ke Rata-ratanya paling dekat
        best_label = None
        min_dist = float('inf')
        for label in matches:
            dist = abs(nilai_pasien - knowledge_base[label]['mean'])
            if dist < min_dist:
                min_dist = dist
                best_label = label
        hasil_diagnosa = best_label
    else:
        # Jika tidak masuk range manapun (Area Abu-abu), cari Rata-rata terdekat
        # Ini logika "Nearest Neighbor"
        for label, stats in knowledge_base.items():
            if stats is not None:
                dist = abs(nilai_pasien - stats['mean'])
                if dist < jarak_terdekat:
                    jarak_terdekat = dist
                    hasil_diagnosa = f"Mirip {label} (Di luar range)"

    return nilai_pasien, img, segmented_image, hasil_diagnosa

# ==========================================================
# BAGIAN 3: GUI & INTERAKSI
# ==========================================================
def start_doctor_check():
    # 1. Load Pengetahuan dulu
    success, result = load_dataset_knowledge()
    
    if not success:
        messagebox.showwarning("Dokter Belum Siap", result)
        return

    knowledge_base = result
    
    # 2. Pilih Gambar Pasien
    file_path = filedialog.askopenfilename(title="Pilih Gambar Pasien X-ray")
    if not file_path: return

    # 3. Lakukan Diagnosa
    nilai, img_asli, img_seg, diagnosa = diagnose_image(file_path, knowledge_base)
    
    # 4. Tampilkan Hasil
    if nilai is not None:
        # Siapkan Teks Laporan
        report_text = (
            f"HASIL PEMERIKSAAN DOKTER DIGITAL\n"
            f"================================\n"
            f"Nama File: {os.path.basename(file_path)}\n"
            f"Nilai Rasio Tulang: {nilai:.4f}\n"
            f"--------------------------------\n"
            f"KESIMPULAN DIAGNOSA:\n"
            f">>> {diagnosa.upper()} <<<\n"
            f"--------------------------------\n"
            f"Acuan Data: Berdasarkan {sum(k['count'] for k in knowledge_base.values() if k)} sampel latih."
        )
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Rontgen Pasien")
        plt.imshow(img_asli, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Visualisasi Area ({diagnosa})")
        plt.imshow(img_seg, cmap='viridis')
        plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        plt.figtext(0.5, 0.88, report_text, 
                    ha='center', va='top', fontsize=11, 
                    bbox={"facecolor":"#ffdddd" if "Osteo" in diagnosa else "#ddffdd", "alpha":1, "pad":10})
        
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Aplikasi Dokter Digital (AI Diagnosis)")
    root.geometry("400x200")
    style = ttk.Style(root)
    style.theme_use('clam')

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")
    
    lbl_title = ttk.Label(main_frame, text="Sistem Pakar Diagnosa Tulang", font=("Arial", 14, "bold"))
    lbl_title.pack(pady=10)

    lbl_desc = ttk.Label(main_frame, text="Sistem akan membaca 'database_fitur.csv'\ndan mencocokkan pasien baru.", justify="center")
    lbl_desc.pack(pady=5)

    btn_action = ttk.Button(main_frame, text="👨‍⚕️ Mulai Pemeriksaan Pasien", command=start_doctor_check)
    btn_action.pack(pady=20, ipady=10, fill='x')

    root.mainloop()
