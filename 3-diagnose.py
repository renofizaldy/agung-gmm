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
# FUNGSI PRE-PROCESSING: CLAHE
# ==========================================================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # Terapkan CLAHE agar standar gambar sama dengan database
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    improved_img = clahe.apply(img)
    
    # Gaussian Blur tipis
    improved_img = cv2.GaussianBlur(improved_img, (3, 3), 0)
    
    return improved_img

# ==========================================================
# BAGIAN 1: MEMBACA DATASET
# ==========================================================
def load_dataset_knowledge():
    filename = "database_fitur.csv"
    if not os.path.exists(filename):
        return False, "Database (CSV) tidak ditemukan.\nHarap kumpulkan data dulu."

    data_store = {"Normal": [], "Osteopenia": [], "Osteoporosis": []}
    total_data = 0

    try:
        with open(filename, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    diag = row['diagnosa'] 
                    val = float(row['rasio_padat_vs_berpori'])
                    if diag in data_store:
                        data_store[diag].append(val)
                        total_data += 1
                except ValueError:
                    continue
    except Exception as e:
        return False, f"Error membaca database: {e}"

    if total_data < 5:
        return False, f"Data belum cukup (Baru ada {total_data}).\nKumpulkan minimal 5-10 data dulu."

    knowledge_base = {}
    for label, values in data_store.items():
        if len(values) > 0:
            knowledge_base[label] = {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "count": len(values)
            }
    return True, knowledge_base

# ==========================================================
# BAGIAN 2: PROSES DIAGNOSA
# ==========================================================
def diagnose_image(image_path, knowledge_base):
    # --- PANGGIL PRE-PROCESSING CLAHE ---
    img = preprocess_image(image_path)
    if img is None: return None, None, None, "Gagal baca gambar"

    # GMM Segmentation
    pixel_values = img.reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(pixel_values)
    labels = gmm.predict(pixel_values)
    
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    sorted_labels = np.zeros_like(labels)
    for i, idx in enumerate(sorted_indices):
        sorted_labels[labels == idx] = i
    segmented_image = sorted_labels.reshape(img.shape)

    # Hitung Fitur
    pixels_padat = np.count_nonzero(segmented_image == 2)
    pixels_berpori = np.count_nonzero(segmented_image == 1)
    nilai_pasien = pixels_padat / pixels_berpori if pixels_berpori > 0 else 0.0

    # Logika Keputusan
    hasil_diagnosa = "TIDAK DIKETAHUI"
    matches = []

    for label, stats in knowledge_base.items():
        if stats['min'] <= nilai_pasien <= stats['max']:
            matches.append(label)
    
    if len(matches) == 1:
        hasil_diagnosa = matches[0]
    elif len(matches) > 1:
        hasil_diagnosa = min(matches, key=lambda l: abs(nilai_pasien - knowledge_base[l]['mean']))
    else:
        hasil_diagnosa = min(knowledge_base.keys(), key=lambda l: abs(nilai_pasien - knowledge_base[l]['mean']))
        hasil_diagnosa = f"{hasil_diagnosa} (Diluar Range)"

    return nilai_pasien, img, segmented_image, hasil_diagnosa

def start_doctor_check():
    success, result = load_dataset_knowledge()
    if not success:
        messagebox.showwarning("Aplikasi Belum Siap", result)
        return

    file_path = filedialog.askopenfilename(title="Pilih Citra X-ray")
    if not file_path: return

    nilai, img_asli, img_seg, diagnosa = diagnose_image(file_path, result)
    
    if nilai is not None:
        report_text = (
            f"HASIL DIAGNOSIS\n"
            f"--------------------------------\n"
            f"Nama File: {os.path.basename(file_path)}\n"
            f"Nilai Rasio Tulang: {nilai:.4f}\n"
            f"--------------------------------\n"
            f"KESIMPULAN DIAGNOSA:\n"
            f">>> {diagnosa.upper()} <<<\n"
            f"--------------------------------\n"
            f"Acuan Data: Berdasarkan {sum(k['count'] for k in result.values() if k)} sampel latih."
        )
        
        # Logika Warna Box
        bg_color = "#ddffdd" # Normal
        if "Osteoporosis" in diagnosa:
            bg_color = "#ffdddd" # Merah
        elif "Osteopenia" in diagnosa:
            bg_color = "#fff4cc" # Kuning

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.title("Rontgen Citra")
        plt.imshow(img_asli, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Visualisasi Area ({diagnosa})")
        plt.imshow(img_seg, cmap='viridis')
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
    root.geometry("400x220")
    root.resizable(False, False)

    style = ttk.Style(root)
    style.theme_use('clam')

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")
    
    lbl_title = ttk.Label(main_frame, text="Diagnosis Citra", font=("Arial", 14, "bold"))
    lbl_title.pack(pady=10)

    lbl_desc = ttk.Label(main_frame, text="Sistem akan membaca 'database_fitur.csv'\ndan mencocokkan citra baru.", justify="center")
    lbl_desc.pack(pady=5)

    btn_action = ttk.Button(main_frame, text="Mulai Pemeriksaan Citra", command=start_doctor_check)
    btn_action.pack(pady=20, ipady=10, fill='x')

    root.mainloop()