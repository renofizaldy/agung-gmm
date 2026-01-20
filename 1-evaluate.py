import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from skimage.feature import graycomatrix, graycoprops
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import seaborn as sns # Opsional: Untuk visualisasi heatmap yang lebih bagus

# ==========================================================
# KONFIGURASI DAN UTILITAS
# ==========================================================
FILE_EVAL_TEMP = "temp_hasil_evaluasi.csv"
DATABASE_LATIH = "database_fitur.csv"
LABELS_ORDER = ["Normal", "Osteopenia", "Osteoporosis"]

# ==========================================================
# BAGIAN 1: MESIN DIAGNOSA (REPLIKA DARI 5-DIAGNOSE.PY)
# ==========================================================
# Fungsi ini disalin agar standar penilaian sama persis dengan sistem utama

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    improved_img = clahe.apply(img)
    improved_img = cv2.GaussianBlur(improved_img, (3, 3), 0)
    return improved_img

def extract_features_complete(img, segmented_image):
    pixels_padat = np.count_nonzero(segmented_image == 2)
    pixels_berpori = np.count_nonzero(segmented_image == 1)
    pixels_total = pixels_padat + pixels_berpori
    rasio_pvb = pixels_padat / pixels_berpori if pixels_berpori > 0 else 0.0
    rasio_pvt = pixels_padat / pixels_total if pixels_total > 0 else 0.0

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

def train_model_on_fly():
    """Melatih model RF berdasarkan database_fitur.csv terkini"""
    if not os.path.exists(DATABASE_LATIH):
        return None
    try:
        df = pd.read_csv(DATABASE_LATIH)
        if len(df) < 5: return None
        
        X = df[['rasio_p_v_b', 'rasio_p_v_t', 'glcm_contrast', 'glcm_homogeneity', 
                'glcm_energy', 'glcm_correlation', 'stat_mean', 'stat_variance']]
        y = df['diagnosa']
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        return clf
    except Exception:
        return None

# ==========================================================
# BAGIAN 2: LOGIKA EVALUASI BATCH
# ==========================================================
def run_batch_test():
    # 1. Cek Model
    model = train_model_on_fly()
    if model is None:
        messagebox.showerror("Error", "Gagal melatih model. Pastikan 'database_fitur.csv' ada dan berisi data.")
        return

    # 2. Ambil Input Kelas Aktual dari Dropdown
    actual_class = combo_actual.get()
    if not actual_class:
        messagebox.showwarning("Peringatan", "Pilih KELAS AKTUAL (Kunci Jawaban) terlebih dahulu!")
        return

    # 3. Pilih Banyak File
    file_paths = filedialog.askopenfilenames(
        title=f"Pilih Gambar untuk kelas ASLI: {actual_class}",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif")]
    )
    if not file_paths: return

    # 4. Loop Proses Diagnosa
    results = []
    total_files = len(file_paths)
    
    progress_bar['maximum'] = total_files
    lbl_status.config(text="Memproses data uji...", foreground="blue")
    root.update()

    feature_names = [
        'rasio_p_v_b', 'rasio_p_v_t', 'glcm_contrast', 'glcm_homogeneity',
        'glcm_energy', 'glcm_correlation', 'stat_mean', 'stat_variance'
    ]

    for i, fpath in enumerate(file_paths):
        try:
            # -- Proses Citra (Sama seperti 5-diagnose.py) --
            img = preprocess_image(fpath)
            if img is None: continue

            pixel_values = img.reshape(-1, 1).astype(np.float64)
            gmm = GaussianMixture(n_components=3, random_state=42).fit(pixel_values)
            labels = gmm.predict(pixel_values)
            
            # Sorting label GMM
            means = gmm.means_.flatten()
            sorted_indices = np.argsort(means)
            sorted_labels = np.zeros_like(labels)
            for k, idx in enumerate(sorted_indices):
                sorted_labels[labels == idx] = k
            segmented_image = sorted_labels.reshape(img.shape)

            # Ekstrak & Prediksi
            feats = extract_features_complete(img, segmented_image)
            df_feat = pd.DataFrame([feats], columns=feature_names)
            prediction = model.predict(df_feat)[0]

            # Simpan: [Nama File, Kelas Asli, Prediksi AI]
            results.append({
                'filename': os.path.basename(fpath),
                'y_true': actual_class,
                'y_pred': prediction
            })
            
            progress_bar['value'] = i + 1
            root.update()

        except Exception as e:
            print(f"Skip file {fpath}: {e}")

    # 5. Simpan ke CSV Sementara (Append Mode)
    if results:
        df_res = pd.DataFrame(results)
        # Jika file belum ada, tulis header. Jika ada, append tanpa header.
        mode = 'a' if os.path.exists(FILE_EVAL_TEMP) else 'w'
        header = not os.path.exists(FILE_EVAL_TEMP)
        df_res.to_csv(FILE_EVAL_TEMP, mode=mode, index=False, header=header)
        
        lbl_status.config(text=f"Berhasil menambahkan {len(results)} data {actual_class}.", foreground="green")
        update_summary()
    else:
        lbl_status.config(text="Gagal memproses gambar.", foreground="red")

def reset_evaluation_data():
    if os.path.exists(FILE_EVAL_TEMP):
        if messagebox.askyesno("Reset", "Hapus semua data evaluasi sementara?"):
            os.remove(FILE_EVAL_TEMP)
            update_summary()
            lbl_status.config(text="Data evaluasi direset.", foreground="black")

def update_summary():
    """Menghitung jumlah data yang sudah terkumpul di CSV sementara"""
    if os.path.exists(FILE_EVAL_TEMP):
        try:
            df = pd.read_csv(FILE_EVAL_TEMP)
            counts = df['y_true'].value_counts()
            summary_txt = "Data Terkumpul:\n"
            for label in LABELS_ORDER:
                n = counts.get(label, 0)
                summary_txt += f"- {label}: {n}\n"
            lbl_summary.config(text=summary_txt)
        except:
            lbl_summary.config(text="Data Terkumpul: Error membaca file")
    else:
        lbl_summary.config(text="Data Terkumpul: Kosong")

# ==========================================================
# BAGIAN 3: KALKULASI & LAPORAN AKHIR
# ==========================================================
def calculate_metrics():
    if not os.path.exists(FILE_EVAL_TEMP):
        messagebox.showwarning("Kosong", "Belum ada data uji. Silakan input gambar dulu.")
        return

    df = pd.read_csv(FILE_EVAL_TEMP)
    y_true = df['y_true']
    y_pred = df['y_pred']

    # --- TAHAP 1: CONFUSION MATRIX ---
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)
    
    # --- TAHAP 2: PRECISION & RECALL PER KELAS ---
    report_dict = classification_report(y_true, y_pred, labels=LABELS_ORDER, output_dict=True, zero_division=0)
    
    # --- TAHAP 3: MACRO AVERAGE ---
    macro_avg = report_dict['macro avg']

    # --- VISUALISASI MATRIKS ---
    plt.figure(figsize=(10, 5))
    
    # Plot Heatmap Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_ORDER, yticklabels=LABELS_ORDER)
    plt.ylabel('Aktual (Kunci Jawaban)')
    plt.xlabel('Prediksi Model')
    plt.title('Tahap 1: Confusion Matrix')

    # Tampilkan Teks Laporan
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    report_text = "TAHAP 2: PRECISION & RECALL\n"
    report_text += "-"*30 + "\n"
    for label in LABELS_ORDER:
        p = report_dict[label]['precision']
        r = report_dict[label]['recall']
        report_text += f"{label.upper()}\n"
        report_text += f"   Precision : {p:.2%}\n"
        report_text += f"   Recall    : {r:.2%}\n\n"
    
    report_text += "-"*30 + "\n"
    report_text += "TAHAP 3: MACRO AVERAGE\n"
    report_text += "-"*30 + "\n"
    report_text += f"Macro Precision : {macro_avg['precision']:.2%}\n"
    report_text += f"Macro Recall    : {macro_avg['recall']:.2%}\n"
    # report_text += f"Macro F1-Score  : {macro_avg['f1-score']:.2%}\n"
    
    plt.text(0.05, 0.95, report_text, fontsize=10, verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    plt.show()

# ==========================================================
# GUI UTAMA
# ==========================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Alat Evaluasi Model")
    root.geometry("500x570")
    
    style = ttk.Style(root)
    style.theme_use('clam')

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")

    # HEADER
    ttk.Label(main_frame, text="Evaluasi Pengujian", font=("Arial", 14, "bold")).pack(pady=5)
    ttk.Label(main_frame, text="Lakukan pengujian bertahap untuk mengisi Confusion Matrix", foreground="gray").pack(pady=(0, 15))

    # AREA INPUT
    input_frame = ttk.LabelFrame(main_frame, text="1. Input Data Uji (Batch)", padding="10")
    input_frame.pack(fill="x", pady=5)

    ttk.Label(input_frame, text="Pilih Kelas Aktual (Kunci Jawaban):").pack(anchor="w")
    combo_actual = ttk.Combobox(input_frame, values=LABELS_ORDER, state="readonly")
    combo_actual.pack(fill="x", pady=5)
    combo_actual.current(0)

    btn_add = ttk.Button(input_frame, text="Pilih Gambar & Proses", command=run_batch_test)
    btn_add.pack(fill="x", pady=5)
    
    progress_bar = ttk.Progressbar(input_frame, orient="horizontal", mode="determinate")
    progress_bar.pack(fill="x", pady=5)

    # AREA MONITORING
    summary_frame = ttk.LabelFrame(main_frame, text="Status Data Uji", padding="10")
    summary_frame.pack(fill="x", pady=10)
    
    lbl_summary = ttk.Label(summary_frame, text="Data Terkumpul: Kosong", justify="left", font=("Consolas", 9))
    lbl_summary.pack(anchor="w")
    
    btn_reset = ttk.Button(summary_frame, text="Reset Data", command=reset_evaluation_data)
    btn_reset.pack(anchor="e", pady=2)

    # AREA EKSEKUSI
    btn_calc = ttk.Button(main_frame, text="HITUNG EVALUASI", command=calculate_metrics)
    btn_calc.pack(fill="x", pady=15, ipady=10)

    lbl_status = ttk.Label(main_frame, text="Siap.", foreground="blue")
    lbl_status.pack(side="bottom")

    update_summary() # Cek apakah ada sisa data lama
    root.mainloop()