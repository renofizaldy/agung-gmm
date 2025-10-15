# ==============================================================================
# Skrip Tahap 2: Ekstraksi Fitur dari Citra Tulang
# ==============================================================================
# Deskripsi:
# Skrip ini memproses kumpulan gambar X-ray dari folder yang terstruktur,
# menjalankan segmentasi GMM, mengekstrak serangkaian fitur, dan
# menyimpan hasilnya dalam satu file CSV untuk digunakan di Tahap 3.
#
# Pustaka yang dibutuhkan:
# - opencv-python-headless
# - scikit-learn
# - matplotlib
# - pandas
# - scikit-image
#
# Cara Menjalankan:
# 1. Siapkan struktur folder gambar Anda.
# 2. Sesuaikan variabel `dataset_folder`.
# 3. Jalankan dari terminal: python extract_features.py
# ==============================================================================

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from skimage.feature import graycomatrix, graycoprops

def extract_features(image_path, diagnosis_label):
    """
    Fungsi untuk memproses satu gambar: segmentasi GMM dan ekstraksi fitur.
    """
    print(f"Processing: {image_path}")

    # --- 1. Segmentasi GMM (dari Tahap 1) ---
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        return None # Lewati jika gambar tidak bisa dibaca

    pixel_values = img_original.reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=3, random_state=42)
    labels = gmm.fit_predict(pixel_values)

    # Mengurutkan label: 0=background, 1=berpori, 2=padat
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    sorted_labels = np.zeros_like(labels)
    for i, idx in enumerate(sorted_indices):
        sorted_labels[labels == idx] = i
        
    segmented_image = sorted_labels.reshape(img_original.shape)
    
    # Asumsi label: 0=background, 1=tulang berpori, 2=tulang padat
    label_berpori = 1
    label_padat = 2
    
    # --- 2. Ekstraksi Fitur ---
    
    # A. Fitur Rasio Proporsi Cluster
    pixels_padat = np.count_nonzero(segmented_image == label_padat)
    pixels_berpori = np.count_nonzero(segmented_image == label_berpori)
    total_pixels_tulang = pixels_padat + pixels_berpori
    
    if total_pixels_tulang == 0:
        return None # Lewati jika tidak ada tulang yang terdeteksi

    rasio_tulang_padat = pixels_padat / total_pixels_tulang
    rasio_tulang_berpori = pixels_berpori / total_pixels_tulang
    
    # B. Fitur Statistik Intensitas
    intensitas_padat = img_original[segmented_image == label_padat]
    intensitas_berpori = img_original[segmented_image == label_berpori]

    intensitas_mean_padat = np.mean(intensitas_padat) if len(intensitas_padat) > 0 else 0
    intensitas_std_padat = np.std(intensitas_padat) if len(intensitas_padat) > 0 else 0
    intensitas_mean_berpori = np.mean(intensitas_berpori) if len(intensitas_berpori) > 0 else 0
    intensitas_std_berpori = np.std(intensitas_berpori) if len(intensitas_berpori) > 0 else 0

    # C. Fitur Tekstur (GLCM)
    # GLCM dihitung pada gambar grayscale asli
    glcm = graycomatrix(img_original, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    
    glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
    glcm_dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    glcm_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    glcm_energy = graycoprops(glcm, 'energy')[0, 0]
    glcm_correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Mengumpulkan semua fitur dalam sebuah dictionary
    features = {
        'image_id': os.path.basename(image_path),
        'diagnosis': diagnosis_label,
        'rasio_tulang_padat': rasio_tulang_padat,
        'rasio_tulang_berpori': rasio_tulang_berpori,
        'intensitas_mean_padat': intensitas_mean_padat,
        'intensitas_std_padat': intensitas_std_padat,
        'intensitas_mean_berpori': intensitas_mean_berpori,
        'intensitas_std_berpori': intensitas_std_berpori,
        'glcm_contrast': glcm_contrast,
        'glcm_dissimilarity': glcm_dissimilarity,
        'glcm_homogeneity': glcm_homogeneity,
        'glcm_energy': glcm_energy,
        'glcm_correlation': glcm_correlation
    }
    
    return features

# --- Main Program ---
if __name__ == "__main__":
    # PENTING: Sesuaikan path ini dengan lokasi folder dataset Anda
    dataset_folder = 'dataset_tulang' 
    
    # Daftar untuk menampung semua hasil fitur
    all_image_features = []
    
    # Loop melalui setiap subfolder diagnosis
    for diagnosis in ['Normal', 'Osteopenia', 'Osteoporosis']:
        folder_path = os.path.join(dataset_folder, diagnosis)
        
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found for diagnosis '{diagnosis}', skipping.")
            continue
            
        # Loop melalui setiap gambar di dalam subfolder
        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                image_path = os.path.join(folder_path, image_file)
                
                # Ekstrak fitur dari satu gambar
                try:
                    features = extract_features(image_path, diagnosis)
                    if features:
                        all_image_features.append(features)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    # Konversi daftar hasil menjadi Pandas DataFrame
    df = pd.DataFrame(all_image_features)
    
    # Simpan DataFrame ke file CSV
    output_csv_path = 'data_fitur.csv'
    df.to_csv(output_csv_path, index=False)
    
    print("\n" + "="*50)
    print(f"🎉 Proses ekstraksi fitur selesai!")
    print(f"Data disimpan di: {output_csv_path}")
    print(f"Total gambar yang diproses: {len(df)}")
    print("Contoh data:")
    print(df.head())
    print("="*50)