# ==============================================================================
# Skrip Analisis Citra Kepadatan Tulang Menggunakan GMM-EM
# ==============================================================================
# Deskripsi:
# Skrip ini melakukan segmentasi pada citra grayscale (seperti hasil X-ray)
# untuk mengidentifikasi area dengan intensitas berbeda. Tujuannya adalah
# untuk riset dan eksperimen, bukan untuk diagnosis medis.
#
# Pustaka yang dibutuhkan:
# - opencv-python-headless : Untuk memproses gambar
# - scikit-learn            : Untuk implementasi GMM-EM
# - matplotlib              : Untuk menampilkan hasil
#
# Cara Menjalankan:
# 1. Simpan skrip ini sebagai file Python (misal: analisis_tulang.py).
# 2. Letakkan file gambar yang akan dianalisis di folder yang sama.
# 3. Ganti nilai variabel `image_path` dengan nama file gambar Anda.
# 4. Jalankan dari terminal: python analisis_tulang.py
# ==============================================================================

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def segment_bone_density(image_path, n_clusters=3):
    """
    Fungsi untuk memuat gambar, melakukan segmentasi dengan GMM,
    dan menampilkan hasilnya.

    Args:
        image_path (str): Path menuju file gambar.
        n_clusters (int): Jumlah cluster yang ingin diidentifikasi.
                          Contoh: 3 untuk (background, jaringan lunak, tulang padat).
    """
    # --- Langkah 1: Memuat dan Memproses Gambar ---
    try:
        # Membaca gambar dalam mode grayscale (skala keabuan)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Cek jika gambar gagal dimuat
        if img is None:
            print(f"Error: Gagal memuat gambar dari path: {image_path}")
            print("Pastikan nama file dan path sudah benar.")
            return

        print("Gambar berhasil dimuat.")
        
        # Mengubah gambar 2D (tinggi x lebar) menjadi array 1D (jumlah_piksel x 1)
        # Ini diperlukan karena scikit-learn mengharapkan input 2D
        pixel_values = img.reshape(-1, 1)
        print(f"Gambar diubah menjadi {pixel_values.shape[0]} piksel untuk analisis.")

    except Exception as e:
        print(f"Terjadi error saat memuat gambar: {e}")
        return

    # --- Langkah 2: Menerapkan Algoritma GMM-EM ---
    print(f"Memulai clustering GMM dengan {n_clusters} komponen...")
    try:
        # Inisialisasi model GMM
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        
        # Melatih model pada data piksel
        gmm.fit(pixel_values)
        
        # Memprediksi label cluster untuk setiap piksel
        labels = gmm.predict(pixel_values)
        print("Clustering GMM selesai.")

    except Exception as e:
        print(f"Terjadi error saat menjalankan GMM: {e}")
        return

    # --- Langkah 3: Mengurutkan Cluster Berdasarkan Kecerahan ---
    # Mengurutkan label cluster agar warnanya konsisten:
    # Cluster paling gelap -> 0, paling terang -> n_clusters-1
    # Membuat visualisasi lebih intuitif (tulang padat = warna paling cerah)
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    sorted_labels = np.zeros_like(labels)
    for i, idx in enumerate(sorted_indices):
        sorted_labels[labels == idx] = i
    
    # Mengubah kembali array label yang sudah diurutkan menjadi bentuk gambar asli
    segmented_image = sorted_labels.reshape(img.shape)
    print("Label cluster telah diurutkan berdasarkan intensitas rata-rata.")

    # --- Langkah 4: Menampilkan Hasil ---
    print("Menampilkan gambar asli dan hasil segmentasi...")
    plt.figure(figsize=(12, 6))

    # Tampilkan gambar asli
    plt.subplot(1, 2, 1)
    plt.title("Gambar Asli")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # Tampilkan gambar hasil segmentasi
    plt.subplot(1, 2, 2)
    plt.title(f"Hasil Segmentasi GMM ({n_clusters} Cluster)")
    # 'viridis' adalah colormap yang memberikan kontras warna yang baik
    plt.imshow(segmented_image, cmap='viridis')
    plt.axis('off')

    plt.suptitle("Analisis Kepadatan Tulang (Eksperimental)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Main Program ---
if __name__ == "__main__":
    # GANTI NAMA FILE DI BAWAH INI dengan nama file gambar Anda
    # Pastikan gambar berada di folder yang sama dengan skrip ini.
    image_path = "example_xray.jpg"
    
    # Anda bisa mengubah jumlah cluster jika diperlukan
    # Misalnya, jika ada lebih banyak variasi jaringan yang ingin dipisahkan
    num_of_clusters = 3 
    
    # Memanggil fungsi utama untuk memulai proses
    segment_bone_density(image_path, num_of_clusters)