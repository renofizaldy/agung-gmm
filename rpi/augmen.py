import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os

# ==========================================================
# FUNGSI PENGOLAHAN CITRA (AUGMENTASI)
# ==========================================================

def rotate_image(image, angle):
    """Memutar gambar tanpa menyisakan ruang hitam (Auto-Crop)"""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    
    # Menghitung sin dan cos dari sudut rotasi
    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])
    
    # Menghitung lebar dan tinggi baru agar tidak terpotong
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # Menyesuaikan matriks rotasi ke pusat yang baru
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
    
    # Kembalikan ke ukuran asli (cropping center) agar dimensi tetap konsisten
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    return rotated[start_y:start_y+h, start_x:start_x+w]

def zoom_image(image, zoom_factor):
    """Melakukan zoom in atau zoom out pada gambar"""
    h, w = image.shape[:2]
    
    if zoom_factor == 1.0:
        return image
    
    # Mengubah ukuran gambar
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if zoom_factor > 1.0:
        # Zoom In: Potong bagian tengah agar ukuran kembali ke asli
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        return resized[start_y:start_y+h, start_x:start_x+w]
    else:
        # Zoom Out: Tambahkan padding agar ukuran kembali ke asli
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        # Menggunakan BORDER_REPLICATE agar pinggiran terlihat natural seperti X-ray
        return cv2.copyMakeBorder(resized, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, 
                                  cv2.BORDER_REPLICATE)

# ==========================================================
# FUNGSI UTAMA PROSES BATCH
# ==========================================================

def start_batch_augmentation():
    # Pilih banyak file sekaligus
    file_paths = filedialog.askopenfilenames(
        title="Pilih Gambar untuk Augmentasi",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_paths:
        return

    # Siapkan folder output
    output_folder = "variasi"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_files = len(file_paths)
    btn_select.config(state="disabled")
    
    # Daftar parameter sesuai kesepakatan
    flips = [False, True]
    rotations = [-10, -5, 0, 5, 10]
    zooms = [0.95, 1.0, 1.05]
    
    try:
        for i, path in enumerate(file_paths):
            # Update label loading
            lbl_status.config(text=f"Memproses {i+1}/{total_files} Gambar Asli...")
            root.update()
            
            # Baca gambar
            original_img = cv2.imread(path)
            if original_img is None: continue
            
            base_name = os.path.splitext(os.path.basename(path))[0]
            
            # Loop kombinasi (2 x 5 x 3 = 30 variasi)
            for flip in flips:
                for angle in rotations:
                    for zoom in zooms:
                        # 1. Terapkan Flip
                        img_processed = cv2.flip(original_img, 1) if flip else original_img.copy()
                        
                        # 2. Terapkan Rotasi
                        if angle != 0:
                            img_processed = rotate_image(img_processed, angle)
                        
                        # 3. Terapkan Zoom
                        if zoom != 1.0:
                            img_processed = zoom_image(img_processed, zoom)
                        
                        # Buat Nama File
                        flip_tag = "hz-flip" if flip else "no-flip"
                        rot_tag = f"rot-{angle}"
                        zoom_tag = f"zoom-{int(zoom*100)}"
                        
                        filename = f"{base_name}-{flip_tag}-{rot_tag}-{zoom_tag}.jpg"
                        save_path = os.path.join(output_folder, filename)
                        
                        # Simpan hasil
                        cv2.imwrite(save_path, img_processed)
        
        messagebox.showinfo("Selesai", f"Berhasil membuat variasi gambar!\nCek folder: '{output_folder}'")
    
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")
    
    finally:
        lbl_status.config(text="Proses Selesai")
        btn_select.config(state="normal")

# ==========================================================
# GUI SETUP
# ==========================================================

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Augmentasi Citra Batch")
    root.geometry("400x200")
    root.resizable(False, False)

    style = ttk.Style(root)
    style.theme_use('clam')

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(expand=True, fill="both")

    ttk.Label(main_frame, text="Data Augmentation Tool", font=("Arial", 12, "bold")).pack(pady=5)
    ttk.Label(main_frame, text="Variasi: Flip x 5 Rotasi x 3 Zoom", font=("Arial", 9)).pack()

    btn_select = ttk.Button(main_frame, text="Pilih Gambar", command=start_batch_augmentation)
    btn_select.pack(pady=20, ipady=5, fill='x')

    lbl_status = ttk.Label(main_frame, text="Siap memproses", font=("Arial", 10, "italic"))
    lbl_status.pack()

    root.mainloop()