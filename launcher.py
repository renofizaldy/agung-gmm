import tkinter as tk
from tkinter import messagebox, ttk
import os
import subprocess
import csv
import sys

SCRIPT_TRAINING = "6-gui-fitur-csv-label.py"
SCRIPT_DIAGNOSA = "7-diagnose.py"
DATABASE_FILE   = "database_fitur.csv"

def get_database_count():
    if not os.path.exists(DATABASE_FILE):
        return 0
    
    try:
        with open(DATABASE_FILE, mode='r') as f:
            row_count = sum(1 for row in f)
            return max(0, row_count - 1)
    except Exception:
        return 0

def update_status_label():
    count = get_database_count()
    if count == 0:
        lbl_status.config(text="Status Database: Kosong (Belum ada data)", foreground="red")
    else:
        lbl_status.config(text=f"Status Database: Ditemukan {count} Data Latih", foreground="green")
    
    root.after(2000, update_status_label)

def launch_training():
    if not os.path.exists(SCRIPT_TRAINING):
        messagebox.showerror("Error", f"File script tidak ditemukan:\n{SCRIPT_TRAINING}")
        return
    subprocess.Popen([sys.executable, SCRIPT_TRAINING])

def launch_diagnosis():
    if not os.path.exists(SCRIPT_DIAGNOSA):
        messagebox.showerror("Error", f"File script tidak ditemukan:\n{SCRIPT_DIAGNOSA}")
        return
    subprocess.Popen([sys.executable, SCRIPT_DIAGNOSA])

def reset_database():
    if not os.path.exists(DATABASE_FILE):
        messagebox.showinfo("Info", "Database sudah kosong.")
        return

    jawaban = messagebox.askyesno(
        "Konfirmasi Hapus", 
        "Hapus seluruh data latih?\n\nTindakan ini tidak bisa dibatalkan."
    )

    if jawaban:
        try:
            os.remove(DATABASE_FILE)
            messagebox.showinfo("Sukses", "Data latih berhasil direset.")
            update_status_label()
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menghapus file:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Main Launcher - Sistem Deteksi Tulang")
    root.geometry("400x350")
    root.resizable(False, False)

    style = ttk.Style(root)
    style.theme_use('clam')
    
    header_frame = ttk.Frame(root, padding="10")
    header_frame.pack(fill="x")

    lbl_title = ttk.Label(header_frame, text="Sistem Deteksi Tulang", font=("Arial", 16, "bold"))
    lbl_title.pack()
    lbl_subtitle = ttk.Label(header_frame, text="Metode Segmentasi GMM-EM", font=("Arial", 10))
    lbl_subtitle.pack()

    btn_frame = ttk.Frame(root, padding="20")
    btn_frame.pack(expand=True, fill="both")

    btn_train = ttk.Button(btn_frame, text="Training Data Latih", command=launch_training)
    btn_train.pack(fill="x", pady=10, ipady=8)

    btn_diagnose = ttk.Button(btn_frame, text="Diagnosis Citra", command=launch_diagnosis)
    btn_diagnose.pack(fill="x", pady=10, ipady=8)

    ttk.Separator(btn_frame, orient='horizontal').pack(fill='x', pady=15)

    btn_reset = ttk.Button(btn_frame, text="⚠️ Reset Data Latih", command=reset_database)
    btn_reset.pack(fill="x", pady=5, ipady=5)

    status_frame = ttk.Frame(root, padding="10", relief="sunken")
    status_frame.pack(fill="x", side="bottom")
    
    lbl_status = ttk.Label(status_frame, text="Memeriksa Database...", font=("Arial", 9, "bold"))
    lbl_status.pack()

    update_status_label()

    root.mainloop()