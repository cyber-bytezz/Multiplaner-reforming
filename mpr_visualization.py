import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter

def load_dicom_series(directory):
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        if not dicom_names:
            raise ValueError("No DICOM files found in the selected directory.")
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load DICOM series: {e}")
        return None

def extract_slice(image, plane, index):
    try:
        if plane == 'axial':
            slice_image = image[:, :, index]
        elif plane == 'coronal':
            slice_image = image[:, index, :]
        elif plane == 'sagittal':
            slice_image = image[index, :, :]
        return sitk.GetArrayFromImage(slice_image)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to extract slice: {e}")
        return None

def display_slice(canvas, image, plane, index, color_map):
    slice_image = extract_slice(image, plane, index)
    if slice_image is None:
        return
    
    slice_image = slice_image.astype(np.float32)
    if color_map != 'gray':
        slice_image = apply_color_map(slice_image, color_map)

    # Clear the existing figure and create a new one
    canvas.figure.clear()  
    ax = canvas.figure.add_subplot(111)
    ax.imshow(slice_image, cmap=color_map)
    ax.axis('off')
    canvas.draw()

def apply_color_map(image, color_map):
    return plt.cm.get_cmap(color_map)(image / np.max(image))[:, :, :3] if np.max(image) > 0 else image

def apply_filter(image, filter_type):
    if filter_type == 'Gaussian':
        return gaussian_filter(image, sigma=1)
    return image

class MPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced MPR Visualizer")
        self.image = None
        self.current_indices = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        self.max_indices = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        self.color_map_var = tk.StringVar(value='gray')
        self.filter_var = tk.StringVar(value='None')
        self.create_widgets()

    def create_widgets(self):
        self.load_button = tk.Button(self.root, text="Load DICOM Folder", command=self.load_dicom)
        self.load_button.pack(pady=10)

        self.color_map_menu = tk.OptionMenu(self.root, self.color_map_var, 'gray', 'jet', 'hot', command=self.update_color_map)
        self.color_map_menu.pack(pady=10)

        self.filter_menu = tk.OptionMenu(self.root, self.filter_var, 'None', 'Gaussian', command=self.update_filter)
        self.filter_menu.pack(pady=10)

        self.capture_axial_button = tk.Button(self.root, text="Capture Axial Snapshot", command=lambda: self.capture_snapshot('axial'))
        self.capture_axial_button.pack(pady=10)

        self.capture_coronal_button = tk.Button(self.root, text="Capture Coronal Snapshot", command=lambda: self.capture_snapshot('coronal'))
        self.capture_coronal_button.pack(pady=10)

        self.capture_sagittal_button = tk.Button(self.root, text="Capture Sagittal Snapshot", command=lambda: self.capture_snapshot('sagittal'))
        self.capture_sagittal_button.pack(pady=10)

        self.views_frame = tk.Frame(self.root)
        self.views_frame.pack(pady=10, padx=10)

        self.axial_frame = self.create_plane_frame('Axial')
        self.coronal_frame = self.create_plane_frame('Coronal')
        self.sagittal_frame = self.create_plane_frame('Sagittal')

        self.axial_canvas = self.create_canvas(self.axial_frame)
        self.coronal_canvas = self.create_canvas(self.coronal_frame)
        self.sagittal_canvas = self.create_canvas(self.sagittal_frame)

        self.axial_scroll = self.create_scrollbar(self.axial_frame, 'axial')
        self.coronal_scroll = self.create_scrollbar(self.coronal_frame, 'coronal')
        self.sagittal_scroll = self.create_scrollbar(self.sagittal_frame, 'sagittal')

        self.snapshot_frame = tk.Frame(self.root)
        self.snapshot_frame.pack(pady=10)

        self.axial_snapshot_canvas = self.create_snapshot_canvas("Axial Snapshot")
        self.coronal_snapshot_canvas = self.create_snapshot_canvas("Coronal Snapshot")
        self.sagittal_snapshot_canvas = self.create_snapshot_canvas("Sagittal Snapshot")

        # Bind keyboard events for navigation
        self.root.bind('<Left>', lambda e: self.change_slice(-1))
        self.root.bind('<Right>', lambda e: self.change_slice(1))

    def create_plane_frame(self, label_text):
        plane_frame = tk.Frame(self.views_frame)
        plane_frame.pack(side=tk.LEFT, padx=10)
        plane_label = tk.Label(plane_frame, text=label_text)
        plane_label.pack()
        return plane_frame
    
    def create_canvas(self, parent_frame):
        fig, ax = plt.subplots(figsize=(5, 5))
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.get_tk_widget().pack()
        return canvas
    
    def create_scrollbar(self, parent_frame, plane):
        scrollbar = tk.Scale(parent_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=lambda val, p=plane: self.on_scroll(p, int(val)))
        scrollbar.pack(fill=tk.X)
        return scrollbar

    def create_snapshot_canvas(self, label_text):
        snapshot_frame = tk.Frame(self.root)
        snapshot_frame.pack(pady=5)

        snapshot_label = tk.Label(snapshot_frame, text=label_text)
        snapshot_label.pack()
        
        snapshot_canvas = tk.Canvas(snapshot_frame, width=300, height=300)
        snapshot_canvas.pack()
        
        return snapshot_canvas

    def load_dicom(self):
        dicom_dir = filedialog.askdirectory(title="Select DICOM Directory")
        
        if dicom_dir:
            self.image = load_dicom_series(dicom_dir)
            
            if self.image is not None:
                size = self.image.GetSize()
                self.max_indices = {'axial': size[2] - 1, 'coronal': size[1] - 1, 'sagittal': size[0] - 1}
                
                for plane in ['axial', 'coronal', 'sagittal']:
                    self.current_indices[plane] = self.max_indices[plane] // 2
                
                self.axial_scroll.config(to=self.max_indices['axial'])
                self.coronal_scroll.config(to=self.max_indices['coronal'])
                self.sagittal_scroll.config(to=self.max_indices['sagittal'])
                
                self.update_all_views()
                messagebox.showinfo("Loaded", "DICOM series loaded successfully!")
            else:
                messagebox.showwarning("Error", "Failed to load the DICOM series.")
        else:
            messagebox.showwarning("Error", "Please select a valid directory.")
    
    def on_scroll(self, plane, index):
        self.current_indices[plane] = index
        self.update_all_views()
    
    def update_all_views(self):
        for plane in ['axial', 'coronal', 'sagittal']:
            display_slice(getattr(self, f"{plane}_canvas"), self.image, plane, self.current_indices[plane], self.color_map_var.get())
    
    def update_filter(self, *args):
        self.update_all_views()

    def capture_snapshot(self, plane):
        index = self.current_indices[plane]
        snapshot_image = extract_slice(self.image, plane, index)
        
        if snapshot_image is not None:
            # Apply selected filter before displaying
            filter_type = self.filter_var.get()
            if filter_type != 'None':
                snapshot_image = apply_filter(snapshot_image, filter_type)
                
            pil_image = Image.fromarray(snapshot_image)
            pil_image = pil_image.resize((300, 300), Image.ANTIALIAS)
            tk_image = ImageTk.PhotoImage(pil_image)

            snapshot_canvas = getattr(self, f"{plane}_snapshot_canvas")
            snapshot_canvas.delete("all")
            snapshot_canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            snapshot_canvas.image = tk_image  # Keep a reference to avoid garbage collection

    def change_slice(self, direction):
        for plane in ['axial', 'coronal', 'sagittal']:
            self.current_indices[plane] += direction
            if self.current_indices[plane] < 0:
                self.current_indices[plane] = 0
            elif self.current_indices[plane] > self.max_indices[plane]:
                self.current_indices[plane] = self.max_indices[plane]
        self.update_all_views()

    def update_color_map(self, *args):
        self.update_all_views()

if __name__ == "__main__":
    root = tk.Tk()
    app = MPRApp(root)
    root.mainloop()
