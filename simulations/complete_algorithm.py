import os.path
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from skimage import io
from skimage.color import rgb2gray
import pandas as pd

# custom calsses with class 😎
from image_processing_AC import AC_detction
from myEllipseRansac import myEllipseRansac
from drag_ellipse import DraggablePlot

# GUI stuff
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
from matplotlib.figure import Figure

from tqdm import tqdm
import time

N_AIR = 1
N_WATER = 1  # 1.33
FOCAL = 20e-3
TARGET_R = 0.15
SENSOR_SIZE = 24e-3  # for 35mm sensor: 24X36mm


# path = "C:/Users/nitay/Desktop/000.png"

class resultsDialog(tk.simpledialog.Dialog):
    def __init__(self, parent, title, meta_params):
        self.meta_params = meta_params
        super().__init__(parent, title)

    def body(self, frame):
        # print(type(frame)) # tkinter.Frame
        dists_title_frame = tk.Frame(master=frame)
        dists_val_frame = tk.Frame(master=frame)
        coeffs_title_frame = tk.Frame(master=frame)
        coeffs_header_frame = tk.Frame(master=frame)
        coeffs_val_frames = [tk.Frame(master=frame) for _ in self.meta_params]

        dists_title_frame.pack(side='top')
        dists_val_frame.pack(side='top')
        coeffs_title_frame.pack(side='top')
        coeffs_header_frame.pack(side='top')
        [f.pack(side='top') for f in coeffs_val_frames]

        tk.Label(dists_title_frame, width=25, text="Target Distances [m]:").pack(side='top')
        tk.Label(dists_val_frame, width=25, text="Target 1:").pack(side='left')
        t1_d = tk.Entry(dists_val_frame, width=25)
        t1_d.insert(0, "{:.2f}".format(self.meta_params[0]['t1_dist']))
        t1_d.pack(side='left')
        t1_d.config(state="readonly")
        tk.Label(dists_val_frame, width=25, text="Target 2:").pack(side='left')
        t2_d = tk.Entry(dists_val_frame, width=25)
        t2_d.insert(0, "{:.2f}".format(self.meta_params[0]['t2_dist']))
        t2_d.pack(side='left')
        t2_d.config(state="readonly")
        tk.Label(coeffs_title_frame, text="Calculated attenuation coefficients").pack(side='top')
        tk.Label(coeffs_header_frame, width=25, text="Image Path").pack(side='left')
        tk.Label(coeffs_header_frame, width=25, text="R [1/m]").pack(side='left')
        tk.Label(coeffs_header_frame, width=25, text="G [1/m]").pack(side='left')
        tk.Label(coeffs_header_frame, width=25, text="B [1/m]").pack(side='left')
        for idx, f in enumerate(coeffs_val_frames):
            name = tk.Entry(f, width=25)
            name.insert(0, "{}".format(self.meta_params[idx]['name']))
            name.pack(side='left')
            name.config(state="readonly")
            
            r_coeff = tk.Entry(f, width=25)
            r_coeff.insert(0, "{:.4f}".format(self.meta_params[idx]['calc_coeff_r']))
            r_coeff.pack(side='left')
            r_coeff.config(state="readonly")
            
            g_coeff = tk.Entry(f, width=25)
            g_coeff.insert(0, "{:.4f}".format(self.meta_params[idx]['calc_coeff_g']))
            g_coeff.pack(side='left')
            g_coeff.config(state="readonly")
            
            b_coeff = tk.Entry(f, width=25)
            b_coeff.insert(0, "{:.4f}".format(self.meta_params[idx]['calc_coeff_b']))
            b_coeff.pack(side='left')
            b_coeff.config(state="readonly")

        return frame

    def close_pressed(self):
        self.destroy()

    def save_run_new_file(self):
        filetypes = (
            ('csv file', '*.csv'),
        )
        f = fd.asksaveasfilename(filetypes=filetypes)
        df = pd.DataFrame(self.meta_params)
        df.to_csv(f, index=False)

    def save_run_existing_file(self):
        filetypes = (
            ('csv file', '*.csv'),
            ('All files', '*.*')
        )
        path = fd.askopenfilename(filetypes=filetypes)
        df = pd.concat([pd.read_csv(path), pd.DataFrame(self.meta_params)]).fillna('')
        df.to_csv(path, index=False)

    def buttonbox(self):
        self.close_button = tk.Button(self, text='close', width=5, command=self.close_pressed)
        self.save_new_button = tk.Button(self, text='save to new file', command=self.save_run_new_file)
        self.save_open_button = tk.Button(self, text='save to existing file', command=self.save_run_existing_file)
        self.close_button.pack(side='right')
        self.save_new_button.pack(side='left')
        self.save_open_button.pack(side='left')


class ParamsDialog(tk.simpledialog.Dialog):
    def __init__(self, parent, title):
        super().__init__(parent, title)
        self.iconphoto(False, tk.PhotoImage(file="./simulations/technion_logo.png"))


    def body(self, frame):
        # print(type(frame)) # tkinter.Frame
        n_air_frame = tk.Frame(master=frame)
        n_water_frame = tk.Frame(master=frame)
        focal_frame = tk.Frame(master=frame)
        target_r_frame = tk.Frame(master=frame)
        sensor_size_frame = tk.Frame(master=frame)

        n_air_frame.pack(side='top')
        n_water_frame.pack(side='top')
        focal_frame.pack(side='top')
        target_r_frame.pack(side='top')
        sensor_size_frame.pack(side='top')

        self.n_air_label = tk.Label(n_air_frame, width=25, text="Air Refraction Coefficient")
        self.n_water_label = tk.Label(n_water_frame, width=25, text="Water Refraction Coefficient")
        self.focal_label = tk.Label(focal_frame, width=25, text="Focal Length [m]")
        self.target_r_label = tk.Label(target_r_frame, width=25, text="Target Radius [m]")
        self.sensor_size_label = tk.Label(sensor_size_frame, width=25, text="Sensor Size [m]")
        self.n_air_box = tk.Entry(n_air_frame, width=25)
        self.n_water_box = tk.Entry(n_water_frame, width=25)
        self.focal_box = tk.Entry(focal_frame, width=25)
        self.target_r_box = tk.Entry(target_r_frame, width=25)
        self.sensor_size_box = tk.Entry(sensor_size_frame, width=25)
        self.n_air_box.insert(0, N_AIR)
        self.n_water_box.insert(0, N_WATER)
        self.focal_box.insert(0, FOCAL)
        self.target_r_box.insert(0, TARGET_R)
        self.sensor_size_box.insert(0, SENSOR_SIZE)

        self.n_air_label.pack(side='left')
        self.n_air_box.pack(side='right')
        self.n_water_label.pack(side='left')
        self.n_water_box.pack(side='right')
        self.focal_label.pack(side='left')
        self.focal_box.pack(side='right')
        self.target_r_label.pack(side='left')
        self.target_r_box.pack(side='right')
        self.sensor_size_label.pack(side='left')
        self.sensor_size_box.pack(side='right')

        return frame

    def ok_pressed(self):
        global N_AIR, N_WATER, FOCAL, TARGET_R, SENSOR_SIZE
        N_AIR = float(self.n_air_box.get())
        N_WATER = float(self.n_water_box.get())
        FOCAL = float(self.focal_box.get())
        TARGET_R = float(self.target_r_box.get())
        SENSOR_SIZE = float(self.sensor_size_box.get())
        self.destroy()

    def buttonbox(self):
        self.ok_button = tk.Button(self, text='OK', width=5, command=self.ok_pressed)
        self.ok_button.pack()
        self.bind("<Return>", lambda event: self.ok_pressed())


class WaterTurbidityApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('Water Turbidity App')
        self.iconphoto(False, tk.PhotoImage(file=os.path.join(os.path.abspath(os.path.dirname(__file__)),"app_assets/technion_logo.png")))
        self.buttonsFrame = tk.Frame(height=50)
        self.buttonsFrame.pack(side='top')
        open_button = ttk.Button(self.buttonsFrame, text="Open image",
                                 command=self.select_image).pack(side='left')
        exit_button = ttk.Button(self.buttonsFrame, text="EXIT",
                                 command=self.quit).pack(side='right')
        params_button = ttk.Button(self.buttonsFrame, text="Default Params",
                                   command=self.set_default_params).pack(side='right')
        
        self.imageFrame = None
        
        # GUI and target autodetection will only refer to first image while calculate
        self.image = None   
        self.image_paths = ''
        
    def set_default_params(self):
        ParamsDialog(parent=self, title='Default_params')

    def autodetect_targets(self):
        snake1, snake2 = AC_detction(self.image)
        target_1 = myEllipseRansac(snake1).get_params()
        target_2 = myEllipseRansac(snake2).get_params()
        self.imageFrame.destroy()
        self.imageFrame = tk.Frame()
        self.imageFrame.pack(side='top')
        figure = Figure(figsize=(6, 4), dpi=100)
        canvas = FigureCanvasTkAgg(figure, self.imageFrame)
        self.drag_plot = DraggablePlot(self.image,
                                       target_1, target_2,
                                       standalone=False)
        self.drag_plot._init_plot(figure=figure, canvas=canvas)

        axes = self.drag_plot.get_axes()
        canvas.get_tk_widget().pack(side='top')
        NavigationToolbar2Tk(canvas, self.imageFrame)
        if not hasattr(self, 'calculate_button'):
            self.calculate_button = ttk.Button(self.buttonsFrame, 
                                               text="Calculate", 
                                               command=self.calculate_coeffs).pack(side='left')

    def calculate_coeffs(self):
        x1, y1, w1, h1 = self.drag_plot._targets[0].get_params()
        x2, y2, w2, h2 = self.drag_plot._targets[1].get_params()
        results_list = []
        for p in self.image_paths:
            img = io.imread(p)
            att_R_w, att_G_w, att_B_w, d1, d2 = calc_coeffs_from_ellipses(((x1, y1), w1, h1, 0),
                                                                          ((x2, y2), w2, h2, 0),
                                                                          img, show_mask=False)
            results_list.append({"name": p, "t1_dist": d1, "t2_dist": d2,
                                 "calc_coeff_r": att_R_w, "calc_coeff_g": att_G_w, "calc_coeff_b": att_B_w,
                                 "t1_x": x1, "t1_y": y1, "t1_w": w1, "t1_h": h1,
                                 "t2_x": x2, "t2_y": y2, "t2_w": w2, "t2_h": h2,
                                 "N_AIR": N_AIR, "N_WATER": N_WATER,
                                 "FOCAL": FOCAL, "TARGET_R": TARGET_R, "SENSOR_SIZE": SENSOR_SIZE
                                 })
        resultsDialog(self, 'results', results_list)

    def quit(self):
        sys.exit(0)

    def open_image(self, img):
        if self.imageFrame is None:
            # to make sure we crearte it only once
            run_button = ttk.Button(self.buttonsFrame, text="Run Autodetection",
                                    command=self.autodetect_targets).pack(side='left')
        else:
            # easier to destroy and recreate than refresh...
            self.imageFrame.destroy()
        self.imageFrame = tk.Frame()
        self.imageFrame.pack(side='top')
        figure = Figure(figsize=(6, 4), dpi=100)
        canvas = FigureCanvasTkAgg(figure, self.imageFrame)
        axes = figure.add_subplot()
        axes.imshow(img)
        canvas.get_tk_widget().pack(side='top')
        NavigationToolbar2Tk(canvas, self.imageFrame)

    def select_image(self):
        filetypes = (
            ('png images', '*.png'),
            ('All files', '*.*')
        )
        paths = ''
        while not paths:
            paths = fd.askopenfilenames(filetypes=filetypes) # returns tuple of paths or empty string
            if paths == '':
                tk.messagebox.showwarning(title="No file selected", message="please select valid image file/files")
        self.image_paths = paths
        if len(paths) > 1:
            tk.messagebox.showinfo(title="Bulk mode", message=f"you selected {len(paths)} image\nBulk mode activated")
        try:
            self.image = io.imread(self.image_paths[0])
            self.open_image(self.image)
        except Exception as e:
            print(e)

    def run_app(self):
        tk.mainloop()

def calc_coeffs_from_ellipses(target1, target2, img, show_mask=True):
    """
    calculate the attenuation coeffs given the marked targets
    called as callback from DraggablePlot, 
    results are displayed in pop-up window
    inputs:
        target1/2 - ellipses params ((x,y),width,height,theta)
        image     - the original rgb image
    """
    # make targets into matplotlib patches so we can use some nice features
    xy1, width1, height1, angle1 = target1
    xy2, width2, height2, angle2 = target2
    t1_patch = Ellipse(xy=xy1, width=width1, height=height1)
    t2_patch = Ellipse(xy=xy2, width=width2, height=height2)

    t1_TL, t1_BR = t1_patch.get_extents().get_points()
    t2_TL, t2_BR = t2_patch.get_extents().get_points()

    targets_TL = np.min(np.vstack((t1_TL, t2_TL)), axis=0).astype(int)
    targets_BR = np.max(np.vstack((t1_BR, t2_BR)), axis=0).astype(int)

    img_gray = rgb2gray(img)
    mask_1 = np.ndarray(img.shape[:2], bool)
    mask_2 = np.ndarray(img.shape[:2], bool)
    t1 = []
    t2 = []

    # TODO: do we still want to do the for loop?
    print('before for loop: {}'.format(time.time()))
    for i in tqdm(range(targets_TL[1], targets_BR[1] + 1), position=0):  # row
        for j in range(targets_TL[0], targets_BR[0] + 1):  # column
            if t1_patch.contains_point((j, i)):
                mask_1[i, j] = True
                t1.append((i, j, img_gray[i, j], *list(img[i, j])))  # (y,x,gray,r,g,b)
            if t2_patch.contains_point((j, i)):
                mask_2[i, j] = True
                t2.append((i, j, img_gray[i, j], *list(img[i, j])))  # (y,x,gray,r,g,b)
    print('after for loop: {}'.format(time.time()))
    t1 = np.array(t1)
    t2 = np.array(t2)

    # TODO: now we are dealing with user marked ellipse, is this relevant?
    # center_t1, r1 = get_center_radius_from_snake(np.array(target_1), t1)
    # center_t2, r2 = get_center_radius_from_snake(np.array(target_2), t2)

    # t1 = t1[np.linalg.norm(t1[:,:2] - center_t1, axis=1) < 0.9*r1]
    # t2 = t2[np.linalg.norm(t2[:,:2] - center_t2, axis=1) < 0.9*r2]

    # TODO: we calculate distance with radius, how to define in ellipse? 
    # height? width?
    r1 = target1[2] / 2
    r2 = target2[2] / 2

    d1 = calc_distance(img.shape[0], r1)
    d2 = calc_distance(img.shape[0], r2)

    avg_t1 = np.mean(t1[:, 2])
    avg_t2 = np.mean(t2[:, 2])
    t1_b = t1[t1[:, 2] < avg_t1]
    t1_w = t1[t1[:, 2] >= avg_t1]
    t2_b = t2[t2[:, 2] < avg_t2]
    t2_w = t2[t2[:, 2] >= avg_t2]

    mask_3 = np.ndarray(img.shape, int)
    mask_3[t1_w[:, 0].astype(int), t1_w[:, 1].astype(int)] = (0, 0, 255)
    mask_3[t1_b[:, 0].astype(int), t1_b[:, 1].astype(int)] = (255, 0, 0)
    mask_3[t2_w[:, 0].astype(int), t2_w[:, 1].astype(int)] = (0, 255, 0)
    mask_3[t2_b[:, 0].astype(int), t2_b[:, 1].astype(int)] = (255, 255, 255)

    avg_t1_b = np.mean(t1_b[:, 3:], axis=0)
    avg_t1_w = np.mean(t1_w[:, 3:], axis=0)
    avg_t2_b = np.mean(t2_b[:, 3:], axis=0)
    avg_t2_w = np.mean(t2_w[:, 3:], axis=0)

    att_R_w, att_G_w, att_B_w = clac_attenuation_coeffs(d1, avg_t1_w, avg_t1_b, d2, avg_t2_w, avg_t2_b)

    if show_mask:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(mask_3)
        print(att_R_w, att_G_w, att_B_w, d1, d2)
        plt.show()
    else:  # probably called from TkInter
        return att_R_w, att_G_w, att_B_w, d1, d2


def clac_attenuation_coeffs(t1_dist, t1_w_avg, t1_b_avg, t2_dist, t2_w_avg, t2_b_avg):
    """
    calculate attenuation coeeficients for RGB chnnel, expects the image to be in RGB
    :return: attenuation coeffs in RGB format
    """
    att_R_w = - np.log((t1_w_avg[0] - t1_b_avg[0]) / (t2_w_avg[0] - t2_b_avg[0])) / (t1_dist - t2_dist)
    att_G_w = - np.log((t1_w_avg[1] - t1_b_avg[1]) / (t2_w_avg[1] - t2_b_avg[1])) / (t1_dist - t2_dist)
    att_B_w = - np.log((t1_w_avg[2] - t1_b_avg[2]) / (t2_w_avg[2] - t2_b_avg[2])) / (t1_dist - t2_dist)
    return att_R_w, att_G_w, att_B_w


def calc_distance(img_height, radius):
    """
    calculate the distance from the camera to the targets in meters
    :param img_height: in pixels
    :param radius: radius of the target in pixels
    :return: for each target return distance in meters
    """
    focal_eff = FOCAL * (N_WATER / N_AIR)
    pix_d = SENSOR_SIZE / img_height
    return (focal_eff * TARGET_R) / (pix_d * radius)


def get_center_radius_from_snake(snake, t):
    """
    calculate the radius and center of a AC detection in pixels.
    expect snake to be np.array(n,2) and t are the pixels inside each target in format [n,(y,x,gray,r,g,b)].
    """
    c = np.mean(t[:, :2], axis=0)
    r = np.mean(np.linalg.norm(np.array(snake) - c, axis=1))

    return c, r


if __name__ == "__main__":
    app = WaterTurbidityApp()
    app.run_app()
