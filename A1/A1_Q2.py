import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# ------------------ Image Ops ------------------
BORDER_MAP = {
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    "wrap": cv2.BORDER_WRAP,
    "reflect101": cv2.BORDER_REFLECT_101,
}

def adjust_brightness(img, beta):
    out = img.astype(np.int16) + int(beta)
    return np.clip(out, 0, 255).astype(np.uint8)

def adjust_contrast(img, alpha):
    out = img.astype(np.float32) * float(alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

def to_grayscale(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_threshold(img, thresh_val, inv=False):
    gray = to_grayscale(img)
    t = int(np.clip(thresh_val, 0, 255))
    mode = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    _, out = cv2.threshold(gray, t, 255, mode)
    return out

def blend_with_image(img, other, alpha):
    a = float(np.clip(alpha, 0.0, 1.0))
    h, w = img.shape[:2]
    other_resized = cv2.resize(other, (w, h), interpolation=cv2.INTER_AREA)

    # Match channels
    if img.ndim == 2 and other_resized.ndim == 3:
        img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(img_c, 1 - a, other_resized, a, 0)
    if img.ndim == 3 and other_resized.ndim == 2:
        other_c = cv2.cvtColor(other_resized, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(img, 1 - a, other_c, a, 0)
    return cv2.addWeighted(img, 1 - a, other_resized, a, 0)

def compute_min_padding_to_ratio(h, w, rw, rh):
    target_ratio = rw / rh
    cur_ratio = w / h
    top = bottom = left = right = 0

    if abs(cur_ratio - target_ratio) < 1e-9:
        return top, bottom, left, right

    if cur_ratio > target_ratio:
        new_h = int(np.ceil(w * rh / rw))
        pad_h = max(0, new_h - h)
        top = pad_h // 2
        bottom = pad_h - top
    else:
        new_w = int(np.ceil(h * rw / rh))
        pad_w = max(0, new_w - w)
        left = pad_w // 2
        right = pad_w - left

    return top, bottom, left, right

def add_padding(img, extra_px, border_type, mode, ratio=(4, 5), constant_color=(0, 0, 0)):
    extra_px = max(0, int(extra_px))
    bt = BORDER_MAP.get(border_type, cv2.BORDER_REFLECT)

    if mode == "square":
        rw, rh = 1, 1
    elif mode == "rectangle":
        rw, rh = 16, 9
    else:
        rw, rh = ratio

    h, w = img.shape[:2]
    t1, b1, l1, r1 = compute_min_padding_to_ratio(h, w, rw, rh)

    out = cv2.copyMakeBorder(img, t1, b1, l1, r1, bt, value=constant_color)
    out = cv2.copyMakeBorder(out, extra_px, extra_px, extra_px, extra_px, bt, value=constant_color)

    hh, ww = out.shape[:2]
    t3, b3, l3, r3 = compute_min_padding_to_ratio(hh, ww, rw, rh)
    out = cv2.copyMakeBorder(out, t3, b3, l3, r3, bt, value=constant_color)
    return out

# ------------------ GUI App ------------------
class PhotoEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini Photo Editor (Tkinter + OpenCV)")
        self.root.geometry("1100x650")

        self.original = None
        self.current = None

        self.history_imgs = []
        self.history_ops = []

        self.blend_img = None

        self._build_ui()

    def _build_ui(self): 
        # Left: preview panel
        left = tk.Frame(self.root, bd=2, relief="groove")
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.lbl_original = tk.Label(left, text="Original", font=("Arial", 12, "bold"))
        self.lbl_original.pack(pady=(8, 4))
        self.canvas_original = tk.Label(left)
        self.canvas_original.pack(padx=10, pady=6)
        self.lbl_preview = tk.Label(left, text="Preview", font=("Arial", 12, "bold"))
        self.lbl_preview.pack(pady=(12, 4))
        self.canvas_preview = tk.Label(left)
        self.canvas_preview.pack(padx=10, pady=6)

        # Right: controls
        right = tk.Frame(self.root, bd=2, relief="groove")
        right.pack(side="right", fill="y", padx=8, pady=8)

        # Buttons row
        btn_row = tk.Frame(right)
        btn_row.pack(fill="x", pady=6)

        tk.Button(btn_row, text="Upload Image", command=self.upload_image, width=14).pack(side="left", padx=4)
        tk.Button(btn_row, text="Save", command=self.save_image, width=10).pack(side="left", padx=4)
        tk.Button(btn_row, text="Undo", command=self.undo, width=10).pack(side="left", padx=4)

        # Brightness
        self.brightness = tk.IntVar(value=0)
        frame_b = ttk.LabelFrame(right, text="Brightness")
        frame_b.pack(fill="x", padx=8, pady=6)

        tk.Scale(frame_b, from_=-100, to=100, orient="horizontal",
                 variable=self.brightness, command=lambda _: self.apply_preview()).pack(fill="x", padx=8, pady=6)

        # Contrast
        self.contrast = tk.DoubleVar(value=1.0)
        frame_c = ttk.LabelFrame(right, text="Contrast")
        frame_c.pack(fill="x", padx=8, pady=6)

        tk.Scale(frame_c, from_=0.5, to=3.0, resolution=0.05, orient="horizontal",
                 variable=self.contrast, command=lambda _: self.apply_preview()).pack(fill="x", padx=8, pady=6)

        # Grayscale + Threshold
        frame_t = ttk.LabelFrame(right, text="Grayscale & Threshold")
        frame_t.pack(fill="x", padx=8, pady=6)

        self.use_gray = tk.BooleanVar(value=False)
        tk.Checkbutton(frame_t, text="Grayscale", variable=self.use_gray, command=self.apply_preview).pack(anchor="w", padx=8, pady=(6, 2))

        self.thresh_val = tk.IntVar(value=127)
        self.thresh_on = tk.BooleanVar(value=False)
        self.thresh_inv = tk.BooleanVar(value=False)

        tk.Checkbutton(frame_t, text="Enable Threshold", variable=self.thresh_on, command=self.apply_preview).pack(anchor="w", padx=8)
        tk.Checkbutton(frame_t, text="Inverse", variable=self.thresh_inv, command=self.apply_preview).pack(anchor="w", padx=8)

        tk.Scale(frame_t, from_=0, to=255, orient="horizontal",
                 variable=self.thresh_val, command=lambda _: self.apply_preview()).pack(fill="x", padx=8, pady=6)

        # Padding
        frame_p = ttk.LabelFrame(right, text="Padding")
        frame_p.pack(fill="x", padx=8, pady=6)

        self.pad_extra = tk.IntVar(value=0)
        tk.Label(frame_p, text="Extra padding px").pack(anchor="w", padx=8, pady=(6, 0))
        tk.Scale(frame_p, from_=0, to=200, orient="horizontal",
                 variable=self.pad_extra, command=lambda _: self.apply_preview()).pack(fill="x", padx=8, pady=(0, 6))

        self.pad_mode = tk.StringVar(value="square")
        tk.Label(frame_p, text="Proportion").pack(anchor="w", padx=8)
        ttk.Combobox(frame_p, textvariable=self.pad_mode, state="readonly",
                     values=["square", "rectangle", "custom"]).pack(fill="x", padx=8, pady=4)

        self.rw = tk.IntVar(value=4)
        self.rh = tk.IntVar(value=5)
        ratio_row = tk.Frame(frame_p)
        ratio_row.pack(fill="x", padx=8, pady=4)
        tk.Label(ratio_row, text="Custom ratio W:H").pack(side="left")
        tk.Entry(ratio_row, textvariable=self.rw, width=5).pack(side="left", padx=6)
        tk.Label(ratio_row, text=":").pack(side="left")
        tk.Entry(ratio_row, textvariable=self.rh, width=5).pack(side="left", padx=6)

        self.border_type = tk.StringVar(value="reflect")
        tk.Label(frame_p, text="Border type").pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Combobox(frame_p, textvariable=self.border_type, state="readonly",
                     values=list(BORDER_MAP.keys())).pack(fill="x", padx=8, pady=4)

        tk.Button(frame_p, text="Apply Padding", command=self.commit_padding).pack(fill="x", padx=8, pady=(6, 8))

        # Blend
        frame_bl = ttk.LabelFrame(right, text="Blend With Another Image")
        frame_bl.pack(fill="x", padx=8, pady=6)

        blend_row = tk.Frame(frame_bl)
        blend_row.pack(fill="x", padx=8, pady=6)
        tk.Button(blend_row, text="Choose Image", command=self.load_blend_image).pack(side="left")
        self.alpha = tk.DoubleVar(value=0.5)
        tk.Label(frame_bl, text="Alpha").pack(anchor="w", padx=8)
        tk.Scale(frame_bl, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
                 variable=self.alpha, command=lambda _: self.apply_preview()).pack(fill="x", padx=8, pady=6)
        tk.Button(frame_bl, text="Commit Blend", command=self.commit_blend).pack(fill="x", padx=8, pady=(0, 8))

        # Commit button + History
        tk.Button(right, text="Commit Current Preview", command=self.commit_preview, height=2).pack(fill="x", padx=8, pady=10)

        frame_h = ttk.LabelFrame(right, text="History")
        frame_h.pack(fill="both", expand=True, padx=8, pady=6)
        self.history_list = tk.Listbox(frame_h, height=10)
        self.history_list.pack(fill="both", expand=True, padx=8, pady=8)

    # ---------- UI actions ----------
    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not open image.")
            return

        self.original = img
        self.current = img.copy()
        self.history_imgs = [self.current.copy()]
        self.history_ops = []
        self.history_list.delete(0, tk.END)

        self.reset_controls()
        self.render_images(self.original, self.current)

    def reset_controls(self):
        self.brightness.set(0)
        self.contrast.set(1.0)
        self.use_gray.set(False)
        self.thresh_on.set(False)
        self.thresh_inv.set(False)
        self.thresh_val.set(127)
        self.pad_extra.set(0)
        self.pad_mode.set("square")
        self.border_type.set("reflect")
        self.alpha.set(0.5)
        self.blend_img = None

    def render_images(self, orig, prev):
        self._show_on_label(self.canvas_original, orig, max_size=(520, 250))
        self._show_on_label(self.canvas_preview, prev, max_size=(520, 250))

    def _show_on_label(self, label, img_bgr, max_size=(500, 250)):
        # Convert to RGB for PIL
        if img_bgr.ndim == 2:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)
        pil.thumbnail(max_size, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)
        label.configure(image=tk_img)
        label.image = tk_img  # keep ref

    def apply_preview(self):
        if self.current is None:
            return
        img = self.current.copy()

        # brightness + contrast
        img = adjust_brightness(img, self.brightness.get())
        img = adjust_contrast(img, self.contrast.get())

        # grayscale (optional)
        if self.use_gray.get():
            img = to_grayscale(img)

        # threshold (optional)
        if self.thresh_on.get():
            img = apply_threshold(img, self.thresh_val.get(), inv=self.thresh_inv.get())

        # preview blend (optional, non-committed)
        if self.blend_img is not None:
            img = blend_with_image(img, self.blend_img, self.alpha.get())
        # preview padding (so the slider actually shows something)
        extra = self.pad_extra.get()
        if extra > 0:
            mode = self.pad_mode.get()
            rw = max(1, int(self.rw.get()))
            rh = max(1, int(self.rh.get()))
            border = self.border_type.get()

            img = add_padding(
                img,
                extra_px=extra,
                border_type=border,
                mode=mode,
                ratio=(rw, rh),
                constant_color=(0, 0, 0)
            )
        self.render_images(self.original, img)

    def reset_edit_controls(self):
        """Reset sliders/toggles after committing so effects don't stack."""
        self.brightness.set(0)
        self.contrast.set(1.0)

        self.use_gray.set(False)
        self.thresh_on.set(False)
        self.thresh_inv.set(False)
        self.thresh_val.set(127)

        self.pad_extra.set(0)
        self.pad_mode.set("square")
        self.border_type.set("reflect")
        self.rw.set(4)
        self.rh.set(5)

        self.alpha.set(0.5)
        self.blend_img = None  # important: stop blending preview

    def commit_preview(self):
        """Commit the current slider/toggle preview as a real operation (push to history)."""
        if self.current is None:
            return

        base = self.current.copy()   # base image = what is already committed
        img = base.copy()
        ops = []

        # Read current UI values ONCE
        b = int(self.brightness.get())
        c = float(self.contrast.get())

        # Apply brightness/contrast first
        if b != 0:
            img = adjust_brightness(img, b)
            ops.append(f"brightness {b:+d}")

        if abs(c - 1.0) > 1e-9:
            img = adjust_contrast(img, c)
            ops.append(f"contrast x{c:.2f}")

        # grayscale
        if self.use_gray.get():
            img = to_grayscale(img)
            ops.append("grayscale")

        # threshold
        if self.thresh_on.get():
            mode = "inv" if self.thresh_inv.get() else "binary"
            img = apply_threshold(img, self.thresh_val.get(), inv=self.thresh_inv.get())
            ops.append(f"threshold {mode} @ {self.thresh_val.get()}")

        # blend (if any)
        if self.blend_img is not None:
            img = blend_with_image(img, self.blend_img, self.alpha.get())
            ops.append(f"blend alpha={self.alpha.get():.2f}")

        # padding preview is inside apply_preview() now,
        # so DO NOT commit padding here — commit padding only with "Apply Padding"
        # (otherwise padding will commit twice or confuse history)

        if not ops:
            messagebox.showinfo("Info", "No changes to commit.")
            return

        self.current = img                      
        self._push_history(img, " + ".join(ops))

        self.reset_edit_controls()             
        self.apply_preview()


    def commit_padding(self):
        if self.current is None:
            return
        mode = self.pad_mode.get()
        rw = max(1, int(self.rw.get()))
        rh = max(1, int(self.rh.get()))
        border = self.border_type.get()
        extra = self.pad_extra.get()

        out = add_padding(
            self.current,
            extra_px=extra,
            border_type=border,
            mode=mode,
            ratio=(rw, rh),
            constant_color=(0, 0, 0)
        )

        if mode == "custom":
            desc = f"pad {extra}px {border} ratio {rw}:{rh}"
        else:
            desc = f"pad {extra}px {border} mode {mode}"

        self._push_history(out, desc)
        self.current = out
        self.reset_edit_controls()
        self.apply_preview()
    def load_blend_image(self):
        if self.current is None:
            messagebox.showinfo("Info", "Upload an image first.")
            return
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not open blend image.")
            return
        self.blend_img = img
        self.apply_preview()

    def commit_blend(self):
        if self.current is None:
            return
        if self.blend_img is None:
            messagebox.showinfo("Info", "Choose a blend image first.")
            return
        out = blend_with_image(self.current, self.blend_img, self.alpha.get())
        desc = f"blend alpha={self.alpha.get():.2f}"
        self._push_history(out, desc)
        self.current = out
        self.reset_edit_controls()
        self.apply_preview()

    def undo(self):
        if len(self.history_imgs) <= 1:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return
        self.history_imgs.pop()
        if self.history_ops:
            self.history_ops.pop()
            self.history_list.delete(tk.END)

        self.current = self.history_imgs[-1].copy()
        self.apply_preview()

    def _push_history(self, img, op_desc):
        self.history_imgs.append(img.copy())
        self.history_ops.append(op_desc)
        self.history_list.insert(tk.END, op_desc)

    def save_image(self):
        if self.current is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg *.jpeg")]
        )
        if not path:
            return
        ok = cv2.imwrite(path, self.current)
        if ok:
            messagebox.showinfo("Saved", f"Saved to:\n{path}")
        else:
            messagebox.showerror("Error", "Failed to save image.")


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoEditorApp(root)
    root.mainloop()
