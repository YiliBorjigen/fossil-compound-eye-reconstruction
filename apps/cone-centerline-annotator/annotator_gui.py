#!/usr/bin/env python3
"""Clickable 3-D centre-line annotation for unfolded compound-eye CT."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter, maximum_filter

from annotation_core import (
    dense_path,
    export_training_labels,
    load_project,
    new_project,
    save_project,
)
from training import train_from_scribbles


class Annotator(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Compound-eye 3D centre-line annotator")
        self.geometry("1120x820")
        self.minsize(960, 700)
        self.volume_path: Path | None = None
        self.volume: np.ndarray | None = None
        self.project: dict | None = None
        self.output_dir: Path | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.mode = tk.StringVar(value="centreline")
        self.depth = tk.IntVar(value=0)
        self.show_candidates = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Open a patch folder to begin.")
        self.radius = tk.DoubleVar(value=2.5)
        self._build()
        self.bind("<Left>", lambda _event: self.step_depth(-1))
        self.bind("<Right>", lambda _event: self.step_depth(1))

    def _build(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Open patch folder", command=self.open_patch).pack(side="left")
        ttk.Button(toolbar, text="Open annotations", command=self.open_annotations).pack(side="left", padx=6)
        ttk.Button(toolbar, text="Save and export", command=self.save_and_export).pack(side="left")
        ttk.Button(toolbar, text="Train preliminary mask", command=self.train_mask).pack(side="left", padx=6)
        ttk.Label(toolbar, textvariable=self.status).pack(side="left", padx=12)

        body = ttk.Frame(self, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(body, width=300, padding=(10, 0))
        right.pack(side="right", fill="y")

        self.canvas = tk.Canvas(left, width=760, height=700, background="#111111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)

        nav = ttk.Frame(left, padding=(0, 8, 0, 0))
        nav.pack(fill="x")
        ttk.Button(nav, text="◀", width=4, command=lambda: self.step_depth(-1)).pack(side="left")
        self.slider = ttk.Scale(nav, from_=0, to=60, variable=self.depth, command=self.slider_changed)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(nav, text="▶", width=4, command=lambda: self.step_depth(1)).pack(side="left")
        self.depth_label = ttk.Label(nav, text="depth 0")
        self.depth_label.pack(side="left", padx=8)

        ttk.Label(right, text="Annotation mode", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Radiobutton(right, text="Cone centre-line", variable=self.mode, value="centreline").pack(anchor="w")
        ttk.Radiobutton(right, text="Explicit background", variable=self.mode, value="background").pack(anchor="w")
        ttk.Checkbutton(right, text="Show automatic candidates", variable=self.show_candidates, command=self.redraw).pack(anchor="w", pady=(2, 10))

        cone_bar = ttk.Frame(right)
        cone_bar.pack(fill="x")
        ttk.Button(cone_bar, text="New cone", command=self.new_cone).pack(side="left")
        ttk.Button(cone_bar, text="Delete", command=self.delete_cone).pack(side="left", padx=5)
        self.cone_list = tk.Listbox(right, height=14, exportselection=False)
        self.cone_list.pack(fill="x", pady=6)
        self.cone_list.bind("<<ListboxSelect>>", lambda _event: self.redraw())

        radius_bar = ttk.Frame(right)
        radius_bar.pack(fill="x", pady=(2, 8))
        ttk.Label(radius_bar, text="Approx. cone radius:").pack(side="left")
        ttk.Spinbox(radius_bar, from_=1.0, to=6.0, increment=0.5, width=5, textvariable=self.radius, command=self.update_radius).pack(side="left", padx=5)

        edit_bar = ttk.Frame(right)
        edit_bar.pack(fill="x")
        ttk.Button(edit_bar, text="Undo point", command=self.undo_point).pack(side="left")
        ttk.Button(edit_bar, text="Clear background", command=self.clear_background).pack(side="left", padx=5)

        instructions = (
            "1. Start a new cone.\n"
            "2. Click its centre on several depths.\n"
            "3. Use ←/→ or the slider to move.\n"
            "4. Add background clicks away from cones.\n"
            "5. Save and inspect the preliminary mask.\n\n"
            "Mark at least three well-separated cones and background points in each region. "
            "The program interpolates between your control points; you do not need to click every slice."
        )
        ttk.Label(right, text=instructions, wraplength=270, justify="left").pack(anchor="w", pady=(14, 0))

    def open_patch(self) -> None:
        folder = filedialog.askdirectory(title="Choose patch_1, patch_2 or patch_3")
        if not folder:
            return
        path = Path(folder) / "unfolded_intensity.npy"
        if not path.exists():
            messagebox.showerror("Wrong folder", "This folder does not contain unfolded_intensity.npy")
            return
        self.volume_path = path
        self.volume = np.load(path, mmap_mode="r")
        self.project = new_project(path, self.volume.shape, 1.08)
        self.output_dir = Path(folder) / "manual_annotations"
        self.depth.set(int(self.volume.shape[2] // 2))
        self.slider.configure(to=self.volume.shape[2] - 1)
        self.refresh_cone_list()
        self.status.set(f"Opened {Path(folder).name}: {self.volume.shape}")
        self.redraw()

    def open_annotations(self) -> None:
        filename = filedialog.askopenfilename(title="Open annotations.json", filetypes=[("Annotation JSON", "*.json")])
        if not filename:
            return
        project = load_project(Path(filename))
        source = Path(project["source_volume"])
        if not source.exists():
            replacement = filedialog.askopenfilename(title="Locate unfolded_intensity.npy", filetypes=[("NumPy volume", "*.npy")])
            if not replacement:
                return
            source = Path(replacement)
            project["source_volume"] = str(source.resolve())
        self.volume_path = source
        self.volume = np.load(source, mmap_mode="r")
        if list(self.volume.shape) != project["shape_uvd"]:
            messagebox.showerror("Shape mismatch", "The selected volume does not match this annotation project.")
            return
        self.project = project
        self.output_dir = Path(filename).parent
        self.slider.configure(to=self.volume.shape[2] - 1)
        self.refresh_cone_list()
        self.status.set(f"Loaded {len(project['cones'])} cones")
        self.redraw()

    def selected_cone(self) -> dict | None:
        if self.project is None:
            return None
        selection = self.cone_list.curselection()
        if not selection:
            return None
        return self.project["cones"][selection[0]]

    def new_cone(self) -> None:
        if self.project is None:
            messagebox.showinfo("Open data first", "Choose a patch folder first.")
            return
        next_id = max([cone["id"] for cone in self.project["cones"]], default=0) + 1
        self.project["cones"].append({"id": next_id, "radius_voxels": float(self.radius.get()), "nodes": []})
        self.refresh_cone_list(select=len(self.project["cones"]) - 1)
        self.mode.set("centreline")
        self.autosave()

    def delete_cone(self) -> None:
        if self.project is None:
            return
        selection = self.cone_list.curselection()
        if not selection:
            return
        del self.project["cones"][selection[0]]
        self.refresh_cone_list(select=min(selection[0], len(self.project["cones"]) - 1))
        self.autosave()
        self.redraw()

    def update_radius(self) -> None:
        cone = self.selected_cone()
        if cone is not None:
            cone["radius_voxels"] = float(self.radius.get())
            self.autosave()

    def undo_point(self) -> None:
        cone = self.selected_cone()
        if cone and cone["nodes"]:
            cone["nodes"].pop()
            self.autosave()
            self.refresh_cone_list(select=self.cone_list.curselection()[0])
            self.redraw()

    def clear_background(self) -> None:
        if self.project is not None:
            self.project["background_points"] = []
            self.autosave()
            self.redraw()

    def refresh_cone_list(self, select: int | None = None) -> None:
        self.cone_list.delete(0, "end")
        if self.project is None:
            return
        for cone in self.project["cones"]:
            self.cone_list.insert("end", f"Cone {cone['id']} — {len(cone['nodes'])} control points")
        if select is not None and self.project["cones"]:
            self.cone_list.selection_set(max(0, select))
            self.cone_list.activate(max(0, select))

    def autosave(self) -> None:
        if self.project is not None and self.output_dir is not None:
            save_project(self.project, self.output_dir)

    def save_and_export(self, silent: bool = False) -> None:
        if self.project is None or self.output_dir is None:
            if not silent:
                messagebox.showinfo("Nothing to save", "Open a patch folder first.")
            return
        project_path = save_project(self.project, self.output_dir)
        outputs = export_training_labels(self.project, self.output_dir)
        self.status.set(f"Saved {project_path.parent}")
        if not silent:
            messagebox.showinfo(
                "Saved",
                f"Annotations: {project_path}\nTraining labels: {outputs['labels']}\nCentrelines: {outputs['centrelines']}",
            )

    def train_mask(self) -> None:
        if self.project is None or self.volume_path is None or self.output_dir is None:
            messagebox.showinfo("Open data first", "Open and annotate a patch first.")
            return
        self.save_and_export(silent=True)
        scribbles = self.output_dir / "training_scribbles.npy"
        self.status.set("Training preliminary mask…")

        def work() -> None:
            try:
                summary = train_from_scribbles(
                    self.volume_path,
                    scribbles,
                    self.output_dir / "preliminary_segmentation",
                )
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Training stopped", str(exc)))
                self.after(0, lambda: self.status.set("Training stopped; add more labels."))
                return
            self.after(0, lambda: self.status.set(f"Preliminary mask ready; OOB={summary['random_forest_oob_score']:.3f}"))
            self.after(0, lambda: messagebox.showinfo("Training complete", "A preliminary mask and preview were saved. Review them before using any centre-line measurements."))

        threading.Thread(target=work, daemon=True).start()

    def step_depth(self, change: int) -> None:
        if self.volume is None:
            return
        self.depth.set(int(np.clip(self.depth.get() + change, 0, self.volume.shape[2] - 1)))
        self.redraw()

    def slider_changed(self, _value: str) -> None:
        self.redraw()

    def mouse_wheel(self, event: tk.Event) -> None:
        self.step_depth(-1 if event.delta > 0 else 1)

    def display_transform(self) -> tuple[float, float, float]:
        if self.volume is None:
            return 1.0, 0.0, 0.0
        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 100)
        scale = min(width / self.volume.shape[1], height / self.volume.shape[0])
        x0 = (width - self.volume.shape[1] * scale) / 2
        y0 = (height - self.volume.shape[0] * scale) / 2
        return scale, x0, y0

    def canvas_click(self, event: tk.Event) -> None:
        if self.volume is None or self.project is None:
            return
        scale, x0, y0 = self.display_transform()
        column = (event.x - x0) / scale
        row = (event.y - y0) / scale
        if not (0 <= row < self.volume.shape[0] and 0 <= column < self.volume.shape[1]):
            return
        depth = int(self.depth.get())
        if self.mode.get() == "background":
            self.project["background_points"].append(
                {"depth": depth, "row": row, "column": column, "radius_voxels": 2.0}
            )
        else:
            cone = self.selected_cone()
            if cone is None:
                self.new_cone()
                cone = self.selected_cone()
            cone["nodes"] = [node for node in cone["nodes"] if int(node["depth"]) != depth]
            cone["nodes"].append({"depth": depth, "row": row, "column": column})
            cone["nodes"].sort(key=lambda node: node["depth"])
        self.autosave()
        selected = self.cone_list.curselection()
        self.refresh_cone_list(select=selected[0] if selected else 0)
        self.redraw()

    def candidate_points(self, image: np.ndarray) -> np.ndarray:
        smooth = gaussian_filter(image.astype(float), 1.2)
        high_pass = smooth - gaussian_filter(smooth, 6.0)
        threshold = np.percentile(high_pass, 75)
        return np.argwhere((high_pass == maximum_filter(high_pass, size=7)) & (high_pass > threshold))

    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.volume is None:
            self.canvas.create_text(380, 320, text="Open a patch folder", fill="white", font=("TkDefaultFont", 18))
            return
        depth = int(np.clip(self.depth.get(), 0, self.volume.shape[2] - 1))
        self.depth.set(depth)
        self.depth_label.configure(text=f"depth {depth}")
        image = self.volume[:, :, depth].astype(float)
        lo, hi = np.percentile(image, (1, 99))
        display = np.clip((image - lo) * (255.0 / (hi - lo + 1e-6)), 0, 255).astype(np.uint8)
        scale, x0, y0 = self.display_transform()
        resized = Image.fromarray(display).resize(
            (max(1, int(self.volume.shape[1] * scale)), max(1, int(self.volume.shape[0] * scale))),
            Image.Resampling.NEAREST,
        )
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(x0, y0, anchor="nw", image=self.photo)

        if self.show_candidates.get():
            for row, column in self.candidate_points(image):
                x, y = x0 + column * scale, y0 + row * scale
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, outline="#FFD700", width=1)

        selected = self.cone_list.curselection()
        active = selected[0] if selected else -1
        if self.project is not None:
            for index, cone in enumerate(self.project["cones"]):
                points = {int(point["depth"]): point for point in dense_path(cone.get("nodes", []))}
                if depth in points:
                    point = points[depth]
                    x, y = x0 + point["column"] * scale, y0 + point["row"] * scale
                    colour = "#FF3B30" if index == active else "#34C759"
                    radius = float(cone.get("radius_voxels", 2.5)) * scale
                    self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=colour, width=2)
                for node in cone.get("nodes", []):
                    if int(node["depth"]) == depth:
                        x, y = x0 + node["column"] * scale, y0 + node["row"] * scale
                        self.canvas.create_line(x - 5, y, x + 5, y, fill="white", width=2)
                        self.canvas.create_line(x, y - 5, x, y + 5, fill="white", width=2)
            for point in self.project.get("background_points", []):
                if int(point["depth"]) == depth:
                    x, y = x0 + point["column"] * scale, y0 + point["row"] * scale
                    self.canvas.create_rectangle(x - 4, y - 4, x + 4, y + 4, outline="#00D4FF", width=2)


if __name__ == "__main__":
    Annotator().mainloop()
