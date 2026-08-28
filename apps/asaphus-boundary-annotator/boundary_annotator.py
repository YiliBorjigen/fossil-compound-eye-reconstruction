#!/usr/bin/env python3
"""Click-based blinded CT-boundary annotator."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageEnhance, ImageTk

from annotation_core import (
    STRUCTURE_CLASSES,
    load_annotations,
    load_pack,
    point_from_canvas,
    save_annotations,
)


DISPLAY = 430


def to_photo(
    array: np.ndarray,
    contrast: float = 1.0,
    size: int = DISPLAY,
) -> ImageTk.PhotoImage:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        scaled = np.zeros(array.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            high = low + 1
        scaled = np.clip((array - low) / (high - low), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    image = Image.fromarray(scaled).resize((size, size), Image.Resampling.NEAREST)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return ImageTk.PhotoImage(image)


class BoundaryAnnotator(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Blinded fossil-eye CT boundary annotator")
        self.geometry("1320x860")
        self.minsize(1150, 760)
        self.pack_folder: Path | None = None
        self.manifest = None
        self.cases = []
        self.records = {}
        self.index = 0
        self.output_path: Path | None = None
        self.photos = []
        self._build()
        bundled = Path(__file__).resolve().parent / "annotation_pack"
        if (bundled / "manifest.json").exists():
            self.after(100, lambda: self.open_pack_path(bundled))

    def _build(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Open annotation pack",
                   command=self.open_pack).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Open existing annotations",
                   command=self.open_existing).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Save and export",
                   command=self.save).pack(side="left", padx=4)
        ttk.Label(toolbar, text="Annotator ID:").pack(side="left", padx=(20, 4))
        self.annotator_id = tk.StringVar()
        ttk.Entry(toolbar, textvariable=self.annotator_id, width=18).pack(side="left")
        self.status = tk.StringVar(value="Open the blinded annotation pack to begin.")
        ttk.Label(toolbar, textvariable=self.status).pack(side="left", padx=20)

        content = ttk.Frame(self, padding=(8, 0, 8, 8))
        content.pack(fill="both", expand=True)
        images = ttk.Frame(content)
        images.pack(side="left", fill="both", expand=True)
        controls = ttk.Frame(content, width=320)
        controls.pack(side="right", fill="y", padx=(10, 0))

        headings = ttk.Frame(images)
        headings.pack(fill="x")
        ttk.Label(headings, text="U–depth section: click visible boundary",
                  anchor="center").pack(side="left", expand=True, fill="x")
        ttk.Label(headings, text="V–depth section: click visible boundary",
                  anchor="center").pack(side="left", expand=True, fill="x")
        canvases = ttk.Frame(images)
        canvases.pack(fill="both", expand=True)
        self.u_canvas = tk.Canvas(canvases, width=DISPLAY, height=DISPLAY,
                                  background="black", cursor="crosshair")
        self.u_canvas.pack(side="left", expand=True, padx=4, pady=4)
        self.v_canvas = tk.Canvas(canvases, width=DISPLAY, height=DISPLAY,
                                  background="black", cursor="crosshair")
        self.v_canvas.pack(side="left", expand=True, padx=4, pady=4)
        self.u_canvas.bind("<Button-1>", lambda event: self.add_point("u", event))
        self.v_canvas.bind("<Button-1>", lambda event: self.add_point("v", event))

        tangent_frame = ttk.Frame(images)
        tangent_frame.pack(fill="x", pady=4)
        ttk.Label(tangent_frame, text="Tangential preview depth:").pack(side="left")
        self.depth_index = tk.IntVar(value=0)
        self.depth_scale = ttk.Scale(tangent_frame, from_=0, to=1,
                                     variable=self.depth_index,
                                     command=lambda _value: self.render())
        self.depth_scale.pack(side="left", fill="x", expand=True, padx=8)
        self.depth_label = ttk.Label(tangent_frame, width=18)
        self.depth_label.pack(side="left")
        self.tangent_canvas = tk.Canvas(images, width=260, height=260,
                                        background="black")
        self.tangent_canvas.pack(pady=4)
        ttk.Label(
            images,
            text="Cross-sections: outer surface at cyan line; inward depth increases downward.",
            anchor="center",
        ).pack(fill="x")

        nav = ttk.Frame(controls)
        nav.pack(fill="x", pady=(0, 10))
        ttk.Button(nav, text="Previous", command=lambda: self.move(-1)).pack(
            side="left", expand=True, fill="x", padx=2)
        ttk.Button(nav, text="Next", command=lambda: self.move(1)).pack(
            side="left", expand=True, fill="x", padx=2)

        self.case_label = ttk.Label(controls, text="No case", font=("TkDefaultFont", 15, "bold"))
        self.case_label.pack(anchor="w", pady=(0, 8))
        ttk.Label(controls, text="Boundary visibility").pack(anchor="w")
        self.visibility = tk.StringVar(value="uncertain")
        for value in ("visible", "uncertain", "not visible"):
            ttk.Radiobutton(controls, text=value.capitalize(), value=value,
                            variable=self.visibility,
                            command=self.capture_fields).pack(anchor="w")

        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(controls, text="Most plausible interpretation").pack(anchor="w")
        self.structure = tk.StringVar(value=STRUCTURE_CLASSES[3])
        for value in STRUCTURE_CLASSES:
            ttk.Radiobutton(controls, text=value.capitalize(), value=value,
                            variable=self.structure,
                            command=self.capture_fields).pack(anchor="w", pady=1)

        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(controls, text="Confidence (1 = low, 5 = high)").pack(anchor="w")
        self.confidence = tk.IntVar(value=1)
        ttk.Spinbox(controls, from_=1, to=5, textvariable=self.confidence,
                    width=5, command=self.capture_fields).pack(anchor="w")

        ttk.Label(controls, text="Notes").pack(anchor="w", pady=(8, 2))
        self.notes = tk.Text(controls, height=7, width=38, wrap="word")
        self.notes.pack(fill="x")
        self.notes.bind("<FocusOut>", lambda _event: self.capture_fields())
        ttk.Button(controls, text="Clear clicked points",
                   command=self.clear_points).pack(fill="x", pady=(10, 4))
        ttk.Label(
            controls,
            text=("Click several points along any visible boundary in each "
                  "cross-section. Do not infer a boundary that is not visible. "
                  "The algorithm prediction is intentionally hidden."),
            wraplength=300,
        ).pack(anchor="w", pady=8)

    def open_pack(self) -> None:
        chosen = filedialog.askdirectory(title="Select blinded annotation pack")
        if not chosen:
            return
        self.open_pack_path(Path(chosen))

    def open_pack_path(self, folder: Path) -> None:
        try:
            self.pack_folder = folder
            self.manifest, self.cases = load_pack(self.pack_folder)
            self.records = load_annotations(Path("__missing__"),
                                            [case.case_id for case in self.cases])
            self.index = 0
            self.output_path = None
            self.depth_scale.configure(to=len(self.cases[0].depth_vox) - 1)
            self.depth_index.set(len(self.cases[0].depth_vox) // 2)
            self.load_fields()
            self.render()
        except Exception as exc:
            messagebox.showerror("Could not open pack", str(exc))

    def open_existing(self) -> None:
        if not self.cases:
            messagebox.showinfo("Open a pack first", "Open the matching annotation pack first.")
            return
        chosen = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not chosen:
            return
        try:
            self.output_path = Path(chosen)
            self.records = load_annotations(
                self.output_path, [case.case_id for case in self.cases]
            )
            self.load_fields()
            self.render()
        except Exception as exc:
            messagebox.showerror("Could not open annotations", str(exc))

    def current(self):
        return self.cases[self.index]

    def capture_fields(self) -> None:
        if not self.cases:
            return
        row = self.records[self.current().case_id]
        row["visibility"] = self.visibility.get()
        row["structure_class"] = self.structure.get()
        row["confidence"] = int(self.confidence.get())
        row["notes"] = self.notes.get("1.0", "end").strip()

    def load_fields(self) -> None:
        if not self.cases:
            return
        row = self.records[self.current().case_id]
        self.visibility.set(row["visibility"])
        self.structure.set(row["structure_class"])
        self.confidence.set(row["confidence"])
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", row.get("notes", ""))
        self.case_label.configure(
            text=f"Case {self.current().case_id}  ({self.index + 1}/{len(self.cases)})"
        )

    def move(self, delta: int) -> None:
        if not self.cases:
            return
        self.capture_fields()
        self.index = min(max(self.index + delta, 0), len(self.cases) - 1)
        self.depth_scale.configure(to=len(self.current().depth_vox) - 1)
        self.load_fields()
        self.render()

    def add_point(self, view: str, event: tk.Event) -> None:
        if not self.cases:
            return
        case = self.current()
        lateral = case.u_vox if view == "u" else case.v_vox
        point = point_from_canvas(event.x, event.y, DISPLAY, DISPLAY,
                                  lateral, case.depth_vox)
        field = "u_depth_points" if view == "u" else "v_depth_points"
        self.records[case.case_id][field].append(point)
        self.render()

    def clear_points(self) -> None:
        if not self.cases:
            return
        row = self.records[self.current().case_id]
        row["u_depth_points"] = []
        row["v_depth_points"] = []
        self.render()

    def draw_points(self, canvas: tk.Canvas, points: list[dict], lateral: np.ndarray,
                    depth: np.ndarray) -> None:
        for point in points:
            x = float(np.interp(point["lateral_vox"], [lateral[0], lateral[-1]],
                                [0, DISPLAY - 1]))
            y = float(np.interp(point["depth_vox"], [depth[0], depth[-1]],
                                [0, DISPLAY - 1]))
            radius = 4
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                               outline="#ff3b30", width=2)

    def render(self) -> None:
        if not self.cases:
            return
        case = self.current()
        center_u = len(case.u_vox) // 2
        center_v = len(case.v_vox) // 2
        slab = 2
        u_section = case.intensity[:, center_v - slab:center_v + slab + 1, :].mean(axis=1).T
        v_section = case.intensity[center_u - slab:center_u + slab + 1, :, :].mean(axis=0).T
        self.photos = [to_photo(u_section), to_photo(v_section)]
        for canvas, photo in ((self.u_canvas, self.photos[0]),
                              (self.v_canvas, self.photos[1])):
            canvas.delete("all")
            canvas.create_image(0, 0, image=photo, anchor="nw")
            surface_y = float(np.interp(
                0.0, [case.depth_vox[0], case.depth_vox[-1]], [0, DISPLAY - 1]
            ))
            canvas.create_line(0, surface_y, DISPLAY, surface_y,
                               fill="#00d9ff", width=1, dash=(5, 4))
        row = self.records[case.case_id]
        self.draw_points(self.u_canvas, row["u_depth_points"], case.u_vox,
                         case.depth_vox)
        self.draw_points(self.v_canvas, row["v_depth_points"], case.v_vox,
                         case.depth_vox)

        index = int(round(float(self.depth_index.get())))
        index = min(max(index, 0), len(case.depth_vox) - 1)
        tangential = case.intensity[:, :, index].T
        tangent = to_photo(tangential, size=260)
        self.photos.append(tangent)
        self.tangent_canvas.delete("all")
        self.tangent_canvas.create_image(0, 0, image=tangent, anchor="nw")
        depth = case.depth_vox[index]
        spacing = float(self.manifest["voxel_spacing_um"])
        self.depth_label.configure(text=f"{depth:.1f} vox / {depth * spacing:.1f} µm")
        completed = sum(
            bool(row["u_depth_points"] or row["v_depth_points"])
            or row["visibility"] == "not visible"
            for row in self.records.values()
        )
        self.status.set(f"{completed}/{len(self.cases)} cases annotated")

    def save(self) -> None:
        if not self.cases:
            messagebox.showinfo("Nothing to save", "Open an annotation pack first.")
            return
        if not self.annotator_id.get().strip():
            messagebox.showinfo("Annotator ID required",
                                "Enter your name or an anonymous annotator code.")
            return
        self.capture_fields()
        if self.output_path is None:
            chosen = filedialog.asksaveasfilename(
                title="Save annotations",
                defaultextension=".json",
                initialfile=f"{self.annotator_id.get().strip()}_annotations.json",
                filetypes=[("JSON", "*.json")],
            )
            if not chosen:
                return
            self.output_path = Path(chosen)
        try:
            save_annotations(self.output_path, self.manifest,
                             self.annotator_id.get(), self.records)
            messagebox.showinfo(
                "Saved",
                f"Saved JSON and CSV exports beside:\n{self.output_path}",
            )
        except Exception as exc:
            messagebox.showerror("Could not save", str(exc))


if __name__ == "__main__":
    BoundaryAnnotator().mainloop()
