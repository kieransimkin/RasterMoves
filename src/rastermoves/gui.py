from __future__ import annotations

import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .downloads import Downloader
from .errors import UpscaleError
from .paths import cache_dir
from .refinement.specs import BUILTINS
from .registry import Registry
from .sweep import run_all_models

SUPPORTED_INPUTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def comparison_entries(summary: dict, output_dir: Path, source: Path) -> list[tuple[str, Path]]:
    """Return the original plus every successful baseline/final image in display order."""
    entries: list[tuple[str, Path]] = [("Original", source)]
    for row in summary.get("results", []):
        if row.get("status") not in {"success", "reused"}:
            continue
        model_id = row.get("model_id", "model")
        artifacts = row.get("refinement_artifacts") or {}
        baseline = artifacts.get("baseline", {}).get("file")
        if baseline:
            path = output_dir / baseline
            if path.is_file():
                entries.append((f"{model_id} — upscale baseline", path))
        final = output_dir / row["output"]
        if final.is_file():
            suffix = " — post-diffusion" if baseline else ""
            entries.append((f"{model_id}{suffix}", final))
    return entries


def unique_output_dir(source: Path) -> Path:
    base = source.parent / f"{source.stem}-rastermoves"
    if not base.exists():
        return base
    index = 2
    while (candidate := source.parent / f"{source.stem}-rastermoves-{index}").exists():
        index += 1
    return candidate


class RasterMovesGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RasterMoves")
        self.root.geometry("1180x820")
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.source: Path | None = None
        self.output_dir: Path | None = None
        self.entries: list[tuple[str, Path]] = []
        self.original_photo = None
        self.result_photo = None

        self.cache = cache_dir()
        self.registry = Registry(self.cache)
        self.specs = self.registry.search()

        self._build()
        self._install_drop_target()
        self._refresh_models()
        self.root.after(80, self._drain_events)

    def _build(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        controls = ttk.Frame(outer)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        viewer = ttk.Frame(outer)
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.columnconfigure(1, weight=1)
        viewer.rowconfigure(2, weight=1)

        ttk.Label(controls, text="Models", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.model_list = tk.Listbox(controls, selectmode=tk.MULTIPLE, width=48, height=17, exportselection=False)
        self.model_list.pack(fill="x", pady=(4, 4))
        row = ttk.Frame(controls)
        row.pack(fill="x")
        ttk.Button(row, text="Select all", command=lambda: self.model_list.selection_set(0, tk.END)).pack(side="left")
        ttk.Button(row, text="Clear", command=lambda: self.model_list.selection_clear(0, tk.END)).pack(side="left", padx=4)
        ttk.Button(row, text="Download selected", command=self.download_selected).pack(side="right")

        ttk.Separator(controls).pack(fill="x", pady=10)
        ttk.Label(controls, text="Post-diffusion refinement").pack(anchor="w")
        self.refiner = tk.StringVar(value="None")
        ttk.Combobox(controls, textvariable=self.refiner, state="readonly",
                     values=["None", *BUILTINS.keys()], width=28).pack(anchor="w", pady=(3, 7))

        options = ttk.Frame(controls)
        options.pack(fill="x")
        ttk.Label(options, text="Device").grid(row=0, column=0, sticky="w")
        self.device = tk.StringVar(value="auto")
        ttk.Combobox(options, textvariable=self.device, state="readonly",
                     values=["auto", "cuda", "mps", "cpu"], width=12).grid(row=0, column=1, sticky="e")
        ttk.Label(options, text="Precision").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.precision = tk.StringVar(value="fp32")
        ttk.Combobox(options, textvariable=self.precision, state="readonly",
                     values=["fp32", "fp16"], width=12).grid(row=1, column=1, sticky="e", pady=(4, 0))

        ttk.Separator(controls).pack(fill="x", pady=10)
        ttk.Label(controls, text="Input image", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.drop = ttk.Label(controls, text="Drop an image here\nor click Browse", anchor="center", relief="groove", padding=18)
        self.drop.pack(fill="x", pady=(4, 5))
        ttk.Button(controls, text="Browse…", command=self.browse).pack(fill="x")
        self.input_label = ttk.Label(controls, text="No file selected", wraplength=340)
        self.input_label.pack(anchor="w", pady=(6, 8))

        self.run_button = ttk.Button(controls, text="Run selected suite", command=self.run_suite)
        self.run_button.pack(fill="x", pady=(2, 8))
        self.progress = ttk.Progressbar(controls, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.status = tk.StringVar(value=f"Models folder: {self.cache}")
        ttk.Label(controls, textvariable=self.status, wraplength=340).pack(anchor="w", pady=(5, 0))

        ttk.Label(viewer, text="Compare result against original", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w")
        self.result_choice = tk.StringVar(value="")
        self.result_box = ttk.Combobox(viewer, textvariable=self.result_choice, state="readonly")
        self.result_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        self.result_box.bind("<<ComboboxSelected>>", lambda _e: self._render_comparison())
        self.original_canvas = tk.Canvas(viewer, background="#1d1d1d", highlightthickness=0)
        self.result_canvas = tk.Canvas(viewer, background="#1d1d1d", highlightthickness=0)
        self.original_canvas.grid(row=2, column=0, sticky="nsew", padx=(0, 3))
        self.result_canvas.grid(row=2, column=1, sticky="nsew", padx=(3, 0))
        self.original_canvas.bind("<Configure>", lambda _e: self._render_comparison())
        self.result_canvas.bind("<Configure>", lambda _e: self._render_comparison())
        ttk.Label(viewer, text="Original", anchor="center").grid(row=3, column=0, sticky="ew", pady=(4, 0))
        self.result_caption = ttk.Label(viewer, text="Result", anchor="center")
        self.result_caption.grid(row=3, column=1, sticky="ew", pady=(4, 0))

    def _install_drop_target(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            self.drop.configure(text="Click Browse to select an image\n(drag/drop available with rastermoves[gui])")

    def _on_drop(self, event):
        try:
            paths = self.root.tk.splitlist(event.data)
        except tk.TclError:
            paths = [event.data.strip("{}")]
        if paths:
            self.set_source(Path(paths[0]))

    def browse(self):
        name = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")])
        if name:
            self.set_source(Path(name))

    def set_source(self, path: Path):
        path = path.expanduser().resolve()
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_INPUTS:
            messagebox.showerror("RasterMoves", "Select a supported image file.")
            return
        self.source = path
        self.input_label.configure(text=str(path))
        self._show_original(path)

    def _selected_specs(self):
        return [self.specs[i] for i in self.model_list.curselection()]

    def _refresh_models(self):
        self.model_list.delete(0, tk.END)
        dl = Downloader(self.cache)
        for spec in self.specs:
            resources = self.registry.get(spec.id).resources()
            cached = bool(resources and dl.target(resources[0]).is_file())
            mark = "✓" if cached else "↓"
            self.model_list.insert(tk.END, f"{mark}  {spec.id}  ({spec.scale}x, {spec.architecture})")
        if self.specs:
            self.model_list.selection_set(0, tk.END)

    def _set_busy(self, value: bool):
        self.busy = value
        self.run_button.configure(state="disabled" if value else "normal")

    def download_selected(self):
        specs = self._selected_specs()
        if not specs:
            messagebox.showinfo("RasterMoves", "Select at least one model.")
            return
        self._start_worker(self._download_models, specs, False)

    def run_suite(self):
        specs = self._selected_specs()
        if not specs:
            messagebox.showinfo("RasterMoves", "Select at least one model.")
            return
        if self.source is None:
            messagebox.showinfo("RasterMoves", "Choose an input image first.")
            return
        run_options = {
            "refiner": self.refiner.get(),
            "device": self.device.get(),
            "precision": self.precision.get(),
        }
        self._start_worker(self._download_models, specs, True, run_options)

    def _start_worker(self, target, *args):
        if self.busy:
            return
        self._set_busy(True)
        self.progress.configure(mode="determinate", value=0, maximum=100)
        threading.Thread(target=self._worker_wrapper, args=(target, *args), daemon=True).start()

    def _worker_wrapper(self, target, *args):
        try:
            target(*args)
        except BaseException as exc:
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self.events.put(("busy", False))

    def _download_models(self, specs, run_after: bool, run_options=None):
        total = len(specs)
        for index, spec in enumerate(specs, 1):
            plugin = self.registry.get(spec.id)
            resources = plugin.resources()
            if not resources:
                self.events.put(("status", f"{spec.id}: no downloadable resource"))
                continue

            def progress(done, expected, label, idx=index, model=spec.id):
                if expected:
                    fraction = max(0.0, min(1.0, done / expected))
                    pct = ((idx - 1) + fraction) / total * 100
                    detail = f"{done / 1024**2:.1f}/{expected / 1024**2:.1f} MiB"
                else:
                    pct = (idx - 1) / total * 100
                    detail = f"{done / 1024**2:.1f} MiB"
                self.events.put(("progress", pct, f"Downloading {model}: {detail} — {label}"))

            Downloader(self.cache, progress=progress).get(resources[0])
            self.events.put(("progress", index / total * 100, f"Ready: {spec.id} ({index}/{total})"))
        self.events.put(("models_refreshed",))
        if run_after:
            self._run_sweep(specs, run_options or {})
        else:
            self.events.put(("status", "Selected model downloads are ready."))

    def _run_sweep(self, specs, run_options):
        assert self.source is not None
        refiner_id = run_options.get("refiner", "None")
        refinement = None
        if refiner_id != "None":
            self.events.put(("status", f"Preparing {refiner_id} diffusion components…"))
            from .refinement.diffusers_backend import download_bundle
            from .refinement.specs import get_spec
            refiner_spec, _ = get_spec(refiner_id)

            def refiner_progress(done, total, role, name, state):
                pct = done / total * 100 if total else 0
                self.events.put(("progress", pct, f"{refiner_id}: {state} {role}/{name} ({done}/{total})"))

            download_bundle(refiner_spec, cache_dir=self.cache, progress=refiner_progress)
            refinement = {
                "refiner": refiner_id,
                "options": {},
                "cache_dir": str(self.cache),
                "offline": False,
                "device": run_options.get("device", "auto"),
                "precision": run_options.get("precision", "fp32"),
                "offload": "none",
                "vae_tiling": True,
                "scheduler": "ddim",
                "strict_checksums": False,
                "external_plugins": False,
            }
        output_dir = unique_output_dir(self.source)
        self.output_dir = output_dir

        def sweep_progress(index, total, model_id, state):
            pct = index / total * 100 if state in {"success", "failed", "reused"} else (index - 1) / total * 100
            self.events.put(("progress", pct, f"{model_id}: {state} ({index}/{total})"))

        summary = run_all_models(
            self.source,
            output_dir,
            specs,
            session_options={"cache_dir": self.cache, "device": run_options.get("device", "auto"),
                             "precision": run_options.get("precision", "fp32")},
            image_options={},
            format="png",
            registry=self.registry,
            progress=sweep_progress,
            refinement=refinement,
        )
        self.events.put(("complete", summary, output_dir, self.source))

    def _drain_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    self.progress.configure(value=event[1])
                    self.status.set(event[2])
                elif kind == "status":
                    self.status.set(event[1])
                elif kind == "busy":
                    self._set_busy(event[1])
                elif kind == "models_refreshed":
                    selected = [self.specs[i].id for i in self.model_list.curselection()]
                    self._refresh_models()
                    for i, spec in enumerate(self.specs):
                        if spec.id not in selected:
                            self.model_list.selection_clear(i)
                elif kind == "complete":
                    _, summary, output_dir, source = event
                    self.entries = comparison_entries(summary, output_dir, source)
                    labels = [label for label, _ in self.entries[1:]]
                    self.result_box.configure(values=labels)
                    if labels:
                        self.result_box.current(0)
                    self.status.set(f"Complete: {output_dir}")
                    self.progress.configure(value=100)
                    self._render_comparison()
                    failed = summary.get("counts", {}).get("failed", 0)
                    if failed:
                        messagebox.showwarning("RasterMoves", f"Finished with {failed} failed model(s). See summary.json.")
                elif kind == "error":
                    self.status.set(event[1])
                    messagebox.showerror("RasterMoves", event[1])
        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _show_original(self, path: Path):
        self.entries = [("Original", path)]
        self._render_comparison()

    @staticmethod
    def _photo(path: Path, width: int, height: int):
        if width < 20 or height < 20:
            return None
        with Image.open(path) as image:
            image.thumbnail((width - 12, height - 12), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image.copy())

    def _render_comparison(self):
        if not self.entries:
            return
        original = self.entries[0][1]
        selected = None
        label = self.result_choice.get()
        for candidate_label, candidate in self.entries[1:]:
            if candidate_label == label:
                selected = candidate
                break
        self.original_photo = self._photo(original, self.original_canvas.winfo_width(), self.original_canvas.winfo_height())
        self.original_canvas.delete("all")
        if self.original_photo:
            self.original_canvas.create_image(self.original_canvas.winfo_width() // 2,
                                              self.original_canvas.winfo_height() // 2,
                                              image=self.original_photo)
        self.result_canvas.delete("all")
        if selected:
            self.result_photo = self._photo(selected, self.result_canvas.winfo_width(), self.result_canvas.winfo_height())
            if self.result_photo:
                self.result_canvas.create_image(self.result_canvas.winfo_width() // 2,
                                                self.result_canvas.winfo_height() // 2,
                                                image=self.result_photo)
            self.result_caption.configure(text=label)


def main() -> int:
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    RasterMovesGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
