"""
Motion Grade Grading Application.

Operator workflow:
  1. Load a CSV (Participant ID, Sample Number, Measurement Number — no grades needed)
  2. App resolves ISQ paths and runs cascade inference for each scan
  3. Review table lets operator approve or override each grade (1-5 scale)
  4. Export saves results CSV + appends to long-running corrections log

Launch with:
  run_grader.bat                  (normal mode — reads live ISQ from Y:\\data)
  run_grader.bat --debug          (debug mode — reads pre-processed .npy files)
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APP = os.path.dirname(os.path.abspath(__file__))
for _p in [_ROOT, _APP]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config

try:
    import customtkinter as ctk
except ImportError:
    raise ImportError("customtkinter not installed. Run: pip install customtkinter")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from cascade_infer import CascadePredictor
from accumulation_log import AccumulationLog
from csv_path_resolver import _extract_bone_type


# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

ROW_COLORS = {
    'pending':   '',
    'approved':  '#d4edda',
    'corrected': '#cce5ff',
    'error':     '#f8d7da',
    'low_conf':  '#fff3cd',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_to_pil(arr: np.ndarray) -> 'Image.Image':
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    return Image.fromarray((arr * 255).astype(np.uint8), mode='L')


def _make_copyable_textbox(parent, height: int = 68) -> ctk.CTkTextbox:
    """CTkTextbox that allows text selection and Ctrl+C but blocks editing."""
    box = ctk.CTkTextbox(parent, height=height)
    inner = box._textbox  # underlying tk.Text

    def _on_key(e):
        if e.state & 4 and e.keysym.lower() in ('c', 'a'):
            return None  # pass Ctrl+C / Ctrl+A
        return 'break'

    inner.bind('<Key>', _on_key)
    inner.bind('<<Paste>>', lambda e: 'break')
    inner.configure(cursor='ibeam')
    return box


def _set_copyable_text(box: ctk.CTkTextbox, text: str):
    box.configure(state='normal')
    box.delete('1.0', 'end')
    box.insert('1.0', text)


# ── Main App ──────────────────────────────────────────────────────────────────

class GradingApp(ctk.CTk):
    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.title("Motion Grade Classifier" + (" [DEBUG]" if debug else ""))
        self.geometry("1400x780")
        self.minsize(1050, 650)

        self.predictor: Optional[CascadePredictor] = None
        self.log = AccumulationLog()
        self.scan_rows: list[dict] = []
        self.selected_idx: Optional[int] = None

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_main()
        self._build_footer()
        self._show_frame('load')

    def _build_header(self):
        hdr = ctk.CTkFrame(self, height=50, corner_radius=0)
        hdr.grid(row=0, column=0, sticky='ew')
        ctk.CTkLabel(hdr, text="Motion Grade Classifier",
                     font=ctk.CTkFont(size=18, weight='bold')).pack(side='left', padx=16, pady=10)
        if self.debug:
            ctk.CTkLabel(hdr, text="DEBUG MODE — using pre-processed .npy data",
                         text_color='orange', font=ctk.CTkFont(size=12)).pack(side='left', padx=8)
        self._status_var = tk.StringVar(value="Load a CSV to begin.")
        ctk.CTkLabel(hdr, textvariable=self._status_var, font=ctk.CTkFont(size=12)).pack(side='right', padx=16)

    def _build_main(self):
        self._frames: dict = {}
        container = ctk.CTkFrame(self, corner_radius=0, fg_color='transparent')
        container.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)
        for FrameClass in (LoadFrame, ProcessingFrame, ReviewFrame):
            frame = FrameClass(container, self)
            frame.grid(row=0, column=0, sticky='nsew')
            self._frames[FrameClass.NAME] = frame

    def _build_footer(self):
        ftr = ctk.CTkFrame(self, height=35, corner_radius=0)
        ftr.grid(row=2, column=0, sticky='ew')
        log_summary = self.log.summary()
        ctk.CTkLabel(ftr,
                     text=f"Log: {log_summary['total']} total graded  |  {log_summary['corrected']} corrections",
                     font=ctk.CTkFont(size=11)).pack(side='left', padx=12)
        ctk.CTkLabel(ftr, text=f"Log file: {config.LOG_FILE}",
                     font=ctk.CTkFont(size=10), text_color='gray').pack(side='right', padx=12)

    def _show_frame(self, name: str):
        self._frames[name].tkraise()

    def _set_status(self, msg: str):
        self._status_var.set(msg)
        self.update_idletasks()

    def start_processing(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read CSV:\n{e}")
            return

        self.scan_rows = self._parse_csv(df)
        if not self.scan_rows:
            messagebox.showerror("Error",
                "No valid scans found in CSV (need Sample Number + Measurement Number).")
            return

        try:
            self.predictor = CascadePredictor()
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not initialize predictor:\n{e}")
            return

        self._show_frame('processing')
        self._frames['processing'].run(self.scan_rows, self._on_processing_done)

    def _on_processing_done(self, results: list[dict]):
        self.scan_rows = results
        self._frames['review'].load_results(results)
        self._show_frame('review')
        done = sum(1 for r in results if r.get('status') != 'error')
        errors = sum(1 for r in results if r.get('status') == 'error')
        self._set_status(f"Done: {done} graded, {errors} errors. Review and export.")

    def export_results(self, rows: list[dict]):
        out_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"motion_grades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        if not out_path:
            return

        out_rows = []
        log_rows = []
        for row in rows:
            if row.get('status') == 'error':
                continue
            out_row = {
                'participant_id':      row.get('participant_id', ''),
                'sample_number':       row.get('sample_number'),
                'measurement_number':  row.get('measurement_number'),
                'bone_type':           row.get('bone_type', ''),
                'machine_grade_bin':   row.get('machine_grade_bin', ''),
                'machine_grade_label': row.get('machine_grade_label', ''),
                'operator_grade':      row.get('operator_grade', ''),
                'operator_approved':   row.get('operator_approved', True),
                'operator_notes':      row.get('operator_notes', ''),
                'confidence':          f"{row.get('confidence', 0):.1%}",
                'graded_at':           row.get('graded_at', datetime.now().isoformat(timespec='seconds')),
            }
            out_rows.append(out_row)
            log_rows.append(out_row)

        pd.DataFrame(out_rows).to_csv(out_path, index=False)
        self.log.append(log_rows)
        s = self.log.summary()
        messagebox.showinfo("Exported",
                            f"Results saved to:\n{out_path}\n\n"
                            f"Log updated — {s['total']} total, {s['corrected']} corrections.")
        self._set_status(f"Exported {len(out_rows)} scans.")

    @staticmethod
    def _parse_csv(df: pd.DataFrame) -> list[dict]:
        import re

        def _find(candidates):
            for c in df.columns:
                norm = re.sub(r"[^a-z0-9]", "", c.lower())
                for cand in candidates:
                    if re.sub(r"[^a-z0-9]", "", cand.lower()) == norm:
                        return c
            return None

        sample_col = _find(["Sample Number", "SampleNumber", "sample"])
        meas_col   = _find(["Measurement Number", "MeasurementNumber", "measurement"])
        pid_col    = _find(["Participant ID", "ParticipantID"])

        if not sample_col or not meas_col:
            return []

        def _to_int(val):
            if pd.isna(val):
                raise ValueError
            m = re.search(r'\d+', str(val))
            if not m:
                raise ValueError
            return int(m.group())

        rows = []
        for _, row in df.iterrows():
            try:
                sample_num = _to_int(row[sample_col])
                meas_num   = _to_int(row[meas_col])
            except (ValueError, TypeError):
                continue
            pid = str(row[pid_col]).strip() if pid_col and not pd.isna(row.get(pid_col)) else ''
            bone_type = _extract_bone_type(pid) or 'Tibia'
            rows.append({
                'participant_id':    pid,
                'sample_number':     sample_num,
                'measurement_number': meas_num,
                'bone_type':         bone_type,
                'status':            'pending',
                'machine_grade_bin':  None,
                'machine_grade_label': '',
                'confidence':        0.0,
                'operator_grade':    None,
                'operator_approved': None,
                'operator_notes':    '',
                'graded_at':         '',
            })
        return rows


# ── Load Frame ────────────────────────────────────────────────────────────────

class LoadFrame(ctk.CTkFrame):
    NAME = 'load'

    def __init__(self, parent, app: GradingApp):
        super().__init__(parent, corner_radius=8)
        self.app = app
        self._csv_var = tk.StringVar()

        inner = ctk.CTkFrame(self, fg_color='transparent')
        inner.place(relx=0.5, rely=0.5, anchor='center')

        ctk.CTkLabel(inner, text="Load Scan CSV",
                     font=ctk.CTkFont(size=20, weight='bold')).grid(
            row=0, column=0, columnspan=3, pady=(0, 20))

        ctk.CTkLabel(inner, text="CSV file:",
                     font=ctk.CTkFont(size=13)).grid(row=1, column=0, sticky='e', padx=8)
        ctk.CTkEntry(inner, textvariable=self._csv_var, width=380).grid(row=1, column=1, sticky='ew')
        ctk.CTkButton(inner, text="Browse...", width=90, command=self._browse).grid(
            row=1, column=2, padx=(6, 0))

        ctk.CTkLabel(inner,
                     text="CSV must have 'Participant ID', 'Sample Number', and 'Measurement Number' columns.\n"
                          "Grade column is optional — if present it will be ignored.",
                     font=ctk.CTkFont(size=11), text_color='gray').grid(
            row=2, column=0, columnspan=3, pady=(10, 20))

        ctk.CTkButton(inner, text="Start Processing",
                      font=ctk.CTkFont(size=14, weight='bold'),
                      height=42, command=self._start).grid(row=3, column=0, columnspan=3, pady=10)

        if app.debug:
            ds_dir = str(config.DEBUG_DATASET_DIR or "Not set in config.py")
            ctk.CTkLabel(inner, text=f"Debug dataset: {ds_dir}",
                         font=ctk.CTkFont(size=10), text_color='orange').grid(
                row=4, column=0, columnspan=3, pady=(0, 8))

    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self._csv_var.set(path)

    def _start(self):
        csv_path = self._csv_var.get().strip()
        if not csv_path:
            messagebox.showwarning("No file", "Please select a CSV file first.")
            return
        if not Path(csv_path).exists():
            messagebox.showerror("Not found", f"File not found:\n{csv_path}")
            return
        self.app.start_processing(csv_path)


# ── Processing Frame ──────────────────────────────────────────────────────────

class ProcessingFrame(ctk.CTkFrame):
    NAME = 'processing'

    def __init__(self, parent, app: GradingApp):
        super().__init__(parent, corner_radius=8)
        self.app = app
        self._cancel_event = threading.Event()
        self._q: queue.Queue = queue.Queue()

        inner = ctk.CTkFrame(self, fg_color='transparent')
        inner.place(relx=0.5, rely=0.5, anchor='center')

        ctk.CTkLabel(inner, text="Processing Scans...",
                     font=ctk.CTkFont(size=18, weight='bold')).pack(pady=(0, 20))

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ctk.CTkProgressBar(inner, width=420, variable=self._progress_var)
        self._progress_bar.pack(pady=8)

        self._scan_label = ctk.CTkLabel(inner, text="Initializing...", font=ctk.CTkFont(size=12))
        self._scan_label.pack(pady=4)

        self._counter_label = ctk.CTkLabel(inner, text="",
                                            font=ctk.CTkFont(size=11), text_color='gray')
        self._counter_label.pack()

        ctk.CTkButton(inner, text="Cancel", width=100,
                      fg_color='#dc3545', hover_color='#a71d2a',
                      command=self._cancel).pack(pady=(20, 0))

    def run(self, scan_rows: list[dict], on_done):
        self._cancel_event.clear()
        self._results = [row.copy() for row in scan_rows]
        self._on_done = on_done
        self._total = len(scan_rows)
        self._processed = 0
        self._progress_var.set(0)
        self._scan_label.configure(text=f"Starting {self._total} scans...")
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        self.after(100, self._poll)

    def _worker(self):
        predictor = self.app.predictor
        debug = self.app.debug
        for i, row in enumerate(self._results):
            if self._cancel_event.is_set():
                break
            try:
                sample_num = row['sample_number']
                meas_num   = row['measurement_number']
                bone_type  = row['bone_type']
                predictor.load_bone(bone_type)
                if debug:
                    result = predictor.predict_debug(sample_num, meas_num, bone_type)
                else:
                    result = predictor.predict(sample_num, meas_num, bone_type)
                row['machine_grade_bin']   = result['grade_bin']
                row['machine_grade_label'] = result['grade_label']
                row['confidence']          = result['confidence']
                row['status'] = ('low_conf'
                                 if result['confidence'] < config.LOW_CONFIDENCE_THRESHOLD
                                 else 'pending')
                row['operator_grade']    = [1, 2, 4][result['grade_bin']]
                row['operator_approved'] = True
                row['graded_at']         = datetime.now().isoformat(timespec='seconds')
            except Exception as e:
                row['status']              = 'error'
                row['machine_grade_label'] = f"Error: {e}"
                row['confidence']          = 0.0
            self._q.put(('progress', i, row))
        self._q.put(('done', None, None))

    def _poll(self):
        try:
            while True:
                event, idx, row = self._q.get_nowait()
                if event == 'done':
                    self._on_done(self._results)
                    return
                if event == 'progress':
                    self._results[idx] = row
                    self._processed += 1
                    frac = self._processed / self._total
                    self._progress_var.set(frac)
                    pid = (row.get('participant_id', '')
                           or f"s{row['sample_number']}_m{row['measurement_number']}")
                    self._scan_label.configure(text=f"Processing: {pid}")
                    self._counter_label.configure(text=f"{self._processed} / {self._total}")
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _cancel(self):
        self._cancel_event.set()
        self.app._set_status("Cancelled.")
        self.app._show_frame('load')


# ── Preview Panel ─────────────────────────────────────────────────────────────

class PreviewPanel(ctk.CTkFrame):
    """Z-stack image viewer for the currently selected scan."""

    def __init__(self, parent, app: GradingApp):
        super().__init__(parent, corner_radius=6)
        self.app = app
        self._slices: list = []       # list of PIL Images (set on main thread only)
        self._slice_idx = 0
        self._overlay_msg = "Select a scan to preview."
        self._cancel_load = threading.Event()
        self._canvas_photo = None     # keep PhotoImage reference alive

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(self, fg_color='transparent')
        hdr.grid(row=0, column=0, sticky='ew', padx=8, pady=(8, 2))
        hdr.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(hdr, text="Image Preview",
                     font=ctk.CTkFont(size=12, weight='bold')).grid(row=0, column=0, sticky='w')
        self._slice_counter = ctk.CTkLabel(hdr, text="",
                                            font=ctk.CTkFont(size=10), text_color='gray')
        self._slice_counter.grid(row=0, column=1, sticky='e')

        if not _PIL_AVAILABLE:
            ctk.CTkLabel(self,
                         text="Image preview requires Pillow.\nRun: pip install Pillow",
                         text_color='orange', font=ctk.CTkFont(size=11)).grid(
                row=1, column=0)
            return

        self._canvas = tk.Canvas(self, bg='#1c1c1c', bd=0, highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky='nsew', padx=8, pady=4)
        self._canvas.bind('<MouseWheel>', self._on_scroll)
        self._canvas.bind('<Configure>', self._on_resize)

        slider_frame = ctk.CTkFrame(self, fg_color='transparent')
        slider_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=(0, 10))
        slider_frame.grid_columnconfigure(0, weight=1)
        self._slider = ctk.CTkSlider(slider_frame, from_=0, to=1, number_of_steps=1,
                                      command=self._on_slider_move, state='disabled')
        self._slider.grid(row=0, column=0, sticky='ew')

    # ── Public API ────────────────────────────────────────────────────────────

    def load_scan(self, row: dict):
        """Start loading slices for this scan asynchronously."""
        if not _PIL_AVAILABLE:
            return

        if row.get('status') == 'error':
            self._overlay_msg = f"Error: {row.get('machine_grade_label', 'scan error')}"
            self._slices = []
            self._slice_counter.configure(text="")
            self._draw_overlay()
            return

        # Cancel any in-flight load
        self._cancel_load.set()
        self._cancel_load = threading.Event()
        cancel = self._cancel_load

        self._slices = []
        self._slice_idx = 0
        self._overlay_msg = "Loading..."
        self._slice_counter.configure(text="")
        if hasattr(self, '_slider'):
            self._slider.configure(state='disabled')
            self._slider.set(0)
        self._draw_overlay()

        t = threading.Thread(target=self._load_worker, args=(row, cancel), daemon=True)
        t.start()

    # ── Background loading ────────────────────────────────────────────────────

    def _load_worker(self, row: dict, cancel: threading.Event):
        try:
            slices = self._read_slices(row, cancel)
            if cancel.is_set():
                return
            # Hand off to main thread
            mid = len(slices) // 2
            self.after(0, lambda s=slices, i=mid: self._on_slices_ready(s, i))
        except Exception as e:
            if not cancel.is_set():
                msg = f"Preview error:\n{e}"
                self.after(0, lambda m=msg: self._set_overlay(m))

    def _read_slices(self, row: dict, cancel: threading.Event) -> list:
        sample_num = row['sample_number']
        meas_num   = row['measurement_number']

        if self.app.debug:
            ds_dir = Path(config.DEBUG_DATASET_DIR)
            scan_folder = ds_dir / f"s{sample_num:08d}_m{meas_num:08d}"
            files = sorted(scan_folder.glob("*.npy"))
            slices = []
            for f in files:
                if cancel.is_set():
                    return []
                slices.append(_normalize_to_pil(np.load(str(f)).astype(np.float32)))
            return slices
        else:
            # Live ISQ — load full volume without crop (preview only)
            from isq_reader import read_isq
            from csv_path_resolver import first_isq
            isq_root = Path(config.ISQ_ROOT)
            scan_dir = isq_root / f"{sample_num:08d}" / f"{meas_num:08d}"
            isq_path = first_isq(scan_dir)
            if isq_path is None:
                raise FileNotFoundError(f"No ISQ file at {scan_dir}")
            volume = read_isq(isq_path, verbose=False)  # (z, y, x)
            slices = []
            for arr in volume:
                if cancel.is_set():
                    return []
                slices.append(_normalize_to_pil(arr.astype(np.float32)))
            return slices

    def _on_slices_ready(self, slices: list, start_idx: int):
        self._slices = slices
        self._slice_idx = start_idx
        if not slices:
            self._set_overlay("No images available.")
            return
        self._overlay_msg = ""
        n = len(slices)
        if hasattr(self, '_slider'):
            steps = max(1, n - 1)
            self._slider.configure(to=steps, number_of_steps=steps, state='normal')
            self._slider.set(start_idx)
        self._slice_counter.configure(text=f"Slice {self._slice_idx + 1} / {n}")
        self._redraw()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _redraw(self):
        if not self._slices:
            self._draw_overlay()
            return
        n = len(self._slices)
        self._slice_idx = max(0, min(self._slice_idx, n - 1))
        self._slice_counter.configure(text=f"Slice {self._slice_idx + 1} / {n}")

        img = self._slices[self._slice_idx].copy()
        w = max(self._canvas.winfo_width(), 10)
        h = max(self._canvas.winfo_height(), 10)
        img.thumbnail((w, h), Image.LANCZOS)

        self._canvas_photo = ImageTk.PhotoImage(img)
        self._canvas.delete('all')
        self._canvas.create_image(w // 2, h // 2, anchor='center', image=self._canvas_photo)

    def _set_overlay(self, msg: str):
        self._overlay_msg = msg
        self._slices = []
        self._slice_counter.configure(text="")
        self._draw_overlay()

    def _draw_overlay(self):
        if not hasattr(self, '_canvas'):
            return
        w = max(self._canvas.winfo_width(), 10)
        h = max(self._canvas.winfo_height(), 10)
        self._canvas.delete('all')
        self._canvas_photo = None
        self._canvas.create_text(
            w // 2, h // 2, text=self._overlay_msg,
            fill='#888888', font=('Segoe UI', 11),
            anchor='center', justify='center', width=max(w - 20, 10),
        )

    def _on_resize(self, _event):
        if self._slices:
            self._redraw()
        else:
            self._draw_overlay()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _on_slider_move(self, value: float):
        if self._slices:
            self._slice_idx = round(value)
            self._redraw()

    def _on_scroll(self, event):
        if not self._slices:
            return
        if event.delta > 0:
            self._slice_idx = max(0, self._slice_idx - 1)
        else:
            self._slice_idx = min(len(self._slices) - 1, self._slice_idx + 1)
        if hasattr(self, '_slider'):
            self._slider.set(self._slice_idx)
        self._redraw()


# ── Review Frame ──────────────────────────────────────────────────────────────

class ReviewFrame(ctk.CTkFrame):
    NAME = 'review'

    def __init__(self, parent, app: GradingApp):
        super().__init__(parent, corner_radius=8)
        self.app = app
        self._rows: list[dict] = []

        # Three-column layout: table | preview | edit panel
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self._build_table()
        self._build_preview()
        self._build_edit_panel()
        self._build_bottom_bar()

    def _build_table(self):
        table_frame = ctk.CTkFrame(self, corner_radius=6)
        table_frame.grid(row=0, column=0, sticky='nsew', padx=(6, 3), pady=6)
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        cols       = ('participant_id', 'bone', 'machine_grade', 'confidence', 'operator_grade', 'status')
        col_labels = ('Participant ID', 'Bone', 'Machine Grade', 'Conf.', 'Operator Grade', 'Status')
        col_widths = (160, 58, 155, 52, 95, 82)

        style = ttk.Style()
        style.configure("Treeview", rowheight=24, font=('Segoe UI', 10))
        style.configure("Treeview.Heading", font=('Segoe UI', 10, 'bold'))

        self._tree = ttk.Treeview(table_frame, columns=cols, show='headings', selectmode='browse')
        for col, label, width in zip(cols, col_labels, col_widths):
            self._tree.heading(col, text=label)
            self._tree.column(col, width=width, minwidth=40, anchor='center')

        vsb = ttk.Scrollbar(table_frame, orient='vertical', command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')

        for tag, color in ROW_COLORS.items():
            if color:
                self._tree.tag_configure(tag, background=color)

        self._tree.bind('<<TreeviewSelect>>', self._on_select)

    def _build_preview(self):
        self._preview = PreviewPanel(self, self.app)
        self._preview.grid(row=0, column=1, sticky='nsew', padx=3, pady=6)

    def _build_edit_panel(self):
        panel = ctk.CTkFrame(self, corner_radius=6)
        panel.grid(row=0, column=2, sticky='nsew', padx=(3, 6), pady=6)
        panel.grid_columnconfigure(0, weight=1)

        # Row 8 is the spacer that absorbs leftover height,
        # keeping Prev/Next pinned to the bottom of the panel.
        panel.grid_rowconfigure(0, weight=0)  # header
        panel.grid_rowconfigure(1, weight=0)  # scan info (copyable)
        panel.grid_rowconfigure(2, weight=0)  # machine prediction (copyable)
        panel.grid_rowconfigure(3, weight=0)  # grade label
        panel.grid_rowconfigure(4, weight=0)  # radio buttons
        panel.grid_rowconfigure(5, weight=0)  # notes label
        panel.grid_rowconfigure(6, weight=0)  # notes textbox
        panel.grid_rowconfigure(7, weight=0)  # approve/override buttons
        panel.grid_rowconfigure(8, weight=1)  # spacer
        panel.grid_rowconfigure(9, weight=0)  # prev/next — always at bottom

        ctk.CTkLabel(panel, text="Review & Override",
                     font=ctk.CTkFont(size=14, weight='bold')).grid(
            row=0, column=0, pady=(12, 6), padx=10)

        # Copyable scan info (3 lines)
        self._info_text = _make_copyable_textbox(panel, height=68)
        self._info_text.grid(row=1, column=0, padx=10, pady=(0, 4), sticky='ew')
        _set_copyable_text(self._info_text, "Select a scan to review.")

        # Copyable machine prediction (2 lines)
        self._machine_text = _make_copyable_textbox(panel, height=46)
        self._machine_text.grid(row=2, column=0, padx=10, pady=(0, 4), sticky='ew')

        ctk.CTkLabel(panel, text="Operator grade (1–5):",
                     font=ctk.CTkFont(size=12, weight='bold')).grid(
            row=3, column=0, padx=10, pady=(10, 4), sticky='w')

        grade_frame = ctk.CTkFrame(panel, fg_color='transparent')
        grade_frame.grid(row=4, column=0, padx=10, sticky='w')
        self._grade_var = tk.IntVar(value=0)
        self._grade_buttons: list = []
        for g in range(1, 6):
            rb = ctk.CTkRadioButton(grade_frame, text=str(g),
                                     variable=self._grade_var, value=g,
                                     font=ctk.CTkFont(size=13),
                                     command=self._on_grade_change)
            rb.grid(row=0, column=g - 1, padx=4)
            self._grade_buttons.append(rb)

        ctk.CTkLabel(panel, text="Notes (optional):",
                     font=ctk.CTkFont(size=12)).grid(
            row=5, column=0, padx=10, pady=(12, 2), sticky='w')

        self._notes_box = ctk.CTkTextbox(panel, height=80, width=200)
        self._notes_box.grid(row=6, column=0, padx=10, sticky='ew')

        btn_frame = ctk.CTkFrame(panel, fg_color='transparent')
        btn_frame.grid(row=7, column=0, padx=10, pady=(8, 4), sticky='ew')
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(btn_frame, text="Approve",
                      fg_color='#28a745', hover_color='#1e7e34',
                      command=self._approve).grid(row=0, column=0, padx=3, sticky='ew')
        ctk.CTkButton(btn_frame, text="Override",
                      fg_color='#0d6efd', hover_color='#0a58ca',
                      command=self._override).grid(row=0, column=1, padx=3, sticky='ew')

        # Row 8 is the spacer (weight=1 above) — no widget needed

        nav_frame = ctk.CTkFrame(panel, fg_color='transparent')
        nav_frame.grid(row=9, column=0, pady=(4, 14))
        ctk.CTkButton(nav_frame, text="◀ Prev", width=80,
                      command=self._prev).grid(row=0, column=0, padx=4)
        ctk.CTkButton(nav_frame, text="Next ▶", width=80,
                      command=self._next).grid(row=0, column=1, padx=4)

    def _build_bottom_bar(self):
        bar = ctk.CTkFrame(self, corner_radius=0, height=44)
        bar.grid(row=1, column=0, columnspan=3, sticky='ew', padx=6, pady=(0, 6))
        bar.grid_columnconfigure(0, weight=1)
        self._summary_label = ctk.CTkLabel(bar, text="", font=ctk.CTkFont(size=12))
        self._summary_label.grid(row=0, column=0, sticky='w', padx=12)
        ctk.CTkButton(bar, text="Export Results",
                      font=ctk.CTkFont(size=13, weight='bold'),
                      height=34, command=self._export).grid(row=0, column=1, padx=12, pady=5)

    # ── Table management ──────────────────────────────────────────────────────

    def load_results(self, rows: list[dict]):
        self._rows = rows
        self._refresh_table()

    def _refresh_table(self):
        self._tree.delete(*self._tree.get_children())
        for i, row in enumerate(self._rows):
            grade_bin    = row.get('machine_grade_bin')
            machine_label = config.GRADE_BIN_LABELS.get(grade_bin, row.get('machine_grade_label', ''))
            op_grade     = row.get('operator_grade', '')
            conf         = row.get('confidence', 0)
            conf_str     = f"{conf:.0%}" if isinstance(conf, float) else ''

            approved = row.get('operator_approved')
            if row['status'] == 'error':
                status_str, tag = 'Error', 'error'
            elif approved is False:
                status_str, tag = 'Modified', 'corrected'
            elif approved is True:
                status_str, tag = 'Approved', 'approved'
            elif row['status'] == 'low_conf':
                status_str, tag = '⚠ Low conf', 'low_conf'
            else:
                status_str, tag = 'Pending', 'pending'

            self._tree.insert('', 'end', iid=str(i), tags=(tag,), values=(
                row.get('participant_id', ''),
                row.get('bone_type', ''),
                machine_label if row['status'] != 'error' else row.get('machine_grade_label', 'Error'),
                conf_str,
                op_grade,
                status_str,
            ))
        self._update_summary()

    def _on_select(self, _event=None):
        sel = self._tree.selection()
        if not sel:
            return
        self._load_into_panel(int(sel[0]))

    def _load_into_panel(self, idx: int):
        row = self._rows[idx]
        self.app.selected_idx = idx

        pid = row.get('participant_id', '') or f"s{row['sample_number']}_m{row['measurement_number']}"
        _set_copyable_text(self._info_text,
            f"Scan: {pid}\n"
            f"Sample: {row['sample_number']}  Meas: {row['measurement_number']}\n"
            f"Bone: {row['bone_type']}")

        grade_bin   = row.get('machine_grade_bin')
        machine_lbl = config.GRADE_BIN_LABELS.get(grade_bin, row.get('machine_grade_label', 'N/A'))
        conf        = row.get('confidence', 0)
        _set_copyable_text(self._machine_text,
            f"Machine: {machine_lbl}\nConfidence: {conf:.1%}")

        op_grade = row.get('operator_grade') or ([1, 2, 4][grade_bin] if grade_bin is not None else 1)
        self._grade_var.set(int(op_grade) if op_grade else 0)

        self._notes_box.delete('1.0', 'end')
        self._notes_box.insert('1.0', row.get('operator_notes', ''))

        state = 'disabled' if row['status'] == 'error' else 'normal'
        for rb in self._grade_buttons:
            rb.configure(state=state)

        self._preview.load_scan(row)

    def _on_grade_change(self):
        pass

    def _get_notes(self) -> str:
        return self._notes_box.get('1.0', 'end').strip()

    def _approve(self):
        idx = self.app.selected_idx
        if idx is None:
            return
        op_grade = self._grade_var.get()
        if op_grade == 0:
            messagebox.showwarning("No grade", "Select a grade (1–5) before approving.")
            return
        row = self._rows[idx]
        row['operator_grade']    = op_grade
        row['operator_approved'] = True
        row['operator_notes']    = self._get_notes()
        row['status']            = 'approved'
        row['graded_at']         = datetime.now().isoformat(timespec='seconds')
        self._refresh_table()
        self._tree.selection_set(str(idx))
        self._next()

    def _override(self):
        idx = self.app.selected_idx
        if idx is None:
            return
        op_grade = self._grade_var.get()
        if op_grade == 0:
            messagebox.showwarning("No grade", "Select a grade (1–5) before overriding.")
            return
        row = self._rows[idx]
        machine_bin        = row.get('machine_grade_bin')
        machine_suggestion = [1, 2, 4][machine_bin] if machine_bin is not None else None
        row['operator_grade']    = op_grade
        row['operator_approved'] = (op_grade == machine_suggestion)
        row['operator_notes']    = self._get_notes()
        row['status']            = 'approved' if row['operator_approved'] else 'corrected'
        row['graded_at']         = datetime.now().isoformat(timespec='seconds')
        self._refresh_table()
        self._tree.selection_set(str(idx))
        self._next()

    def _prev(self):
        idx = self.app.selected_idx
        if idx is None or idx == 0:
            return
        new_idx = idx - 1
        self._tree.selection_set(str(new_idx))
        self._tree.see(str(new_idx))
        self._load_into_panel(new_idx)

    def _next(self):
        idx = self.app.selected_idx
        if idx is None or idx >= len(self._rows) - 1:
            return
        new_idx = idx + 1
        self._tree.selection_set(str(new_idx))
        self._tree.see(str(new_idx))
        self._load_into_panel(new_idx)

    def _update_summary(self):
        total     = len(self._rows)
        approved  = sum(1 for r in self._rows if r.get('operator_approved') is True)
        corrected = sum(1 for r in self._rows if r.get('operator_approved') is False)
        errors    = sum(1 for r in self._rows if r['status'] == 'error')
        pending   = total - approved - corrected - errors
        self._summary_label.configure(
            text=f"{total} scans  |  {approved} approved  |  {corrected} modified  "
                 f"|  {errors} errors  |  {pending} pending")

    def _export(self):
        unapproved = sum(1 for r in self._rows
                         if r.get('status') not in ('approved', 'corrected', 'error')
                         and r.get('operator_approved') is None)
        if unapproved > 0:
            if not messagebox.askyesno("Pending reviews",
                                       f"{unapproved} scans have not been reviewed yet.\n"
                                       f"Export anyway?"):
                return
        self.app.export_results(self._rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Motion Grade Grading Application")
    parser.add_argument("--debug", action="store_true",
                        help="Use pre-processed .npy dataset instead of live ISQ reading")
    args = parser.parse_args()

    if args.debug:
        ds = config.DEBUG_DATASET_DIR
        if not ds:
            print("Error: DEBUG_DATASET_DIR is not set in config.py")
            sys.exit(1)
        if not Path(ds).exists():
            print(f"Error: DEBUG_DATASET_DIR does not exist: {ds}")
            sys.exit(1)

    app = GradingApp(debug=args.debug)
    app.mainloop()


if __name__ == "__main__":
    main()
