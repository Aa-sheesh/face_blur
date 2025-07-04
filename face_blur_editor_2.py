#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face-Blur Video Editor
Incremental GPU/CPU Blur + Live Tracking + NVENC Export
+ JSON persistence (central annotations/ folder)
+ Shift-drag add-bbox
+ Button-add / Del-add-bbox
+ Corner-drag resize
+ Drag move
+ Click toggle blur
+ Confidence slider (display & export)
+ Undo/Redo (Ctrl+Z / Ctrl+Y)
+ Copy/Paste bbox (Ctrl+C / Ctrl+V)
+ Live preview + thumbnail generation
"""
import os
import sys
import cv2
import shutil
import subprocess
import copy
import json
from pathlib import Path
from ultralytics import YOLO
from PyQt5.QtCore    import Qt, QThread, pyqtSignal
from PyQt5.QtGui     import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QLabel, QPushButton, QListWidget, QProgressBar,
    QSlider, QWidget, QHBoxLayout, QVBoxLayout, QShortcut
)

# ───────────────────── CONFIG ───────────────────── #
YOLO_MODEL_PATH = r"D:\Desktop\Face Recognition 2\vfs-forensic-latest-custom\face_blur\yolov8n-face.pt"
DEVICE_ID       = 0
BLUR_KERNEL     = (45, 45)
CORNER_THRESH   = 8    # px (video coords)
CLICK_THRESH    = 5    # px (display coords)
THUMB_SCALE     = 0.125  # 1/8 size thumbs

# central annotations folder
ANNOTATIONS_DIR = os.path.join(os.getcwd(), "annotations")
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
# ─────────────────────────────────────────────────── #

# ───────── data containers ───────── #
class FrameDetections:
    __slots__ = ("boxes","ids","confs")
    def __init__(self):
        self.boxes = []; self.ids = []; self.confs = []
    def add(self, bbox, tid, conf):
        self.boxes.append(bbox); self.ids.append(tid); self.confs.append(conf)
    def __iter__(self):
        return zip(self.boxes, self.ids, self.confs)

class VideoInfo:
    __slots__ = ("path","nF","fps","w","h","frames_dir","thumb_dir")
    def __init__(self, path, nF, fps, w, h, frames_dir, thumb_dir):
        self.path, self.nF, self.fps = path, nF, fps
        self.w, self.h = w, h
        self.frames_dir, self.thumb_dir = frames_dir, thumb_dir

# ───────────── persistence ───────────── #
def make_video_id(path: str) -> str:
    """Use file size + mtime as stable video ID."""
    st = Path(path).stat()
    return f"{st.st_size}_{int(st.st_mtime)}"

def json_path_for(video_id: str) -> str:
    return os.path.join(ANNOTATIONS_DIR, f"{video_id}.json")

def backup_json(video_id: str):
    src = json_path_for(video_id)
    if os.path.exists(src):
        dst = os.path.join(ANNOTATIONS_DIR, f"{video_id}_backup.json")
        shutil.copy2(src, dst)

def load_annotations(video_id: str):
    p = json_path_for(video_id)
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return json.load(f)

def save_annotations(video_id: str, rec: dict):
    backup_json(video_id)
    with open(json_path_for(video_id), "w") as f:
        json.dump(rec, f, indent=2)

# ───────────── Incremental Tracker Thread ───────────── #
class TrackerThread(QThread):
    stepped = pyqtSignal(int)
    prog    = pyqtSignal(int)
    done    = pyqtSignal(VideoInfo)

    def __init__(self, video_path, frames_dir, thumb_dir, shared_detections):
        super().__init__()
        self.video_path        = video_path
        self.last_tid_clicked = None   # track which track‐ID was last clicked
        self.last_bbox_clicked = None  # its bbox (x1,y1,x2,y2)
        self.frames_dir        = frames_dir
        self.thumb_dir         = thumb_dir
        self.shared_detections = shared_detections

    def run(self):
        # prepare dirs
        shutil.rmtree(self.frames_dir, ignore_errors=True)
        shutil.rmtree(self.thumb_dir, ignore_errors=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.thumb_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.critical(None, "Error", f"Cannot open {self.video_path}")
            return
        nF  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        model = YOLO(YOLO_MODEL_PATH).to(f"cuda:{DEVICE_ID}")
        vr    = cv2.VideoCapture(self.video_path)

        for idx, res in enumerate(model.track(
            source=self.video_path,
            device=DEVICE_ID,
            stream=True, verbose=False,
            conf=0.01,
            tracker = r"D:\Desktop\Face Recognition 2\vfs-forensic-latest-custom\face_blur\botsort.yaml"
        )):
            vr.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = vr.read()
            if not ok:
                continue

            # save full frame
            frm_path = os.path.join(self.frames_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(frm_path, frame)
            # save thumbnail
            thumb = cv2.resize(frame, (0,0), fx=THUMB_SCALE, fy=THUMB_SCALE)
            cv2.imwrite(os.path.join(self.thumb_dir, f"thumb_{idx:06d}.jpg"), thumb)

            # collect detections
            fd = FrameDetections()
            if res.boxes is not None and len(res.boxes):
                ids   = res.boxes.id.cpu().numpy()   if res.boxes.id   is not None else [None]*len(res.boxes)
                confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else [0.0]*len(res.boxes)
                xyxy  = res.boxes.xyxy.cpu().numpy()
                for (x1,y1,x2,y2), tid, cf in zip(xyxy, ids, confs):
                    if tid is not None:
                        fd.add((int(x1),int(y1),int(x2),int(y2)), int(tid), float(cf))
            self.shared_detections[idx] = fd

            # signals
            if idx % 5 == 0:
                self.prog.emit(int((idx+1)/nF*100))
            self.stepped.emit(idx)

        vr.release()
        self.prog.emit(100)
        info = VideoInfo(self.video_path, nF, fps, w, h, self.frames_dir, self.thumb_dir)
        self.done.emit(info)

# ───────────── Export Thread ───────────── #
class ExportThread(QThread):
    prog = pyqtSignal(int)
    done = pyqtSignal(str)

    def __init__(self, info, dets, blur_ids, out_path, threshold):
        super().__init__()
        self.last_tid_clicked = None   # track which track‐ID was last clicked
        self.last_bbox_clicked = None  # its bbox (x1,y1,x2,y2)
        self.info      = info
        self.dets      = dets
        self.blur_ids  = blur_ids
        self.out       = out_path
        self.threshold = threshold
        try:
            self.blur_filter = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC3, -1, BLUR_KERNEL, sigmaX=0)
            self.use_gpu = True
        except:
            self.blur_filter, self.use_gpu = None, False

    def run(self):
        tmp = f"{self.info.frames_dir}_blurred"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)

        for idx in range(self.info.nF):
            frame = cv2.imread(os.path.join(self.info.frames_dir, f"frame_{idx:06d}.jpg"))
            if frame is None:
                continue

            for bbox, tid, conf in self.dets.get(idx, FrameDetections()):
                if conf < self.threshold or tid not in self.blur_ids:
                    continue
                x1,y1,x2,y2 = bbox
                roi = frame[y1:y2, x1:x2]
                if roi.size:
                    if self.use_gpu:
                        try:
                            gm = cv2.cuda_GpuMat(); gm.upload(roi)
                            roi = self.blur_filter.apply(gm).download()
                        except:
                            roi = cv2.GaussianBlur(roi, BLUR_KERNEL, 0)
                    else:
                        roi = cv2.GaussianBlur(roi, BLUR_KERNEL, 0)
                    frame[y1:y2, x1:x2] = roi

            cv2.imwrite(os.path.join(tmp, f"frame_{idx:06d}.png"), frame)
            if idx % 10 == 0:
                self.prog.emit(int((idx+1)/self.info.nF*50))

        cmd = [
            "ffmpeg","-y","-hwaccel","cuda",
            "-framerate", str(self.info.fps),
            "-i", os.path.join(tmp, "frame_%06d.png"),
            "-c:v","h264_nvenc","-preset","fast",
            self.out
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.done.emit(self.out)

# ───────────── Save-Frames Thread ───────────── #
class SaveFramesThread(QThread):
    prog = pyqtSignal(int)
    done = pyqtSignal(str)

    def __init__(self, info, dets, blur_ids, threshold):
        super().__init__()
        self.info      = info
        self.dets      = dets
        self.blur_ids  = blur_ids
        self.threshold = threshold
        try:
            self.blur_filter = cv2.cuda.createGaussianFilter(
                cv2.CV_8UC3, -1, BLUR_KERNEL, sigmaX=0)
            self.use_gpu = True
        except Exception:
            self.blur_filter, self.use_gpu = None, False

    def run(self):
        try:
            dst_dir = f"{self.info.frames_dir}_blurred"
            os.makedirs(dst_dir, exist_ok=True)

            for idx in range(self.info.nF):
                src = os.path.join(self.info.frames_dir,
                                   f"frame_{idx:06d}.jpg")
                if not os.path.exists(src):
                    continue
                frame = cv2.imread(src)
                if frame is None:
                    continue

                # apply blur where needed
                for bbox, tid, conf in self.dets.get(
                        idx, FrameDetections()):
                    if conf < self.threshold or tid not in self.blur_ids:
                        continue
                    x1, y1, x2, y2 = bbox
                    roi = frame[y1:y2, x1:x2]
                    if roi.size:
                        if self.use_gpu:
                            try:
                                g = cv2.cuda_GpuMat(); g.upload(roi)
                                roi = self.blur_filter.apply(g).download()
                            except Exception:
                                roi = cv2.GaussianBlur(roi, BLUR_KERNEL, 0)
                        else:
                            roi = cv2.GaussianBlur(roi, BLUR_KERNEL, 0)
                        frame[y1:y2, x1:x2] = roi

                cv2.imwrite(os.path.join(dst_dir,
                             f"frame_{idx:06d}.jpg"), frame)

                if idx % 10 == 0:
                    self.prog.emit(int((idx + 1) / self.info.nF * 100))

            self.prog.emit(100)
            self.done.emit(dst_dir)

        except Exception as exc:
            # bubble up fatal errors in a safe way
            self.done.emit(f"ERROR::{exc}")


# ───────────── Main Window ───────────── #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # state
        self.detections      = {}    # frame→FrameDetections
        self.blur_ids        = set()
        self.info            = None
        self.current_idx     = 0
        self.last_tid_clicked = None   # track which track‐ID was last clicked
        self.last_bbox_clicked = None  # its bbox (x1,y1,x2,y2)
        self.undo_stack      = []
        self.redo_stack      = []
        self.mode            = None
        self.act_tid         = None
        self.corner_idx      = None
        self.start_pt        = None
        self.orig_boxes      = {}
        self.pending_add     = False
        self.pending_delete  = False
        self.copied_bbox     = None  # for Ctrl+C/V

        # JSON persistence
        self.video_id = None   # set on open

        # widgets
        self.open_btn    = QPushButton("Open Video")
        self.export_btn  = QPushButton("Export Video")
        self.add_btn     = QPushButton("Add BBox")
        self.del_btn     = QPushButton("Delete BBox")
        self.prev_btn    = QPushButton("◀ Prev [A]")
        self.save_frames_btn = QPushButton("Save Blurred Frames")
        self.next_btn    = QPushButton("Next ▶ [D]")
        self.slider      = QSlider(Qt.Horizontal)
        self.frame_lbl   = QLabel("Frame 0/0")
        self.conf_lbl    = QLabel("Conf ≥ 20%")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0,100)
        self.conf_slider.setValue(20)
        self.conf_slider.setSingleStep(1)
        self.progress    = QProgressBar()
        self.log_list    = QListWidget()
        self.img_label   = QLabel(alignment=Qt.AlignCenter)

        # mouse
        self.img_label.mousePressEvent   = self._press
        self.img_label.mouseMoveEvent    = self._move
        self.img_label.mouseReleaseEvent = self._release

        # shortcuts
        QShortcut(QKeySequence("Ctrl+A"), self, self.enable_add)
        QShortcut(QKeySequence("Del"),    self, self.enable_delete)
        QShortcut(QKeySequence("Ctrl+D"), self, self.enable_delete)
        QShortcut(QKeySequence("Ctrl+C"), self, self.copy_bbox)
        QShortcut(QKeySequence("Ctrl+V"), self, self.paste_bbox)
        QShortcut(QKeySequence("A"),      self, lambda: self._seek(-1))
        QShortcut(QKeySequence("D"),      self, lambda: self._seek(+1))
        #  ── NEW: propagate all bboxes from previous frame into current frame ──
        QShortcut(QKeySequence("R"), self, self.replicate_prev_frame)

        # layout
        ctrl = QHBoxLayout()
        for w in (self.open_btn,self.add_btn,self.del_btn,
                  self.export_btn,self.prev_btn,self.next_btn,
                  self.export_btn, self.save_frames_btn,
                  self.prev_btn, self.next_btn):
            ctrl.addWidget(w)
        conf_l = QHBoxLayout()
        conf_l.addWidget(self.conf_lbl)
        conf_l.addWidget(self.conf_slider)
        scrub  = QHBoxLayout()
        scrub.addWidget(self.slider,5)
        scrub.addWidget(self.frame_lbl,1)
        left   = QVBoxLayout()
        left.addWidget(QLabel("Logs:")); left.addWidget(self.log_list); left.addWidget(self.progress)
        right  = QVBoxLayout()
        right.addLayout(ctrl)
        right.addLayout(conf_l)
        right.addWidget(self.img_label,stretch=1)
        right.addLayout(scrub)
        root   = QHBoxLayout()
        root.addLayout(left,1); root.addLayout(right,4)
        cw = QWidget(); cw.setLayout(root)
        self.setCentralWidget(cw)

        # disable until first frame
        for w in (self.export_btn, self.save_frames_btn,     # ← NEW
                  self.prev_btn,self.next_btn,self.slider):
            w.setEnabled(False)

        # signals
        self.open_btn.clicked.connect(self.open_video)
        self.export_btn.clicked.connect(self.start_export)
        self.add_btn.clicked.connect(self.enable_add)
        self.del_btn.clicked.connect(self.enable_delete)
        self.save_frames_btn.clicked.connect(self.start_resave_blurred)
        self.prev_btn.clicked.connect(lambda: self._seek(-1))
        self.next_btn.clicked.connect(lambda: self._seek(+1))
        self.slider.sliderReleased.connect(lambda: self._show(self.slider.value()))
        self.slider.valueChanged.connect(lambda v: self.frame_lbl.setText(
            f"Frame {v+1}/{self.info.nF}" if self.info else ""
        ))
        self.conf_slider.valueChanged.connect(self._on_conf_changed)

    # ─ open / load existing annotations ─
    def open_video(self):
        path,_ = QFileDialog.getOpenFileName(self,"Select Video","","Videos (*.mp4 *.avi *.mkv *.mov)")
        self.setWindowTitle(f"Loading: {os.path.basename(path)}")
        if not path: return
        self.video_id = make_video_id(path)
        frames_dir = os.path.splitext(path)[0] + "_frames"
        thumb_dir  = frames_dir + "_thumbs"

        rec = load_annotations(self.video_id)
        if rec:
            # resume from JSON
            self.info = VideoInfo(
                path, rec["nF"], rec["fps"], rec["w"], rec["h"],
                frames_dir, thumb_dir
            )
            self.blur_ids = set(rec["blur_ids"])
            for idx, arr in rec["frames"].items():
                fd = FrameDetections()
                for e in arr:
                    fd.add(tuple(e["bbox"]), e["tid"], e["conf"])
                self.detections[int(idx)] = fd
            for w in (self.export_btn, self.add_btn, self.del_btn,
                  self.save_frames_btn,
                  self.prev_btn, self.next_btn, self.slider):
                w.setEnabled(True)
            self.slider.setRange(0, self.info.nF-1)
            self._show(0)
        else:
            # fresh track
            self.detections.clear()
            self.blur_ids.clear()
            self.info = None
            self.tracker = TrackerThread(path, frames_dir, thumb_dir, self.detections)
            self.tracker.stepped.connect(self._on_frame_ready)
            self.tracker.prog.connect(self.progress.setValue)
            self.tracker.done.connect(self._on_track_done)
            self.tracker.start()

    def start_resave_blurred(self):
        if not self.info:
            return

        # Wait for tracker to finish if still running
        if hasattr(self, 'tracker') and self.tracker.isRunning():
            self.tracker.wait()

        thr = self.conf_slider.value() / 100.0
        self.frame_saver = SaveFramesThread(
            self.info, self.detections, self.blur_ids, thr)
        self.frame_saver.prog.connect(self.progress.setValue)
        self.frame_saver.done.connect(self._on_frames_saved)
        self.progress.setValue(0)
        self.frame_saver.start()

    def _on_frames_saved(self, result):
        if result.startswith("ERROR::"):
            QMessageBox.critical(
                self, "Save Error",
                f"Couldn’t save blurred frames:\n{result[7:]}")
        else:
            QMessageBox.information(
                self, "Finished",
                f"All blurred frames written to:\n{result}")
        self.progress.setValue(0)


    def _on_frame_ready(self, idx):
        if idx == 0:
            for w in (self.export_btn,self.add_btn,self.del_btn, self.save_frames_btn,
                      self.prev_btn,self.next_btn,self.slider):
                w.setEnabled(True)
        if self.info is None:
            self.slider.setMaximum(idx)
        if idx == self.current_idx:
            self._show(idx)

    # ──────────────────────────────────────────────────────────────
    #  Propagate every bbox from “previous frame” (idx-1) to “current frame”.
    #  Triggered by pressing the R key.
    # ──────────────────────────────────────────────────────────────

    def replicate_prev_frame(self) -> None:
        """
        Copy every bbox from frame (current_idx-1) into the current
        frame. Each pasted box receives a NEW unique track-ID so it
        can be manipulated without affecting the original track.
        """
        try:
            # Guard: nothing loaded or we’re on the first frame
            if not self.info or self.current_idx == 0:
                return

            prev_idx = self.current_idx - 1
            src_fd   = self.detections.get(prev_idx)
            if not src_fd or not src_fd.boxes:
                return                          # nothing to copy

            self._push_undo()                   # enable Ctrl-Z

            # Ensure destination FrameDetections object exists
            dst_fd = self.detections.setdefault(
                self.current_idx, FrameDetections()
            )

            # Start new TIDs after current max (or at 1)
            next_tid = (max(self.blur_ids) if self.blur_ids else 0) + 1

            for bbox, _tid, _conf in src_fd:    # ignore orig tid/conf
                dst_fd.add(bbox, next_tid, 1.0) # high conf so it shows
                self.blur_ids.add(next_tid)     # blur by default
                next_tid += 1

            self._save_json()
            self._show(self.current_idx)
            self.statusBar().showMessage(
                "Copied bboxes from previous frame", 1500
            )

        except Exception as exc:
            # Failsafe: never crash the UI
            QMessageBox.critical(
                self, "Replicate Error",
                f"Couldn’t replicate bboxes:\n{exc}"
            )

    def _on_track_done(self, info):
        self.info = info
        # save initial JSON metadata
        rec = {
            "video_id":  self.video_id,
            "nF":        info.nF,
            "fps":       info.fps,
            "w":         info.w,
            "h":         info.h,
            "frames_dir":info.frames_dir,
            "thumb_dir": info.thumb_dir,
            "frames":    {},
            "blur_ids":  []
        }
        save_annotations(self.video_id, rec)
        self.slider.setRange(0, info.nF-1)
        self._show(0)
        video_name = os.path.basename(info.path)
        self.setWindowTitle(f"Editing: {video_name} – Face-Blur Editor")

    # ─ export ─
    def start_export(self):
        if hasattr(self,'tracker') and self.tracker.isRunning():
            self.tracker.wait()
        path,_ = QFileDialog.getSaveFileName(self,"Save Video","edited.mp4","MP4 Video (*.mp4)")
        if not path: return
        thr = self.conf_slider.value()/100.0
        self.exporter = ExportThread(self.info, self.detections, self.blur_ids, path, thr)
        self.exporter.prog.connect(self.progress.setValue)
        self.exporter.done.connect(lambda p: QMessageBox.information(self,"Done",f"Saved → {p}"))
        self.exporter.start()

    # ─ persistence helper ─
    def _save_json(self):
        rec = load_annotations(self.video_id)
        if not rec:
            return
        # serialize frames
        frames_ser = {}
        for idx, fd in self.detections.items():
            arr = []
            for bbox, tid, conf in zip(fd.boxes, fd.ids, fd.confs):
                arr.append({"bbox":bbox, "tid":tid, "conf":conf})
            frames_ser[str(idx)] = arr
        rec["frames"]   = frames_ser
        rec["blur_ids"] = list(self.blur_ids)
        save_annotations(self.video_id, rec)

    # ─ add / delete helpers ─
    def enable_add(self):
        if not self.info: return
        self.pending_add = True
        # self.setWindowTitle("Add mode: click-drag to draw")

    def enable_delete(self):
        if not self.info: return
        self.pending_delete = True
        # self.setWindowTitle("Delete mode: click a box to delete")

    # ─ copy/paste ─
    def copy_bbox(self):
        # nothing selected?
        if self.last_bbox_clicked is None:
            QMessageBox.information(self, "Copy", "Click a box first, then press Ctrl-C")
            return
        self.copied_bbox = (self.last_bbox_clicked, self.last_tid_clicked)
        self.statusBar().showMessage("BBox copied", 1000)


    def paste_bbox(self):
        if not self.copied_bbox:
            return                    # nothing copied yet
        bbox, orig_tid = self.copied_bbox

        # new, unique track-ID (keep existing if you want same id)
        new_tid = (max(self.blur_ids) if self.blur_ids else 0) + 1

        self._push_undo()
        fd = self.detections.setdefault(self.current_idx, FrameDetections())
        fd.add(bbox, new_tid, 1.0)
        self.blur_ids.add(new_tid)

        self._save_json()
        self._show(self.current_idx)


    # ─ undo/redo ─
    def _push_undo(self):
        self.undo_stack.append((copy.deepcopy(self.detections), copy.deepcopy(self.blur_ids)))
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack: return
        self.redo_stack.append((copy.deepcopy(self.detections), copy.deepcopy(self.blur_ids)))
        self.detections, self.blur_ids = self.undo_stack.pop()
        self._save_json()
        self._show(self.current_idx)

    def redo(self):
        if not self.redo_stack: return
        self.undo_stack.append((copy.deepcopy(self.detections), copy.deepcopy(self.blur_ids)))
        self.detections, self.blur_ids = self.redo_stack.pop()
        self._save_json()
        self._show(self.current_idx)

    def keyPressEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            if e.key() == Qt.Key_Z:
                self.undo(); return
            if e.key() == Qt.Key_Y:
                self.redo(); return
        super().keyPressEvent(e)

    # ─ conf slider ─
    def _on_conf_changed(self, v):
        self.conf_lbl.setText(f"Conf ≥ {v}%")
        self._show(self.current_idx)

    # ─ display ─
    def _show(self, idx):
        if not self.info: return
        idx = max(0, min(idx, self.info.nF-1))
        self.current_idx = idx
        path = os.path.join(self.info.frames_dir, f"frame_{idx:06d}.jpg")
        frame = cv2.imread(path)
        if frame is None:
            return
        canvas = frame.copy()
        thr = self.conf_slider.value()/100.0

        for bbox, tid, conf in self.detections.get(idx, FrameDetections()):
            if conf < thr:
                continue
            x1,y1,x2,y2 = bbox
            label = f"{conf:.2f}"
            cv2.putText(canvas, label, (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            if tid in self.blur_ids:
                canvas[y1:y2, x1:x2] = cv2.GaussianBlur(
                    canvas[y1:y2, x1:x2], BLUR_KERNEL, 0)
            cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,0), 2)

        h,w = canvas.shape[:2]
        img = QImage(canvas.data, w, h, 3*w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(img)
        scaled = pix.scaled(self.img_label.size(),
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(scaled)
        dw,dh = scaled.width(), scaled.height()
        self._scale_x, self._scale_y = dw/w, dh/h
        lw,lh = self.img_label.width(), self.img_label.height()
        self._dx, self._dy = (lw-dw)/2, (lh-dh)/2
        self.frame_lbl.setText(f"Frame {idx+1}/{self.info.nF}")

    def _seek(self, step):
        self._show(self.current_idx + step)

    def _to_vid(self, ex, ey):
        return int((ex - self._dx)/self._scale_x), int((ey - self._dy)/self._scale_y)

    # ─ mouse interactions ─
    def _press(self, ev):
        if not self.info:
            return

        x, y = self._to_vid(ev.x(), ev.y())
        shift = bool(ev.modifiers() & Qt.ShiftModifier)

        # ──────────────────────────────────────────────
        # Store the last clicked bbox (for copy support)
        # ──────────────────────────────────────────────
        clicked_tid = None
        clicked_bbox = None
        for bbox, tid, conf in self.detections.get(self.current_idx, FrameDetections()):
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_tid = tid
                clicked_bbox = bbox
                break

        self.last_tid_clicked = clicked_tid
        self.last_bbox_clicked = clicked_bbox

        if self.pending_add:
            self._push_undo()
            self.mode = "add"
            self.start_pt = (x, y)
            self.pending_add = False
            return

        if self.pending_delete:
            self._push_undo()
            for bbox, tid, conf in self.detections.get(self.current_idx, FrameDetections()):
                x1, y1, x2, y2 = bbox
                if x1 <= x <= x2 and y1 <= y <= y2:
                    for f, fd in list(self.detections.items()):
                        kept = FrameDetections()
                        for b, t, c in zip(fd.boxes, fd.ids, fd.confs):
                            if t != tid:
                                kept.add(b, t, c)
                        self.detections[f] = kept
                    self.blur_ids.discard(tid)
                    break
            self.pending_delete = False
            self._save_json()
            video_name = os.path.basename(self.info.path) if self.info else "Unknown"
            self.setWindowTitle(f"Editing: {video_name} – Face-Blur Editor")
            self._show(self.current_idx)
            return

        # ───── corner resize ─────
        for bbox, tid, conf in self.detections.get(self.current_idx, FrameDetections()):
            x1, y1, x2, y2 = bbox
            for i, (cx, cy) in enumerate([(x1, y1), (x2, y1), (x1, y2), (x2, y2)]):
                if abs(cx - x) <= CORNER_THRESH and abs(cy - y) <= CORNER_THRESH:
                    self._push_undo()
                    self.mode = "resize"
                    self.corner_idx = i
                    self.act_tid = tid
                    self.start_pt = (x, y)
                    self.orig_boxes = {
                        f: b                                     # ← gathers ALL frames
                        for f, fd in self.detections.items()
                        for b, t, _ in zip(fd.boxes, fd.ids, fd.confs)
                        if t == tid
                    }
                    return

        # ───── drag move ─────
        for bbox, tid, conf in self.detections.get(self.current_idx, FrameDetections()):
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._push_undo()
                self.mode = "move"
                self.act_tid = tid
                self.start_pt = (x, y)
                self.orig_boxes = {
                    f: b                                     # ← gathers ALL frames
                    for f, fd in self.detections.items()
                    for b, t, _ in zip(fd.boxes, fd.ids, fd.confs)
                    if t == tid
                }
                return

        # ───── shift-click add shortcut ─────
        if shift:
            self._push_undo()
            self.mode = "add"
            self.start_pt = (x, y)
        else:
            self.mode = None
            self.act_tid = None


    def _move(self, ev):
        if self.mode in (None,"add"): return
        x,y = self._to_vid(ev.x(), ev.y())

        if self.mode=="move":
            dx,dy = x-self.start_pt[0], y-self.start_pt[1]
            for f,orig in self.orig_boxes.items():
                x1,y1,x2,y2=orig
                new=(max(0,x1+dx),max(0,y1+dy),
                     min(self.info.w,x2+dx),min(self.info.h,y2+dy))
                fd=self.detections[f]
                for i,(b,t,c) in enumerate(zip(fd.boxes,fd.ids,fd.confs)):
                    if t==self.act_tid: fd.boxes[i]=new
            self._show(self.current_idx)

        elif self.mode=="resize":
            sx,sy=self.start_pt; dx,dy=x-sx,y-sy
            for f,orig in self.orig_boxes.items():
                x1,y1,x2,y2=orig
                if self.corner_idx==0: new=(x1+dx,y1+dy,x2,y2)
                elif self.corner_idx==1: new=(x1,y1+dy,x2+dx,y2)
                elif self.corner_idx==2: new=(x1+dx,y1,x2,y2+dy)
                else: new=(x1,y1,x2+dx,y2+dy)
                nx1,ny1,nx2,ny2=new
                new=(max(0,min(nx1,nx2)),max(0,min(ny1,ny2)),
                     min(self.info.w,max(nx1,nx2)),min(self.info.h,max(ny1,ny2)))
                fd=self.detections[f]
                for i,(b,t,c) in enumerate(zip(fd.boxes,fd.ids,fd.confs)):
                    if t==self.act_tid: fd.boxes[i]=new
            self._show(self.current_idx)

    def _release(self, ev):
        if self.mode=="add" and self.start_pt:
            x0,y0=self.start_pt; x1,y1=self._to_vid(ev.x(),ev.y())
            if abs(x1-x0)>CLICK_THRESH and abs(y1-y0)>CLICK_THRESH:
                bbox=(min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1))
                new_tid=(max(self.blur_ids) if self.blur_ids else 0)+1
                fd=self.detections.setdefault(self.current_idx,FrameDetections())
                fd.add(bbox,new_tid,1.0)
                self.blur_ids.add(new_tid)

        elif self.mode in ("move","resize"):
            dx=ev.x()-(self.start_pt[0]*self._scale_x+self._dx)
            dy=ev.y()-(self.start_pt[1]*self._scale_y+self._dy)
            if (dx*dx+dy*dy)**0.5<CLICK_THRESH and self.act_tid:
                self._push_undo()
                if self.act_tid in self.blur_ids:
                    self.blur_ids.remove(self.act_tid)
                else:
                    self.blur_ids.add(self.act_tid)

        # save after every edit
        self._save_json()

        # reset
        self.mode=self.act_tid=self.corner_idx=self.start_pt=None
        self.pending_add=self.pending_delete=False
        self._show(self.current_idx)

    def closeEvent(self, ev):
        if hasattr(self,'tracker') and self.tracker.isRunning():
            self.tracker.quit(); self.tracker.wait()
        if hasattr(self,'exporter') and self.exporter.isRunning():
            self.exporter.quit(); self.exporter.wait()
        super().closeEvent(ev)

if __name__=="__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())
