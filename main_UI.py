import os
import sys
import shutil
import time
import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QListWidgetItem, QDialog, QTableWidgetItem, QVBoxLayout,
    QStackedWidget, QTabWidget, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTextEdit, QGroupBox, QSlider, QListWidget,
    QProgressBar, QSplitter, QTreeView, QFileSystemModel, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QDir, QFile, QIODevice
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QDragEnterEvent, QDropEvent, QColor, QAction
from PySide6.QtUiTools import QUiLoader

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from logic import (
    BASE_DIR, load_settings, save_settings_to_file, DEFAULT_SETTINGS,
    get_hardware_info, resolve_device, load_ai_models,
    process_audio_pipeline, calculate_audio_metering, compute_audio_fingerprint,
    cache_mgr, prepare_tensor_for_vad, get_padded_speech_timestamps,
    refine_transcribed_slate_text, parse_slate_info, format_destination_filename,
    run_stt_no_hallucination, vad_model, stt_model
)

UI_FILE_PATH = os.path.join(BASE_DIR, "main.ui")
if not os.path.exists(UI_FILE_PATH):
    UI_FILE_PATH = os.path.join(BASE_DIR, "Main.ui")


class InteractiveWaveformCanvas(FigureCanvasQTAgg):
    seek_requested = Signal(float)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 2.2), dpi=100, facecolor="#1e1e2e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#12121a")
        self.fig.tight_layout()

        super().__init__(self.fig)
        self.setParent(parent)

        self.total_duration = 0.0
        self.playhead_line = None
        self.view_start = 0.0
        self.view_end = 1.0

    def plot_waveform_with_vad(self, audio_data, sr, speech_spans, crop_pct):
        self.ax.clear()
        self.ax.set_facecolor("#12121a")

        if len(audio_data) == 0:
            self.draw()
            return

        self.total_duration = len(audio_data) / float(sr)
        self.view_start = 0.0
        self.view_end = self.total_duration

        step = max(1, len(audio_data) // 2000)
        sub_data = audio_data[::step]
        time_axis = np.linspace(0, self.total_duration, len(sub_data))

        self.ax.plot(time_axis, sub_data, color="#78909c", linewidth=0.8)

        crop_ratio = crop_pct / 100.0
        crop_start = self.total_duration * crop_ratio
        crop_end = self.total_duration * (1.0 - crop_ratio)
        self.ax.axvspan(0, crop_start, color='#424242', alpha=0.5)
        self.ax.axvspan(crop_end, self.total_duration, color='#424242', alpha=0.5)

        for span in speech_spans:
            self.ax.axvspan(span['start'], span['end'], color='#81c784', alpha=0.55)

        self.playhead_line = self.ax.axvline(x=0.0, color='#ff5252', linewidth=2.0)

        self.ax.set_xlim(self.view_start, self.view_end)
        self.ax.set_yticks([])
        self.ax.tick_params(colors='#aaa', labelsize=8)
        self.fig.tight_layout()
        self.draw()

    def update_playhead_pos(self, current_sec):
        if self.playhead_line and self.total_duration > 0:
            self.playhead_line.set_xdata([current_sec, current_sec])
            self.draw_idle()

    def wheelEvent(self, event):
        if self.total_duration <= 0: return
        delta = event.angleDelta().y()
        scale_factor = 0.85 if delta > 0 else 1.25

        cur_center = (self.view_start + self.view_end) / 2.0
        cur_range = (self.view_end - self.view_start) * scale_factor
        cur_range = max(1.0, min(cur_range, self.total_duration))

        self.view_start = max(0.0, cur_center - cur_range / 2.0)
        self.view_end = min(self.total_duration, self.view_start + cur_range)

        self.ax.set_xlim(self.view_start, self.view_end)
        self.draw_idle()

    def mousePressEvent(self, event):
        if self.total_duration > 0 and event.button() == Qt.LeftButton:
            widget_width = self.width()
            pos_x = event.position().x()
            rel_ratio = pos_x / float(widget_width)
            clicked_sec = self.view_start + rel_ratio * (self.view_end - self.view_start)
            clicked_sec = max(0.0, min(clicked_sec, self.total_duration))
            self.seek_requested.emit(clicked_sec)

        super().mousePressEvent(event)


class LogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detailed Processing Logs")
        self.resize(750, 500)

        layout = QVBoxLayout(self)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 9))
        layout.addWidget(self.txt_log)

        lay_btn = QHBoxLayout()
        btn_copy = QPushButton("Copy All Logs")
        btn_copy.clicked.connect(self.copy_logs)
        btn_close = QPushButton("Close Window")
        btn_close.clicked.connect(self.close)

        lay_btn.addWidget(btn_copy)
        lay_btn.addStretch()
        lay_btn.addWidget(btn_close)
        layout.addLayout(lay_btn)

    def append_log(self, text):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.txt_log.append(timestamp + text)

    def clear_log(self):
        self.txt_log.clear()

    def copy_logs(self):
        QApplication.clipboard().setText(self.txt_log.toPlainText())
        QMessageBox.information(self, "Copied", "Logs copied to clipboard.")


class AutoClassifierThread(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, str)
    stats_signal = Signal(int, int, int, int, int, int)
    finished_signal = Signal()

    def __init__(self, src_dir, dst_dir, min_dur, crop_pct, speech_thresh, vad_choice, stt_choice, dev_choice, settings):
        super().__init__()
        self.src_dir = os.path.abspath(src_dir)
        self.dst_dir = os.path.abspath(dst_dir)
        self.min_dur = min_dur
        self.crop_pct = crop_pct / 100.0
        self.speech_thresh = speech_thresh / 100.0
        self.vad_choice = vad_choice
        self.stt_choice = stt_choice
        self.dev_choice = dev_choice
        self.settings = settings

    def run(self):
        self.log_signal.emit("[START] Auto classification started...")

        if not os.path.exists(self.src_dir):
            self.log_signal.emit("[ERROR] Source directory does not exist.")
            self.finished_signal.emit()
            return

        load_ai_models(
            model_size=self.stt_choice,
            vad_choice=self.vad_choice,
            device_choice=self.dev_choice,
            log_func=lambda m: self.log_signal.emit(m)
        )

        supported_exts = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aiff', '.aac')
        all_files = []
        for root_path, _, filenames in os.walk(self.src_dir):
            for fname in filenames:
                if fname.lower().endswith(supported_exts):
                    rel_path = os.path.relpath(os.path.join(root_path, fname), self.src_dir)
                    all_files.append((fname, os.path.abspath(os.path.join(root_path, fname)), rel_path))

        if not all_files:
            self.log_signal.emit("[WARNING] No audio files found in source directory.")
            self.finished_signal.emit()
            return

        total_count = len(all_files)
        cnt_processed, cnt_roomtone, cnt_short, cnt_match, cnt_review = 0, 0, 0, 0, 0

        self.log_signal.emit(f"\n[SCAN] Total {total_count} files found...\n" + "="*55)
        target_dev = resolve_device(self.dev_choice)

        is_copy_mode = "Copy" in self.settings.get("file_action_mode", "Copy")
        prefix = self.settings.get("filename_prefix", "")
        postfix = self.settings.get("filename_postfix", "")
        strategy = self.settings.get("naming_strategy", "Parsed Scene Name")
        gain_db = float(self.settings.get("gain_boost_db", 0.0))

        for idx, (file_name, full_src_path, rel_path) in enumerate(all_files):
            pct = int(((idx + 1) / float(total_count)) * 100)
            self.progress_signal.emit(pct, f"Processing [{idx+1}/{total_count}]: {file_name}")

            try:
                cache_data = cache_mgr.load_cache(file_name)

                if cache_data is not None:
                    total_duration_sec = cache_data['duration_sec']
                    speech_timestamps_json = cache_data['speech_timestamps']
                    transcribed_text = cache_data.get('transcribed_text', '')
                    self.log_signal.emit(f"[CACHE] {file_name}")
                else:
                    audio_data, sr = process_audio_pipeline(full_src_path, gain_db=gain_db)
                    total_duration_sec = len(audio_data) / float(sr)

                    wav_tensor = prepare_tensor_for_vad(audio_data, sr, device=target_dev)
                    speech_timestamps_json = get_padded_speech_timestamps(wav_tensor, vad_model, sr=16000, pad_sec=1.0)

                    if speech_timestamps_json:
                        first_start = speech_timestamps_json[0]['start']
                        crop_start = int(first_start * sr)
                        crop_end = min(len(audio_data), int((first_start + 8.0) * sr))
                        audio_clip = audio_data[crop_start:crop_end]
                    else:
                        audio_clip = audio_data[:int(min(15.0, total_duration_sec) * sr)]

                    transcribe_kwargs = {
                        "language": "ko",
                        "condition_on_previous_text": False,
                        "no_speech_threshold": 0.5,
                        "beam_size": 5
                    }

                    if self.settings.get("stt_prompt_enabled", True):
                        prompt_text = self.settings.get("stt_initial_prompt", "").strip()
                        if prompt_text:
                            transcribe_kwargs["initial_prompt"] = prompt_text

                    try:
                        segments, _ = stt_model.transcribe(
                            audio_clip, hotwords="씬 컷 테이크 scene cut take", **transcribe_kwargs
                        )
                    except TypeError:
                        segments, _ = stt_model.transcribe(audio_clip, **transcribe_kwargs)

                    raw_text = "".join([s.text for s in segments]).strip()
                    transcribed_text = refine_transcribed_slate_text(raw_text)

                    cache_data = {
                        'filename': file_name,
                        'duration_sec': round(total_duration_sec, 2),
                        'speech_timestamps': speech_timestamps_json,
                        'transcribed_text': transcribed_text
                    }
                    cache_mgr.save_cache(file_name, cache_data)

                # Classification Step
                if total_duration_sec < self.min_dur:
                    subfolder_name = "Short_Duration"
                    parsed_scene = None
                    cnt_short += 1
                    self.log_signal.emit(f"[SHORT] {file_name} ({total_duration_sec:.1f}s)")
                else:
                    crop_start_sec = total_duration_sec * self.crop_pct
                    crop_end_sec = total_duration_sec * (1.0 - self.crop_pct)
                    middle_duration_sec = max(0.1, crop_end_sec - crop_start_sec)

                    middle_speech_sec = 0.0
                    for seg in speech_timestamps_json:
                        overlap_start = max(seg['start'], crop_start_sec)
                        overlap_end = min(seg['end'], crop_end_sec)
                        if overlap_end > overlap_start:
                            middle_speech_sec += (overlap_end - overlap_start)

                    middle_speech_ratio = middle_speech_sec / middle_duration_sec

                    if middle_speech_ratio <= self.speech_thresh:
                        subfolder_name = "RoomTone"
                        parsed_scene = None
                        cnt_roomtone += 1
                        self.log_signal.emit(f"[ROOMTONE] {file_name} (Mid Speech: {middle_speech_ratio*100:.2f}%)")
                    else:
                        subfolder_name = "Unclassified_Manual_Review"
                        parsed_scene = parse_slate_info(transcribed_text)
                        if parsed_scene:
                            subfolder_name = parsed_scene
                            cnt_match += 1
                            self.log_signal.emit(f"[MATCH] [{subfolder_name}] {file_name} (Text: \"{transcribed_text}\")")
                        else:
                            cnt_review += 1
                            self.log_signal.emit(f"[REVIEW] {file_name} (Mid Speech: {middle_speech_ratio*100:.1f}%, Text: \"{transcribed_text}\")")

                cnt_processed += 1
                target_folder = os.path.abspath(os.path.join(self.dst_dir, subfolder_name))
                os.makedirs(target_folder, exist_ok=True)

                out_filename = format_destination_filename(file_name, parsed_scene, prefix, postfix, strategy, idx+1)
                dst_path = os.path.join(target_folder, out_filename)

                if is_copy_mode:
                    shutil.copy2(full_src_path, dst_path)
                else:
                    shutil.move(full_src_path, dst_path)

            except Exception as e:
                self.log_signal.emit(f"[ERROR] {file_name}: {e}")

            self.stats_signal.emit(total_count, cnt_processed, cnt_roomtone, cnt_short, cnt_match, cnt_review)

        self.log_signal.emit("="*55 + "\n[FINISHED] Auto classification completed!")
        self.finished_signal.emit()


class AudioClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()

        if not os.path.exists(UI_FILE_PATH) or os.path.getsize(UI_FILE_PATH) == 0:
            QMessageBox.critical(None, "UI File Error", f"main.ui file is missing or empty at:\n{UI_FILE_PATH}\n\nPlease ensure main.ui contains valid XML.")
            sys.exit(1)

        ui_file = QFile(UI_FILE_PATH)
        if not ui_file.open(QIODevice.ReadOnly):
            QMessageBox.critical(None, "UI Read Error", f"Cannot open main.ui file.")
            sys.exit(1)

        loader = QUiLoader()
        self.ui = loader.load(ui_file, self)
        ui_file.close()

        if self.ui is None:
            QMessageBox.critical(None, "XML Parse Error", "Failed to parse main.ui file.")
            sys.exit(1)

        self.setCentralWidget(self.ui)
        self.setWindowTitle("Smart Audio Classifier V0.1.3")
        self.resize(self.ui.size())
        self.setAcceptDrops(True)

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)

        init_db = float(self.settings.get("default_volume_db", 0.0))
        self.apply_volume_db(init_db)

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(16)
        self.play_timer.timeout.connect(self.sync_playhead)

        self.current_audio_file = ""
        self.total_duration_sec = 0.0
        self.last_moved_action = None

        self.log_dialog = LogDialog(self)

        self.bind_menu_actions()
        self.bind_ui_elements()
        self.setup_canvas_placeholders()
        self.setup_keyboard_shortcuts()
        self.apply_dark_theme()

    def bind_menu_actions(self):
        main_stack = self.ui.findChild(QStackedWidget, "main_stack") or self.ui.findChild(QTabWidget, "main_tabs")

        def switch_page(idx):
            if main_stack:
                if isinstance(main_stack, QStackedWidget): main_stack.setCurrentIndex(idx)
                elif isinstance(main_stack, QTabWidget): main_stack.setCurrentIndex(idx)

        menu_mappings = {
            "actionAudio_sort": 0,
            "actionManual_Sort": 1,
            "actionAudio_Enhancement": 2,
            "actionSetting": 3
        }

        for act_name, page_idx in menu_mappings.items():
            act = self.ui.findChild(QAction, act_name)
            if act:
                act.triggered.connect(lambda idx=page_idx: switch_page(idx))

    def bind_ui_elements(self):
        def get_w(name, w_type):
            return self.ui.findChild(w_type, name)

        # Auto Tab
        self.txt_src = get_w("txt_src", QLineEdit)
        self.txt_dst = get_w("txt_dst", QLineEdit)
        self.btn_src = get_w("btn_src", QPushButton)
        self.btn_dst = get_w("btn_dst", QPushButton)

        if self.btn_src: self.btn_src.clicked.connect(lambda: self.browse_folder(self.txt_src))
        if self.btn_dst: self.btn_dst.clicked.connect(lambda: self.browse_folder(self.txt_dst))

        if self.txt_src: self.txt_src.setText(os.path.abspath(self.settings["src_dir"]))
        if self.txt_dst: self.txt_dst.setText(os.path.abspath(self.settings["dst_dir"]))

        self.spn_dur = get_w("spn_dur", QSpinBox)
        self.spn_crop = get_w("spn_crop", QSpinBox)
        self.spn_speech = get_w("spn_speech", QDoubleSpinBox)
        self.spn_gain = get_w("spn_gain", QDoubleSpinBox)
        self.cmb_stt = get_w("cmb_stt", QComboBox)
        self.cmb_dev = get_w("cmb_dev", QComboBox)

        if self.spn_dur: self.spn_dur.setValue(int(self.settings["min_dur_thresh"]))
        if self.spn_crop: self.spn_crop.setValue(int(self.settings["crop_pct"]))
        if self.spn_speech: self.spn_speech.setValue(self.settings["speech_thresh_pct"])
        if self.spn_gain: self.spn_gain.setValue(float(self.settings.get("gain_boost_db", 0.0)))

        if self.cmb_stt and self.cmb_stt.count() == 0:
            self.cmb_stt.addItems(["large-v3-turbo (Recommended High Accuracy)", "base (Fast)", "small (Balanced)", "medium (High Accuracy)"])
        if self.cmb_dev and self.cmb_dev.count() == 0:
            self.cmb_dev.addItems(["Auto (GPU Priority)", "GPU (CUDA)", "CPU"])

        self.btn_run_auto = get_w("btn_run_auto", QPushButton)
        if self.btn_run_auto: self.btn_run_auto.clicked.connect(self.start_auto_classification)

        self.lbl_dash_status = get_w("lbl_dash_status", QLabel)
        self.progress_bar = get_w("progress_bar", QProgressBar)
        self.lbl_dash_file = get_w("lbl_dash_file", QLabel)

        self.lbl_stat_total = get_w("lbl_stat_total", QLabel)
        self.lbl_stat_processed = get_w("lbl_stat_processed", QLabel)
        self.lbl_stat_roomtone = get_w("lbl_stat_roomtone", QLabel)
        self.lbl_stat_short = get_w("lbl_stat_short", QLabel)
        self.lbl_stat_match = get_w("lbl_stat_match", QLabel)
        self.lbl_stat_review = get_w("lbl_stat_review", QLabel)

        # Manual Tab
        self.btn_play = get_w("btn_play", QPushButton)
        self.btn_stop = get_w("btn_stop", QPushButton)

        if self.btn_play: self.btn_play.clicked.connect(self.toggle_play)
        if self.btn_stop: self.btn_stop.clicked.connect(self.stop_audio)

        self.lbl_vol_db = get_w("lbl_vol_db", QLabel)
        self.slider_vol_db = get_w("slider_vol_db", QSlider)
        if self.slider_vol_db:
            self.slider_vol_db.setRange(-30, 18)
            self.slider_vol_db.setValue(int(self.settings.get("default_volume_db", 0.0)))
            self.slider_vol_db.valueChanged.connect(self.on_volume_db_changed)

        self.chk_autoplay = get_w("chk_autoplay", QCheckBox)
        if self.chk_autoplay: self.chk_autoplay.setChecked(self.settings.get("auto_play", False))

        self.lbl_timer = get_w("lbl_timer", QLabel)
        self.lbl_lufs = get_w("lbl_lufs", QLabel)
        self.lbl_tp = get_w("lbl_tp", QLabel)
        self.lbl_peak = get_w("lbl_peak", QLabel)
        self.lbl_rms = get_w("lbl_rms", QLabel)

        self.lbl_queue_count = get_w("lbl_queue_count", QLabel)
        self.list_files = get_w("list_files", QListWidget)
        if self.list_files: self.list_files.itemSelectionChanged.connect(self.on_file_selected)

        self.btn_refresh = get_w("btn_refresh", QPushButton)
        if self.btn_refresh: self.btn_refresh.clicked.connect(self.refresh_manual_file_list)

        self.lbl_file_info = get_w("lbl_file_info", QLabel)
        self.txt_stt = get_w("txt_stt", QLineEdit)
        self.txt_sc = get_w("txt_sc", QLineEdit)
        self.txt_ct = get_w("txt_ct", QLineEdit)

        self.btn_move_sc = get_w("btn_move_sc", QPushButton)
        if self.btn_move_sc: self.btn_move_sc.clicked.connect(self.move_to_scene)

        self.btn_roomtone = get_w("btn_roomtone", QPushButton)
        if self.btn_roomtone: self.btn_roomtone.clicked.connect(lambda: self.move_to_folder("RoomTone"))

        self.btn_short = get_w("btn_short", QPushButton)
        if self.btn_short: self.btn_short.clicked.connect(lambda: self.move_to_folder("Short_Duration"))

        self.btn_skip = get_w("btn_skip", QPushButton)
        if self.btn_skip: self.btn_skip.clicked.connect(self.skip_file)

        self.btn_undo = get_w("btn_undo", QPushButton)
        if self.btn_undo: self.btn_undo.clicked.connect(self.undo_last_action)

        # Dynamic QTreeView Binding (Source / Destination / Enhancement)
        self.tree_view_src = get_w("tree_view_src", QTreeView)
        self.tree_view_dst = get_w("tree_view", QTreeView) or get_w("tree_view_dst", QTreeView)
        self.tree_view_enh = get_w("tree_view_enh", QTreeView)

        if self.tree_view_src:
            self.tree_model_src = QFileSystemModel()
            self.tree_model_src.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files)
            src_path = os.path.abspath(self.settings["src_dir"])
            os.makedirs(src_path, exist_ok=True)
            self.tree_model_src.setRootPath(src_path)
            self.tree_view_src.setModel(self.tree_model_src)
            self.tree_view_src.setRootIndex(self.tree_model_src.index(src_path))
            self.tree_view_src.setHeaderHidden(True)
            self.tree_view_src.setColumnHidden(1, True)
            self.tree_view_src.setColumnHidden(2, True)
            self.tree_view_src.setColumnHidden(3, True)
            self.tree_view_src.doubleClicked.connect(self.on_tree_file_double_clicked)

        if self.tree_view_dst:
            self.tree_model = QFileSystemModel()
            self.tree_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files)
            dst_path = os.path.abspath(self.settings["dst_dir"])
            os.makedirs(dst_path, exist_ok=True)
            self.tree_model.setRootPath(dst_path)
            self.tree_view_dst.setModel(self.tree_model)
            self.tree_view_dst.setRootIndex(self.tree_model.index(dst_path))
            self.tree_view_dst.setHeaderHidden(True)
            self.tree_view_dst.setColumnHidden(1, True)
            self.tree_view_dst.setColumnHidden(2, True)
            self.tree_view_dst.setColumnHidden(3, True)
            self.tree_view_dst.doubleClicked.connect(self.on_tree_file_double_clicked)

        if self.tree_view_enh:
            if not hasattr(self, 'tree_model'):
                self.tree_model = QFileSystemModel()
                self.tree_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files)
                dst_path = os.path.abspath(self.settings["dst_dir"])
                os.makedirs(dst_path, exist_ok=True)
                self.tree_model.setRootPath(dst_path)
            self.tree_view_enh.setModel(self.tree_model)
            self.tree_view_enh.setRootIndex(self.tree_model.index(os.path.abspath(self.settings["dst_dir"])))
            self.tree_view_enh.setHeaderHidden(True)
            self.tree_view_enh.setColumnHidden(1, True)
            self.tree_view_enh.setColumnHidden(2, True)
            self.tree_view_enh.setColumnHidden(3, True)
            self.tree_view_enh.doubleClicked.connect(self.on_tree_file_double_clicked)

        # Settings Tab
        self.chk_prompt_enable = get_w("chk_prompt_enable", QCheckBox)
        if self.chk_prompt_enable: self.chk_prompt_enable.setChecked(self.settings.get("stt_prompt_enabled", True))

        self.txt_prompt = get_w("txt_prompt", QTextEdit)
        if self.txt_prompt: self.txt_prompt.setPlainText(self.settings.get("stt_initial_prompt", ""))

        self.btn_save = get_w("btn_save", QPushButton)
        if self.btn_save: self.btn_save.clicked.connect(self.save_settings)

        self.btn_show_logs = get_w("btn_show_logs", QPushButton)
        if self.btn_show_logs: self.btn_show_logs.clicked.connect(self.log_dialog.show)

    def on_tree_file_double_clicked(self, index):
        """Play and preview audio when double clicked in TreeView"""
        sender_tree = self.sender()
        if not sender_tree: return
        model = sender_tree.model()
        file_path = model.filePath(index)

        if os.path.isfile(file_path):
            supported_exts = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aiff', '.aac')
            if file_path.lower().endswith(supported_exts):
                self.stop_audio()
                self.current_audio_file = os.path.abspath(file_path)

                filename = os.path.basename(file_path)
                cache_data = cache_mgr.load_cache(filename)
                speech_spans = cache_data.get('speech_timestamps', []) if cache_data else []
                transcribed_text = cache_data.get('transcribed_text', '') if cache_data else ''

                if self.txt_stt: self.txt_stt.setText(transcribed_text)

                gain_v = self.spn_gain.value() if self.spn_gain else 0.0
                crop_v = self.spn_crop.value() if self.spn_crop else 15

                audio_data, sr = process_audio_pipeline(file_path, gain_db=gain_v)
                self.total_duration_sec = len(audio_data) / float(sr)

                lufs, tp, pk, rms = calculate_audio_metering(audio_data, sr)
                if self.lbl_lufs: self.lbl_lufs.setText(f"LUFS: {lufs:.1f} dB")
                if self.lbl_tp: self.lbl_tp.setText(f"True Peak: {tp:.1f} dBTP")
                if self.lbl_peak: self.lbl_peak.setText(f"Peak: {pk:.1f} dB")
                if self.lbl_rms: self.lbl_rms.setText(f"RMS: {rms:.1f} dB")

                if hasattr(self, 'canvas') and self.canvas:
                    self.canvas.plot_waveform_with_vad(audio_data, sr, speech_spans, crop_v)

                if self.lbl_file_info: self.lbl_file_info.setText(f"File: {filename} | Duration: {self.total_duration_sec:.1f}s")

                self.media_player.setSource(QUrl.fromLocalFile(self.current_audio_file))
                self.update_timer_label(0.0)
                self.toggle_play()

    def setup_canvas_placeholders(self):
        self.canvas = InteractiveWaveformCanvas(self)
        self.canvas.seek_requested.connect(self.seek_audio)

        container = self.ui.findChild(QWidget, "waveform_container")
        if container:
            if container.layout() is None:
                lay = QVBoxLayout(container)
                lay.setContentsMargins(0, 0, 0, 0)
                container.setLayout(lay)
            container.layout().addWidget(self.canvas)

        self.canvas_enh = InteractiveWaveformCanvas(self)
        container_enh = self.ui.findChild(QWidget, "waveform_container_enh")
        if container_enh:
            if container_enh.layout() is None:
                lay_enh = QVBoxLayout(container_enh)
                lay_enh.setContentsMargins(0, 0, 0, 0)
                container_enh.setLayout(lay_enh)
            container_enh.layout().addWidget(self.canvas_enh)

    def apply_volume_db(self, db_val):
        linear_gain = 10 ** (db_val / 20.0)
        self.audio_output.setVolume(min(1.0, max(0.0, linear_gain)))

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls and self.txt_src:
            folder_path = os.path.abspath(urls[0].toLocalFile())
            if os.path.isdir(folder_path):
                self.txt_src.setText(folder_path)
                QMessageBox.information(self, "Folder Dropped", f"Source directory set to:\n{folder_path}")

    def setup_keyboard_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_play)
        QShortcut(QKeySequence(Qt.Key_R), self, lambda: self.move_to_folder("RoomTone"))
        QShortcut(QKeySequence(Qt.Key_S), self, lambda: self.move_to_folder("Short_Duration"))
        QShortcut(QKeySequence(Qt.Key_Right), self, self.skip_file)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_last_action)
        QShortcut(QKeySequence(Qt.Key_Return), self, self.move_to_scene)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self.move_to_scene)

    def browse_folder(self, target_widget):
        if not target_widget: return
        dir_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if dir_path:
            target_widget.setText(os.path.abspath(dir_path))
            if target_widget == self.txt_src and hasattr(self, 'tree_model_src') and self.tree_model_src:
                self.tree_model_src.setRootPath(os.path.abspath(dir_path))
                if self.tree_view_src:
                    self.tree_view_src.setRootIndex(self.tree_model_src.index(os.path.abspath(dir_path)))
            elif target_widget == self.txt_dst and hasattr(self, 'tree_model') and self.tree_model:
                self.tree_model.setRootPath(os.path.abspath(dir_path))
                if self.tree_view_dst:
                    self.tree_view_dst.setRootIndex(self.tree_model.index(os.path.abspath(dir_path)))

    def start_auto_classification(self):
        if self.btn_run_auto: self.btn_run_auto.setEnabled(False)
        self.log_dialog.clear_log()
        if self.lbl_dash_status: self.lbl_dash_status.setText("System Status: Processing Batch...")
        if self.progress_bar: self.progress_bar.setValue(0)

        src = self.txt_src.text() if self.txt_src else self.settings["src_dir"]
        dst = self.txt_dst.text() if self.txt_dst else self.settings["dst_dir"]
        dur = self.spn_dur.value() if self.spn_dur else 5
        crop = self.spn_crop.value() if self.spn_crop else 15
        speech = self.spn_speech.value() if self.spn_speech else 2.0
        stt_m = self.cmb_stt.currentText() if self.cmb_stt else "large-v3-turbo"
        dev_m = self.cmb_dev.currentText() if self.cmb_dev else "Auto"
        gain_v = self.spn_gain.value() if self.spn_gain else 0.0

        self.settings["gain_boost_db"] = gain_v

        self.thread = AutoClassifierThread(
            src_dir=src, dst_dir=dst, min_dur=dur, crop_pct=crop,
            speech_thresh=speech, vad_choice="Silero VAD v5", stt_choice=stt_m,
            dev_choice=dev_m, settings=self.settings
        )
        self.thread.log_signal.connect(self.log_dialog.append_log)
        self.thread.progress_signal.connect(self.update_progress_dashboard)
        self.thread.stats_signal.connect(self.update_stats_dashboard)
        self.thread.finished_signal.connect(self.on_auto_finished)
        self.thread.start()

    def update_progress_dashboard(self, pct, status_msg):
        if self.progress_bar: self.progress_bar.setValue(pct)
        if self.lbl_dash_file: self.lbl_dash_file.setText(status_msg)

    def update_stats_dashboard(self, total, processed, roomtone, short, match, review):
        if self.lbl_stat_total: self.lbl_stat_total.setText(f"Total Files Found: {total}")
        if self.lbl_stat_processed: self.lbl_stat_processed.setText(f"Processed: {processed} / {total}")
        if self.lbl_stat_roomtone: self.lbl_stat_roomtone.setText(f"RoomTone: {roomtone}")
        if self.lbl_stat_short: self.lbl_stat_short.setText(f"Short Duration: {short}")
        if self.lbl_stat_match: self.lbl_stat_match.setText(f"Matched Scenes: {match}")
        if self.lbl_stat_review: self.lbl_stat_review.setText(f"Unclassified (Review Needed): {review}")

    def on_auto_finished(self):
        if self.btn_run_auto: self.btn_run_auto.setEnabled(True)
        if self.lbl_dash_status: self.lbl_dash_status.setText("System Status: Batch Classification Completed!")
        dst = self.txt_dst.text() if self.txt_dst else self.settings["dst_dir"]
        if hasattr(self, 'tree_model'): self.tree_model.setRootPath(os.path.abspath(dst))

    def on_volume_db_changed(self, db_value):
        if self.lbl_vol_db: self.lbl_vol_db.setText(f" Gain: {db_value:+.1f} dB")
        self.apply_volume_db(db_value)

    def on_tab_changed(self, index):
        if index == 1:
            self.refresh_manual_file_list()

    def refresh_manual_file_list(self):
        self.stop_audio()
        dst = self.txt_dst.text() if self.txt_dst else self.settings["dst_dir"]
        unclassified_dir = os.path.abspath(os.path.join(dst, "Unclassified_Manual_Review"))

        if not self.list_files: return
        self.list_files.blockSignals(True)
        self.list_files.clear()

        supported_exts = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aiff', '.aac')
        if os.path.exists(unclassified_dir):
            files = [f for f in os.listdir(unclassified_dir) if f.lower().endswith(supported_exts)]

            fingerprints = {}
            for f in files:
                fpath = os.path.abspath(os.path.join(unclassified_dir, f))
                try:
                    d, s = sf.read(fpath)
                    fp = compute_audio_fingerprint(d, s)
                    if fp in fingerprints:
                        item = QListWidgetItem(f"[DUPLICATE] {f}")
                        item.setForeground(QColor("#ffb74d"))
                    else:
                        fingerprints[fp] = f
                        item = QListWidgetItem(f)
                    self.list_files.addItem(item)
                except Exception:
                    self.list_files.addItem(QListWidgetItem(f))

            if self.lbl_queue_count: self.lbl_queue_count.setText(f"Remaining: {len(files)} Files")
        else:
            if self.lbl_queue_count: self.lbl_queue_count.setText("Remaining: 0 Files")

        self.list_files.blockSignals(False)
        if self.list_files.count() > 0:
            self.list_files.setCurrentRow(0)
        else:
            self.on_file_selected()

    def on_file_selected(self):
        self.stop_audio()
        if not self.list_files: return
        item = self.list_files.currentItem()
        if not item:
            if self.lbl_file_info: self.lbl_file_info.setText("No unclassified files remaining.")
            if self.txt_stt: self.txt_stt.clear()
            return

        raw_text = item.text()
        filename = raw_text.replace("[DUPLICATE] ", "").strip()

        dst = self.txt_dst.text() if self.txt_dst else self.settings["dst_dir"]
        unclassified_dir = os.path.abspath(os.path.join(dst, "Unclassified_Manual_Review"))
        file_path = os.path.abspath(os.path.join(unclassified_dir, filename))
        self.current_audio_file = file_path

        cache_data = cache_mgr.load_cache(filename)
        speech_spans = cache_data.get('speech_timestamps', []) if cache_data else []
        transcribed_text = cache_data.get('transcribed_text', '') if cache_data else ''

        stt_m = self.cmb_stt.currentText() if self.cmb_stt else "large-v3-turbo"
        if not transcribed_text:
            transcribed_text = run_stt_no_hallucination(file_path, model_size=stt_m)
            if cache_data:
                cache_data['transcribed_text'] = transcribed_text
                cache_mgr.save_cache(filename, cache_data)

        if self.txt_stt: self.txt_stt.setText(transcribed_text)

        gain_v = self.spn_gain.value() if self.spn_gain else 0.0
        crop_v = self.spn_crop.value() if self.spn_crop else 15

        audio_data, sr = process_audio_pipeline(file_path, gain_db=gain_v)
        self.total_duration_sec = len(audio_data) / float(sr)

        lufs, tp, pk, rms = calculate_audio_metering(audio_data, sr)
        if self.lbl_lufs: self.lbl_lufs.setText(f"LUFS: {lufs:.1f} dB")
        if self.lbl_tp: self.lbl_tp.setText(f"True Peak: {tp:.1f} dBTP")
        if self.lbl_peak: self.lbl_peak.setText(f"Peak: {pk:.1f} dB")
        if self.lbl_rms: self.lbl_rms.setText(f"RMS: {rms:.1f} dB")

        self.canvas.plot_waveform_with_vad(audio_data, sr, speech_spans, crop_v)
        if self.lbl_file_info: self.lbl_file_info.setText(f"File: {filename} | Duration: {self.total_duration_sec:.1f}s")

        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        self.update_timer_label(0.0)

        if self.chk_autoplay and self.chk_autoplay.isChecked():
            self.toggle_play()

    def sync_playhead(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            cur_ms = self.media_player.position()
            cur_sec = cur_ms / 1000.0
            self.canvas.update_playhead_pos(cur_sec)
            self.update_timer_label(cur_sec)

    def update_timer_label(self, cur_sec):
        cur_m, cur_s = int(cur_sec // 60), int(cur_sec % 60)
        tot_m, tot_s = int(self.total_duration_sec // 60), int(self.total_duration_sec % 60)
        if self.lbl_timer: self.lbl_timer.setText(f"{cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d}")

    def toggle_play(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_timer.stop()
            if self.btn_play: self.btn_play.setText("Play")
        else:
            self.media_player.play()
            self.play_timer.start()
            if self.btn_play: self.btn_play.setText("Pause")

    def stop_audio(self):
        self.media_player.stop()
        self.media_player.setSource(QUrl())
        self.play_timer.stop()
        if self.btn_play: self.btn_play.setText("Play")
        self.canvas.update_playhead_pos(0.0)
        self.update_timer_label(0.0)

    def seek_audio(self, target_sec):
        self.media_player.setPosition(int(target_sec * 1000))
        self.canvas.update_playhead_pos(target_sec)
        self.update_timer_label(target_sec)

    def move_to_scene(self):
        sc = self.txt_sc.text().strip() if self.txt_sc else ""
        ct = self.txt_ct.text().strip() if self.txt_ct else ""
        if not sc.isdigit():
            QMessageBox.warning(self, "Warning", "Scene number must be numeric.")
            return

        cut_num = int(ct) if ct.isdigit() else 1
        parts = [f"Scene_{int(sc):02d}", f"Cut_{cut_num:02d}"]

        self.move_to_folder("_".join(parts))

        if self.txt_sc: self.txt_sc.setText(sc)
        if self.txt_ct: self.txt_ct.setText(str(cut_num + 1))

    def move_to_folder(self, target_subfolder):
        if not self.list_files: return
        item = self.list_files.currentItem()
        if not item: return
        raw_text = item.text()
        filename = raw_text.replace("[DUPLICATE] ", "").strip()

        self.stop_audio()
        QApplication.processEvents()

        dst = self.txt_dst.text() if self.txt_dst else self.settings["dst_dir"]
        unclassified_dir = os.path.abspath(os.path.join(dst, "Unclassified_Manual_Review"))
        src_path = os.path.join(unclassified_dir, filename)

        dst_dir = os.path.abspath(os.path.join(dst, target_subfolder))
        os.makedirs(dst_dir, exist_ok=True)

        is_copy_mode = True
        out_name = format_destination_filename(filename, target_subfolder if "Scene" in target_subfolder else None, "", "", "Parsed Scene Name")
        dst_path = os.path.join(dst_dir, out_name)

        try:
            if is_copy_mode:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)
            self.last_moved_action = (src_path, dst_path, filename, is_copy_mode)
        except PermissionError:
            time.sleep(0.15)
            if is_copy_mode:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)
            self.last_moved_action = (src_path, dst_path, filename, is_copy_mode)

        self.refresh_manual_file_list()

    def undo_last_action(self):
        if not self.last_moved_action:
            QMessageBox.information(self, "Undo", "No previous action to undo.")
            return

        src_path, dst_path, filename, is_copy = self.last_moved_action
        if os.path.exists(dst_path):
            self.stop_audio()
            QApplication.processEvents()

            if is_copy:
                os.remove(dst_path)
            else:
                os.makedirs(os.path.dirname(src_path), exist_ok=True)
                shutil.move(dst_path, src_path)

            self.last_moved_action = None
            QMessageBox.information(self, "Undo Success", f"Restored {filename} to Unclassified queue.")
            self.refresh_manual_file_list()

    def skip_file(self):
        if not self.list_files: return
        row = self.list_files.currentRow()
        if row < self.list_files.count() - 1:
            self.list_files.setCurrentRow(row + 1)

    def save_settings(self):
        if self.txt_src: self.settings["src_dir"] = os.path.abspath(self.txt_src.text())
        if self.txt_dst: self.settings["dst_dir"] = os.path.abspath(self.txt_dst.text())
        if self.slider_vol_db: self.settings["default_volume_db"] = float(self.slider_vol_db.value())
        if self.chk_autoplay: self.settings["auto_play"] = self.chk_autoplay.isChecked()
        if self.spn_gain: self.settings["gain_boost_db"] = self.spn_gain.value()

        if self.chk_prompt_enable: self.settings["stt_prompt_enabled"] = self.chk_prompt_enable.isChecked()
        if self.txt_prompt: self.settings["stt_initial_prompt"] = self.txt_prompt.toPlainText().strip()

        if save_settings_to_file(self.settings):
            QMessageBox.information(self, "Success", "Settings saved successfully to settings.json.")

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #12121a; color: #e0e0e0; font-family: 'Segoe UI', '맑은 고딕'; }
            QTabWidget::pane { border: 1px solid #333; background-color: #1e1e2e; }
            QTabBar::tab { background: #252538; padding: 8px 18px; color: #aaa; font-weight: bold; }
            QTabBar::tab:selected { background: #1976d2; color: white; }
            QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 8px; font-weight: bold; color: #64b5f6; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QListWidget, QTableWidget { background-color: #1e1e2f; border: 1px solid #444; padding: 4px; color: #fff; border-radius: 4px; }
            QListWidget::item:selected, QTableWidget::item:selected { background-color: #1976d2; color: white; }
            QPushButton { background-color: #333; border: 1px solid #555; padding: 6px 12px; border-radius: 4px; color: #fff; }
            QPushButton:hover { background-color: #444; }
            QSlider::groove:horizontal { border: 1px solid #444; height: 6px; background: #222; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #2196f3; border-radius: 3px; }
            QSlider::handle:horizontal { background: #ffffff; border: 1px solid #777; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px; }
            QSplitter::handle { background-color: #333; }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioClassifierApp()
    window.show()
    sys.exit(app.exec())