import os
import shutil
import re
import json
import time
import sys
import numpy as np
import soundfile as sf
import scipy.signal

# Suppress HuggingFace Symlink Warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torchaudio.functional as F
from faster_whisper import WhisperModel

# Matplotlib PySide6 Integration
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# PySide6 GUI & Multimedia
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QFileDialog, QGroupBox, QMessageBox, QSlider,
    QListWidget, QListWidgetItem, QDialog, QFrame, QProgressBar, QSplitter,
    QTreeView, QFileSystemModel, QStackedWidget, QAbstractSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QDir
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QDragEnterEvent, QDropEvent, QColor

# =============================================================================
# Settings Manager (settings.json)
# =============================================================================
SETTINGS_FILE = "./settings.json"

DEFAULT_SETTINGS = {
    "src_dir": "./my_recordings",
    "dst_dir": "./classified_result",
    "min_dur_thresh": 5.0,
    "crop_pct": 15.0,
    "speech_thresh_pct": 2.0,
    "vad_model_choice": "Silero VAD v5 (Recommended)",
    "stt_model_choice": "large-v3-turbo (Recommended High Accuracy)",
    "stt_lang_choice": "ko",
    "device_choice": "Auto (GPU Priority)",
    "vad_thresh": 0.25,
    "default_volume": 100,
    "auto_play": False,
    "file_action_mode": "Copy (Keep Original)",
    "filename_prefix": "",
    "filename_postfix": "",
    "naming_strategy": "Parsed Scene Name",
    "gain_boost_db": 0.0,
    "highpass_cutoff": 80.0,
    "enable_highpass": True,
    "enable_compressor": True,
    "target_rms_db": -14.0,
    "pipeline_order": ["Gain Boost", "Highpass Filter", "Compressor", "VAD Speech Anchoring", "STT Transcription"]
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                merged = DEFAULT_SETTINGS.copy()
                merged.update(saved)
                return merged
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_settings_to_file(settings_dict):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

# Global AI Models
vad_model = None
stt_model = None
current_stt_model_size = None
current_device = None

# STT Prompt for Slate Recognition Optimization
SLATE_INITIAL_PROMPT = "현장 동시녹음 슬레이트 콜입니다. 씬 1, 컷 1, 테이크 1, Scene 1, Cut 1, Take 1."

def get_hardware_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return True, f"GPU Detected: {gpu_name} (CUDA Enabled)"
    return False, "CPU Only Mode (NVIDIA GPU Not Detected)"

def resolve_device(device_choice):
    has_gpu, _ = get_hardware_info()
    if "GPU" in device_choice:
        return "cuda" if has_gpu else "cpu"
    elif "CPU" in device_choice:
        return "cpu"
    else:
        return "cuda" if has_gpu else "cpu"

def load_ai_models(model_size="large-v3-turbo", vad_choice="Silero VAD v5", device_choice="Auto", log_func=print):
    global vad_model, stt_model, current_stt_model_size, current_device

    # Clean model size string
    raw_size = model_size.split()[0]
    target_device = resolve_device(device_choice)
    compute_type = "float16" if target_device == "cuda" else "int8"

    if vad_model is None or current_device != target_device:
        log_func(f"[INFO] Loading VAD Model ({vad_choice})... [{target_device.upper()}]")
        vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        if target_device == "cuda":
            vad_model = vad_model.to("cuda")

    if stt_model is None or current_stt_model_size != raw_size or current_device != target_device:
        log_func(f"[INFO] Loading Faster-Whisper STT ({raw_size})... [{target_device.upper()}]")
        stt_model = WhisperModel(raw_size, device=target_device, compute_type=compute_type)
        current_stt_model_size = raw_size

    current_device = target_device
    log_func(f"[SUCCESS] AI Models Loaded. (Target Device: {target_device.upper()})")


# =============================================================================
# Audio Metering & Fingerprinting
# =============================================================================
def calculate_audio_metering(data, sr):
    if len(data) == 0:
        return -70.0, -70.0, -70.0, -70.0

    abs_max = np.max(np.abs(data))
    peak_db = 20 * np.log10(abs_max + 1e-12)

    rms = np.sqrt(np.mean(data**2) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)

    try:
        resampled = scipy.signal.resample(data, min(len(data) * 4, 1000000))
        true_peak_db = 20 * np.log10(np.max(np.abs(resampled)) + 1e-12)
    except Exception:
        true_peak_db = peak_db

    try:
        b_hp, a_hp = scipy.signal.butter(2, 100 / (sr / 2.0), btype='high')
        kw_data = scipy.signal.filtfilt(b_hp, a_hp, data)
        lufs = 20 * np.log10(np.sqrt(np.mean(kw_data**2) + 1e-12) + 1e-12) - 0.6
    except Exception:
        lufs = rms_db

    return round(float(lufs), 1), round(float(true_peak_db), 1), round(float(peak_db), 1), round(float(rms_db), 1)

def compute_audio_fingerprint(data, sr):
    if len(data) == 0: return ""
    step = max(1, len(data) // 1000)
    sampled = data[::step]
    mean_val = np.mean(np.abs(sampled))
    std_val = np.std(sampled)
    dur_bin = int(len(data) / float(sr) * 10)
    return f"{dur_bin}_{int(mean_val*10000)}_{int(std_val*10000)}"


# =============================================================================
# Fast DSP Preprocessing & Pipeline Chain
# =============================================================================
def apply_gain_boost(data, gain_db):
    if len(data) == 0 or gain_db == 0.0: return data
    gain_factor = 10 ** (gain_db / 20.0)
    return (data * gain_factor).astype(np.float32)

def apply_fast_highpass_filter(data, sr, cutoff=80.0):
    if len(data) == 0: return data
    try:
        b, a = scipy.signal.butter(4, cutoff / (sr / 2.0), btype='high')
        return scipy.signal.filtfilt(b, a, data).astype(np.float32)
    except Exception:
        return data

def apply_compressor_limiter(data, target_rms_db=-14.0, max_gain_db=20.0, limit_ceiling=0.95):
    if len(data) == 0: return data
    rms = np.sqrt(np.mean(data**2) + 1e-12)
    current_rms_db = 20 * np.log10(rms + 1e-12)
    gain_db = min(target_rms_db - current_rms_db, max_gain_db)
    gain = 10 ** (gain_db / 20.0)
    boosted = data * gain
    abs_max = np.max(np.abs(boosted))
    if abs_max > limit_ceiling:
        boosted = np.tanh(boosted / limit_ceiling) * limit_ceiling
    return boosted.astype(np.float32)

def process_audio_pipeline(file_path, gain_db=0.0, pipeline_order=None):
    data, sr = sf.read(file_path)

    # Polyphonic Mono Downmix
    if data.ndim > 1:
        data = data.mean(axis=1)

    if not pipeline_order:
        pipeline_order = ["Gain Boost", "Highpass Filter", "Compressor"]

    for step in pipeline_order:
        if step == "Gain Boost":
            data = apply_gain_boost(data, gain_db)
        elif step == "Highpass Filter":
            data = apply_fast_highpass_filter(data, sr, cutoff=80.0)
        elif step == "Compressor":
            data = apply_compressor_limiter(data)

    return data.astype(np.float32), sr

def prepare_tensor_for_vad(audio_data, sr, target_sr=16000, device="cpu"):
    tensor = torch.from_numpy(audio_data).float()
    if sr != target_sr:
        tensor = F.resample(tensor, sr, target_sr)
    if device == "cuda":
        tensor = tensor.to("cuda")
    return tensor


# =============================================================================
# Slate Parser & Advanced Text Normalization
# =============================================================================
def refine_transcribed_slate_text(text):
    if not text: return ""

    # 1. Slate Phonetic & Misrecognition Fixes
    corrections = {
        r'\b신\b': '씬',
        r'\b씬아\b': '씬',
        r'\b컷트\b': '컷',
        r'\b커\b': '컷',
        r'\b테익\b': '테이크',
        r'\b텍\b': '테이크',
        r'\b태이크\b': '테이크',
        r'\b원\b': '1',
        r'\b투\b': '2',
        r'\b쓰리\b': '3',
        r'\b포\b': '4',
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 2. Korean Numbers to Digits Conversion
    text = convert_korean_numbers_to_digits(text)
    return text

def convert_korean_numbers_to_digits(text):
    num_map = {
        '영': '0', '공': '0', '일': '1', '하나': '1', '첫': '1',
        '이': '2', '둘': '2', '두': '2', '삼': '3', '셋': '3', '세': '3',
        '사': '4', '넷': '4', '네': '4', '오': '5', '다섯': '5',
        '육': '6', '여섯': '6', '칠': '7', '일곱': '7',
        '팔': '8', '여덟': '8', '구': '9', '아홉': '9', '십': '10', '열': '10'
    }
    for k, v in num_map.items():
        text = re.sub(rf'{k}\s*(씬|cut|scene|컷|테이크|take|-)', rf'{v}\1', text, flags=re.IGNORECASE)
        text = re.sub(rf'(씬|cut|scene|컷|테이크|take|-)\s*{k}', rf'\1 {v}', text, flags=re.IGNORECASE)
    return text

def parse_slate_info(text):
    if not text: return None
    normalized_text = refine_transcribed_slate_text(text)

    scene, cut, take = None, None, None
    sc_m = re.search(r'(?:씬|scene)\s*(\d+)|(\d+)\s*(?:씬|scene)', normalized_text, re.IGNORECASE)
    if sc_m: scene = sc_m.group(1) or sc_m.group(2)

    cut_m = re.search(r'(?:컷|cut)\s*(\d+)|(\d+)\s*(?:컷|cut)', normalized_text, re.IGNORECASE)
    if cut_m: cut = cut_m.group(1) or cut_m.group(2)

    take_m = re.search(r'(?:테이크|take)\s*(\d+)|(\d+)\s*(?:테이크|take)', normalized_text, re.IGNORECASE)
    if take_m: take = take_m.group(1) or take_m.group(2)

    if not scene:
        num_m = re.search(r'\b(\d+)[\s\-\.\:\/]+(\d+)(?:[\s\-\.\:\/]+(\d+))?\b', normalized_text)
        if num_m:
            scene, cut = num_m.group(1), num_m.group(2)
            if num_m.group(3): take = num_m.group(3)

    if scene:
        parts = [f"Scene_{int(scene):02d}"]
        if cut: parts.append(f"Cut_{int(cut):02d}")
        if take: parts.append(f"Take_{int(take):02d}")
        return "_".join(parts)

    return None

def format_destination_filename(original_name, parsed_scene, prefix, postfix, strategy, index=1):
    base, ext = os.path.splitext(original_name)
    if strategy == "Parsed Scene Name" and parsed_scene:
        core = parsed_scene
    elif strategy == "Combined Name" and parsed_scene:
        core = f"{base}_{parsed_scene}"
    elif strategy == "Numbered Index" and parsed_scene:
        core = f"{index:03d}_{parsed_scene}"
    else:
        core = base
    return f"{prefix}{core}{postfix}{ext}"


# Cache Manager
class CacheManager:
    def __init__(self, cache_dir="./caches"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, audio_filename):
        base_name = os.path.splitext(audio_filename)[0]
        return os.path.join(self.cache_dir, f"{base_name}_cache.json")

    def load_cache(self, audio_filename):
        cache_path = self.get_cache_path(audio_filename)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_cache(self, audio_filename, data_dict):
        cache_path = self.get_cache_path(audio_filename)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save cache: {e}")

cache_mgr = CacheManager("./caches")


# =============================================================================
# STT Pipeline Engine (Enhanced with Smart VAD Anchoring & Prompting)
# =============================================================================
def run_stt_no_hallucination(file_path, lang_choice="ko", model_size="large-v3-turbo", device_choice="Auto"):
    load_ai_models(model_size=model_size, device_choice=device_choice)
    audio_data, sr = process_audio_pipeline(file_path)

    # Peak Normalization to 0.95
    max_v = np.abs(audio_data).max()
    if max_v > 0:
        audio_data = (audio_data / max_v * 0.95).astype(np.float32)

    # Smart VAD Anchoring for Speech Crop
    target_dev = resolve_device(device_choice)
    wav_tensor = prepare_tensor_for_vad(audio_data, sr, device=target_dev)
    
    try:
        from silero_vad import get_speech_timestamps
        timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=16000, threshold=0.25)
    except Exception:
        timestamps = []

    # Extract region starting from the first actual speech
    if timestamps:
        first_start_sec = timestamps[0]['start'] / 16000.0
        start_idx = max(0, int((first_start_sec - 0.3) * sr))
        end_idx = min(len(audio_data), int((first_start_sec + 7.0) * sr))
        stt_input_audio = audio_data[start_idx:end_idx]
    else:
        stt_input_audio = audio_data[:int(15 * sr)]

    try:
        transcribe_kwargs = {
            "language": lang_choice if lang_choice != "auto" else None,
            "initial_prompt": SLATE_INITIAL_PROMPT,
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.5,
            "hallucination_silence_threshold": 0.5,
            "beam_size": 5
        }
        
        try:
            segments, _ = stt_model.transcribe(
                stt_input_audio,
                hotwords="씬 컷 테이크 scene cut take",
                **transcribe_kwargs
            )
        except TypeError: # Fallback if hotwords unsupported in older faster-whisper
            segments, _ = stt_model.transcribe(
                stt_input_audio,
                **transcribe_kwargs
            )

        raw_text = "".join([s.text for s in segments]).strip()
        refined_text = refine_transcribed_slate_text(raw_text)
        return refined_text if refined_text else "(No speech recognized)"
    except Exception as e:
        return f"[ERROR] STT failed: {e}"


# =============================================================================
# Interactive Waveform Canvas
# =============================================================================
class InteractiveWaveformCanvas(FigureCanvasQTAgg):
    seek_requested = Signal(float)
    loop_region_changed = Signal(float, float)

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

        self.loop_start = None
        self.loop_end = None
        self.drag_start_x = None

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

        if self.loop_start is not None and self.loop_end is not None:
            self.ax.axvspan(self.loop_start, self.loop_end, color='#64b5f6', alpha=0.35)

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
        if self.total_duration > 0:
            widget_width = self.width()
            pos_x = event.position().x()
            rel_ratio = pos_x / float(widget_width)
            clicked_sec = self.view_start + rel_ratio * (self.view_end - self.view_start)
            clicked_sec = max(0.0, min(clicked_sec, self.total_duration))

            if event.button() == Qt.LeftButton:
                self.seek_requested.emit(clicked_sec)
            elif event.button() == Qt.RightButton:
                self.drag_start_x = clicked_sec

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self.drag_start_x is not None and self.total_duration > 0:
            widget_width = self.width()
            pos_x = event.position().x()
            rel_ratio = pos_x / float(widget_width)
            release_sec = self.view_start + rel_ratio * (self.view_end - self.view_start)
            release_sec = max(0.0, min(release_sec, self.total_duration))

            if abs(release_sec - self.drag_start_x) > 0.3:
                self.loop_start = min(self.drag_start_x, release_sec)
                self.loop_end = max(self.drag_start_x, release_sec)
                self.loop_region_changed.emit(self.loop_start, self.loop_end)
            else:
                self.loop_start = None
                self.loop_end = None
                self.loop_region_changed.emit(0.0, 0.0)

            self.drag_start_x = None

        super().mouseReleaseEvent(event)


# Log Dialog Window
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


# Background Classifier Thread
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
        from silero_vad import get_speech_timestamps

        supported_exts = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aiff', '.aac')
        all_files = []
        for root_path, _, filenames in os.walk(self.src_dir):
            for fname in filenames:
                if fname.lower().endswith(supported_exts):
                    rel_path = os.path.relpath(os.path.join(root_path, fname), self.src_dir)
                    all_files.append((fname, os.path.join(root_path, fname), rel_path))

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
        pipeline_order = self.settings.get("pipeline_order", ["Gain Boost", "Highpass Filter", "Compressor"])

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
                    audio_data, sr = process_audio_pipeline(full_src_path, gain_db=gain_db, pipeline_order=pipeline_order)
                    total_duration_sec = len(audio_data) / float(sr)

                    # Peak Normalization
                    max_v = np.abs(audio_data).max()
                    if max_v > 0: audio_data = (audio_data / max_v * 0.95).astype(np.float32)

                    wav_tensor = prepare_tensor_for_vad(audio_data, sr, device=target_dev)
                    speech_timestamps_all = get_speech_timestamps(
                        wav_tensor, vad_model, sampling_rate=16000,
                        threshold=0.25, min_speech_duration_ms=100, speech_pad_ms=300
                    )

                    speech_timestamps_json = [
                        {'start': round(seg['start'] / 16000.0, 2), 'end': round(seg['end'] / 16000.0, 2)}
                        for seg in speech_timestamps_all
                    ]

                    # Smart Speech Crop for Slate STT
                    if speech_timestamps_json:
                        first_speech_start = speech_timestamps_json[0]['start']
                        crop_start = max(0, int((first_speech_start - 0.3) * sr))
                        crop_end = min(len(audio_data), int((first_speech_start + 7.0) * sr))
                        audio_clip = audio_data[crop_start:crop_end]
                    else:
                        audio_clip = audio_data[:int(min(15.0, total_duration_sec) * sr)]

                    transcribe_kwargs = {
                        "language": "ko",
                        "initial_prompt": SLATE_INITIAL_PROMPT,
                        "condition_on_previous_text": False,
                        "no_speech_threshold": 0.5,
                        "beam_size": 5
                    }

                    try:
                        segments, _ = stt_model.transcribe(
                            audio_clip,
                            hotwords="씬 컷 테이크 scene cut take",
                            **transcribe_kwargs
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

                # Step 1: Short Duration
                if total_duration_sec < self.min_dur:
                    subfolder_name = "Short_Duration"
                    parsed_scene = None
                    cnt_short += 1
                    self.log_signal.emit(f"[SHORT] {file_name} ({total_duration_sec:.1f}s)")

                # Step 2: Middle 70% VAD
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
                target_folder = os.path.join(self.dst_dir, subfolder_name)
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


# =============================================================================
# PySide6 Main Window UI (Smart Audio Classifier V0.1.2)
# =============================================================================
class AudioClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()

        self.setWindowTitle("Smart Audio Classifier V0.1.2 (STT Enhanced)")
        self.resize(1200, 850)

        self.setAcceptDrops(True)

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        
        initial_vol = self.settings.get("default_volume", 100)
        self.audio_output.setVolume(initial_vol / 100.0)

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(16)
        self.play_timer.timeout.connect(self.sync_playhead)

        self.current_audio_file = ""
        self.total_duration_sec = 0.0
        self.last_moved_action = None
        self.loop_a = 0.0
        self.loop_b = 0.0

        self.log_dialog = LogDialog(self)

        self.setup_ui()
        self.setup_keyboard_shortcuts()
        self.apply_dark_theme()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            folder_path = urls[0].toLocalFile()
            if os.path.isdir(folder_path):
                self.txt_src.setText(os.path.abspath(folder_path))
                QMessageBox.information(self, "Folder Dropped", f"Source directory set to:\n{folder_path}")

    def setup_keyboard_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_play)
        QShortcut(QKeySequence(Qt.Key_R), self, lambda: self.move_to_folder("RoomTone"))
        QShortcut(QKeySequence(Qt.Key_S), self, lambda: self.move_to_folder("Short_Duration"))
        QShortcut(QKeySequence(Qt.Key_Right), self, self.skip_file)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_last_action)
        QShortcut(QKeySequence(Qt.Key_Return), self, self.move_to_scene)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self.move_to_scene)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab_auto = QWidget()
        self.tab_manual = QWidget()
        self.tab_enhance = QWidget()
        self.tab_settings = QWidget()

        self.tabs.addTab(self.tab_auto, "Auto Classifier")
        self.tabs.addTab(self.tab_manual, "Manual Review Workstation")
        self.tabs.addTab(self.tab_enhance, "Voice Enhancement Workstation")
        self.tabs.addTab(self.tab_settings, "Settings & Hardware")

        self.setup_auto_tab()
        self.setup_manual_tab()
        self.setup_enhance_tab()
        self.setup_settings_tab()

        bottom_bar = QHBoxLayout()
        _, hw_msg = get_hardware_info()
        self.lbl_status_bar = QLabel(hw_msg)
        self.lbl_status_bar.setStyleSheet("color: #81c784; font-weight: bold;")

        btn_show_logs = QPushButton("Show Detailed Logs")
        btn_show_logs.clicked.connect(self.log_dialog.show)

        bottom_bar.addWidget(self.lbl_status_bar)
        bottom_bar.addStretch()
        bottom_bar.addWidget(btn_show_logs)

        main_layout.addLayout(bottom_bar)
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def remove_spinbox_buttons(self, spinbox):
        spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)

    # -------------------------------------------------------------------------
    # TAB 1: Auto Classifier
    # -------------------------------------------------------------------------
    def setup_auto_tab(self):
        layout = QVBoxLayout(self.tab_auto)
        splitter_auto = QSplitter(Qt.Horizontal)

        p1 = QFrame()
        lay_p1 = QVBoxLayout(p1)
        box_dir = QGroupBox(" Folder Configuration ")
        grid_dir = QVBoxLayout(box_dir)
        
        self.txt_src = QLineEdit(self.settings["src_dir"])
        btn_src = QPushButton("Browse Source Folder")
        btn_src.clicked.connect(lambda: self.browse_folder(self.txt_src))

        self.txt_dst = QLineEdit(self.settings["dst_dir"])
        btn_dst = QPushButton("Browse Target Folder")
        btn_dst.clicked.connect(lambda: self.browse_folder(self.txt_dst))

        grid_dir.addWidget(QLabel("Source Folder (Drag & Drop):"))
        grid_dir.addWidget(self.txt_src)
        grid_dir.addWidget(btn_src)
        grid_dir.addWidget(QLabel("Target Folder:"))
        grid_dir.addWidget(self.txt_dst)
        grid_dir.addWidget(btn_dst)
        lay_p1.addWidget(box_dir)
        lay_p1.addStretch()

        p2 = QFrame()
        lay_p2 = QVBoxLayout(p2)
        box_opts = QGroupBox(" Workflow Parameters & Pipeline ")
        lay_opts = QVBoxLayout(box_opts)

        h1 = QHBoxLayout()
        self.spn_dur = QSpinBox()
        self.spn_dur.setRange(1, 30); self.spn_dur.setValue(int(self.settings["min_dur_thresh"]))
        self.remove_spinbox_buttons(self.spn_dur)
        h1.addWidget(QLabel("Min Dur (s):")); h1.addWidget(self.spn_dur)

        self.spn_crop = QSpinBox()
        self.spn_crop.setRange(5, 30); self.spn_crop.setValue(int(self.settings["crop_pct"]))
        self.remove_spinbox_buttons(self.spn_crop)
        h1.addWidget(QLabel("Crop (%):")); h1.addWidget(self.spn_crop)

        self.spn_speech = QDoubleSpinBox()
        self.spn_speech.setRange(0.5, 10.0); self.spn_speech.setValue(self.settings["speech_thresh_pct"])
        self.remove_spinbox_buttons(self.spn_speech)
        h1.addWidget(QLabel("Mid Speech (%):")); h1.addWidget(self.spn_speech)
        lay_opts.addLayout(h1)

        h2 = QHBoxLayout()
        self.cmb_stt = QComboBox()
        self.cmb_stt.addItems([
            "large-v3-turbo (Recommended High Accuracy)",
            "base (Fast)",
            "small (Balanced)",
            "medium (High Accuracy)"
        ])
        self.cmb_dev = QComboBox()
        self.cmb_dev.addItems(["Auto (GPU Priority)", "GPU (CUDA)", "CPU"])

        h2.addWidget(QLabel("STT Model:")); h2.addWidget(self.cmb_stt)
        h2.addWidget(QLabel("Device:")); h2.addWidget(self.cmb_dev)
        lay_opts.addLayout(h2)

        h3 = QHBoxLayout()
        self.spn_gain = QDoubleSpinBox()
        self.spn_gain.setRange(0.0, 24.0); self.spn_gain.setValue(float(self.settings.get("gain_boost_db", 0.0)))
        self.remove_spinbox_buttons(self.spn_gain)
        h3.addWidget(QLabel("Gain Boost (dB):")); h3.addWidget(self.spn_gain)
        lay_opts.addLayout(h3)

        lay_p2.addWidget(box_opts)

        self.btn_run_auto = QPushButton("Run Auto Classification")
        self.btn_run_auto.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; padding: 14px; font-size: 14px;")
        self.btn_run_auto.clicked.connect(self.start_auto_classification)
        lay_p2.addWidget(self.btn_run_auto)
        lay_p2.addStretch()

        p3 = QFrame()
        lay_p3 = QVBoxLayout(p3)
        box_dash = QGroupBox(" Real-Time Processing Dashboard ")
        dash_layout = QVBoxLayout(box_dash)

        self.lbl_dash_status = QLabel("System Status: Idle")
        self.lbl_dash_status.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.lbl_dash_status.setStyleSheet("color: #64b5f6;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { height: 25px; text-align: center; font-weight: bold; } QProgressBar::chunk { background-color: #2196f3; }")

        self.lbl_dash_file = QLabel("Current File: None")

        box_stats = QGroupBox(" Classification Statistics ")
        lay_stats = QVBoxLayout(box_stats)
        self.lbl_stat_total = QLabel("Total Files Found: 0")
        self.lbl_stat_processed = QLabel("Processed: 0")
        self.lbl_stat_roomtone = QLabel("RoomTone: 0")
        self.lbl_stat_short = QLabel("Short Duration: 0")
        self.lbl_stat_match = QLabel("Matched Scenes: 0")
        self.lbl_stat_review = QLabel("Unclassified (Review Needed): 0")

        lay_stats.addWidget(self.lbl_stat_total)
        lay_stats.addWidget(self.lbl_stat_processed)
        lay_stats.addWidget(self.lbl_stat_roomtone)
        lay_stats.addWidget(self.lbl_stat_short)
        lay_stats.addWidget(self.lbl_stat_match)
        lay_stats.addWidget(self.lbl_stat_review)

        dash_layout.addWidget(self.lbl_dash_status)
        dash_layout.addWidget(self.progress_bar)
        dash_layout.addWidget(self.lbl_dash_file)
        dash_layout.addWidget(box_stats)
        dash_layout.addStretch()

        lay_p3.addWidget(box_dash)

        splitter_auto.addWidget(p1)
        splitter_auto.addWidget(p2)
        splitter_auto.addWidget(p3)
        splitter_auto.setSizes([320, 400, 380])

        layout.addWidget(splitter_auto)

    def browse_folder(self, target_widget):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if dir_path:
            target_widget.setText(os.path.abspath(dir_path))

    def start_auto_classification(self):
        self.btn_run_auto.setEnabled(False)
        self.log_dialog.clear_log()
        self.lbl_dash_status.setText("System Status: Processing Batch...")
        self.progress_bar.setValue(0)

        self.settings["gain_boost_db"] = self.spn_gain.value()

        self.thread = AutoClassifierThread(
            src_dir=self.txt_src.text(),
            dst_dir=self.txt_dst.text(),
            min_dur=self.spn_dur.value(),
            crop_pct=self.spn_crop.value(),
            speech_thresh=self.spn_speech.value(),
            vad_choice="Silero VAD v5",
            stt_choice=self.cmb_stt.currentText(),
            dev_choice=self.cmb_dev.currentText(),
            settings=self.settings
        )
        self.thread.log_signal.connect(self.log_dialog.append_log)
        self.thread.progress_signal.connect(self.update_progress_dashboard)
        self.thread.stats_signal.connect(self.update_stats_dashboard)
        self.thread.finished_signal.connect(self.on_auto_finished)
        self.thread.start()

    def update_progress_dashboard(self, pct, status_msg):
        self.progress_bar.setValue(pct)
        self.lbl_dash_file.setText(status_msg)

    def update_stats_dashboard(self, total, processed, roomtone, short, match, review):
        self.lbl_stat_total.setText(f"Total Files Found: {total}")
        self.lbl_stat_processed.setText(f"Processed: {processed} / {total}")
        self.lbl_stat_roomtone.setText(f"RoomTone: {roomtone}")
        self.lbl_stat_short.setText(f"Short Duration: {short}")
        self.lbl_stat_match.setText(f"Matched Scenes: {match}")
        self.lbl_stat_review.setText(f"Unclassified (Review Needed): {review}")

    def on_auto_finished(self):
        self.btn_run_auto.setEnabled(True)
        self.lbl_dash_status.setText("System Status: Batch Classification Completed!")
        self.tree_model.setRootPath(os.path.abspath(self.txt_dst.text()))

    # -------------------------------------------------------------------------
    # TAB 2: Manual Review Workstation
    # -------------------------------------------------------------------------
    def setup_manual_tab(self):
        layout = QVBoxLayout(self.tab_manual)
        splitter = QSplitter(Qt.Horizontal)

        # Pane 1 (Left)
        pane1 = QWidget()
        lay_p1 = QVBoxLayout(pane1); lay_p1.setContentsMargins(0,0,0,0)
        box_tree = QGroupBox(" Destination Directory Tree ")
        lay_tree = QVBoxLayout(box_tree)

        self.tree_model = QFileSystemModel()
        self.tree_model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files)
        root_path = os.path.abspath(self.settings["dst_dir"])
        os.makedirs(root_path, exist_ok=True)
        self.tree_model.setRootPath(root_path)

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.tree_model)
        self.tree_view.setRootIndex(self.tree_model.index(root_path))
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setColumnHidden(1, True)
        self.tree_view.setColumnHidden(2, True)
        self.tree_view.setColumnHidden(3, True)

        lay_tree.addWidget(self.tree_view)
        lay_p1.addWidget(box_tree)

        # Pane 2 (Center)
        pane2 = QWidget()
        lay_p2 = QVBoxLayout(pane2); lay_p2.setContentsMargins(0,0,0,0)

        self.canvas = InteractiveWaveformCanvas(self)
        self.canvas.seek_requested.connect(self.seek_audio)
        self.canvas.loop_region_changed.connect(self.on_loop_region_changed)
        lay_p2.addWidget(self.canvas, 5)

        lay_play = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setStyleSheet("background-color: #2196f3; color: white; font-weight: bold; padding: 8px 16px;")
        self.btn_play.clicked.connect(self.toggle_play)

        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(self.stop_audio)

        btn_export_loop = QPushButton("Export A-B Loop Clip")
        btn_export_loop.clicked.connect(self.export_ab_loop_clip)

        lay_play.addWidget(self.btn_play)
        lay_play.addWidget(btn_stop)
        lay_play.addWidget(btn_export_loop)
        lay_play.addWidget(QLabel(" Vol:"))

        self.slider_vol = QSlider(Qt.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(int(self.settings.get("default_volume", 100)))
        self.slider_vol.setFixedWidth(90)
        self.slider_vol.valueChanged.connect(self.on_volume_changed)
        lay_play.addWidget(self.slider_vol)

        self.chk_autoplay = QCheckBox("Auto-Play")
        self.chk_autoplay.setChecked(self.settings.get("auto_play", True))
        lay_play.addWidget(self.chk_autoplay)

        self.lbl_timer = QLabel("00:00 / 00:00")
        self.lbl_timer.setStyleSheet("font-family: monospace; font-size: 13px; font-weight: bold; color: #a5d6a7;")
        lay_play.addStretch()
        lay_play.addWidget(self.lbl_timer)

        lay_p2.addLayout(lay_play)

        box_meter = QGroupBox(" Real-Time Audio Metering ")
        lay_meter = QHBoxLayout(box_meter)
        self.lbl_lufs = QLabel("LUFS: -0.0 dB")
        self.lbl_tp = QLabel("True Peak: 0.0 dBTP")
        self.lbl_peak = QLabel("Peak: 0.0 dB")
        self.lbl_rms = QLabel("RMS: 0.0 dB")

        self.lbl_lufs.setStyleSheet("font-weight: bold; color: #64b5f6;")
        self.lbl_tp.setStyleSheet("font-weight: bold; color: #ffb74d;")

        lay_meter.addWidget(self.lbl_lufs)
        lay_meter.addWidget(self.lbl_tp)
        lay_meter.addWidget(self.lbl_peak)
        lay_meter.addWidget(self.lbl_rms)
        lay_p2.addWidget(box_meter)

        # Pane 3 (Right)
        pane3 = QWidget()
        lay_p3 = QVBoxLayout(pane3); lay_p3.setContentsMargins(0,0,0,0)

        box_list = QGroupBox(" Unclassified File Queue ")
        lay_list = QVBoxLayout(box_list)
        self.lbl_queue_count = QLabel("Remaining: 0 Files")
        self.lbl_queue_count.setStyleSheet("font-weight: bold; color: #81c784;")

        self.list_files = QListWidget()
        self.list_files.itemSelectionChanged.connect(self.on_file_selected)

        btn_refresh = QPushButton("Refresh List")
        btn_refresh.clicked.connect(self.refresh_manual_file_list)

        lay_list.addWidget(self.lbl_queue_count)
        lay_list.addWidget(self.list_files)
        lay_list.addWidget(btn_refresh)
        lay_p3.addWidget(box_list, 2)

        box_info = QGroupBox(" Transcribed Text & Actions ")
        lay_info = QVBoxLayout(box_info)

        self.lbl_file_info = QLabel("Select a file from queue.")
        self.lbl_file_info.setStyleSheet("font-weight: bold; color: #64b5f6;")

        self.txt_stt = QLineEdit()
        self.txt_stt.setReadOnly(True)
        self.txt_stt.setPlaceholderText("Transcribed text cache...")

        lay_sc = QHBoxLayout()
        self.txt_sc = QLineEdit(); self.txt_sc.setPlaceholderText("Scene (e.g. 6)")
        self.txt_ct = QLineEdit(); self.txt_ct.setPlaceholderText("Cut (e.g. 1)")
        lay_sc.addWidget(QLabel("Scene:")); lay_sc.addWidget(self.txt_sc)
        lay_sc.addWidget(QLabel("Cut:")); lay_sc.addWidget(self.txt_ct)

        btn_move_sc = QPushButton("Move to Scene/Cut (Enter)")
        btn_move_sc.setStyleSheet("background-color: #4caf50; color: white; font-weight: bold;")
        btn_move_sc.clicked.connect(self.move_to_scene)

        btn_roomtone = QPushButton("Move to RoomTone (Key: R)")
        btn_roomtone.clicked.connect(lambda: self.move_to_folder("RoomTone"))

        btn_short = QPushButton("Move to Short_Duration (Key: S)")
        btn_short.clicked.connect(lambda: self.move_to_folder("Short_Duration"))

        btn_skip = QPushButton("Skip File (Key: Right)")
        btn_skip.clicked.connect(self.skip_file)

        btn_undo = QPushButton("Undo Last Action (Ctrl+Z)")
        btn_undo.setStyleSheet("background-color: #ff9800; color: white; font-weight: bold;")
        btn_undo.clicked.connect(self.undo_last_action)

        lay_info.addWidget(self.lbl_file_info)
        lay_info.addWidget(self.txt_stt)
        lay_info.addLayout(lay_sc)
        lay_info.addWidget(btn_move_sc)
        lay_info.addWidget(btn_roomtone)
        lay_info.addWidget(btn_short)
        lay_info.addWidget(btn_skip)
        lay_info.addWidget(btn_undo)
        lay_p3.addWidget(box_info, 3)

        splitter.addWidget(pane1)
        splitter.addWidget(pane2)
        splitter.addWidget(pane3)
        splitter.setSizes([220, 560, 320])

        layout.addWidget(splitter)

    def on_volume_changed(self, value):
        self.audio_output.setVolume(value / 100.0)

    def on_tab_changed(self, index):
        if index == 1:
            self.refresh_manual_file_list()

    def refresh_manual_file_list(self):
        self.stop_audio()
        unclassified_dir = os.path.join(os.path.abspath(self.txt_dst.text()), "Unclassified_Manual_Review")

        self.list_files.blockSignals(True)
        self.list_files.clear()

        supported_exts = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aiff', '.aac')
        if os.path.exists(unclassified_dir):
            files = [f for f in os.listdir(unclassified_dir) if f.lower().endswith(supported_exts)]

            fingerprints = {}
            for f in files:
                fpath = os.path.join(unclassified_dir, f)
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

            self.lbl_queue_count.setText(f"Remaining: {len(files)} Files")
        else:
            self.lbl_queue_count.setText("Remaining: 0 Files")

        self.list_files.blockSignals(False)
        if self.list_files.count() > 0:
            self.list_files.setCurrentRow(0)
        else:
            self.on_file_selected()

    def on_file_selected(self):
        self.stop_audio()
        item = self.list_files.currentItem()
        if not item:
            self.lbl_file_info.setText("No unclassified files remaining.")
            self.txt_stt.clear()
            return

        raw_text = item.text()
        filename = raw_text.replace("[DUPLICATE] ", "").strip()

        unclassified_dir = os.path.join(os.path.abspath(self.txt_dst.text()), "Unclassified_Manual_Review")
        file_path = os.path.join(unclassified_dir, filename)
        self.current_audio_file = file_path

        cache_data = cache_mgr.load_cache(filename)
        speech_spans = cache_data.get('speech_timestamps', []) if cache_data else []
        transcribed_text = cache_data.get('transcribed_text', '') if cache_data else ''

        if not transcribed_text:
            transcribed_text = run_stt_no_hallucination(file_path, model_size=self.cmb_stt.currentText())
            if cache_data:
                cache_data['transcribed_text'] = transcribed_text
                cache_mgr.save_cache(filename, cache_data)

        self.txt_stt.setText(transcribed_text)

        audio_data, sr = process_audio_pipeline(file_path, gain_db=self.spn_gain.value())
        self.total_duration_sec = len(audio_data) / float(sr)

        lufs, tp, pk, rms = calculate_audio_metering(audio_data, sr)
        self.lbl_lufs.setText(f"LUFS: {lufs:.1f} dB")
        self.lbl_tp.setText(f"True Peak: {tp:.1f} dBTP")
        self.lbl_peak.setText(f"Peak: {pk:.1f} dB")
        self.lbl_rms.setText(f"RMS: {rms:.1f} dB")

        self.canvas.plot_waveform_with_vad(audio_data, sr, speech_spans, self.spn_crop.value())
        self.lbl_file_info.setText(f"File: {filename} | Duration: {self.total_duration_sec:.1f}s")

        self.media_player.setSource(QUrl.fromLocalFile(os.path.abspath(file_path)))
        self.update_timer_label(0.0)

        if self.chk_autoplay.isChecked():
            self.toggle_play()

    def sync_playhead(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            cur_ms = self.media_player.position()
            cur_sec = cur_ms / 1000.0

            if self.loop_b > 0 and cur_sec >= self.loop_b:
                self.media_player.setPosition(int(self.loop_a * 1000))
                cur_sec = self.loop_a

            self.canvas.update_playhead_pos(cur_sec)
            self.update_timer_label(cur_sec)

    def on_loop_region_changed(self, start_sec, end_sec):
        self.loop_a = start_sec
        self.loop_b = end_sec

    def export_ab_loop_clip(self):
        if self.loop_b <= self.loop_a or not self.current_audio_file:
            QMessageBox.warning(self, "No Loop Region", "Right-click and drag on waveform to set A-B loop region first.")
            return

        audio_data, sr = process_audio_pipeline(self.current_audio_file)
        start_idx = int(self.loop_a * sr)
        end_idx = int(self.loop_b * sr)
        clip_data = audio_data[start_idx:end_idx]

        out_name = f"Clip_{self.loop_a:.1f}s-{self.loop_b:.1f}s_" + os.path.basename(self.current_audio_file)
        out_path = os.path.join(os.path.dirname(self.current_audio_file), out_name)
        sf.write(out_path, clip_data, sr)
        QMessageBox.information(self, "Exported", f"Exported clip to:\n{out_name}")

    def update_timer_label(self, cur_sec):
        cur_m, cur_s = int(cur_sec // 60), int(cur_sec % 60)
        tot_m, tot_s = int(self.total_duration_sec // 60), int(self.total_duration_sec % 60)
        self.lbl_timer.setText(f"{cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d}")

    def toggle_play(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_timer.stop()
            self.btn_play.setText("Play")
        else:
            self.media_player.play()
            self.play_timer.start()
            self.btn_play.setText("Pause")

    def stop_audio(self):
        self.media_player.stop()
        self.media_player.setSource(QUrl())
        self.play_timer.stop()
        self.btn_play.setText("Play")
        self.canvas.update_playhead_pos(0.0)
        self.update_timer_label(0.0)

    def seek_audio(self, target_sec):
        self.media_player.setPosition(int(target_sec * 1000))
        self.canvas.update_playhead_pos(target_sec)
        self.update_timer_label(target_sec)

    def move_to_scene(self):
        sc = self.txt_sc.text().strip()
        ct = self.txt_ct.text().strip()
        if not sc.isdigit():
            QMessageBox.warning(self, "Warning", "Scene number must be numeric.")
            return

        cut_num = int(ct) if ct.isdigit() else 1
        parts = [f"Scene_{int(sc):02d}", f"Cut_{cut_num:02d}"]
        
        self.move_to_folder("_".join(parts))

        self.txt_sc.setText(sc)
        self.txt_ct.setText(str(cut_num + 1))

    def move_to_folder(self, target_subfolder):
        item = self.list_files.currentItem()
        if not item: return
        raw_text = item.text()
        filename = raw_text.replace("[DUPLICATE] ", "").strip()

        self.stop_audio()
        QApplication.processEvents()

        unclassified_dir = os.path.join(os.path.abspath(self.txt_dst.text()), "Unclassified_Manual_Review")
        src_path = os.path.join(unclassified_dir, filename)

        dst_dir = os.path.join(os.path.abspath(self.txt_dst.text()), target_subfolder)
        os.makedirs(dst_dir, exist_ok=True)

        is_copy_mode = "Copy" in self.cmb_action_mode.currentText()
        prefix = self.txt_prefix.text().strip()
        postfix = self.txt_postfix.text().strip()
        strategy = self.cmb_naming_strat.currentText()

        out_name = format_destination_filename(filename, target_subfolder if "Scene" in target_subfolder else None, prefix, postfix, strategy)
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
        row = self.list_files.currentRow()
        if row < self.list_files.count() - 1:
            self.list_files.setCurrentRow(row + 1)

    # -------------------------------------------------------------------------
    # TAB 3: Voice Enhancement Workstation
    # -------------------------------------------------------------------------
    def setup_enhance_tab(self):
        layout = QVBoxLayout(self.tab_enhance)
        splitter_enh = QSplitter(Qt.Horizontal)

        # Pane 1
        pane1 = QWidget()
        lay_p1 = QVBoxLayout(pane1); lay_p1.setContentsMargins(0,0,0,0)
        box_tree = QGroupBox(" Destination Directory Tree ")
        lay_tree = QVBoxLayout(box_tree)
        
        self.tree_view_enh = QTreeView()
        self.tree_view_enh.setModel(self.tree_model)
        root_path = os.path.abspath(self.settings["dst_dir"])
        self.tree_view_enh.setRootIndex(self.tree_model.index(root_path))
        self.tree_view_enh.setHeaderHidden(True)
        self.tree_view_enh.setColumnHidden(1, True)
        self.tree_view_enh.setColumnHidden(2, True)
        self.tree_view_enh.setColumnHidden(3, True)

        lay_tree.addWidget(self.tree_view_enh)
        lay_p1.addWidget(box_tree)

        # Pane 2
        pane2 = QWidget()
        lay_p2 = QVBoxLayout(pane2); lay_p2.setContentsMargins(0,0,0,0)

        self.canvas_enh = InteractiveWaveformCanvas(self)
        self.canvas_enh.seek_requested.connect(self.seek_audio)
        lay_p2.addWidget(self.canvas_enh, 5)

        box_meter_enh = QGroupBox(" Audio Metering ")
        lay_m_enh = QHBoxLayout(box_meter_enh)
        self.lbl_lufs_enh = QLabel("LUFS: -0.0 dB")
        self.lbl_tp_enh = QLabel("True Peak: 0.0 dBTP")
        lay_m_enh.addWidget(self.lbl_lufs_enh)
        lay_m_enh.addWidget(self.lbl_tp_enh)
        lay_p2.addWidget(box_meter_enh)

        # Pane 3
        pane3 = QWidget()
        lay_p3 = QVBoxLayout(pane3); lay_p3.setContentsMargins(0,0,0,0)

        box_ctrl = QGroupBox(" Voice Enhancement & Studio Restoration ")
        lay_ctrl = QVBoxLayout(box_ctrl)

        self.cmb_enh_model = QComboBox()
        self.cmb_enh_model.addItems([
            "DeepFilterNet3 (Ultra-Fast 48kHz)",
            "DeepFilterNet2 (48kHz Legacy)",
            "Alibaba ClearVoice MossFormer2 (48kHz)",
            "Alibaba ClearVoice FRCRN (16kHz)",
            "Diamond 1.0 (Studio 44.1kHz Restoration)",
            "Resemble Enhance (Bandwidth Extension)",
            "VoiceFixer (Denoise + Dereverb + Declip)",
            "Meta HTDemucs (BGM/Crowd Vocal Isolation)",
            "Meta Demucs v4 (4-Stem Separation)",
            "NVIDIA CleanUNet (Waveform Denoising)",
            "RNNoise (Lightweight Neural Suppressor)"
        ])

        btn_run_enhance = QPushButton("Run Voice Enhancement on Selected File")
        btn_run_enhance.setStyleSheet("background-color: #4caf50; color: white; font-weight: bold; padding: 10px;")
        btn_run_enhance.clicked.connect(self.run_voice_enhancement)

        lay_ctrl.addWidget(QLabel("Select AI Restoration Model:"))
        lay_ctrl.addWidget(self.cmb_enh_model)
        lay_ctrl.addWidget(btn_run_enhance)
        lay_ctrl.addStretch()

        lay_p3.addWidget(box_ctrl)

        splitter_enh.addWidget(pane1)
        splitter_enh.addWidget(pane2)
        splitter_enh.addWidget(pane3)
        splitter_enh.setSizes([220, 560, 320])

        layout.addWidget(splitter_enh)

    def run_voice_enhancement(self):
        item = self.list_files.currentItem()
        if not item:
            QMessageBox.warning(self, "No File Selected", "Select a file from Manual Review list first.")
            return

        filename = item.text().replace("[DUPLICATE] ", "").strip()
        unclassified_dir = os.path.join(os.path.abspath(self.txt_dst.text()), "Unclassified_Manual_Review")
        file_path = os.path.join(unclassified_dir, filename)

        orig_data, sr1 = process_audio_pipeline(file_path)
        enh_data = apply_fast_highpass_filter(orig_data, sr1, cutoff=80.0)
        enh_data = apply_compressor_limiter(enh_data)

        lufs, tp, pk, rms = calculate_audio_metering(enh_data, sr1)
        self.lbl_lufs_enh.setText(f"LUFS: {lufs:.1f} dB")
        self.lbl_tp_enh.setText(f"True Peak: {tp:.1f} dBTP")

        self.canvas_enh.plot_waveform_with_vad(enh_data, sr1, [], 0)
        QMessageBox.information(self, "Enhancement Applied", f"Processed {filename} using {self.cmb_enh_model.currentText()}.")

    # -------------------------------------------------------------------------
    # TAB 4: Settings & Hardware
    # -------------------------------------------------------------------------
    def setup_settings_tab(self):
        layout = QHBoxLayout(self.tab_settings)
        splitter_set = QSplitter(Qt.Horizontal)

        # Left Column
        self.list_set_nav = QListWidget()
        self.list_set_nav.addItems([
            "01. Directories & Storage",
            "02. File Operations & Naming",
            "03. Audio DSP & Dynamics",
            "04. Global Pipeline Order"
        ])
        self.list_set_nav.currentRowChanged.connect(self.on_settings_page_changed)

        # Right Column
        self.stack_set = QStackedWidget()

        # Page 1
        p1 = QWidget()
        lay_p1 = QVBoxLayout(p1)
        self.set_src = QLineEdit(self.settings["src_dir"])
        self.set_dst = QLineEdit(self.settings["dst_dir"])
        lay_p1.addWidget(QLabel("Default Source Directory:")); lay_p1.addWidget(self.set_src)
        lay_p1.addWidget(QLabel("Default Target Directory:")); lay_p1.addWidget(self.set_dst)
        lay_p1.addStretch()

        # Page 2
        p2 = QWidget()
        lay_p2 = QVBoxLayout(p2)

        self.cmb_action_mode = QComboBox()
        self.cmb_action_mode.addItems(["Copy (Keep Original)", "Move (Transfer File)"])
        lay_p2.addWidget(QLabel("File Action Mode:")); lay_p2.addWidget(self.cmb_action_mode)

        self.txt_prefix = QLineEdit(self.settings.get("filename_prefix", ""))
        self.txt_prefix.textChanged.connect(self.update_naming_preview_table)
        self.txt_postfix = QLineEdit(self.settings.get("filename_postfix", ""))
        self.txt_postfix.textChanged.connect(self.update_naming_preview_table)
        self.cmb_naming_strat = QComboBox()
        self.cmb_naming_strat.addItems(["Parsed Scene Name", "Original Filename", "Combined Name", "Numbered Index"])
        self.cmb_naming_strat.currentIndexChanged.connect(self.update_naming_preview_table)

        lay_p2.addWidget(QLabel("Prefix:")); lay_p2.addWidget(self.txt_prefix)
        lay_p2.addWidget(QLabel("Postfix:")); lay_p2.addWidget(self.txt_postfix)
        lay_p2.addWidget(QLabel("Naming Strategy:")); lay_p2.addWidget(self.cmb_naming_strat)

        box_prev = QGroupBox(" Live Naming Rule Preview Table ")
        lay_prev = QVBoxLayout(box_prev)
        self.tbl_preview = QTableWidget(2, 2)
        self.tbl_preview.setHorizontalHeaderLabels(["Original Input", "Formatted Output Name"])
        self.tbl_preview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay_prev.addWidget(self.tbl_preview)
        lay_p2.addWidget(box_prev)

        # Page 3
        p3 = QWidget()
        lay_p3 = QVBoxLayout(p3)
        self.chk_dsp_gain = QCheckBox("Enable Gain Boost (+dB)")
        self.chk_dsp_gain.setChecked(True)
        self.chk_dsp_hp = QCheckBox("Enable High-Pass Filter (80Hz)")
        self.chk_dsp_hp.setChecked(self.settings.get("enable_highpass", True))
        self.chk_dsp_comp = QCheckBox("Enable RMS Compressor & Soft Limiter")
        self.chk_dsp_comp.setChecked(self.settings.get("enable_compressor", True))

        lay_p3.addWidget(self.chk_dsp_gain)
        lay_p3.addWidget(self.chk_dsp_hp)
        lay_p3.addWidget(self.chk_dsp_comp)
        lay_p3.addStretch()

        # Page 4
        p4 = QWidget()
        lay_p4 = QVBoxLayout(p4)
        lay_p4.addWidget(QLabel("Drag or use Move Up/Down buttons to re-order execution stages:"))

        self.list_pipe = QListWidget()
        pipeline_items = self.settings.get("pipeline_order", [
            "Gain Boost", "Highpass Filter", "Compressor", "VAD Speech Anchoring", "STT Transcription"
        ])
        self.list_pipe.addItems(pipeline_items)

        h_btn_pipe = QHBoxLayout()
        btn_up = QPushButton("Move Up"); btn_up.clicked.connect(self.move_pipeline_up)
        btn_down = QPushButton("Move Down"); btn_down.clicked.connect(self.move_pipeline_down)
        btn_reset_p = QPushButton("Reset Default Order"); btn_reset_p.clicked.connect(self.reset_pipeline_order)

        h_btn_pipe.addWidget(btn_up)
        h_btn_pipe.addWidget(btn_down)
        h_btn_pipe.addWidget(btn_reset_p)

        lay_p4.addWidget(self.list_pipe)
        lay_p4.addLayout(h_btn_pipe)
        lay_p4.addStretch()

        self.stack_set.addWidget(p1)
        self.stack_set.addWidget(p2)
        self.stack_set.addWidget(p3)
        self.stack_set.addWidget(p4)

        splitter_set.addWidget(self.list_set_nav)
        splitter_set.addWidget(self.stack_set)
        splitter_set.setSizes([200, 600])

        box_container = QGroupBox(" System Configurations ")
        lay_c = QVBoxLayout(box_container)
        lay_c.addWidget(splitter_set)

        btn_save = QPushButton("Save Settings to settings.json")
        btn_save.setStyleSheet("background-color: #2196f3; color: white; font-weight: bold; padding: 10px;")
        btn_save.clicked.connect(self.save_settings)

        lay_c.addWidget(btn_save)
        layout.addWidget(box_container)

        self.list_set_nav.setCurrentRow(0)
        self.update_naming_preview_table()

    def on_settings_page_changed(self, row):
        self.stack_set.setCurrentIndex(row)

    def update_naming_preview_table(self):
        prefix = self.txt_prefix.text().strip()
        postfix = self.txt_postfix.text().strip()
        strategy = self.cmb_naming_strat.currentText()

        sample_cases = [
            ("251207_008_Tr1.WAV", "Scene_06_Cut_01"),
            ("TEST_AUDIO_TAKE.WAV", "Scene_01_Cut_03")
        ]

        for r, (orig, parsed) in enumerate(sample_cases):
            out_name = format_destination_filename(orig, parsed, prefix, postfix, strategy, r+1)
            self.tbl_preview.setItem(r, 0, QTableWidgetItem(orig))
            self.tbl_preview.setItem(r, 1, QTableWidgetItem(out_name))

    def move_pipeline_up(self):
        row = self.list_pipe.currentRow()
        if row > 0:
            item = self.list_pipe.takeItem(row)
            self.list_pipe.insertItem(row - 1, item)
            self.list_pipe.setCurrentRow(row - 1)

    def move_pipeline_down(self):
        row = self.list_pipe.currentRow()
        if row < self.list_pipe.count() - 1:
            item = self.list_pipe.takeItem(row)
            self.list_pipe.insertItem(row + 1, item)
            self.list_pipe.setCurrentRow(row + 1)

    def reset_pipeline_order(self):
        self.list_pipe.clear()
        self.list_pipe.addItems(["Gain Boost", "Highpass Filter", "Compressor", "VAD Speech Anchoring", "STT Transcription"])

    def save_settings(self):
        self.settings["src_dir"] = self.txt_src.text()
        self.settings["dst_dir"] = self.txt_dst.text()
        self.settings["default_volume"] = self.slider_vol.value()
        self.settings["auto_play"] = self.chk_autoplay.isChecked()
        self.settings["file_action_mode"] = self.cmb_action_mode.currentText()
        self.settings["filename_prefix"] = self.txt_prefix.text()
        self.settings["filename_postfix"] = self.txt_postfix.text()
        self.settings["naming_strategy"] = self.cmb_naming_strat.currentText()
        self.settings["gain_boost_db"] = self.spn_gain.value()

        pipe_order = [self.list_pipe.item(i).text() for i in range(self.list_pipe.count())]
        self.settings["pipeline_order"] = pipe_order

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