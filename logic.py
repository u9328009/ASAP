import os
import re
import json
import time
import sys
import numpy as np
import soundfile as sf
import scipy.signal
import torch
import torchaudio.functional as F
from faster_whisper import WhisperModel

# Suppress HuggingFace Symlink Warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# =============================================================================
# Absolute Path Anchoring
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
DEFAULT_CACHE_DIR = os.path.join(BASE_DIR, "caches")

DEFAULT_SETTINGS = {
    "src_dir": os.path.join(BASE_DIR, "my_recordings"),
    "dst_dir": os.path.join(BASE_DIR, "classified_result"),
    "min_dur_thresh": 5.0,
    "crop_pct": 15.0,
    "speech_thresh_pct": 2.0,
    "vad_model_choice": "Silero VAD v5 (Recommended)",
    "stt_model_choice": "large-v3-turbo (Recommended High Accuracy)",
    "stt_lang_choice": "ko",
    "device_choice": "Auto (GPU Priority)",
    "vad_thresh": 0.25,
    "default_volume_db": 0.0,
    "auto_play": False,
    "file_action_mode": "Copy (Keep Original)",
    "filename_prefix": "",
    "filename_postfix": "",
    "naming_strategy": "Parsed Scene Name",
    "gain_boost_db": 0.0,
    "stt_prompt_enabled": True,
    "stt_initial_prompt": "현장 동시녹음 슬레이트 콜: 씬, 컷, 테이크, Scene, Cut, Take.",
    "pipeline_order": ["Gain Boost", "Soft Highpass (40Hz)", "VAD Speech Anchoring (+1s Pad)", "STT Transcription"]
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

# Global AI Model Singletons
vad_model = None
stt_model = None
current_stt_model_size = None
current_device = None

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
# Audio Preprocessing & Pipeline
# =============================================================================
def apply_gain_boost(data, gain_db):
    if len(data) == 0 or gain_db == 0.0: return data
    gain_factor = 10 ** (gain_db / 20.0)
    return (data * gain_factor).astype(np.float32)

def apply_soft_highpass_filter(data, sr, cutoff=40.0):
    if len(data) == 0: return data
    try:
        b, a = scipy.signal.butter(2, cutoff / (sr / 2.0), btype='high')
        return scipy.signal.filtfilt(b, a, data).astype(np.float32)
    except Exception:
        return data

def process_audio_pipeline(file_path, gain_db=0.0):
    abs_file_path = os.path.abspath(file_path)
    data, sr = sf.read(abs_file_path)

    if data.ndim > 1:
        data = data.mean(axis=1)

    data = apply_soft_highpass_filter(data, sr, cutoff=40.0)
    data = apply_gain_boost(data, gain_db)

    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = (data / max_val * 0.95).astype(np.float32)

    return data, sr

def prepare_tensor_for_vad(audio_data, sr, target_sr=16000, device="cpu"):
    tensor = torch.from_numpy(audio_data).float()
    if sr != target_sr:
        tensor = F.resample(tensor, sr, target_sr)
    if device == "cuda":
        tensor = tensor.to("cuda")
    return tensor

def get_padded_speech_timestamps(wav_tensor, vad_model, sr=16000, pad_sec=1.0, threshold=0.25):
    from silero_vad import get_speech_timestamps
    raw_spans = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=sr, threshold=threshold)
    if not raw_spans:
        return []

    total_sec = wav_tensor.shape[-1] / float(sr)
    padded_spans = []

    for seg in raw_spans:
        start_s = max(0.0, (seg['start'] / float(sr)) - pad_sec)
        end_s = min(total_sec, (seg['end'] / float(sr)) + pad_sec)

        if not padded_spans:
            padded_spans.append({'start': round(start_s, 2), 'end': round(end_s, 2)})
        else:
            last = padded_spans[-1]
            if start_s <= last['end']:
                last['end'] = round(max(last['end'], end_s), 2)
            else:
                padded_spans.append({'start': round(start_s, 2), 'end': round(end_s, 2)})

    return padded_spans


# =============================================================================
# Slate Parsing & Normalization
# =============================================================================
def refine_transcribed_slate_text(text):
    if not text: return ""

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


# Cache Manager with Absolute Pathing
class CacheManager:
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR
        self.cache_dir = os.path.abspath(cache_dir)
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

cache_mgr = CacheManager(DEFAULT_CACHE_DIR)


# =============================================================================
# STT Pipeline Execution
# =============================================================================
def run_stt_no_hallucination(file_path, lang_choice="ko", model_size="large-v3-turbo", device_choice="Auto"):
    settings = load_settings()
    load_ai_models(model_size=model_size, device_choice=device_choice)
    audio_data, sr = process_audio_pipeline(os.path.abspath(file_path), gain_db=settings.get("gain_boost_db", 0.0))

    target_dev = resolve_device(device_choice)
    wav_tensor = prepare_tensor_for_vad(audio_data, sr, device=target_dev)

    padded_spans = get_padded_speech_timestamps(wav_tensor, vad_model, sr=16000, pad_sec=1.0)

    if padded_spans:
        first_start = padded_spans[0]['start']
        start_idx = int(first_start * sr)
        end_idx = min(len(audio_data), int((first_start + 8.0) * sr))
        stt_input = audio_data[start_idx:end_idx]
    else:
        stt_input = audio_data[:int(15 * sr)]

    transcribe_kwargs = {
        "language": lang_choice if lang_choice != "auto" else None,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.5,
        "beam_size": 5
    }

    if settings.get("stt_prompt_enabled", True):
        prompt_text = settings.get("stt_initial_prompt", "").strip()
        if prompt_text:
            transcribe_kwargs["initial_prompt"] = prompt_text

    try:
        try:
            segments, _ = stt_model.transcribe(stt_input, hotwords="씬 컷 테이크 scene cut take", **transcribe_kwargs)
        except TypeError:
            segments, _ = stt_model.transcribe(stt_input, **transcribe_kwargs)

        raw_text = "".join([s.text for s in segments]).strip()
        refined_text = refine_transcribed_slate_text(raw_text)
        return refined_text if refined_text else "(No speech recognized)"
    except Exception as e:
        return f"[ERROR] STT failed: {e}"