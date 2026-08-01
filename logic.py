import json
import os
import shutil
import soundfile as sf
import numpy as np
import torch
from scipy.signal import resample

class AudioLogicProcessor:
     # [시스템 설정 파일 경로 획득 / Get system settings file path]
    def get_setting_filepath(self):
        base_dir = os.path.join(os.getenv("APPDATA"), "asap")
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, "setting.json")

    # [글로벌 시스템 설정 파일 읽기 / Load global system settings]
    def load_app_settings(self):
        setting_path = self.get_setting_filepath()
        default_settings = {
            "highpass_cutoff": 80.0
        }
        if os.path.exists(setting_path):
            try:
                with open(setting_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default_settings
        return default_settings


    def __init__(self):

        self.slate_margin_ratio = 0.15
        self.sample_rate_target = 16000  
        self.highpass_cutoff = 80  
        self.vad_model = None
        self.stt_model = None
        self.stt_model_canary = None
        self.stt_model_qwen3 = None
        settings = self.load_app_settings()
        self.highpass_cutoff = settings.get("highpass_cutoff", 80.0)
        
    def default_trim_sort_value_changed(self, value):

        print(f"Trim Sort value updated to: {value}")
        self.slate_margin_ratio = value / 100.0

    def prepare_audio_for_vad(self, audio_data, sample_rate):
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        target_sr = 16000
        if sample_rate != target_sr:
            num_target_samples = int(len(audio_data) * target_sr / sample_rate)
            audio_data = resample(audio_data, num_target_samples)
            sample_rate = target_sr
            
        return audio_data, sample_rate

    def load_offline_vad_model(self):
        if self.vad_model is None:
            from silero_vad import load_silero_vad
            self.vad_model = load_silero_vad()
        return self.vad_model

    def apply_highpass(self, audio_data, sample_rate):

        from scipy.signal import butter, lfilter
     
        nyquist = 0.5 * sample_rate
        
        if self.highpass_cutoff <= 0 or self.highpass_cutoff >= nyquist:
            return audio_data

        normalized_cutoff = self.highpass_cutoff / nyquist
        b, a = butter(N=5, Wn=normalized_cutoff, btype='high', analog=False)

        return lfilter(b, a, audio_data)
    
    def preprocess_and_detect(self, file_path):

        data, sr = sf.read(file_path)
        filtered_data = self.apply_highpass(data, sr)
        max_peak = np.max(np.abs(filtered_data))
        if max_peak > 0:
            scaled_data = filtered_data * (0.9 / max_peak)
        else:
            scaled_data = filtered_data
        return scaled_data, sr

    def calculate_effective_voice_ratio(self, speech_segments, total_duration):
        margin_ratio = self.slate_margin_ratio
        start_margin = total_duration * margin_ratio
        end_margin = total_duration * (1.0 - margin_ratio)
        effective_duration = total_duration * (1.0 - (2 * margin_ratio))
        
        if effective_duration <= 0:
            return 0.0
            
        effective_voice_time = 0.0

        for seg in speech_segments:
            valid_start = max(seg["start"], start_margin)
            valid_end = min(seg["end"], end_margin)
            
            if valid_start < valid_end:
                effective_voice_time += (valid_end - valid_start)

        effective_ratio = (effective_voice_time / effective_duration) * 100.0
        return round(effective_ratio, 2)

    # [짧은 파일 여부 검사 / Check if audio file is too short]
    def is_audio_too_short(self, total_duration, min_limit=1.0):
        return total_duration < min_limit

    # [목적지 캐시 파일 경로 획득 / Get destination cache path]
    def get_cache_path(self, dst_dir, file_name):
        cache_dir = os.path.join(dst_dir, "caches")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{file_name}.json")

    # [1대1 JSON 캐시 데이터 저장 / Save 1-to-1 JSON cache data]
    def save_cache_json(self, dst_dir, file_name, cache_data):
        cache_path = self.get_cache_path(dst_dir, file_name)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)

    # [1대1 JSON 캐시 데이터 읽기 / Load 1-to-1 JSON cache data]
    def load_cache_json(self, dst_dir, file_name):
        cache_path = self.get_cache_path(dst_dir, file_name)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def extract_speech_clips(self, audio_data, sample_rate, speech_segments):
        import numpy as np
        
        if not speech_segments:
            return np.array([], dtype=np.float32)
            
        clips = []
        for seg in speech_segments:
            start_sec = max(0.0, seg["start"] - 1.0)
            end_sec = seg["end"] + 1.0
            
            start_idx = int(start_sec * sample_rate)
            end_idx = int(end_sec * sample_rate)
            
            clips.append(audio_data[start_idx:end_idx])
            
        return np.concatenate(clips) if clips else np.array([], dtype=np.float32)

     # [선택된 VAD 모델에 맞춰 음성 감지 실행 / Execute voice activity detection with the selected VAD model]
    def run_selected_vad(self, audio_data, sample_rate, vad_type):
        if "FireRed" in vad_type:
            return self.run_firered_vad(audio_data, sample_rate)
        elif "Silero" in vad_type:
            return self.run_silero_vad(audio_data, sample_rate)

    # [로컬 모델 디렉토리 경로 획득 및 자동 다운로드 / Get local model directory path and download automatically]
    def get_model_directory(self, model_name):
        base_dir = os.path.join(os.getenv("APPDATA"), "asap", "models")
        local_dir = os.path.join(base_dir, model_name)
        os.makedirs(local_dir, exist_ok=True)
        
        if not os.listdir(local_dir):
            print(f"Model missing. Downloading {model_name} automatically...")
            from huggingface_hub import snapshot_download
              
            snapshot_download(repo_id="FireRedTeam/FireRedVAD", local_dir=local_dir)
            
        return local_dir


    # [FireRed VAD 모델 구동 및 음성 구간 추출 / Run FireRed VAD model and extract speech segments]
    def run_firered_vad(self, audio_data, sample_rate):
        from fireredvad import FireRedVad, FireRedVadConfig
        
        vad_audio, vad_sr = self.prepare_audio_for_vad(audio_data, sample_rate)
        
        config = FireRedVadConfig(
            use_gpu=False,
            speech_threshold=0.4,
            min_speech_frame=20
        )
        
        # 1. 자동 다운로드 기능이 결합된 모델 디렉토리 경로 획득 (VAD 하위 폴더 자동 매핑)
        local_base_dir = self.get_model_directory("FireRedVAD")
        local_model_dir = os.path.join(local_base_dir, "VAD")
        
        # 2. 로드 및 연산 수행
        vad_model = FireRedVad.from_pretrained(local_model_dir, config)
        
        audio_input = (vad_sr, vad_audio)
        result, _ = vad_model.detect(audio_input)
        
        speech_segments = []
        raw_segments = result.get("event2timestamps", {}).get("speech", [])
        
        for start, end in raw_segments:
            speech_segments.append({"start": start, "end": end})
            
        return speech_segments
    
    # [silero VAD 모델 구동 / Run silero VAD model]
    def run_silero_vad(self, audio_data, sample_rate):
        try:
           
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
                
           
            target_sr = 16000
            if sample_rate != target_sr:
                num_samples = int(len(audio_data) * target_sr / sample_rate)
                audio_data = resample(audio_data, num_samples)
                
          
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val

            tensor_audio = torch.from_numpy(audio_data).float()

            from silero_vad import load_silero_vad, get_speech_timestamps
            model = self.load_offline_vad_model()
            
            speech_timestamps = get_speech_timestamps(
                tensor_audio,
                model,
                sampling_rate=target_sr,
                return_seconds=True
            )
            return speech_timestamps

        except Exception as e:
            
            print(f"[DEBUG VAD ERROR] {str(e)}")
            return []

    @staticmethod
    def copy_to_room_tone(file_name, src_dir, dst_dir):
        base_dir = os.path.join(os.getenv("APPDATA"), "asap")
        os.makedirs(base_dir, exist_ok=True)
        settings_path = os.path.join(base_dir, "setting.json")
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                room_tone_folder_name = settings.get("general_room_tone_folder_name", "room_tone")
        
        room_tone_dir = os.path.join(dst_dir, room_tone_folder_name)
        
        if not os.path.exists(room_tone_dir):
            os.makedirs(room_tone_dir)

        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(room_tone_dir, file_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
    @staticmethod
    def copy_to_classified(file_name, src_dir, dst_dir):
        base_dir = os.path.join(os.getenv("APPDATA"), "asap")
        os.makedirs(base_dir, exist_ok=True)
        settings_path = os.path.join(base_dir, "setting.json")
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                classified_folder_name = settings.get("general_classified_folder_name", "classified")
        classified_dir = os.path.join(dst_dir, classified_folder_name)
        
        if not os.path.exists(classified_dir):
            os.makedirs(classified_dir)

        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(classified_dir, file_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    @staticmethod
    def copy_to_short_duration(file_name, src_dir, dst_dir):
        
        base_dir = os.path.join(os.getenv("APPDATA"), "asap")
        os.makedirs(base_dir, exist_ok=True)
        settings_path = os.path.join(base_dir, "setting.json")
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
                short_duration_folder_name = settings.get("general_short_duration_folder_name", "short_duration")
        short_duration_dir = os.path.join(dst_dir, short_duration_folder_name)

        if not os.path.exists(short_duration_dir):
            os.makedirs(short_duration_dir)


        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(short_duration_dir, file_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    # [선택된 STT 모델 및 언어 설정에 맞춰 대사 추출 실행 / Execute speech-to-text with the selected model and language]
    def run_selected_stt(self, audio_data, stt_type, stt_lang):
        lang_code = stt_lang[:2].lower()
        if lang_code == "au":
            lang_code = None
        elif lang_code == "kr":
            lang_code = "ko"
            
        if "Whisper tiny" in stt_type:
            return self.run_whisper_tiny_stt(audio_data, lang_code)
        elif "Whisper base" in stt_type:
            return self.run_whisper_base_stt(audio_data, lang_code)
        elif "Whisper large-v3-turbo" in stt_type:
            return self.run_whisper_large_v3_turbo_stt(audio_data, lang_code)
        elif "Whisper large-v3" in stt_type:
            return self.run_whisper_large_v3_stt(audio_data, lang_code)
        elif "Canary-Qwen" in stt_type:
            return self.run_canary_qwen_stt(audio_data, lang_code)
        elif "Qwen3" in stt_type:
            return self.run_qwen3_stt(audio_data, lang_code)
        return ""


    # [로컬 Whisper tiny 모델 구동 및 대사 추출 / Run offline Whisper tiny model and transcribe text]
    def run_whisper_tiny_stt(self, audio_data, lang_code):
        from faster_whisper import WhisperModel
        import torch
        base_dir = os.path.join(os.getenv("APPDATA"), "asap", "models")
        local_dir = os.path.join(base_dir, "Whisper")
        model_path = os.path.join(local_dir, "tiny")
        os.makedirs(model_path, exist_ok=True)
        transcription = ""
        if self.stt_model is None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device_type == "cuda" else "float32"
            self.stt_model = WhisperModel(model_path, device=device_type, compute_type=compute_type,cache_dir = base_dir)
        segments, _ = self.stt_model.transcribe(audio_data, language=lang_code, beam_size=5)
        for segment in segments:
            transcription += segment.text
        return transcription.strip()

    # [로컬 Whisper base 모델 구동 및 대사 추출 / Run offline Whisper base model and transcribe text]
    def run_whisper_base_stt(self, audio_data, lang_code):
        from faster_whisper import WhisperModel
        import torch
        base_dir = os.path.join(os.getenv("APPDATA"), "asap", "models")
        local_dir = os.path.join(base_dir, "Whisper")
        model_path = os.path.join(local_dir, "base")
        os.makedirs(model_path, exist_ok=True)
        transcription = ""
        if self.stt_model is None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device_type == "cuda" else "float32"
            self.stt_model = WhisperModel("base", device=device_type, compute_type=compute_type, download_root=local_dir)
        segments, _ = self.stt_model.transcribe(audio_data, language=lang_code, beam_size=5)
        for segment in segments:
            transcription += segment.text
        return transcription.strip()
    
    # [로컬 Whisper large v3 모델 구동 및 대사 추출 / Run offline Whisper large v3 model and transcribe text]
    def run_whisper_large_v3_stt(self, audio_data, lang_code):
        from faster_whisper import WhisperModel
        import torch
        base_dir = os.path.join(os.getenv("APPDATA"), "asap", "models")
        local_dir = os.path.join(base_dir, "Whisper")
        model_path = os.path.join(local_dir, "large-v3")
        os.makedirs(model_path, exist_ok=True)
        transcription = ""
        if self.stt_model is None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device_type == "cuda" else "float32"
            self.stt_model = WhisperModel("large-v3", device=device_type, compute_type=compute_type, download_root=local_dir)
        segments, _ = self.stt_model.transcribe(audio_data, language=lang_code, beam_size=5)
        for segment in segments:
            transcription += segment.text
        return transcription.strip()
    
    # [로컬 Whisper large v3 turbo 모델 구동 및 대사 추출 / Run offline Whisper large v3 turbo`` model and transcribe text]
    def run_whisper_large_v3_turbo_stt(self, audio_data, lang_code):
        from faster_whisper import WhisperModel
        import torch
        base_dir = os.path.join(os.getenv("APPDATA"), "asap", "models")
        local_dir = os.path.join(base_dir, "Whisper")
        model_path = os.path.join(local_dir, "large-v3-turbo")
        transcription = ""
        if self.stt_model is None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device_type == "cuda" else "float32"
            self.stt_model = WhisperModel("large-v3-turbo", device=device_type, compute_type=compute_type, download_root=local_dir)
        segments, _ = self.stt_model.transcribe(audio_data, language=lang_code, beam_size=5)
        for segment in segments:
            transcription += segment.text
        return transcription.strip()

    # [Canary-Qwen 2.5B 모델 구동 / Run Canary-Qwen 2.5B model]
    def run_canary_qwen_stt(self, audio_data, lang_code):
        from transformers import pipeline
        import torch

        model_dir = self.get_model_directory("Canary-Qwen-2.5B")
        
        if self.stt_model_canary is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.stt_model_canary = pipeline(
                "automatic-speech-recognition",
                model="Qwen/Qwen2-Audio-7B-Instruct",
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        generated_kwargs = {"generated_kwargs" :{"language": lang_code} if lang_code else {}}

        result = self.stt_model_canary(audio_data, **generated_kwargs)
        return result["text"].strip()


    # [Qwen3-ASR 1.7B 모델 구동 / Run Qwen3-ASR 1.7B model]
    def run_qwen3_stt(self, audio_data, lang_code):
        from transformers import pipeline
        import torch

        model_dir = self.get_model_directory("Qwen3-ASR-1.7B")

        if self.stt_model_qwen3 is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.stt_model_qwen3 = pipeline(
                "automatic-speech-recognition",
                model="Qwen/Qwen2-Audio-Instruct",
                device=device,
                model_kwargs={"local_files_only": True} if os.listdir(model_dir) else {}
            )

        generated_kwargs = {"generated_kwargs" :{"language": lang_code} if lang_code else {}}
        result = self.stt_model_qwen3(audio_data, **generated_kwargs)
        return result["text"].strip()