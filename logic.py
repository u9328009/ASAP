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
            "highpass_cutoff": 80.0,
            "min_file_duration": 1.0,
            "last_used_preset": "default"
        }
        
        if os.path.exists(setting_path):
            try:
                with open(setting_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default_settings
                
        self.save_app_settings(default_settings)
        return default_settings

    # [글로벌 시스템 설정 파일 저장 / Save global system settings]
    def save_app_settings(self, settings_dict):
        setting_path = self.get_setting_filepath()
        with open(setting_path, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, ensure_ascii=False, indent=4)


    def __init__(self):

        self.slate_margin_ratio = 0.15
        self.sample_rate_target = 16000  
        self.highpass_cutoff = 80  
        self.vad_model = None

        settings = self.load_app_settings()
        self.highpass_cutoff = settings.get("highpass_cutoff", 80.0)
        
    def default_trim_sort_value_changed(self, value):

        print(f"Trim Sort value updated to: {value}")
        self.slate_margin_ratio = value / 100.0

    def prepare_audio_for_vad(self, audio_data, sample_rate):
        # 1. 2채널(스테레오) 이상인 경우 1채널(모노)로 평균 통합
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # 2. 샘플 레이트가 16000Hz가 아니면 16000Hz로 리샘플링
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



     # [선택된 VAD 모델에 맞춰 음성 감지 실행 / Execute voice activity detection with the selected VAD model]
    def run_selected_vad(self, audio_data, sample_rate, vad_type):
        if "FireRed" in vad_type:
            return self.run_firered_vad(audio_data, sample_rate)
        elif "WebRTC" in vad_type:
            return self.run_webrtc_vad(audio_data, sample_rate)
        elif "Silero" in vad_type:
            return self.run_silero_vad(audio_data, sample_rate)

    # [로컬 모델 디렉토리 경로 획득 및 자동 다운로드 / Get local model directory path and download automatically]
    def get_model_directory(self, model_name):
        base_dir = os.path.join(os.getenv("APPDATA"), "asap", "models")
        local_dir = os.path.join(base_dir, model_name)
        os.makedirs(local_dir, exist_ok=True)
        
        # 폴더가 비어 있는 최초 구동 시점에만 자동으로 원격 허브에서 파일을 다운로드
        if not os.listdir(local_dir):
            print(f"Model missing. Downloading {model_name} automatically...")
            from huggingface_hub import snapshot_download
            
            # FireRedVAD 공식 저장소 전체를 지정된 Roaming 폴더 하위에 로컬로 온전하게 다운로드
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

    # [WebRTC VAD 모델 구동 / Run WebRTC VAD model]
    def run_webrtc_vad(self, audio_data, sample_rate):
        import webrtcvad
        import numpy as np

        # 1. 16kHz 모노 데이터로 변환 후 int16 PCM(정수형)으로 스케일 변경
        vad_audio, vad_sr = self.prepare_audio_for_vad(audio_data, sample_rate)
        pcm_data = (vad_audio * 32767).astype(np.int16)
        
        # 2. WebRTC VAD 설정 (감도 범위: 0~3, 3이 가장 엄격하게 노이즈를 제함)
        vad = webrtcvad.Vad(3)
        
        # 3. 30ms 프레임 슬라이싱 수치 계산 (16000Hz * 0.03s = 480샘플)
        frame_ms = 30
        frame_size = int(vad_sr * (frame_ms / 1000.0))
        
        # 4. 프레임별로 루프를 순회하며 음성 감지 상태 제어
        speech_segments = []
        in_speech = False
        speech_start = 0.0
        
        for i in range(0, len(pcm_data) - frame_size, frame_size):
            frame = pcm_data[i : i + frame_size]
            frame_bytes = frame.tobytes()
            
            # WebRTC C++ 엔진에 바이트 데이터를 공급해 음성 여부 판정
            is_speech = vad.is_speech(frame_bytes, vad_sr)
            current_time = i / vad_sr
            
            if is_speech and not in_speech:
                in_speech = True
                speech_start = current_time
            elif not is_speech and in_speech:
                in_speech = False
                speech_segments.append({"start": speech_start, "end": current_time})
                
        # 대사 도중에 파일이 끝나는 경우 예외 세그먼트 마감 처리
        if in_speech:
            speech_segments.append({"start": speech_start, "end": len(pcm_data) / vad_sr})
            
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
        # 1. 대상 경로 아래에 'room_tone' 폴더 경로 조립
        room_tone_dir = os.path.join(dst_dir, "room_tone")
        
        if not os.path.exists(room_tone_dir):
            os.makedirs(room_tone_dir)

        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(room_tone_dir, file_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)