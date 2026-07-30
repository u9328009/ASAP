import os
import json
import time
from datetime import datetime 
import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import QModelIndex, Qt , QThread, Signal
import logic
import soundfile as sf
processor = logic.AudioLogicProcessor()


def format_size(size_in_bytes):
    if size_in_bytes < 1024:
        return f"{size_in_bytes} Bytes"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    elif size_in_bytes < 1024 * 1024 * 1024:
        return f"{size_in_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_in_bytes / (1024 * 1024 * 1024):.1f} GB"

# [오디오 정렬 작업용 백그라운드 스레드 / Background thread for audio sorting operations]
class AudioSortWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(str, str) 
    progress_signal = Signal(int)
    finished_signal = Signal(int, float)

    # [작업자 인스턴스 초기화 / Initialize worker instance]
    def __init__(self, selected_files, src_dir, dst_dir, processor, threshold, vad_type, stt_type):
        super().__init__()
        self.selected_files = selected_files
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.processor = processor
        self.vad_type = vad_type
        self.threshold = threshold
        self.stt_type = stt_type

    # [스레드 실제 실행 본체 / Run actual thread body]
    def run(self):
        import time
        start_time = time.time()
        roomtone_count = 0
        total_count = len(self.selected_files)
        
        for index, file_name in enumerate(self.selected_files):
            full_path = os.path.join(self.src_dir, file_name)
            total_duration = sf.info(full_path).duration

            if self.processor.is_audio_too_short(total_duration, min_limit=1.0):
                cache_data = {"file_name": file_name, "status": "Too Short"}
                self.processor.save_cache_json(self.dst_dir, file_name, cache_data)
                self.status_signal.emit(file_name, "Too Short")
                self.log_signal.emit(f"{file_name} SHORT DURATION")
                continue

            scaled_data, sr = self.processor.preprocess_and_detect(full_path)
            speech_segments = self.processor.run_selected_vad(scaled_data, sr, self.vad_type)
            voice_ratio = self.processor.calculate_effective_voice_ratio(speech_segments, total_duration)
            
            has_voice = voice_ratio >= self.threshold
            status_str = "Classified" if has_voice else "Room Tone"
            stt_text = "" 

            if has_voice:
                speech_clip = self.processor.extract_speech_clips(scaled_data, sr, speech_segments)
                
                if len(speech_clip) > 0:
                    stt_text = self.processor.run_selected_stt(speech_clip, self.stt_type)

            cache_data = {
                "file_name": file_name,
                "total_duration": total_duration,
                "speech_segments": speech_segments,
                "voice_ratio": voice_ratio,
                "has_voice": has_voice,
                "status": status_str,
                "stt_text": stt_text
            }
            self.processor.save_cache_json(self.dst_dir, file_name, cache_data)

            if not has_voice:
                self.processor.copy_to_room_tone(file_name, self.src_dir, self.dst_dir)
                roomtone_count += 1
                log_msg = f"{file_name} VAD {voice_ratio}% ROOMTONE"
            else:
                log_msg = f"{file_name} VAD {voice_ratio}% STT result: {stt_text}"

            self.status_signal.emit(file_name, status_str)
            self.log_signal.emit(log_msg)
            self.progress_signal.emit(int(((index + 1) / total_count) * 100))
            
        elapsed_time = time.time() - start_time
        self.finished_signal.emit(roomtone_count, elapsed_time)


# [기본 디렉토리 및 설정 파일 초기화 / Initialize base directories and config files]
def initialize_asap_environment():
    appdata_path = os.getenv("APPDATA")
    base_dir = os.path.join(appdata_path, "asap")
    
    preset_dir = os.path.join(base_dir, "preset")
    caches_dir = os.path.join(base_dir, "caches")

    os.makedirs(preset_dir, exist_ok=True)
    os.makedirs(caches_dir, exist_ok=True)

    setting_file = os.path.join(base_dir, "setting.json")
    if not os.path.exists(setting_file):
        default_settings = {
            "highpass_cutoff": 40.0,
            "last_used_preset": "default"
        }
        with open(setting_file, "w", encoding="utf-8") as f:
            json.dump(default_settings, f, ensure_ascii=False, indent=4)

    default_preset_file = os.path.join(preset_dir, "default.json")
    if not os.path.exists(default_preset_file):
        default_preset = {
            "cut_margin": 15,
            "voice_detect": 2.0,
            "vad_model": "silero-vad",
            "stt_model": "Whisper"
        }
        with open(default_preset_file, "w", encoding="utf-8") as f:
            json.dump(default_preset, f, ensure_ascii=False, indent=4)

#---------------------------------------------------------------------------------------------------------------------------------------
#audio sort tab

# 폴더 로더에 사용하는 트리뷰 모델 생성 및 필터링 folder loader model creation and filtering
tree_file = QStandardItemModel()
tree_file.setHorizontalHeaderLabels(["File Name", "Status"])
ALLOWED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".alac", ".aiff", ".opus")

# 폴더 로드  folder loader 
def load_folder():
    folder_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder")
    if folder_path:
        print("Selected folder:", folder_path)
        window.txt_src.setText(folder_path)
        load_treeview(folder_path)
        return folder_path
    else:
        return None

# [트리 뷰 상태 칸 텍스트 갱신 / Update status column text in tree view]
def update_item_status(file_name, status_text):
    for row in range(tree_file.rowCount()):
        item = tree_file.item(row, 0)
        if item.text() == file_name:
            # Duration 열 추가로 인해 Status 위치가 4번 열로 변경됨
            tree_file.item(row, 4).setText(status_text)
            break

# [트리 뷰 내용 로드 및 오디오 시간/캐시 상태 복원 / Load contents, duration, and restore cache status into tree view]
def load_treeview(folder_path):
    if not folder_path:
        return
        
    dst_dir = window.txt_dst.text()
    tree_file.clear()
    tree_file.setHorizontalHeaderLabels(["File Name", "Duration", "File Date", "File Size", "Status"])
    
    try:
        files = os.listdir(folder_path)
    except Exception as e:
        return
        
    for file_name in files:
        if file_name.lower().endswith(ALLOWED_EXTENSIONS):
            full_path = os.path.join(folder_path, file_name)
            
            file_item = QStandardItem(file_name)
            file_item.setCheckable(True)
            file_item.setCheckState(Qt.CheckState.Checked)
            
            duration_sec = sf.info(full_path).duration
            mins, secs = int(duration_sec // 60), int(duration_sec % 60)
            duration_str = f"{mins:02d}:{secs:02d}"
            
            file_size_str = format_size(os.path.getsize(full_path))
            dt = datetime.fromtimestamp(os.path.getmtime(full_path))
            hour_12 = dt.hour % 12 or 12
            am_pm = "AM" if dt.hour < 12 else "PM"
            file_date_str = f"{dt.strftime('%Y.%m.%d')} {am_pm} {hour_12}:{dt.strftime('%M:%S')}"
            
            cache_data = processor.load_cache_json(dst_dir, file_name)
            status_text = cache_data.get("status", "Ready") if cache_data else "Ready"
            
            status_item = QStandardItem(status_text)

            if status_text != "Ready":
                file_item.setCheckable(False)
                file_item.setCheckState(Qt.CheckState.Unchecked)

            tree_file.appendRow([
                file_item, 
                QStandardItem(duration_str), 
                QStandardItem(file_date_str), 
                QStandardItem(file_size_str), 
                status_item
            ])
            
    window.TreeViewFiles.setModel(tree_file)
    window.TreeViewFiles.resizeColumnToContents(0)
    window.TreeViewFiles.resizeColumnToContents(1)
    window.TreeViewFiles.resizeColumnToContents(2)
    window.TreeViewFiles.resizeColumnToContents(3)
    window.TreeViewFiles.resizeColumnToContents(4)

# 트리뷰 필터링  treeview filtering
def filter_treeview_by_name(search_text):
    query = search_text.lower() 
    for row in range(tree_file.rowCount()):
        item = tree_file.item(row, 0)
        is_match = query in item.text().lower()
        window.TreeViewFiles.setRowHidden(row, QModelIndex(), not is_match)

def load_export_folder():
    folder_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder")
    if folder_path:
        print("Selected folder:", folder_path)
        window.txt_dst.setText(folder_path)
        
        # 소스 폴더가 이미 채워져 있다면 캐시 상태를 즉시 복원하도록 트리 뷰 갱신
        src_dir = window.txt_src.text()
        if src_dir:
            load_treeview(src_dir)
            
        return folder_path
    else:
        return None

# [보이는 파일만 전체 선택 / Select all visible files]
def select_all_visible():
    for row in range(tree_file.rowCount()):
        if not window.TreeViewFiles.isRowHidden(row, QModelIndex()):
            tree_file.item(row, 0).setCheckState(Qt.CheckState.Checked)

# [보이는 파일만 전체 해제 / Unselect all visible files]
def unselect_all_visible():
    for row in range(tree_file.rowCount()):
        if not window.TreeViewFiles.isRowHidden(row, QModelIndex()):
            tree_file.item(row, 0).setCheckState(Qt.CheckState.Unchecked)

# [선택된 오디오 파일 목록 수집 / Collect selected audio files]
def get_checked_audio_files():
    checked_files = []
    for row in range(tree_file.rowCount()):
        item = tree_file.item(row, 0)
        if item.checkState() == Qt.CheckState.Checked:
            checked_files.append(item.text())
            
    return checked_files

# [정렬 전체 공정 시작 처리 / Start the entire sorting process]
def start_sorting_process():
    src_dir = window.txt_src.text()
    dst_dir = window.txt_dst.text()
    selected_files = get_checked_audio_files()
    vad_type = window.combo_vad.currentText()
    stt_type = window.combo_stt.currentText()

    if not selected_files:
        write_log("[Warning] No files selected.")
        return

    threshold = window.doublespin_VocDect.value()
    window.AudioSortProgress.setValue(0)
    window.AudioSortStart.setEnabled(False)

    global worker
    worker = AudioSortWorker(selected_files, src_dir, dst_dir, processor, threshold, vad_type, stt_type)
    
    worker.log_signal.connect(write_log)
    worker.status_signal.connect(update_item_status)  # UI Status 칸 실시간 업데이트 연결
    worker.progress_signal.connect(window.AudioSortProgress.setValue)
    worker.finished_signal.connect(on_sorting_finished)
    
    worker.start()


# [모든 작업 완료 시 UI 원복 처리 / Handle complete sorting process]
def on_sorting_finished(roomtone_count, elapsed_time):
    window.AudioSortStart.setEnabled(True)
    
    src_dir = window.txt_src.text()
    load_treeview(src_dir)
    
    mins = int(elapsed_time // 60)
    secs = int(elapsed_time % 60)
    time_str = f"{mins:02d}:{secs:02d}"
    
    total_files = len(get_checked_audio_files())
    
    write_log("COMPLETED")
    write_log(f"TOTAL FILE {total_files} ROOMTONE {roomtone_count} TOTAL TIME {time_str}")

# [모든 파일 상태 초기화 / Reset status of all files]
def reset_status_of_all_files():
    if window.txt_dst.text() == "":
        write_log("[Warning] Please select a destination folder first.")
        return
    else:
       dist_dir = window.txt_dst.text()
       cache_dir = os.path.join(dist_dir, "caches")
       for filename in os.listdir(cache_dir):
           if filename.endswith(".json"):
               os.remove(os.path.join(cache_dir, filename))
       load_treeview(window.txt_src.text())

        
    

# [단일 파일 가공 및 상태 반영 / Process single file and reflect status]
def process_audio_with_cache(file_name, src_dir, dst_dir):
    full_path = os.path.join(src_dir, file_name)
    total_duration = sf.info(full_path).duration
    
    # 1. 짧은 파일 처리 및 캐시 저장
    if processor.is_audio_too_short(total_duration, min_limit=1.0):
        cache_data = {"file_name": file_name, "status": "Too Short"}
        processor.save_cache_json(dst_dir, file_name, cache_data)
        update_item_status(file_name, "Too Short")
        write_log(f"[Skip] File too short: {file_name}")
        return

    # 2. VAD 연산
    scaled_data, sr = processor.preprocess_and_detect(full_path)
    speech_segments = processor.run_silero_vad(scaled_data, sr)
    voice_ratio = processor.calculate_effective_voice_ratio(speech_segments, total_duration)
    
    threshold = window.doublespin_VocDect.value()
    has_voice = voice_ratio >= threshold
    status_str = "Classified" if has_voice else "Room Tone"

    # 3. 1대1 JSON 캐시 무조건 저장 (dst_dir/caches/ 폴더 생성됨)
    cache_data = {
        "file_name": file_name,
        "total_duration": total_duration,
        "speech_segments": speech_segments,
        "voice_ratio": voice_ratio,
        "has_voice": has_voice,
        "status": status_str,
        "stt_text": ""
    }
    processor.save_cache_json(dst_dir, file_name, cache_data)

    # 4. 공음 분류 시 복사 및 상태(Status) 갱신
    if not has_voice:
        processor.copy_to_room_tone(file_name, src_dir, dst_dir)
        update_item_status(file_name, "Room Tone")
        write_log(f"[Classified] Copied to Room Tone: {file_name}")
    else:
        update_item_status(file_name, "Classified")

def write_log(message):
    current_time = datetime.now().strftime("[%H:%M:%S]")
    formatted_log = f"{current_time} {message}"

    window.audiosortLog.appendPlainText(formatted_log)

loader = QUiLoader()
app = QtWidgets.QApplication(sys.argv)
window = loader.load("main.ui", None)

window.btn_src.clicked.connect(load_folder)
window.btn_dst.clicked.connect(load_export_folder)
window.txt_src.returnPressed.connect(lambda: load_treeview(window.txt_src.text()))
window.txt_dst.returnPressed.connect(lambda: load_export_folder)

window.btn_Selall.clicked.connect(select_all_visible)
window.btn_Unselall.clicked.connect(unselect_all_visible)
window.btn_Resetstat.clicked.connect(reset_status_of_all_files)
window.btn_find.clicked.connect(lambda: filter_treeview_by_name(window.txt_find.text()))
window.txt_find.textChanged.connect(lambda: filter_treeview_by_name(window.txt_find.text()))
window.btn_viewall.clicked.connect(lambda: filter_treeview_by_name(""))
window.spin_TrimSort.valueChanged.connect(processor.default_trim_sort_value_changed)
window.AudioSortStart.clicked.connect(start_sorting_process)


window.show()
app.exec()