import os
import json
import time
from datetime import datetime 
import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import QModelIndex, Qt , QThread, Signal
from moderngl_window import window
import logic
import soundfile as sf
import pyqtgraph as pg
import numpy as np

processor = logic.AudioLogicProcessor()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 

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
    # (총 파일 수, 룸톤 수, 짧은 파일 수, 소요 시간) 시그널 전달
    finished_signal = Signal(int, int, int, int, float)

    # [작업자 초기화 / Initialize worker]
    def __init__(self, selected_files, src_dir, dst_dir, processor, threshold, vad_type, stt_type, stt_lan):
        super().__init__() 
        
        self.selected_files = selected_files
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.processor = processor
        self.threshold = threshold
        self.vad_type = vad_type
        self.stt_type = stt_type
        self.stt_lan = stt_lan
        
    # [스레드 실제 실행 본체 / Run actual thread body]

    def run(self):
        import time
        start_time = time.time()
        roomtone_count = 0
        short_count = 0
        classified_count = 0
        total_count = len(self.selected_files)
        
        for index, file_name in enumerate(self.selected_files):
            full_path = os.path.join(self.src_dir, file_name)
            total_duration = sf.info(full_path).duration

            if self.processor.is_audio_too_short(total_duration, min_limit=1.0):
                cache_data = {"file_name": file_name, "status": "Too Short"}
                self.processor.save_cache_json(self.dst_dir, file_name, cache_data)
                self.status_signal.emit(file_name, "Too Short")
                self.log_signal.emit(f"{file_name} SHORT DURATION")
                self.processor.copy_to_short_duration(file_name, self.src_dir, self.dst_dir)
                short_count += 1
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
                    stt_text = self.processor.run_selected_stt(speech_clip, self.stt_type, self.stt_lan)

            cache_data = {
                "file_name": file_name,
                "total_duration": total_duration,
                "speech_segments": speech_segments,
                "voice_ratio": voice_ratio,
                "has_voice": has_voice,
                "status": status_str,
                "stt_text": stt_text,
                "vad_type": self.vad_type,
                "stt_type": self.stt_type,
                "stt_lan": self.stt_lan
            }
            self.processor.save_cache_json(self.dst_dir, file_name, cache_data)

            if not has_voice:
                self.processor.copy_to_room_tone(file_name, self.src_dir, self.dst_dir)
                roomtone_count += 1
                log_msg = f"{file_name} VAD {voice_ratio}% ROOMTONE"
            else:
                self.processor.copy_to_classified(file_name, self.src_dir, self.dst_dir)
                classified_count += 1
                log_msg = f"{file_name} VAD {voice_ratio}% STT result: {stt_text}"
                

            self.status_signal.emit(file_name, status_str)
            self.log_signal.emit(log_msg)
            self.progress_signal.emit(int(((index + 1) / total_count) * 100))
            
        elapsed_time = time.time() - start_time
        self.finished_signal.emit(total_count, roomtone_count, short_count, classified_count, elapsed_time)


# [모든 작업 완료 처리 / Handle complete sorting process]
def on_sorting_finished(total_count, roomtone_count, short_count, classified_count, elapsed_time):
    window.btn_sort_start.setEnabled(True)
    
    src_dir = window.txt_src.text()
    load_treeview(src_dir)
    
    mins = int(elapsed_time // 60)
    secs = int(elapsed_time % 60)
    time_str = f"{mins:02d}:{secs:02d}"
    
    write_log("COMPLETED")
    write_log(f"TOTAL FILE {total_count} ROOMTONE {roomtone_count} SHORT {short_count} CLASSIFIED {classified_count} TOTAL TIME {time_str},EVG FILE TIME {elapsed_time/total_count:.2f} sec")


# [기본 디렉토리 및 설정 파일 초기화 / Initialize base directories and config files]
def initialize_ASAP_environment():
    appdata_path = os.getenv("APPDATA")
    base_dir = os.path.join(appdata_path, "ASAP")
    
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


# [설정 파일 로드 및 저장 / Load and save settings]
def load_settings():
    base_dir = os.path.join(os.getenv("APPDATA"), "ASAP")
    setting_file = os.path.join(base_dir, "setting.json")
    if not os.path.exists(setting_file):
        default_settings = {
            "highpass":"80",
            "file_prefix":"",
            "file_suffix":"", 
            "file_name":"",
            "custom_file_name":"",
            "copyrighter":"",
            "overwrite_existing":"Overwrite",
            "backup_source":"Do not Backup",
            "general_source_folder":"",
            "general_export_folder":"",
            "general_backup_folder":"",
            "general_room_tone_folder_name":"01_RoomTone",
            "general_classified_folder_name":"00_Classified",
            "general_shortduration_folder_name":"02_ShortClips",
            "general_folder_Structure":".Class/scene SN/Cut CN/Take TN/File",
            "interface_language":"English",
            "theme":"system Default",
            "basic_font":"pretendard",
            "basic_font_size":11
        }
        with open(setting_file, "w", encoding="utf-8") as f:
            json.dump(default_settings, f, ensure_ascii=False, indent=4)
    with open(setting_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_settings(settings):
    base_dir = os.path.join(os.getenv("APPDATA"), "ASAP")
    setting_file = os.path.join(base_dir, "setting.json")
    with open(setting_file, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)

# [프리셋 로드 및 저장 / Load and save presets]
def load_preset(preset_name):
    base_dir = os.path.join(os.getenv("APPDATA"), "ASAP")
    preset_dir = os.path.join(base_dir, "preset")
    os.makedirs(preset_dir, exist_ok=True)
    if not os.path.exists(os.path.join(preset_dir, "default.json")):
        save_preset("default", {
        "cut_margin": 15,
        "voice_detect": 2.0,
        "vad_model": "silero (Balenced)",
        "stt_model": "Whisper",
        "stt_language": "Auto"
        })
    if not os.path.exists(os.path.join(preset_dir, f"{preset_name}.json")):
        return None

    preset_file = os.path.join(preset_dir, f"{preset_name}.json")
    with open(preset_file, "r", encoding="utf-8") as f:
        preset_data = json.load(f)
        window.spin_vad_trim.setValue(preset_data.get("cut_margin","15"))
        window.doublespin_VocDect.setValue(preset_data.get("voice_detect","2.0"))
        window.combo_vad.setCurrentText(preset_data.get("vad_model","silero (Balenced)"))
        window.combo_stt.setCurrentText(preset_data.get("stt_model","Whisper"))
        window.combo_stt_lan.setCurrentText(preset_data.get("stt_language","Auto"))

    

def save_preset(preset_name, preset_data):
    base_dir = os.path.join(os.getenv("APPDATA"), "ASAP")
    preset_dir = os.path.join(base_dir, "preset")
    os.makedirs(preset_dir, exist_ok=True)
    preset_file = os.path.join(preset_dir, f"{preset_name}.json")
    with open(preset_file, "w", encoding="utf-8") as f:
        json.dump(preset_data, f, ensure_ascii=False, indent=4)


# [수동 탭 파형 위젯 초기화 / Initialize manual tab waveform widget]
def setup_waveform_widget():
    layout = QtWidgets.QVBoxLayout(window.widget_audio_man)
    layout.setContentsMargins(0, 0, 0, 0)
    
    plot_widget = pg.PlotWidget()
    plot_widget.setBackground('#1e1e1e')
    plot_widget.showGrid(x=True, y=True, alpha=0.3)
    layout.addWidget(plot_widget)
    
    return plot_widget


# [파형 그리기 및 VAD 구간 오버랩 시각화 / Render waveform and overlap VAD segments]
def render_waveform_and_vad(plot_widget, file_name, dst_dir):
    classified_dir = os.path.join(dst_dir, window.txt_general_classfied.text())
    caches_dir = os.path.join(dst_dir, "caches")
    dst_dir = dst_dir.replace("\\", "/")
    classified_dir = classified_dir.replace("\\", "/")
    caches_dir = caches_dir.replace("\\", "/")
    full_path = os.path.join(classified_dir, file_name)
    data, sr = sf.read(full_path)
    
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    plot_widget.clear()
    
    total_duration = len(data) / sr
    time_axis = np.linspace(0, total_duration, len(data))
    
    # 1. 오디오 파형 그리기 (청록색 파형 선)
    plot_widget.plot(time_axis, data, pen=pg.mkPen('#00bfff', width=1))
    
    # 2. 캐시에서 VAD speech_segments 읽어와서 파형 위에 반투명 오버랩 추가
    cache_data = processor.load_cache_json(caches_dir, file_name)
    if cache_data and "speech_segments" in cache_data:
        for seg in cache_data["speech_segments"]:
            region = pg.LinearRegionItem(
                values=[seg["start"], seg["end"]],
                movable=False,
                brush=pg.mkBrush(0, 255, 128, 60)
            )
            plot_widget.addItem(region)

# [수동 탭 파형 위젯 초기화 / Initialize manual tab waveform widget]
def setup_waveform_widget(ui_window):
    layout = QtWidgets.QVBoxLayout(ui_window.widget_audio_man)
    layout.setContentsMargins(0, 0, 0, 0)
    
    plot_widget = pg.PlotWidget()
    plot_widget.setBackground('#1e1e1e')
    plot_widget.showGrid(x=True, y=True, alpha=0.3)
    layout.addWidget(plot_widget)
    
    return plot_widget


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
            
    window.treeview_source_file.setModel(tree_file)
    window.treeview_source_file.resizeColumnToContents(0)
    window.treeview_source_file.resizeColumnToContents(1)
    window.treeview_source_file.resizeColumnToContents(2)
    window.treeview_source_file.resizeColumnToContents(3)
    window.treeview_source_file.resizeColumnToContents(4)

# 트리뷰 필터링  treeview filtering
def filter_treeview_by_name(search_text):
    query = search_text.lower() 
    for row in range(tree_file.rowCount()):
        item = tree_file.item(row, 0)
        is_match = query in item.text().lower()
        window.treeview_source_file.setRowHidden(row, QModelIndex(), not is_match)

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
        if not window.treeview_source_file.isRowHidden(row, QModelIndex()):
            tree_file.item(row, 0).setCheckState(Qt.CheckState.Checked)

# [보이는 파일만 전체 해제 / Unselect all visible files]
def unselect_all_visible():
    for row in range(tree_file.rowCount()):
        if not window.treeview_source_file.isRowHidden(row, QModelIndex()):
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
    stt_lan = window.combo_stt_lan.currentText()

    if not selected_files:
        write_log("[Warning] No files selected.")
        return

    threshold = window.doublespin_VocDect.value()
    window.progress_sort.setValue(0)
    window.btn_sort_start.setVisible(False)
    window.btn_stop.setVisible(True)

    global worker
    worker = AudioSortWorker(selected_files, src_dir, dst_dir, processor, threshold, vad_type, stt_type,stt_lan)
    
    worker.log_signal.connect(write_log)
    worker.status_signal.connect(update_item_status)  # UI Status 칸 실시간 업데이트 연결
    worker.progress_signal.connect(window.progress_sort.setValue)
    worker.finished_signal.connect(on_sorting_finished)
    
    worker.start()

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
            "vad_model": self.vad_type,
            "stt_model": self.stt_type,
            "stt_lang": self.stt_lan,
            "speech_segments": speech_segments,
            "voice_ratio": voice_ratio,
            "has_voice": has_voice,
            "status": status_str,
            "stt_text": stt_text
        }
    self.processor.save_cache_json(self.dst_dir, file_name, cache_data)


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

# [정렬 작업 강제 중단 / Terminate sorting process]
def stop_sorting_process():
    if "worker" in globals() and worker.isRunning():
        worker.terminate()
        window.btn_sort_start.setVisible(True)
        window.btn_stop.setVisible(False)
        write_log("PROCESS TERMINATED BY USER")

def edit_preset(edit_mode):
    if edit_mode == True:
        switch_preset_tab("load")
        window.txt_preset_edit.setText(window.combo_preset.currentText())
    else:
        switch_preset_tab("edit")
        window.txt_preset_edit.clear()

def save_preset_data():
    preset_name = window.txt_preset_edit.text()
    preset_data = {
        "cut_margin": window.spin_TrimSort.value(),
        "voice_detect": window.doublespin_VocDect.value(),
        "vad_model": window.combo_vad.currentText(),
        "stt_model": window.combo_stt.currentText(),
        "stt_language": window.combo_stt_lan.currentText()
    }
    save_preset(preset_name, preset_data)
    switch_preset_tab("edit")

def switch_preset_tab(current):
    if current == "load":
        window.lab_preset.setVisible(False)
        window.combo_preset.setVisible(False)
        window.btn_prset_edit_mode.setVisible(False)
        window.btn_preset_load.setVisible(False)
        window.btn_preset_save.setVisible(True)
        window.lab_editpreset.setVisible(True)
        window.txt_preset_edit.setVisible(True)
        window.btn_prset_edit_mode_exit.setVisible(True)
    else:
        window.lab_preset.setVisible(True)
        window.combo_preset.setVisible(True)
        window.btn_prset_edit_mode.setVisible(True)
        window.btn_preset_load.setVisible(True)
        window.btn_preset_save.setVisible(False)
        window.lab_editpreset.setVisible(False)
        window.txt_preset_edit.setVisible(False)
        window.btn_prset_edit_mode_exit.setVisible(False)
        update_combo_preset()

def update_combo_preset():
    base_dir = os.path.join(os.getenv("APPDATA"), "ASAP")
    preset_dir = os.path.join(base_dir, "preset")
    os.makedirs(preset_dir, exist_ok=True)
    window.combo_preset.clear()
    files = [f[:-5] for f in os.listdir(preset_dir) if f.endswith(".json")]
    window.combo_preset.addItems(files)

def collect_save_settings():
    highpass_value = window.doublespin_cutoff_hz.value()
    file_prefix_value = window.txt_prefix.text()
    file_suffix_value = window.txt_suffix.text()
    file_name_value = window.combo_filename.currentText()
    custom_file_name_value = window.txt_custom_file_name.text()
    copyrighter_value = window.txt_copyrighter.text()
    overwrite_existing_value = window.combo_overwrite_existing.currentText()
    backup_source_value = window.combo_backup_source.currentText()
    general_source_folder_value = window.txt_general_source.text()
    general_export_folder_value = window.txt_general_export.text()
    general_backup_folder_value = window.txt_general_backup.text()
    general_room_tone_folder_name_value = window.txt_general_roomtone.text()
    general_classified_folder_name_value = window.txt_general_classfied.text()
    general_shortduration_folder_name_value = window.txt_general_short.text()
    general_folder_Structure_value = window.txt_general_folder_structure.text()
    interface_language_value = window.combo_interpface_language.currentText()
    theme_value = window.combo_theme.currentText()
    basic_font_value = window.combo_general_font.currentText()
    basic_font_size_value = window.spin_general_font_size.value()
    changed_settings = {
            "highpass": highpass_value,
            "file_prefix":file_prefix_value,
            "file_suffix":file_suffix_value, 
            "file_name":file_name_value,
            "custom_file_name":custom_file_name_value,
            "copyrighter":copyrighter_value,
            "overwrite_existing":overwrite_existing_value,
            "backup_source":backup_source_value,
            "general_source_folder":general_source_folder_value,
            "general_export_folder":general_export_folder_value,
            "general_backup_folder":general_backup_folder_value,
            "general_room_tone_folder_name":general_room_tone_folder_name_value,
            "general_classified_folder_name":general_classified_folder_name_value,
            "general_shortduration_folder_name":general_shortduration_folder_name_value,
            "general_folder_Structure":general_folder_Structure_value,
            "interface_language":interface_language_value,
            "theme":theme_value,
            "basic_font":basic_font_value,
            "basic_font_size":basic_font_size_value
        }
    save_settings(changed_settings)
    load_settings()
    return ""
    
def apply_settings():
    load_data = load_settings()
    window.doublespin_cutoff_hz.setValue(load_data.get("highpass", 40.0))
    window.txt_prefix.setText(load_data.get("file_prefix", ""))
    window.txt_suffix.setText(load_data.get("file_suffix", ""))
    window.combo_filename.setCurrentText(load_data.get("file_name", ""))
    window.txt_custom_file_name.setText(load_data.get("custom_file_name", ""))
    window.txt_copyrighter.setText(load_data.get("copyrighter", ""))
    window.combo_overwrite_existing.setCurrentText(load_data.get("overwrite_existing", ""))
    window.txt_general_backup.setText(load_data.get("general_backup_folder", ""))
    window.txt_general_source.setText(load_data.get("general_source_folder", ""))
    window.txt_general_export.setText(load_data.get("general_export_folder", ""))
    window.combo_backup_source.setCurrentText(load_data.get("backup_source", ""))
    window.txt_general_roomtone.setText(load_data.get("general_room_tone_folder_name", ""))
    window.txt_general_classfied.setText(load_data.get("general_classified_folder_name", ""))
    window.txt_general_short.setText(load_data.get("general_shortduration_folder_name", ""))
    window.txt_general_folder_structure.setText(load_data.get("general_folder_Structure", ""))
    window.combo_interpface_language.setCurrentText(load_data.get("interface_language", ""))
    window.combo_theme.setCurrentText(load_data.get("theme", ""))
    window.combo_general_font.setCurrentText(load_data.get("basic_font", ""))
    window.spin_general_font_size.setValue(load_data.get("basic_font_size", 11))
    return ""

# [수동 탭 콤보박스 옵션 동기화 / Synchronize manual tab combobox options]
def sync_manual_comboboxes():
    window.combo_vad_man.clear()
    for i in range(window.combo_vad.count()):
        window.combo_vad_man.addItem(window.combo_vad.itemText(i))
    window.combo_vad_man.setCurrentIndex(window.combo_vad.currentIndex())
    
    window.combo_stt_man.clear()
    for i in range(window.combo_stt.count()):
        window.combo_stt_man.addItem(window.combo_stt.itemText(i))
    window.combo_stt_man.setCurrentIndex(window.combo_stt.currentIndex())
    
    window.combo_stt_lan_man.clear()
    for i in range(window.combo_stt_lan.count()):
        window.combo_stt_lan_man.addItem(window.combo_stt_lan.itemText(i))
    window.combo_stt_lan_man.setCurrentIndex(window.combo_stt_lan.currentIndex())

# [수동 탭 EXIF 및 가공 이력 목록 업데이트 / Update manual tab EXIF and processing history list]
def update_exif_list(file_name, dst_dir):
    classified_dir = os.path.join(dst_dir, window.txt_general_classfied.text())
    caches_dir = os.path.join(dst_dir, "caches")
    dst_dir = dst_dir.replace("\\", "/")
    classified_dir = classified_dir.replace("\\", "/")
    caches_dir = caches_dir.replace("\\", "/")
    full_path = os.path.join(classified_dir, file_name)
    info = sf.info(full_path)
    cache_data = processor.load_cache_json(caches_dir, file_name)
    
    exif_model = QStandardItemModel()
    
    exif_model.appendRow(QStandardItem(f"File Name: {file_name}"))
    exif_model.appendRow(QStandardItem(f"Sample Rate: {info.samplerate} Hz"))
    exif_model.appendRow(QStandardItem(f"Channels: {info.channels}"))
    exif_model.appendRow(QStandardItem(f"Duration: {info.duration:.2f}s"))
    
    if cache_data:
        exif_model.appendRow(QStandardItem(f"VAD Model: {cache_data.get('vad_type', 'N/A')}"))
        exif_model.appendRow(QStandardItem(f"STT Model: {cache_data.get('stt_type', 'N/A')}"))
        exif_model.appendRow(QStandardItem(f"STT Lang: {cache_data.get('stt_lan', 'N/A')}"))
        exif_model.appendRow(QStandardItem(f"Voice Ratio: {cache_data.get('voice_ratio', 0.0)}%"))
        exif_model.appendRow(QStandardItem(f"Status: {cache_data.get('status', 'Ready')}"))
        exif_model.appendRow(QStandardItem(f"Text: {cache_data.get('stt_text', '')}"))
        
    window.list_exif.setModel(exif_model)

# [수동 탭 트리뷰 업데이트 / Update manual tab tree view]
def update_manual_treeview(dst_dir):
    roomtone_dir = os.path.join(dst_dir, window.txt_general_roomtone.text())
    classified_dir = os.path.join(dst_dir, window.txt_general_classfied.text())
    shortduration_dir = os.path.join(dst_dir, window.txt_general_short.text())
    cach_dir = os.path.join(dst_dir, "caches")

    tree_man_file = QStandardItemModel()
    tree_man_file.clear()
    tree_man_file.setColumnCount(2)
    tree_man_file.setHorizontalHeaderLabels(["File Name", "Status"])

    try:
        files = os.listdir(classified_dir)
    except Exception as e:
        print(f"Error accessing directory: {e}")
        return

    for file_name in files:
        if file_name.lower().endswith(ALLOWED_EXTENSIONS):
            full_path = os.path.join(classified_dir, file_name)
            file_item = QStandardItem(file_name)
            
            cache_data = processor.load_cache_json(cach_dir, file_name)
            status_text = cache_data.get("status", "Ready") if cache_data else "Ready"
            status_item = QStandardItem(status_text)
            
            tree_man_file.appendRow([file_item, status_item])

    window.tree_audio_man.setModel(tree_man_file)
    window.tree_audio_man.resizeColumnToContents(0)
    window.tree_audio_man.resizeColumnToContents(1)

def process_double_click(index):
    dst_dir = window.txt_dst_man.text()
    if not index.isValid():
        return

    file_name = index.siblingAtColumn(0).data()
    if not file_name:
        return

    update_exif_list(file_name, dst_dir)
    render_waveform_and_vad(waveform_plot, file_name, dst_dir)

loader = QUiLoader()
app = QtWidgets.QApplication(sys.argv)
window = loader.load("main.ui", None)

load_preset("default")
load_settings()
update_combo_preset()
apply_settings()

window.btn_stop.setVisible(False)
switch_preset_tab("edit")
sync_manual_comboboxes()
waveform_plot = setup_waveform_widget(window)



# Audio sort tab button connections
#File selection and loading
window.btn_src.clicked.connect(load_folder)

window.btn_dst.clicked.connect(lambda: [load_export_folder(), window.txt_dst_man.setText(window.txt_dst.text())])

window.txt_src.returnPressed.connect(lambda: load_treeview(window.txt_src.text()))

window.txt_dst.textChanged.connect(lambda: load_treeview(window.txt_src.text()))
window.txt_dst.textChanged.connect(lambda: window.txt_dst_man.setText(window.txt_dst.text()))

window.btn_sel_all.clicked.connect(select_all_visible)

window.btn_unsel_all.clicked.connect(unselect_all_visible)

window.btn_reset_stat.clicked.connect(reset_status_of_all_files)

window.btn_find.clicked.connect(lambda: filter_treeview_by_name(window.txt_find.text()))

window.txt_find.textChanged.connect(lambda: filter_treeview_by_name(window.txt_find.text()))

window.btn_view_all.clicked.connect(lambda: filter_treeview_by_name(""))

#Model settings

window.spin_vad_trim.valueChanged.connect(processor.default_trim_sort_value_changed)

window.btn_sort_start.clicked.connect(start_sorting_process)

window.btn_stop.clicked.connect(stop_sorting_process)

window.combo_preset.currentTextChanged.connect(lambda: load_preset(window.combo_preset.currentText()))

window.btn_prset_edit_mode.clicked.connect(lambda: edit_preset(True))

window.btn_prset_edit_mode_exit.clicked.connect(lambda: edit_preset(False))

window.btn_preset_save.clicked.connect(save_preset_data)

window.btn_preset_load.clicked.connect(lambda: [load_preset(window.combo_preset.currentText()), update_combo_preset()])

# Manual sort tab
#loading manual treeview
window.txt_dst_man.returnPressed.connect(lambda: update_manual_treeview(window.txt_dst_man.text()))
window.btn_load_dst_man.clicked.connect(lambda: [load_export_folder(), update_manual_treeview(window.txt_dst_man.text())])
window.tree_audio_man.doubleClicked.connect(lambda index: process_double_click(index))


# Setting tab button connections
window.btn_apply_set.clicked.connect(collect_save_settings)
window.btn_noapply_set.clicked.connect(apply_settings)

window.show()
app.exec()