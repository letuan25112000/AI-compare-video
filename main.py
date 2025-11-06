import os
import socket
import webbrowser
from threading import Timer
from flask import Flask, render_template, request, send_from_directory
from datetime import datetime
from VideoComparatorNew import VideoObjectAnalyzer
from config import FFMPEG_PATH, RESULT_DIR, STATIC_DIR, TEMPLATE_DIR, UPLOAD_DIR
from utils.help import clean_folder
import threading
import time
import os
import sys
import subprocess
import json


# Make sure runtime folders exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------
# Initialize Flask
# -----------------------------
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["RESULT_FOLDER"] = RESULT_DIR

def find_free_port(default_port=5000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # 0 means: let OS pick a free port
        return s.getsockname()[1]

def open_browser(port):
    url = f"http://127.0.0.1:{port}"
    webbrowser.open_new(url)

# 動画処理
def process_video_feature(filepath_org, folderpath_des, result_path, fps, process_fps, threshold, ai_model_path):
    clean_folder(app.config["RESULT_FOLDER"])
    comparator = VideoObjectAnalyzer(model_path=ai_model_path,fps=fps, process_fps=process_fps, threshold=threshold)
    try:
        excel_name, org_path, saved_videos, frame_datas = comparator.main(filepath_org, folderpath_des, result_path)
        return excel_name, org_path, saved_videos, frame_datas
    except Exception as e:
        print(f"処理エラー: {e}")
        raise

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=True)

@app.route("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)

@app.route("/feature", methods=["GET", "POST"])
def feature():
    if request.method == "POST":
        clean_folder(app.config["UPLOAD_FOLDER"])
        
        fps = int(request.form.get("fps"))               
        process_fps = int(request.form.get("process_fps"))
        threshold = (int(request.form.get("threshold")) / 1000) * process_fps

        # === AI モデル ===
        ai_model_file = request.files.get("ai_model")
        if not ai_model_file or ai_model_file.filename == "":
            return "AIモデルファイルを選択してください。"

        ai_model_name = datetime.now().strftime("%Y%m%d_%H%M%S_") + ai_model_file.filename
        ai_model_path = os.path.join(app.config["UPLOAD_FOLDER"], ai_model_name)
        ai_model_file.save(ai_model_path)

        # === 元動画と先動画 ===
        file_org = request.files["video1"]
        files_des = request.files.getlist("videos")

        if not file_org or file_org.filename == "" or not files_des:
            return "動画ファイルを選択してください。"

        filename_org = datetime.now().strftime("%Y%m%d_%H%M%S_") + file_org.filename
        filepath_org = os.path.join(app.config["UPLOAD_FOLDER"], filename_org)
        file_org.save(filepath_org)

        filepaths_des = []
        for file_des in files_des:
            filename_des = datetime.now().strftime("%Y%m%d_%H%M%S_") + file_des.filename
            filepath_des = os.path.join(app.config["UPLOAD_FOLDER"], filename_des)
            file_des.save(filepath_des)
            filepaths_des.append(filepath_des)

        try:
            excel_name, org_path, saved_videos, frame_datas = process_video_feature(filepath_org, filepaths_des, app.config["RESULT_FOLDER"], fps, process_fps, threshold, ai_model_path)

            return render_template(
                "result_feature.html",
                excel_file=excel_name,
                org_path=org_path,
                video_paths=saved_videos,
                frame_data=frame_datas or []
            )
        except Exception as e:
            return f"処理中にエラーが発生しました: {str(e)}"

    return render_template("feature.html")

@app.route("/diff_frames")
def diff_frames():
    """差分フレーム表示ページ"""
    frame_json_path = os.path.join(app.config["RESULT_FOLDER"], "diff_frames.json")
    frame_data = []
    if os.path.exists(frame_json_path):
        with open(frame_json_path, 'r', encoding='utf-8') as f:
            frame_data = json.load(f)
    
    return render_template("diff_frames.html", frame_data=frame_data)

# ====== SHUTDOWN SERVER ======
@app.route("/shutdown", methods=["POST"])
def shutdown():
    def stop_server():
        time.sleep(1)  
        os._exit(0)  

    threading.Thread(target=stop_server).start()
    return "終了されました..."

if __name__ == "__main__":
    # Example usage
    subprocess.run([FFMPEG_PATH, "-version"])
    port = find_free_port(5000)
    Timer(1, lambda: open_browser(port)).start()
    print(f"✅ Server running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)