import os
import socket
import webbrowser
import threading
import time
import subprocess
import json
from threading import Timer
from flask import Flask, jsonify, render_template, request, send_from_directory, session
from datetime import datetime
from VideoComparatorNew import VideoObjectAnalyzer
from config import API_MODELS, API_PROMPT, FFMPEG_PATH, RESULT_DIR, STATIC_DIR, TEMPLATE_DIR, UPLOAD_DIR
from diff_frame_analyzer import ImageDiffAI
from utils.help import clean_folder

# ランタイムフォルダが存在することを確認する
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------
# Flaskを初期化する
# -----------------------------
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["RESULT_FOLDER"] = RESULT_DIR

current_analysis_thread = None
analysis_stop_event = threading.Event()

def find_free_port(default_port=5000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0)) # 0 は OS に空きポートを選択させることを意味します
        return s.getsockname()[1]

def open_browser(port):
    url = f"http://127.0.0.1:{port}"
    webbrowser.open_new(url)

# 動画処理
def process_video_feature(filepath_org, folderpath_des, result_path, fps, process_fps, threshold, ai_model_path):
    clean_folder(RESULT_DIR)
    comparator = VideoObjectAnalyzer(model_path=ai_model_path,fps=fps, process_fps=process_fps, threshold=threshold)
    try:
        excel_name, org_path, saved_videos, frame_datas = comparator.main(filepath_org, folderpath_des, result_path)
        return excel_name, org_path, saved_videos, frame_datas
    except Exception as e:
        print(f"処理エラー: {e}")
        raise

def background_diff_process(result_folder, api_model, api_prompt, analysis_stop_event=None):
    try:
        ai_diff = ImageDiffAI(result_folder, api_model=api_model, api_prompt=api_prompt, stop_event=analysis_stop_event)
        ai_diff.handle_diff_frames()
    except Exception as e:
        print(f"背景差分フレームの処理中にエラーが発生しました: {e}")
        
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(RESULT_DIR, filename, as_attachment=True)

@app.route("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(RESULT_DIR, filename)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(RESULT_DIR, filename)

@app.route("/feature", methods=["GET", "POST"])
def feature():
    if request.method == "POST":
        clean_folder(UPLOAD_DIR)
        
        fps = int(request.form.get("fps"))               
        process_fps = int(request.form.get("process_fps"))
        threshold = (int(request.form.get("threshold")) / 1000) * process_fps

        # === AI モデル ===
        ai_model_file = request.files.get("ai_model")
        if not ai_model_file or ai_model_file.filename == "":
            return "AIモデルファイルを選択してください。"

        ai_model_name = datetime.now().strftime("%Y%m%d_%H%M%S_") + ai_model_file.filename
        ai_model_path = os.path.join(UPLOAD_DIR, ai_model_name)
        ai_model_file.save(ai_model_path)

        # === 元動画と先動画 ===
        file_org = request.files["video1"]
        files_des = request.files.getlist("videos")

        if not file_org or file_org.filename == "" or not files_des:
            return "動画ファイルを選択してください。"

        filename_org = datetime.now().strftime("%Y%m%d_%H%M%S_") + file_org.filename
        filepath_org = os.path.join(UPLOAD_DIR, filename_org)
        file_org.save(filepath_org)

        filepaths_des = []
        for file_des in files_des:
            filename_des = datetime.now().strftime("%Y%m%d_%H%M%S_") + file_des.filename
            filepath_des = os.path.join(UPLOAD_DIR, filename_des)
            file_des.save(filepath_des)
            filepaths_des.append(filepath_des)

        try:
            excel_name, org_path, saved_videos, frame_datas = process_video_feature(filepath_org, filepaths_des, RESULT_DIR, fps, process_fps, threshold, ai_model_path)
        
            # AI APIを使って、2枚のフレームを比較
            api_model = request.form.get("ai_model_api")
            api_prompt = request.form.get("ai_prompt")

            session['current_ai_model'] = api_model
            session['current_ai_prompt'] = api_prompt

            threading.Thread(
                target=background_diff_process,
                args=(RESULT_DIR, api_model, api_prompt,),
                daemon=True
            ).start()

            return render_template(
                "result_feature.html",
                excel_file=excel_name,
                org_path=org_path,
                video_paths=saved_videos,
                frame_data=frame_datas or []
            )
        except Exception as e:
            return f"処理中にエラーが発生しました: {str(e)}"


    return render_template("feature.html", 
                         ai_models=API_MODELS, 
                         default_prompt=API_PROMPT)

@app.route("/diff_frames")
def diff_frames():
    """差分フレーム表示ページ"""
    frame_data_path = os.path.join(RESULT_DIR, "diff_frames.json")
    frame_data = []
    if os.path.exists(frame_data_path):
        with open(frame_data_path, "r", encoding="utf-8") as f:
            frame_data = json.load(f)
    
    # Kiểm tra xem tất cả analysis đã hoàn thành chưa
    all_analysis_complete = True
    for video_group in frame_data:
        for frame_pair in video_group:
            if not frame_pair.get('diff_text_ai'):
                all_analysis_complete = False
                break
        if not all_analysis_complete:
            break
    
    # Lấy model và prompt hiện tại (từ session hoặc mặc định)
    current_ai_model = session.get('current_ai_model', API_MODELS[0])
    current_ai_prompt = session.get('current_ai_prompt', API_PROMPT)
    
    return render_template('diff_frames.html', 
                         frame_data=frame_data,
                         ai_models=API_MODELS,
                         current_ai_model=current_ai_model,
                         current_ai_prompt=current_ai_prompt,
                         all_analysis_complete=all_analysis_complete)

# -----------------------------
# API: diff_frames ステータス
# -----------------------------
@app.route("/api/diff_status")
def api_diff_status():
    json_path = os.path.join(RESULT_DIR, "diff_frames_ai.json")
    if not os.path.exists(json_path):
        return jsonify({"status": "processing", "data": []})
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify({"status": "done", "data": data})

@app.route('/reanalyze_frames', methods=['POST'])
def reanalyze_frames():
    global current_analysis_thread, analysis_stop_event
    
    data = request.get_json()
    ai_model = data.get('ai_model')
    ai_prompt = data.get('ai_prompt')
    
    # Lưu cài đặt hiện tại vào session
    session['current_ai_model'] = ai_model
    session['current_ai_prompt'] = ai_prompt
    
    # Dừng thread cũ
    if current_analysis_thread and current_analysis_thread.is_alive():
        analysis_stop_event.set()
        current_analysis_thread.join(timeout=5)
        analysis_stop_event.clear()
    
    # Khởi chạy thread mới
    current_analysis_thread = threading.Thread(
        target=background_diff_process,
        args=(RESULT_DIR, ai_model, ai_prompt, analysis_stop_event),
        daemon=True
    )
    current_analysis_thread.start()
    
    return jsonify({'success': True, 'message': 'AI再分析を開始しました。'})

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    global analysis_stop_event
    analysis_stop_event.set()
    return jsonify({'success': True, 'message': '分析を停止しました。'})

# ====== サーバーのシャットダウン ======
@app.route("/shutdown", methods=["POST"])
def shutdown():
    def stop_server():
        time.sleep(1)  
        os._exit(0)  

    threading.Thread(target=stop_server).start()
    return "終了されました..."

if __name__ == "__main__":
    # 使用例
    subprocess.run([FFMPEG_PATH, "-version"])
    port = 29700
    Timer(1, lambda: open_browser(port)).start()
    print(f"✅ Server running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)