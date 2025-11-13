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

# -------------------------------------
# 必要なフォルダを作成
# -------------------------------------
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------------------
# Flaskアプリケーション設定
# -------------------------------------
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["RESULT_FOLDER"] = RESULT_DIR

current_analysis_thread = None
analysis_stop_event = threading.Event()
analysis_lock = threading.Lock()


def open_browser(port):
    """起動時にブラウザを自動的に開く"""
    url = f"http://127.0.0.1:{port}"
    webbrowser.open_new(url)


# -------------------------------------
# 動画処理メイン関数
# -------------------------------------
def process_video_feature(filepath_org, filepaths_des, result_path, fps, process_fps, threshold, ai_model_path):
    """動画特徴比較処理を実行"""
    clean_folder(RESULT_DIR)

    comparator = VideoObjectAnalyzer(
        model_path=ai_model_path,
        fps=fps,
        process_fps=process_fps,
        threshold=threshold
    )

    excel_name, org_path, saved_videos, frame_datas = comparator.main(
        filepath_org, filepaths_des, result_path
    )

    return excel_name, org_path, saved_videos, frame_datas


# -------------------------------------
# 背景でAI画像比較処理を実行
# -------------------------------------
def background_diff_process(result_folder, api_model, api_prompt):
    """diff_frames.json をAIで解析"""
    global current_analysis_thread, analysis_stop_event

    with analysis_lock:
        # 既存スレッドが動作中の場合、停止要求を送る
        if current_analysis_thread and current_analysis_thread.is_alive():
            print("[reanalyze_frames] 既存のAIスレッドを停止中...")
            analysis_stop_event.set()
            current_analysis_thread.join(timeout=15)

            if current_analysis_thread.is_alive():
                print("[reanalyze_frames] 15秒以内に停止できませんでした。スキップします。")
            else:
                print("[reanalyze_frames] 旧スレッドは停止しました。")

            analysis_stop_event.clear()

        print("[background_diff_process] ", analysis_stop_event.is_set())
        ai_diff = ImageDiffAI(
            result_folder,
            api_model=api_model,
            api_prompt=api_prompt,
            stop_event=analysis_stop_event
        )

        # 既存のAI結果をクリア
        ai_diff.clear_api_result_json()

        # 新しいスレッドを開始
        current_analysis_thread = threading.Thread(
            target=ai_diff.handle_diff_frames,
            daemon=True
        )
        current_analysis_thread.start()


# -------------------------------------
# ルートページ
# -------------------------------------
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


# -------------------------------------
# 特徴比較ページ
# -------------------------------------
@app.route("/feature", methods=["GET", "POST"])
def feature():
    if request.method == "POST":
        clean_folder(UPLOAD_DIR)

        fps = int(request.form.get("fps"))
        process_fps = int(request.form.get("process_fps"))
        threshold = (int(request.form.get("threshold")) / 1000) * process_fps

        # === AIモデルファイル ===
        ai_model_file = request.files.get("ai_model")
        if not ai_model_file or ai_model_file.filename == "":
            return "AIモデルファイルを選択してください。"

        ai_model_name = datetime.now().strftime("%Y%m%d_%H%M%S_") + ai_model_file.filename
        ai_model_path = os.path.join(UPLOAD_DIR, ai_model_name)
        ai_model_file.save(ai_model_path)

        # === 元動画と比較対象動画 ===
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
            excel_name, org_path, saved_videos, frame_datas = process_video_feature(
                filepath_org, filepaths_des, RESULT_DIR, fps, process_fps, threshold, ai_model_path
            )

            # === AI画像比較 ===
            api_model = request.form.get("ai_model_api")
            api_prompt = request.form.get("ai_prompt")

            session['current_ai_model'] = api_model
            session['current_ai_prompt'] = api_prompt

            background_diff_process(RESULT_DIR, api_model, api_prompt)

            return render_template(
                "result_feature.html",
                excel_file=excel_name,
                org_path=org_path,
                video_paths=saved_videos,
                frame_data=frame_datas or []
            )
        except Exception as e:
            return f"処理中にエラーが発生しました: {str(e)}"

    return render_template(
        "feature.html",
        ai_models=API_MODELS,
        default_prompt=API_PROMPT
    )


# -------------------------------------
# 差分フレーム表示ページ
# -------------------------------------
@app.route("/diff_frames")
def diff_frames():
    frame_data_path = os.path.join(RESULT_DIR, "diff_frames.json")
    frame_data = []

    if os.path.exists(frame_data_path):
        with open(frame_data_path, "r", encoding="utf-8") as f:
            frame_data = json.load(f)

    # AI解析が完了したか確認
    all_analysis_complete = not (current_analysis_thread and current_analysis_thread.is_alive())

    current_ai_model = session.get('current_ai_model', API_MODELS[0])
    current_ai_prompt = session.get('current_ai_prompt', API_PROMPT)

    return render_template(
        'diff_frames.html',
        frame_data=frame_data,
        ai_models=API_MODELS,
        current_ai_model=current_ai_model,
        current_ai_prompt=current_ai_prompt,
        all_analysis_complete=all_analysis_complete
    )


# -------------------------------------
# AI再解析リクエスト
# -------------------------------------
@app.route('/reanalyze_frames', methods=['POST'])
def reanalyze_frames():
    try:
        data = request.get_json()
        ai_model = data.get('ai_model')
        ai_prompt = data.get('ai_prompt')

        session['current_ai_model'] = ai_model
        session['current_ai_prompt'] = ai_prompt

        background_diff_process(RESULT_DIR, ai_model, ai_prompt)

        return jsonify({'success': True, 'message': 'AI再分析を開始しました。'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'サーバーエラー: {str(e)}'}), 500


# -------------------------------------
# AI解析停止API
# -------------------------------------
@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    try:
        if analysis_stop_event:
            analysis_stop_event.set()
            return jsonify({'success': True, 'message': '分析を停止しました。'})
        else:
            return jsonify({'success': False, 'message': '分析プロセスが見つかりません。'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'停止中にエラーが発生しました: {str(e)}'}), 500


# -------------------------------------
# サーバー終了API
# -------------------------------------
@app.route("/shutdown", methods=["POST"])
def shutdown():
    def stop_server():
        time.sleep(1)
        os._exit(0)

    threading.Thread(target=stop_server).start()
    return "終了しました。"


# -------------------------------------
# メイン処理
# -------------------------------------
if __name__ == "__main__":
    subprocess.run([FFMPEG_PATH, "-version"])
    port = 29700
    Timer(1, lambda: open_browser(port)).start()
    print(f"サーバー起動: http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
