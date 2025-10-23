import os
from flask import Flask, render_template, request, send_from_directory
from datetime import datetime
from VideoComparatorNew import VideoObjectAnalyzer
from utils.help import clean_folder

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

# 動画処理
def process_video_feature_1(filepath_org, folderpath_des, result_path):
    clean_folder(app.config["RESULT_FOLDER"])
    comparator = VideoObjectAnalyzer()
    result_path = comparator.main(filepath_org, folderpath_des, result_path)
    return result_path

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=True)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)

@app.route("/feature1", methods=["GET", "POST"])
def feature1():
    if request.method == "POST":
        clean_folder(app.config["UPLOAD_FOLDER"])
        
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

        excel_path, saved_videos = process_video_feature_1(filepath_org, filepaths_des, app.config["RESULT_FOLDER"])

        return render_template(
            "result_feature1.html",
            excel_file=excel_path,
            video_paths=saved_videos
        )

    return render_template("feature1.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True, threaded=True)
