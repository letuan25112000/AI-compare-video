import os
import cv2
import ssl
import smtplib
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from dotenv import load_dotenv
from datetime import datetime
from email.message import EmailMessage
from VideoComparator import VideoComparator
from VideoProcessor import VideoProcessor

# 環境変数をロード
load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

def send_mail(subject, body, receiver_email, images=None):
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")
    EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")

    em = EmailMessage()
    em["From"] = EMAIL_SENDER
    em["To"] = receiver_email
    em["Subject"] = subject
    em.set_content(body)

    # 添付画像
    if images:
        for i, img in enumerate(images):
            if img is None:
                continue
            _, buffer = cv2.imencode('.jpg', img)
            img_data = buffer.tobytes()
            filename = f'image_{i+1}.jpg'
            em.add_attachment(img_data, maintype='image', subtype='jpeg', filename=filename)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(em)
            print(f"*** メールを {receiver_email} に送信しました。")
    except Exception as e:
        print(f"[メール送信エラー]: {e}")


# 結果フォルダ内の古い動画を削除
def clear_video(folder):
    for f in os.listdir(folder):
        file_path = os.path.join(folder, f)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"[古いファイル削除エラー]: {e}")

# 動画処理関数
def process_video_feature_1(input_path, output_path):
    processor = VideoProcessor()
    return processor.process_video(input_path, output_path)

def process_video_feature_2(filepath_org, filepath_des, result_path):
    clear_video(app.config["RESULT_FOLDER"])
    comparator = VideoComparator()
    changes, snapshots = comparator.compare_videos(filepath_org, filepath_des)
    return changes, snapshots

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/feature1", methods=["GET", "POST"])
def feature1():
    if request.method == "POST":
        clear_video(app.config["UPLOAD_FOLDER"])
        
        email = request.form.get("email")
        file = request.files["video"]

        if not email:
            return "メールアドレスを入力してください。"
        if not file or file.filename == "":
            return "動画ファイルを選択してください。"

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result_filename = "result_" + filename
        result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)

        changes, snapshots = process_video_feature_1(filepath, result_path)

        if changes:
            body = "動画内で異常が検出されました：\n"
            for c in changes:
                body += f"- {c['start']}秒 ～ {c['end']}秒\n"
            print("BODY: ", body)
            send_mail("*** 異常検出レポート", body, email, snapshots)

        return render_template("result.html",
                               result_video=result_filename,
                               changes=changes,
                               email=email)
    return render_template("feature1.html")

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=True)

@app.route("/feature2", methods=["GET", "POST"])
def feature2():
    if request.method == "POST":
        clear_video(app.config["UPLOAD_FOLDER"])
        
        email = request.form.get("email")
        file_org = request.files["video1"]
        file_des = request.files["video2"]

        if not email:
            return "メールアドレスを入力してください。"
        if not file_org or file_org.filename == "" or not file_des or file_des.filename == "":
            return "動画ファイルを選択してください。"
        

        filename_org = datetime.now().strftime("%Y%m%d_%H%M%S_") + file_org.filename
        filepath_org = os.path.join(app.config["UPLOAD_FOLDER"], filename_org)
        file_org.save(filepath_org)
        
        filename_des = datetime.now().strftime("%Y%m%d_%H%M%S_") + file_des.filename
        filepath_des = os.path.join(app.config["UPLOAD_FOLDER"], filename_des)
        file_des.save(filepath_des)

        result_filename = "result_" + filename_org
        result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)

        changes, snapshots = process_video_feature_2(filepath_org, filepath_des, result_path)


        if changes:
            body = "動画内で異常が検出されました：\n"
            for c in changes:
                body += f"- {c['start']}秒 ～ {c['end']}秒\n ・ アイコン異常：{c['error_point']}"
            print("BODY: ", body)
            send_mail("*** 異常検出レポート", body, email, snapshots)

        snapshot_files = []
        for i, img in enumerate(snapshots):
            if img is None:
                continue
            snapshot_name = f"snapshot_{i+1}.jpg"
            snapshot_path = os.path.join(app.config["RESULT_FOLDER"], snapshot_name)
            cv2.imwrite(snapshot_path, img)
            snapshot_files.append(snapshot_name)

        # Rồi chỉnh lại return:
        return render_template(
            "result_feature2.html",
            changes=changes,
            snapshots=snapshot_files,
            email=email,
            combined=list(zip(changes, snapshot_files)) 
)
            
    return render_template("feature2.html")

if __name__ == "__main__":
    app.run(debug=True)
