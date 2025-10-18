import os
import cv2
import ssl
import smtplib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
from dotenv import load_dotenv
from datetime import datetime
from email.message import EmailMessage
from skimage.metrics import structural_similarity as ssim
from VideoComparator import VideoComparator

# 環境変数をロード
load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["RESULT_FOLDER"] = "static/results"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

model = YOLO("models/best_main.pt")

# 定数設定
ERROR_CLASS_ID = [2, 3]
MODEL_CLASS_IDS = ["BT", "Wifi", "Cel", "Hots", "Bri", "{}"]
CONF_THRESH = 0.6
NO_ERROR_LIMIT = 5


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

def compare_video_pixel(video1, video2, threshold=0.8):
    """SSIMでピクセルレベル比較、差分フレームを返す"""
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    diff_frames = []
    frame_index = 0

    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Cannot open video")

    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 or not ret2:
            break

        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray1, gray2, full=True)

        if score < threshold:
            diff_frames.append(frame_index)
        frame_index += 1

    cap1.release()
    cap2.release()
    return diff_frames

def process_detections(frame, results, CONF_THRESH=0.5):
    """YOLO結果を処理してbboxを描画、class_idを返す"""
    frame_class_ids = []
    error_found = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if conf < CONF_THRESH:
            continue
        frame_class_ids.append(cls_id)

        color = (0, 0, 255) if cls_id in ERROR_CLASS_ID else (0, 255, 0)
        if cls_id in ERROR_CLASS_ID:
            error_found = True

        label = MODEL_CLASS_IDS[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return frame_class_ids, error_found

def class_ids_diff(ids1, ids2):
    """ids2にだけ存在するclass_idを返す"""
    return set(ids2) - set(ids1)

# 動画処理関数
def process_video_feature_1(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    clear_video(app.config["RESULT_FOLDER"])
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_id = 0
    in_abnormal = False
    error_frame_count = 0
    start_time = None
    changes = []
    snapshot_images = []
    
    frame_delay_time = 200 / 1000      # 0.2秒
    frame_forward = round(fps * frame_delay_time)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        time_sec = frame_id / fps
        results = model(frame, imgsz=640, verbose=False)[0]

        error_found = False
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf < CONF_THRESH:
                continue

            color = (0, 255, 0)
            if cls_id in ERROR_CLASS_ID:
                color = (0, 0, 255)
                error_found = True
                frame_forward -= 1

            detections.append((x1, y1, x2, y2, color, MODEL_CLASS_IDS[cls_id]))

        # 異常状態を検出
        if error_found:
            error_frame_count = 0
            if not in_abnormal and frame_forward == 0:
                in_abnormal = True
                start_time = time_sec
                snapshot_images.append(frame)
        else:
            if in_abnormal:
                error_frame_count += 1
                if error_frame_count >= NO_ERROR_LIMIT:
                    in_abnormal = False
                    end_time = time_sec
                    changes.append({"start": round(start_time, 1), "end": round(end_time, 1)})
                    error_frame_count = 0
                    frame_forward = round(fps * frame_delay_time)

        # 検出結果を描画
        for x1, y1, x2, y2, color, label in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    if in_abnormal:
        end_time = frame_id / fps
        changes.append({
            "start": round(start_time, 1),
            "end": round(end_time, 1),
        })

    return changes, snapshot_images

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
                body += f"- {c['start']}秒 ～ {c['end']}秒\n"
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
            combined=list(zip(changes, snapshot_files))  # 👈 thêm dòng này
)
            
    return render_template("feature2.html")

if __name__ == "__main__":
    app.run(debug=True)
