import cv2
import numpy as np
from ultralytics import YOLO
import os
import ssl
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

model = YOLO("models/best_621.pt", task='detect')

ERROR_CLASS_ID = [2]
MODEL_CLASS_IDS = [ "BT", "Wifi", "Cel", "Bri", "{}"]

VIDEO_PATH = "./videos/C.mp4"
CONF_THRESH = 0.4
PROCESS_FPS = 3

frame_id = 0
in_abnormal = False
start_time = None
last_detections = []
changes = []  # 変更リスト
images_save = []
class_wifi = []
class_wifi_score = []

frame_forward = 10

is_wifi = True

no_person_count = 0
NO_ERROR_LIMIT = 5

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / PROCESS_FPS) if fps > 0 else 10

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame_id += 1
    time_sec = frame_id / fps
    
    # 指定間隔でのみ処理
    if frame_id % frame_interval == 0:
        is_wifi = False
        results = model(frame, imgsz=640, verbose=False)[0]
        detections = []

        error_found = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf < CONF_THRESH:
                continue

            if cls_id in ERROR_CLASS_ID:
                color = (0, 0, 255)
                error_found = True
                frame_forward -= 1
            else:
                color = (0, 255, 0)

            detections.append((x1, y1, x2, y2, color, MODEL_CLASS_IDS[cls_id]))
        
        # 差分の開始と終了を追跡
        if error_found:
            no_person_count = 0
            if not in_abnormal and frame_forward == 0:
                in_abnormal = True
                start_time = time_sec
                print(f"=== 差分開始: {start_time:.1f}秒")
                images_save.append(frame)
        else:
            if in_abnormal:
                no_person_count += 1
                if no_person_count >= NO_ERROR_LIMIT:
                    in_abnormal = False
                    end_time = time_sec
                    print(f"=== 差分終了: {end_time:.1f}秒")
                    changes.append({
                        "start": round(start_time, 1),
                        "end": round(end_time, 1),
                    })
                    no_person_count = 0  # reset

        # Frameの結果を保存
        last_detections = detections
    
    # last_detections に従って常にフレームを再描画します
    for x1, y1, x2, y2, color, label in last_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.imshow("動画比較 - YOLO", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 動画の最後でまだ差分が続いている場合
if in_abnormal:
    end_time = frame_id / fps
    print(f"=== 差分終了: {end_time:.1f}秒")
    changes.append({
        "start": round(start_time, 1),
        "end": round(end_time, 1),
    })
    # images_save.append(diff_regions)


# ===============================
# メール送信関数
# ===============================
def send_mail(subject, body, images=None):
    EMAIL_SENDER = os.getenv('EMAIL_SENDER')
    EMAIL_PASSWORD = os.getenv('EMAIL_APP_PASSWORD')
    EMAIL_RECEIVER = ['leetuan0388@gmail.com']
    # EMAIL_RECEIVER = ['letuan2k1125@gmail.com', 't_yuki@lsi-dev.co.jp', 'shigeo_tomori0952176@lsi-dev.co.jp']

    em = EmailMessage()
    em['From'] = EMAIL_SENDER
    em['To'] = EMAIL_RECEIVER
    em['Subject'] = subject
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
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(em)
            print("*** メール送信成功")
    except Exception as e:
        print(f"[エラー] メール送信失敗: {e}")


# ===============================
# レポートメール送信
# ===============================
if changes:
    print(f"\n*** 差分が {len(changes)} 件検出されました。メール送信中...")

    subject = "*** 動画差分検出レポート"
    body_lines = ["違いポイントを検出された結果:"]
    i = 1
    for c in changes:
        body_lines.append(f"- {c['start']}秒 ～ {c['end']}秒 ・ image_{i}.jpg")
        i += 1
    body = "\n".join(body_lines)

    try:
        send_mail(subject, body, images_save)
        
        # print(f"*** メール: {subject}, {body}, {images_save}")
    except Exception as e:
        print(f"[エラー] メール送信中のエラー: {e}")
else:
    print("*** 目立った差分は検出されませんでした。")