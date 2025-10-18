import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

class VideoProcessor:
    def __init__(self, model_path="models/best_main.pt", conf_thresh=0.6, frame_delay_time=0.2, no_error_limit=5):
        # YOLOモデルをロード
        self.model = YOLO(model_path, task="detect")
        self.CONF_THRESH = conf_thresh
        self.NO_ERROR_LIMIT = no_error_limit
        self.FRAME_DELAY_TIME = frame_delay_time  # 秒単位
        self.MODEL_CLASS_IDS = ["BT", "Wifi", "Cel", "Hots", "Bri", "Dev"]
        self.MODEL_CLASS_IDS_JP = [
            "ブルートゥース", "Wi-Fi", "セルラー",
            "テザリング", "輝度", "開発"
        ]
        self.ERROR_CLASS_ID = [2, 3]  # 異常とみなすクラスID

    def process_detections(self, frame, results):
        """YOLOの結果を処理し、バウンディングボックスを描画"""
        detections = []
        error_found = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf < self.CONF_THRESH:
                continue

            color = (0, 255, 0)
            if cls_id in self.ERROR_CLASS_ID:
                color = (0, 0, 255)
                error_found = True

            label = self.MODEL_CLASS_IDS[cls_id]
            detections.append((x1, y1, x2, y2, color, label))

        return detections, error_found

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_id = 0
        in_abnormal = False
        error_frame_count = 0
        start_time = None
        changes = []
        snapshot_images = []

        frame_forward = round(fps * self.FRAME_DELAY_TIME)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            time_sec = frame_id / fps
            results = self.model(frame, imgsz=640, verbose=False)[0]

            detections, error_found = self.process_detections(frame, results)

            # 異常状態を検出
            if error_found:
                error_frame_count = 0
                frame_forward -= 1
                if not in_abnormal and frame_forward <= 0:
                    in_abnormal = True
                    start_time = time_sec
                    snapshot_images.append(frame)
            else:
                if in_abnormal:
                    error_frame_count += 1
                    if error_frame_count >= self.NO_ERROR_LIMIT:
                        in_abnormal = False
                        end_time = time_sec
                        changes.append({
                            "start": round(start_time, 1),
                            "end": round(end_time, 1)
                        })
                        error_frame_count = 0
                        frame_forward = round(fps * self.FRAME_DELAY_TIME)

            # 結果を描画
            for x1, y1, x2, y2, color, label in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # 動画が終了してもまだ異常状態の場合
        if in_abnormal:
            end_time = frame_id / fps
            changes.append({
                "start": round(start_time, 1),
                "end": round(end_time, 1),
            })

        return changes, snapshot_images
