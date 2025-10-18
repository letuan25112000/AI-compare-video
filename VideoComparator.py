import cv2
import numpy as np
from ultralytics import YOLO
import os
from skimage.metrics import structural_similarity as ssim
from dotenv import load_dotenv

load_dotenv()


class VideoComparator:
    def __init__(self, model_path="models/best_main.pt", conf_thresh=0.6, process_fps=3, no_error_limit=5):
        self.MODEL_CLASS_IDS = ["BT", "Wifi", "Cel", "Hots", "Bri", "Dev"]
        self.MODEL_CLASS_IDS_JP = [
            "ブルートゥース", "Wi-Fi", "セルラー",
            "テザリング", "輝度", "開発"
        ]
        self.ERROR_CLASS_ID = [2, 3]
        self.CONF_THRESH = conf_thresh
        self.PROCESS_FPS = process_fps
        self.NO_ERROR_LIMIT = no_error_limit

        self.model = YOLO(model_path, task='detect')

    # ===============================
    # ピクセル比較
    # ===============================
    def compare_video_pixel(self, video1, video2, threshold=0.8):
        """SSIMでピクセル比較し、差分フレームindexを返す"""
        cap1 = cv2.VideoCapture(video1)
        cap2 = cv2.VideoCapture(video2)
        diff_frames = []
        frame_index = 0

        if not cap1.isOpened() or not cap2.isOpened():
            raise ValueError("動画を開けませんでした。")

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

    # ===============================
    # YOLO検出描画
    # ===============================
    def process_detections(self, frame, results):
        """YOLO結果からclass_idを抽出し、bbox描画"""
        frame_class_ids = []
        error_found = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf < self.CONF_THRESH:
                continue
            frame_class_ids.append(cls_id)

            color = (0, 0, 255) if cls_id in self.ERROR_CLASS_ID else (0, 255, 0)
            if cls_id in self.ERROR_CLASS_ID:
                error_found = True

            label = self.MODEL_CLASS_IDS[cls_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return frame_class_ids, error_found

    # ===============================
    # 差分Class ID比較
    # ===============================
    def class_ids_diff(self, ids1, ids2):
        """ids2にしかないClass IDを返す"""
        return set(ids2) - set(ids1)

    # ===============================
    # メイン比較処理
    # ===============================
    def compare_videos(self, video_org, video_des):
        cap1 = cv2.VideoCapture(video_org)
        cap2 = cv2.VideoCapture(video_des)

        fps = cap1.get(cv2.CAP_PROP_FPS)
        frame_id = 0
        in_abnormal = False
        start_time = None
        no_error_count = 0
        images_save = []
        changes = []
        icon_name = []

        print("=== Compare pixel ===")
        diff_frames = self.compare_video_pixel(video_org, video_des)

        print("=== Compare AI ===")
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break
            frame_id += 1
            time_sec = frame_id / fps

            if frame_id in diff_frames:
                results1 = self.model(frame1, imgsz=640, verbose=False)[0]
                results2 = self.model(frame2, imgsz=640, verbose=False)[0]

                frame1_ids, _ = self.process_detections(frame1, results1)
                frame2_ids, error_found = self.process_detections(frame2, results2)

                cls_diff = self.class_ids_diff(frame1_ids, frame2_ids)
                if cls_diff:
                    error_found = True

                # 差分検知開始
                if error_found:
                    no_error_count = 0
                    if not in_abnormal:
                        icon_name = cls_diff
                        in_abnormal = True
                        start_time = time_sec
                        print(f"=== 差分開始: {start_time:.1f}秒")
                        images_save.append(frame2)
                else:
                    if in_abnormal:
                        no_error_count += 1
                        if no_error_count >= self.NO_ERROR_LIMIT:
                            in_abnormal = False
                            end_time = time_sec
                            print(f"=== 差分終了: {end_time:.1f}秒")

                            temp = ", ".join([self.MODEL_CLASS_IDS_JP[i] for i in icon_name])
                            changes.append({
                                "start": round(start_time, 1),
                                "end": round(end_time, 1),
                                "error_point": temp
                            })
                            no_error_count = 0

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

        # 最後の差分が続いている場合
        if in_abnormal:
            end_time = frame_id / fps
            temp = ", ".join([self.MODEL_CLASS_IDS_JP[i] for i in icon_name])
            changes.append({
                "start": round(start_time, 1),
                "end": round(end_time, 1),
                "error_point": temp
            })

        return changes, images_save
