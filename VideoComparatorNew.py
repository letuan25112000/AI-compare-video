import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()


class VideoObjectAnalyzer:
    def __init__(self, model_path="models/best_main.pt", conf_thresh=0.6, process_fps=5):
        self.MODEL_CLASS_IDS = ["BT", "Wifi", "Cel", "Hots", "Bri", "Dev"]
        self.MODEL_CLASS_IDS_JP = [
            "ブルートゥース", "Wi-Fi", "セルラー",
            "テザリング", "輝度", "開発"
        ]
        self.CONF_THRESH = conf_thresh
        self.PROCESS_FPS = process_fps
        self.model = YOLO(model_path, task='detect')

    # ============================================
    # 解析: video → frameごとのclass id一覧
    # ============================================
    def analyze_video(self, video_path):
        """Videoをフレーム単位で解析し、{ frame_index: [class_id1, ...] } を返す"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けませんでした: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        interval = int(fps // self.PROCESS_FPS) if fps > self.PROCESS_FPS else 1

        frame_objects = {}
        frame_index = 0

        print(f"\n=== {video_path} の解析開始 ===")
        print(f"FPS: {fps:.1f}, 総フレーム数: {frame_count}, 再生時間: {duration:.2f} 秒")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 指定間隔で処理（軽量化）
            if frame_index % interval != 0:
                frame_index += 1
                continue

            results = self.model(frame, imgsz=640, verbose=False)[0]
            class_ids = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= self.CONF_THRESH:
                    class_ids.append(cls_id)

            frame_objects[frame_index] = list(set(class_ids))
            frame_index += 1

        cap.release()
        print(f"=== 解析完了: {len(frame_objects)} フレーム ===")
        return frame_objects, duration

    # ============================================
    # 比較: 2つのvideo解析結果を比較（ログ付き）
    # ============================================
    def compare_analysis(self, data_org, data_des):
        """2つの解析結果を比較し、frameごとの差分を返す"""
        # cắt phần dư: chỉ so sánh trong phạm vi chung
        max_index = min(max(data_org.keys()), max(data_des.keys()))
        all_keys = sorted(k for k in data_org.keys() if k <= max_index)

        diff_result = {}

        for key in all_keys:
            ids_org = set(data_org.get(key, []))
            ids_des = set(data_des.get(key, []))

            added = ids_des - ids_org
            removed = ids_org - ids_des

            if added or removed:
                added_names = [self.MODEL_CLASS_IDS_JP[i] for i in added]
                removed_names = [self.MODEL_CLASS_IDS_JP[i] for i in removed]

                diff_result[key] = {
                    "frame_index": key,
                    "added": added_names,
                    "removed": removed_names,
                    "org_ids": [self.MODEL_CLASS_IDS_JP[i] for i in ids_org],
                    "des_ids": [self.MODEL_CLASS_IDS_JP[i] for i in ids_des]
                }

                print(f"\n=== Frame {key} 差分検出 ===")
                print(f"元動画: {', '.join(diff_result[key]['org_ids']) or 'なし'}")
                print(f"比較動画: {', '.join(diff_result[key]['des_ids']) or 'なし'}")
                if added_names:
                    print(f"＋追加: {', '.join(added_names)}")
                if removed_names:
                    print(f"－削除: {', '.join(removed_names)}")

        if not diff_result:
            print("差分なし。")

        return diff_result

    # ============================================
    # 総合処理: 2 video入力 → 差分出力
    # ============================================
    def compare_videos(self, video_org, video_des):
        print("=== 元動画を解析中 ===")
        data_org, duration_org = self.analyze_video(video_org)

        print("=== 比較対象動画を解析中 ===")
        data_des, duration_des = self.analyze_video(video_des)

        print("=== 差分比較中 ===")
        diff = self.compare_analysis(data_org, data_des)

        # Tính thời lượng & xác định video nào dài hơn
        if duration_org > duration_des:
            longer = "org"
        elif duration_des > duration_org:
            longer = "des"
        else:
            longer = "equal"

        duration_diff = abs(duration_org - duration_des)

        summary = {
            "duration_org": round(duration_org, 2),
            "duration_des": round(duration_des, 2),
            "duration_diff": round(duration_diff, 2),
            "longer_video": longer,
            "diff_frames": len(diff),
            "diff_detail": diff
        }

        print("\n=== 比較結果サマリ ===")
        print(f"元動画: {duration_org:.2f}s, 比較動画: {duration_des:.2f}s")
        print(f"差分時間: {duration_diff:.2f}s, 長い方: {longer}")
        print(f"差分フレーム: {len(diff)}")

        return summary


def main():
    A = "videos/A_fixed.mp4"
    B = "videos/B_fixed.mp4"

    comparator = VideoObjectAnalyzer()
    result = comparator.compare_videos(A, B)


if __name__ == "__main__":
    main()
