import cv2
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from ultralytics import YOLO
from dotenv import load_dotenv
import openpyxl
import shutil
from checkPC import SystemConfig

load_dotenv()

def clean_folder(results_dir):
    for f in results_dir.glob("*"):
        try:
            if f.is_file() or f.is_symlink():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        except Exception as e:
            print(f"Không thể xóa {f}: {e}")
    print("Đã dọn sạch thư mục results/")

class VideoObjectAnalyzer:
    def __init__(self, model_path="models/best_main.pt", conf_thresh=0.6, config=None):
        self.MODEL_CLASS_IDS = ["BT", "Wifi", "Cel", "Hots", "Bri", "Dev"]
        self.MODEL_CLASS_IDS_JP = ["ブルートゥース", "Wi-Fi", "セルラー", "テザリング", "輝度", "開発"]
        self.CONF_THRESH = conf_thresh
        self.config = config or SystemConfig()
        self.model = YOLO(model_path, task='detect')

    def analyze_video(self, video_path):
        print("Video path: ", video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けませんでした: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        interval = int(fps // self.config.PROCESS_FPS) if fps > self.config.PROCESS_FPS else 1

        frame_objects = {}
        frame_index = 0

        print(f"\n{video_path} の解析開始")
        print(f"FPS: {fps:.1f}, 総フレーム数: {frame_count}, 再生時間: {duration:.2f} 秒")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % interval != 0:
                frame_index += 1
                continue

            results = self.model(frame, imgsz=self.config.IMGSZ, verbose=False)[0]
            class_ids = [int(box.cls[0]) for box in results.boxes if float(box.conf[0]) >= self.CONF_THRESH]
            frame_objects[frame_index] = list(set(class_ids))
            frame_index += 1

        cap.release()
        print(f"解析完了: {len(frame_objects)} フレーム")
        return frame_objects, duration, fps

    def compare_analysis(self, data_org, data_des):
        max_index = min(max(data_org.keys()), max(data_des.keys()))
        all_keys = sorted(k for k in data_org.keys() if k <= max_index)
        diff_result = {}

        for key in all_keys:
            ids_org = set(data_org.get(key, []))
            ids_des = set(data_des.get(key, []))
            added = ids_des - ids_org
            removed = ids_org - ids_des

            if added or removed:
                diff_result[key] = {
                    "frame_index": key,
                    "added": [self.MODEL_CLASS_IDS[i] for i in added],
                    "removed": [self.MODEL_CLASS_IDS[i] for i in removed]
                }

        return diff_result

    def compare_videos(self, video_org, video_des):
        data_org, duration_org, fps_org = self.analyze_video(video_org)
        data_des, duration_des, fps_des = self.analyze_video(video_des)
        diff = self.compare_analysis(data_org, data_des)

        summary = {
            "duration_org": round(duration_org, 2),
            "duration_des": round(duration_des, 2),
            "diff_frames": len(diff),
            "diff_detail": diff,
            "fps": round(fps_des or fps_org, 2)
        }
        return summary


    def process_video_pair(self, video_org, video_des, config):
        analyzer = VideoObjectAnalyzer(config=config)
        summary = analyzer.compare_videos(video_org, video_des)
        return Path(video_des).name, summary


    # ============================================
    #  Excel出力用関数
    # ============================================
    def export_to_excel(self, results, output_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "比較結果"
        headers = ["先動画", "確認結果", "時刻", "違いポイント"]
        ws.append(headers)

        for name, summary in results.items():
            diff_detail = summary["diff_detail"]
            fps = summary.get("fps", 30)  # FPS thực từ video

            for frame_index, info in diff_detail.items():
                start_s = frame_index / fps
                end_s = (frame_index + 1) / fps
                time_str = f"{start_s:.1f}s ~ {end_s:.1f}s"

                added_txt = "＋追加: " + ", ".join(info["added"]) if info["added"] else ""
                removed_txt = "－削除: " + ", ".join(info["removed"]) if info["removed"] else ""
                diff_text = "\n".join([x for x in [added_txt, removed_txt] if x])

                ws.append([name, "NG", time_str, diff_text])

        wb.save(output_path)
        print(f"Excel出力完了: {output_path}")

    def main(self, org_video, video_list, result_path):
        config = self.config

        results_dir = Path("results/VideoComparator")
        results_dir.mkdir(exist_ok=True)

        clean_folder(results_dir)

        print("\n開始: 一括比較モード")
        results = {}

        if config.MAX_WORKERS == 1:
            for v in video_list:
                name, summary = self.process_video_pair(org_video, v, config)
                results[name] = summary
                json.dump(summary, open(results_dir / f"{name}_diff.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                print(f"{name}: 差分 {summary['diff_frames']} フレーム")
        else:
            with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                futures = [executor.submit(self.process_video_pair, org_video, v, config) for v in video_list]
                for f in as_completed(futures):
                    name, summary = f.result()
                    results[name] = summary
                    json.dump(summary, open(results_dir / f"{name}_diff.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                    print(f"{name}: 差分 {summary['diff_frames']} フレーム")

        # --- Excelに出力 ---
        excel_path = result_path + "/compare_summary.xlsx"
        self.export_to_excel(results, excel_path)

        return "compare_summary.xlsx"


