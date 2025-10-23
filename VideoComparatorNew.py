import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from ultralytics import YOLO
import openpyxl
from checkPC import SystemConfig
from utils.help import clean_folder, make_web_ready
from config import MODEL_PATH, MODEL_CLASS_IDS, MODEL_CLASS_IDS_JP, SCREEN_CLASSES, MIN_DIFF_FRAMES

# ===============================
#  動画オブジェクト解析クラス
# ===============================
class VideoObjectAnalyzer:
    def __init__(self, model_path=MODEL_PATH, conf_thresh=0.6, config=None):
        self.CONF_THRESH = conf_thresh
        self.pc_config = config or SystemConfig()
        self.model = YOLO(model_path, task="detect")

    # -------------------------------
    #  動画を解析（フレームごとのオブジェクトを返す）
    # -------------------------------
    def analyze_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けませんでした: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        interval = max(int(fps // self.pc_config.PROCESS_FPS), 1) if fps > 0 else 1

        frame_objects = {}
        frames_dict = {}  # 描画用フレーム保存
        frame_index = 0

        print(f"\n{video_path} の解析を開始します (FPS: {fps:.1f}, 総フレーム: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % interval != 0:
                frame_index += 1
                continue

            results = self.model(frame, imgsz=self.pc_config.IMGSZ, verbose=False)[0]
            class_ids = [int(box.cls[0]) for box in results.boxes if float(box.conf[0]) >= self.CONF_THRESH]

            frame_objects[frame_index] = list(set(class_ids))
            frames_dict[frame_index] = (frame.copy(), results.boxes)
            frame_index += 1

        cap.release()
        print(f"解析完了: {len(frame_objects)} フレーム")
        return frame_objects, frames_dict, fps, (w, h), frame_count

    # -------------------------------
    #  フレームごとの差分比較
    # -------------------------------
    def compare_frame_objects(self, data_org, data_des, min_diff_frames=MIN_DIFF_FRAMES):
        max_index = min(max(data_org.keys()), max(data_des.keys()))
        all_keys = sorted(k for k in data_org.keys() if k <= max_index)
        diff_raw = {}

        # --- ステップ1: 各フレームでの差分を抽出 ---
        for key in all_keys:
            ids_org = set(data_org.get(key, []))
            ids_des = set(data_des.get(key, []))
            added = ids_des - ids_org
            removed = ids_org - ids_des

            # SCREEN_CLASSESに属するものだけを抽出
            added_screen = [MODEL_CLASS_IDS[i] for i in added if MODEL_CLASS_IDS[i] in SCREEN_CLASSES]
            removed_screen = [MODEL_CLASS_IDS[i] for i in removed if MODEL_CLASS_IDS[i] in SCREEN_CLASSES]

            # 該当クラスがある場合のみ記録
            if added_screen or removed_screen:
                diff_raw[key] = {
                    "frame_index": key,
                    "added": [MODEL_CLASS_IDS_JP[i] for i in added if MODEL_CLASS_IDS[i] in SCREEN_CLASSES],
                    "removed": [MODEL_CLASS_IDS_JP[i] for i in removed if MODEL_CLASS_IDS[i] in SCREEN_CLASSES],
                    "cls_en": added_screen + removed_screen,
                }

        # --- ステップ2: 連続しているフレームをグループ化 ---
        diff_result = {}
        if not diff_raw:
            return diff_result

        sorted_keys = sorted(diff_raw.keys())
        current_group = [sorted_keys[0]]

        for i in range(1, len(sorted_keys)):
            # 前のフレームと連続しているか判定
            if sorted_keys[i] == sorted_keys[i - 1] + 1:
                current_group.append(sorted_keys[i])
            else:
                # グループ終了 → 一定以上連続していれば結果に追加
                if len(current_group) >= min_diff_frames:
                    for k in current_group:
                        diff_result[k] = diff_raw[k]
                current_group = [sorted_keys[i]]

        # 最後のグループもチェック
        if len(current_group) >= min_diff_frames:
            for k in current_group:
                diff_result[k] = diff_raw[k]

        return diff_result

    # -------------------------------
    #  差分を描画した動画を保存
    # -------------------------------
    def save_video_with_boxes(self, frames_dict, diff_frames=None, output_path="output.mp4", fps=30, total_frames=None):
        """
        frames_dict: {frame_index: (frame, boxes)}
        diff_frames: {frame_index: {"added":[], "removed":[]}}
        total_frames: 元動画の総フレーム数
        """
        if not frames_dict:
            return

        # 1つのフレームから動画サイズ取得
        h, w = frames_dict[next(iter(frames_dict))][0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        sorted_indices = sorted(frames_dict.keys())
        last_frame_data = None
        last_boxes = None
        last_diff_ids = set()  # 前回の差分オブジェクト

        max_frame = total_frames or sorted_indices[-1]

        for frame_idx in range(max_frame):
            if frame_idx in frames_dict:
                frame, boxes = frames_dict[frame_idx]
                last_frame_data = frame.copy()
                last_boxes = boxes
                diff_info = diff_frames.get(frame_idx) if diff_frames else None
                last_diff_ids = set(diff_info.get("added", []) + diff_info.get("removed", [])) if diff_info else set()
            elif last_frame_data is not None:
                frame = last_frame_data.copy()
                boxes = last_boxes
            else:
                continue

            frame_draw = frame.copy()
            for box in boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = MODEL_CLASS_IDS[cls_id]
                color = (0, 255, 0)
                if label in last_diff_ids:
                    color = (0, 0, 255)
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_draw, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame_draw)

        out.release()
        print(f"動画保存完了: {output_path}")

    # -------------------------------
    #  基準動画との比較処理
    # -------------------------------
    def compare_with_base(self, base_data, video_path: str, result_path: str):
        des_data, frames_dict, fps, _, total_frames = self.analyze_video(video_path)
        diff = self.compare_frame_objects(base_data[0], des_data)

        output_video = Path(result_path) / (Path(video_path).stem + "_diff.mp4")
        self.save_video_with_boxes(frames_dict, diff_frames=diff, output_path=output_video, fps=fps, total_frames=total_frames)

        summary = {
            "diff_frames": len(diff),
            "diff_detail": diff,
            "fps": round(fps, 2)
        }
        return Path(video_path).name, summary

    # -------------------------------
    #  Excel出力
    # -------------------------------
    def export_to_excel(self, results, output_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "比較結果"
        ws.append(["動画名", "判定", "時間範囲", "差分内容"])

        for name, summary in results.items():
            fps = summary.get("fps", 30)
            sorted_frames = sorted(summary["diff_detail"].items())

            merged_entries = []
            prev_text = None
            start_frame = None
            end_frame = None

            for frame_index, info in sorted_frames:
                added_txt = "＋追加: " + ", ".join(info["added"]) if info["added"] else ""
                removed_txt = "－削除: " + ", ".join(info["removed"]) if info["removed"] else ""
                diff_text = "\n".join([x for x in [added_txt, removed_txt] if x])

                if diff_text == prev_text:
                    end_frame = frame_index
                else:
                    if prev_text is not None:
                        merged_entries.append([
                            name,
                            "NG",
                            f"{start_frame/fps:.1f}s ～ {(end_frame+1)/fps:.1f}s",
                            prev_text
                        ])
                    prev_text = diff_text
                    start_frame = frame_index
                    end_frame = frame_index

            if prev_text is not None:
                merged_entries.append([
                    name,
                    "NG",
                    f"{start_frame/fps:.1f}s ～ {(end_frame+1)/fps:.1f}s",
                    prev_text
                ])

            for row in merged_entries:
                ws.append(row)

        wb.save(output_path)
        print(f"Excel出力完了: {output_path}")

    # -------------------------------
    #  メイン処理
    # -------------------------------
    def main(self, org_video: str, video_list: list[str], result_path: str):
        results_dir = Path(result_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        clean_folder(results_dir)

        # 基準動画を一度解析
        base_data = self.analyze_video(org_video)

        # 基準動画を描画して保存
        base_video_path = Path(result_path) / "base_video_boxes.mp4"
        self.save_video_with_boxes(
            base_data[1],
            output_path=base_video_path,
            fps=base_data[2],
            total_frames=base_data[4]
        )

        saved_videos = [str(base_video_path)]
        results = {}

        if self.pc_config.MAX_WORKERS == 1:
            for v in video_list:
                name, summary = self.compare_with_base(base_data, v, result_path)
                results[name] = summary
                saved_videos.append(str(Path(result_path) / (Path(v).stem + "_diff.mp4")))
        else:
            with ProcessPoolExecutor(max_workers=self.pc_config.MAX_WORKERS) as executor:
                futures = [executor.submit(self.compare_with_base, base_data, v, result_path) for v in video_list]
                for f in as_completed(futures):
                    name, summary = f.result()
                    results[name] = summary
                    vname = name.rsplit(".", 1)[0]
                    saved_videos.append(str(Path(result_path) / (vname + "_diff.mp4")))

        excel_path = Path(result_path) / "compare_summary.xlsx"
        self.export_to_excel(results, excel_path)

        video_paths = [make_web_ready(p) for p in saved_videos]
        video_paths = [str(p).replace("\\", "/") for p in video_paths]

        return excel_path, video_paths


# ===============================
#  実行例
# ===============================
if __name__ == "__main__":
    comparator = VideoObjectAnalyzer()
    excel_path, saved_videos = comparator.main(
        org_video="videos/A_fixed.mp4",
        video_list=["videos/B_fixed.mp4"],
        result_path="results/VideoComparator"
    )

    print(excel_path, saved_videos)
