import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
import openpyxl
from checkPC import SystemConfig
from utils.help import apply_excel_format, clean_folder, convert_to_nfps, make_web_ready
from config import MODEL_PATH, MODEL_CLASS_IDS, MODEL_CLASS_IDS_JP, SCREEN_CLASSES, MIN_DIFF_FRAMES, FPS, PROCESS_FPS
import time
import json
import shutil

# ===============================
#  動画オブジェクト解析クラス
# ===============================
class VideoObjectAnalyzer:
    def __init__(self, model_path=MODEL_PATH, conf_thresh=0.6, config=None, fps=FPS, process_fps=PROCESS_FPS, threshold=MIN_DIFF_FRAMES):
        self.FPS = fps
        self.PROCESS_FPS = process_fps
        self.MIN_DIFF_FRAMES = threshold
        self.CONF_THRESH = conf_thresh
        self.pc_config = config or SystemConfig()
        self.model = YOLO(model_path, task="detect")
        self.temp_files = []  # 一時ファイルを管理
    
    def cleanup_temp_files(self):
        """一時ファイルを削除"""
        for temp_file in self.temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink(missing_ok=True)
        self.temp_files = []
    
    # -------------------------------
    #  動画を解析（フレームごとのオブジェクトを返す）
    # -------------------------------
    def analyze_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けませんでした: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(int(self.FPS // self.PROCESS_FPS), 1) if self.FPS > 0 else 1

        frame_objects = {}
        frames_dict = {}
        frame_index = 0

        print(f"\n{video_path} の解析を開始します (FPS: {self.FPS:.1f}, 総フレーム: {frame_count})")

        # --- Timing start ---
        start_time = time.time()
        processed_frames = 0
        total_frame_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % interval != 0:
                frame_index += 1
                continue

            t1 = time.time()
            results = self.model(frame, imgsz=self.pc_config.IMGSZ, verbose=False, device=self.pc_config.DEVICE)[0]
            t2 = time.time()
            total_frame_time += (t2 - t1)
            processed_frames += 1

            class_ids = [int(box.cls[0]) for box in results.boxes if float(box.conf[0]) >= self.CONF_THRESH]
            frame_objects[frame_index] = list(set(class_ids))
            frames_dict[frame_index] = (frame.copy(), results.boxes)
            frame_index += 1

        cap.release()

        # --- Timing end ---
        end_time = time.time()
        total_time = end_time - start_time
        avg_frame_time = total_frame_time / processed_frames if processed_frames > 0 else 0

        print(f"解析完了: {processed_frames} フレーム / {frame_count} フレーム中")
        print(f"総処理時間: {total_time:.2f} 秒 ({total_time/60:.2f} 分)")
        print(f"1フレームあたり平均処理時間: {avg_frame_time:.3f} 秒")

        return frame_objects, frames_dict, frame_count

    # -------------------------------
    #  フレームごとの差分比較
    # -------------------------------
    def compare_frame_objects(self, data_org, data_des):
        """
        SCREEN_CLASSESの差分はNフレーム連続で発生した場合のみ有効とする
        """
        max_index = min(max(data_org.keys()), max(data_des.keys()))
        all_keys = sorted(k for k in data_org.keys() if k <= max_index)
        diff_result = {}

        # SCREEN_CLASSESの変化追跡
        screen_diff_counter = {cls: 0 for cls in SCREEN_CLASSES}

        for key in all_keys:
            ids_org = set(data_org.get(key, []))
            ids_des = set(data_des.get(key, []))
            added = ids_des - ids_org
            removed = ids_org - ids_des

            # --- SCREEN_CLASSESの変化を処理 ---
            confirmed_added = set()
            confirmed_removed = set()

            for cls in SCREEN_CLASSES:
                in_added = cls in [MODEL_CLASS_IDS[i] for i in added if i < len(MODEL_CLASS_IDS)]
                in_removed = cls in [MODEL_CLASS_IDS[i] for i in removed if i < len(MODEL_CLASS_IDS)]

                if in_added or in_removed:
                    screen_diff_counter[cls] += 1
                else:
                    screen_diff_counter[cls] = 0

                # 連続Nフレーム超えたら確定
                if screen_diff_counter[cls] >= self.MIN_DIFF_FRAMES:
                    if in_added:
                        confirmed_added.add(cls)
                    elif in_removed:
                        confirmed_removed.add(cls)
            # --- 通常クラス ---
            normal_added = [MODEL_CLASS_IDS[i] for i in added if MODEL_CLASS_IDS[i] not in SCREEN_CLASSES]
            normal_removed = [MODEL_CLASS_IDS[i] for i in removed if MODEL_CLASS_IDS[i] not in SCREEN_CLASSES]

            added_final = list(set(normal_added) | confirmed_added)
            removed_final = list(set(normal_removed) | confirmed_removed)

            if added_final or removed_final:
                diff_result[key] = {
                    "frame_index": key,
                    "added": [MODEL_CLASS_IDS[MODEL_CLASS_IDS.index(a)] for a in added_final],
                    "removed": [MODEL_CLASS_IDS[MODEL_CLASS_IDS.index(r)] for r in removed_final]
                }

        return diff_result

    # -------------------------------
    #  差分を描画した動画を保存
    # -------------------------------
    def save_video_with_boxes(self, frames_dict, diff_frames=None, output_path="output.mp4", total_frames=None):
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
        out = cv2.VideoWriter(str(output_path), fourcc, self.FPS, (w, h))

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
    #  特定のフレームを画像として保存
    # -------------------------------
    def extract_frame_as_image(self, video_path: str, frame_time: float, output_path: str):
        """指定した時間のフレームを画像として保存"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けませんでした: {video_path}")
        
        # フレーム時間をフレーム番号に変換
        frame_number = int(frame_time * self.FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(output_path), frame)
            print(f"フレーム保存: {output_path} (時間: {frame_time}s)")
        else:
            print(f"フレーム抽出失敗: {video_path} at {frame_time}s")
        
        cap.release()
        return ret

    # -------------------------------
    #  差分フレームの画像を抽出（全て）
    # -------------------------------
    def extract_diff_frames(self, org_video: str, compare_video: str, diff_summary: dict, result_path: Path, video_name: str):
        """差分があるすべてのフレームを画像として抽出"""
        if not diff_summary.get("diff_detail"):
            return []

        fps = diff_summary.get("fps", self.FPS)
        sorted_frames = sorted(diff_summary["diff_detail"].items())

        results = []

        prev_text = None
        start_frame = None
        end_frame = None
        jp_map = dict(zip(MODEL_CLASS_IDS, MODEL_CLASS_IDS_JP))

        # --- 差分内容をグループ化 ---
        grouped_entries = []
        for frame_index, info in sorted_frames:
            added_txt = "＋追加: " + ", ".join(jp_map.get(x, x) for x in info["added"]) if info["added"] else ""
            removed_txt = "－削除: " + ", ".join(jp_map.get(x, x) for x in info["removed"]) if info["removed"] else ""
            diff_text = "\n".join([x for x in [added_txt, removed_txt] if x])

            if diff_text == prev_text:
                end_frame = frame_index
            else:
                if prev_text is not None:
                    grouped_entries.append((start_frame, end_frame, prev_text))
                prev_text = diff_text
                start_frame = frame_index
                end_frame = frame_index

        if prev_text is not None:
            grouped_entries.append((start_frame, end_frame, prev_text))

        # --- 各グループの代表フレームを抽出 ---
        for idx, (start_frame, end_frame, diff_text) in enumerate(grouped_entries, start=1):
            mid_frame = (start_frame + end_frame) // 2
            frame_time = mid_frame / fps

            # 出力ファイル名
            org_frame_path = result_path / f"{Path(video_name).stem}_org_{idx:02d}.jpg"
            compare_frame_path = result_path / f"{Path(video_name).stem}_compare_{idx:02d}.jpg"

            # 抽出
            self.extract_frame_as_image(org_video, frame_time, org_frame_path)
            self.extract_frame_as_image(compare_video, frame_time, compare_frame_path)

            results.append({
                "video_name": video_name,
                "range": f"{start_frame/fps:.1f}s ～ {(end_frame+1)/fps:.1f}s",
                "org_frame": org_frame_path.name,
                "compare_frame": compare_frame_path.name,
                "frame_time": frame_time,
                "diff_text": diff_text,
            })

        return results


    # -------------------------------
    #  基準動画との比較処理（シングルスレッド用）
    # -------------------------------
    def compare_with_origin_single(self, origin_data, video_path: str, result_path: str, org_video_path: str):
        """シングルスレッド用の比較処理"""
        # --- 対象動画をN fpsに変換 ---
        video_nfps = convert_to_nfps(video_path, self.FPS)
        self.temp_files.append(video_nfps)  # 一時ファイルとして登録
        
        try:
            des_data, frames_dict, total_frames = self.analyze_video(video_nfps)
            diff = self.compare_frame_objects(origin_data[0], des_data)

            output_video = Path(result_path) / (Path(video_path).stem + "_diff.mp4")
            self.save_video_with_boxes(frames_dict, diff_frames=diff, output_path=output_video, total_frames=total_frames)

            summary = {
                "diff_frames": len(diff),
                "diff_detail": diff,
                "fps": round(self.FPS, 2)
            }
            
            # 差分フレーム画像を抽出
            frame_data = self.extract_diff_frames(org_video_path, video_path, summary, Path(result_path), Path(video_path).name)
            
        except Exception as e:
            raise e
            
        return Path(video_path).name, summary, frame_data

    # -------------------------------
    #  基準動画との比較処理（マルチスレッド用）
    # -------------------------------
    def compare_with_origin_multi(self, origin_data, video_path: str, result_path: str, org_video_nfps: str):
        """マルチスレッド用の比較処理 - 元動画は事前変換済み"""
        # --- 対象動画をN fpsに変換 ---
        video_nfps = convert_to_nfps(video_path, self.FPS)
        self.temp_files.append(video_nfps)  # 一時ファイルとして登録
        
        try:
            des_data, frames_dict, total_frames = self.analyze_video(video_nfps)
            diff = self.compare_frame_objects(origin_data[0], des_data)

            output_video = Path(result_path) / (Path(video_path).stem + "_diff.mp4")
            self.save_video_with_boxes(frames_dict, diff_frames=diff, output_path=output_video, total_frames=total_frames)

            summary = {
                "diff_frames": len(diff),
                "diff_detail": diff,
                "fps": round(self.FPS, 2)
            }
            
            # 差分フレーム画像を抽出（元動画は変換済みのものを使う）
            frame_data = self.extract_diff_frames(org_video_nfps, video_nfps, summary, Path(result_path), Path(video_path).name)
            
        except Exception as e:
            raise e
            
        return Path(video_path).name, summary, frame_data

    # -------------------------------
    #  Excel出力
    # -------------------------------
    def export_to_excel(self, results, output_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "比較結果"
        ws.append(["動画名", "判定", "時間範囲", "差分内容"])

        jp_map = dict(zip(MODEL_CLASS_IDS, MODEL_CLASS_IDS_JP))

        for name, summary in results.items():
            fps = summary.get("fps", 30)
            sorted_frames = sorted(summary["diff_detail"].items())

            merged_entries = []
            prev_text = None
            start_frame = None
            end_frame = None

            for frame_index, info in sorted_frames:
                added_txt  = "＋追加: " + ", ".join(jp_map.get(x, x) for x in info["added"]) if info["added"] else ""
                removed_txt = "－削除: " + ", ".join(jp_map.get(x, x) for x in info["removed"]) if info["removed"] else ""
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

        apply_excel_format(ws)
        wb.save(output_path)
        print(f"Excel出力完了: {output_path}")

    # -------------------------------
    #  メイン処理
    # -------------------------------
    def main(self, org_video: str, video_list: list[str], result_path: str):
        results_dir = Path(result_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        clean_folder(results_dir)

        # --- 基準動画をN fpsに変換 ---
        org_video_nfps = convert_to_nfps(org_video, self.FPS)
        self.temp_files.append(org_video_nfps)  # 一時ファイルとして登録
        
        # 基準動画を一度解析
        origin_data = self.analyze_video(org_video_nfps)

        # 基準動画を描画して保存
        origin_video_path = Path(result_path) / "origin_video_boxes.mp4"
        self.save_video_with_boxes(
            origin_data[1],
            output_path=origin_video_path,
            total_frames=origin_data[2]
        )

        saved_videos = []
        results = {}
        frame_data_list = []

        try:
            if self.pc_config.MAX_WORKERS == 1:
                # シングルスレッド処理
                for v in video_list:
                    name, summary, frame_data = self.compare_with_origin_single(origin_data, v, result_path, org_video)
                    results[name] = summary
                    saved_videos.append(str(Path(result_path) / (Path(v).stem + "_diff.mp4")))
                    if frame_data:
                        frame_data_list.append(frame_data)
            else:
                # マルチスレッド処理
                with ThreadPoolExecutor(max_workers=self.pc_config.MAX_WORKERS) as executor:
                    futures = [executor.submit(self.compare_with_origin_multi, origin_data, v, result_path, org_video_nfps) for v in video_list]
                    for f in as_completed(futures):
                        name, summary, frame_data = f.result()
                        results[name] = summary
                        vname = name.rsplit(".", 1)[0]
                        saved_videos.append(str(Path(result_path) / (vname + "_diff.mp4")))
                        if frame_data:
                            frame_data_list.append(frame_data)

            # 結果処理
            excel_path = Path(result_path) / "compare_summary.xlsx"
            self.export_to_excel(results, excel_path)

            # フレームデータをJSONに保存
            if frame_data_list:
                frame_json_path = Path(result_path) / "diff_frames.json"
                with open(frame_json_path, 'w', encoding='utf-8') as f:
                    json.dump(frame_data_list, f, ensure_ascii=False, indent=2)

            excel_name = excel_path.name
            org_path = make_web_ready(str(origin_video_path))
            org_path = str(org_path).replace("\\", "/")
            video_paths = [make_web_ready(p) for p in saved_videos]
            video_paths = [str(p).replace("\\", "/") for p in video_paths]

            return excel_name, org_path, video_paths, frame_data_list

        finally:
            # 一時ファイルをクリーンアップ
            self.cleanup_temp_files()

# ===============================
#  実行例
# ===============================
if __name__ == "__main__":
    comparator = VideoObjectAnalyzer(fps=30, process_fps=5, threshold=5)
    excel_path, org_path, saved_videos, frame_data = comparator.main(
        org_video="videos/A_fixed.mp4",
        video_list=["videos/B_fixed.mp4"],
        result_path="results"
    )

    print(excel_path, org_path, saved_videos, frame_data)