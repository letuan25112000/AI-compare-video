import os
import json
import time
import base64
import io
import requests
from PIL import Image
from config import API_MODELS, API_PROMPT


class ImageDiffAI:
    """
    2枚の画像をAIに送信して比較し、diff_frames.jsonを更新するクラス。
    """

    def __init__(
        self,
        result_dir,
        api_model=API_MODELS[0],
        api_prompt=API_PROMPT,
        api_url="https://lsi-dvc.aa0.netvolante.jp/ollama/api/generate",
        max_size=(512, 512),
        quality=80,
        delay=1.0,
        stop_event=None
    ):
        self.result_dir = result_dir
        self.api_model = api_model
        self.api_prompt = api_prompt
        self.api_url = api_url
        self.max_size = max_size
        self.quality = quality
        self.delay = delay
        self.stop_event = stop_event
        self.headers = {"Content-Type": "application/json"}
        self.json_file = "diff_frames.json"

    # --------------------------------------------------
    # 画像を圧縮してBase64文字列に変換
    # --------------------------------------------------
    def compress_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.thumbnail(self.max_size)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=self.quality)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"[圧縮エラー] {image_path}: {e}")
            return None

    # --------------------------------------------------
    # 2枚の画像をAIで比較
    # --------------------------------------------------
    def compare_images(self, img1_path, img2_path):
        if self.stop_event and self.stop_event.is_set():
            print("停止要求を検出。compare_images() を中断。")
            return {"error": "処理が停止されました。"}

        img1_base64 = self.compress_image(img1_path)
        img2_base64 = self.compress_image(img2_path)
        if not img1_base64 or not img2_base64:
            return {"error": "画像を圧縮できませんでした。"}

        data = {
            "model": self.api_model,
            "prompt": self.api_prompt,
            "images": [img1_base64, img2_base64],
            "stream": False,
        }

        try:
            api_start = time.time()
            with requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=10,
                stream=True
            ) as response:
                while not response.ok:
                    if self.stop_event and self.stop_event.is_set():
                        print("停止要求を検出。AIリクエストを中断。")
                        return {"error": "処理が停止されました。"}
                    time.sleep(0.5)

                result = response.json()
                api_time = round(time.time() - api_start, 2)

            return {
                "response": result.get("response", "AIからの応答がありません。"),
                "timing": {"api_time": api_time},
            }

        except requests.Timeout:
            return {"error": "AIリクエストがタイムアウトしました。"}
        except Exception as e:
            if self.stop_event and self.stop_event.is_set():
                print("AI比較中に停止要求を検出。")
                return {"error": "処理が停止されました。"}
            return {"error": f"接続エラー: {str(e)}"}


    # --------------------------------------------------
    # diff_frames.jsonを処理してAIの結果を追加
    # --------------------------------------------------
    def handle_diff_frames(self):
        frame_data_path = os.path.join(self.result_dir, self.json_file)

        if not os.path.exists(frame_data_path):
            print(f"{self.json_file} が見つかりません。")
            return None

        with open(frame_data_path, "r", encoding="utf-8") as f:
            frame_data = json.load(f)

        print(f"AI分析開始: グループ数 = {len(frame_data)}")

        for group_idx, video_group in enumerate(frame_data, start=1):
            for frame_idx, frame_pair in enumerate(video_group, start=1):
                if self.stop_event and self.stop_event.is_set():
                    print("停止要求を検出。処理を終了します。")
                    return

                org_frame_path = os.path.join(self.result_dir, frame_pair["org_frame"])
                compare_frame_path = os.path.join(self.result_dir, frame_pair["compare_frame"])
                print(f"[{group_idx}-{frame_idx}] {frame_pair['org_frame']} vs {frame_pair['compare_frame']}")

                ai_result = self.compare_images(org_frame_path, compare_frame_path)

                if self.stop_event and self.stop_event.is_set():
                    print("停止要求を検出。compare_images後に終了します。")
                    return

                frame_pair["diff_text_ai"] = ai_result.get("response") if not ai_result.get("error") else ai_result["error"]
                frame_pair["timing"] = ai_result.get("timing", {})

                with open(frame_data_path, "w", encoding="utf-8") as f:
                    json.dump(frame_data, f, ensure_ascii=False, indent=2)

                for _ in range(int(self.delay * 10)): 
                    if self.stop_event and self.stop_event.is_set():
                        print("停止要求を検出。スリープ中に終了します。")
                        return
                    time.sleep(0.1)

        print(f"{self.json_file} を更新しました。")
        return frame_data_path

    # --------------------------------------------------
    # diff_text_aiをクリアして再分析に備える
    # --------------------------------------------------
    def clear_api_result_json(self):
        print("[再分析準備] diff_text_aiをリセットします。")

        frame_data_path = os.path.join(self.result_dir, self.json_file)
        if os.path.exists(frame_data_path):
            with open(frame_data_path, "r", encoding="utf-8") as f:
                frame_data = json.load(f)

            for video_group in frame_data:
                for frame_pair in video_group:
                    frame_pair.pop("diff_text_ai", None)
                    frame_pair.pop("timing", None)

            with open(frame_data_path, "w", encoding="utf-8") as f:
                json.dump(frame_data, f, ensure_ascii=False, indent=2)

            print("[完了] diff_text_aiをリセットしました。")


# --------------------------------------------------
# 実行例
# --------------------------------------------------
if __name__ == "__main__":
    result_dir = "./results"
    ai_diff = ImageDiffAI(result_dir)
    ai_diff.handle_diff_frames()
