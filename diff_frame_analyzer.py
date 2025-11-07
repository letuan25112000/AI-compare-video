import os
import json
import time
import base64
import io
import requests
from PIL import Image

def compress_image(image_path, max_size=(512, 512), quality=80):
    """
    画像を圧縮してサイズを小さくし、JPEG形式に変換してBase64文字列として返す。
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail(max_size)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)

            # Base64にエンコード
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return img_base64
    except Exception as e:
        print(f"❌ 画像圧縮エラー: {image_path} -> {e}")
        return None
    

def compare_images(img1_path, img2_path, prompt="2枚の画像を比較してください。"):
    url = "https://lsi-dvc.aa0.netvolante.jp/ollama/api/generate"  # Ollama APIエンドポイント
    headers = {"Content-Type": "application/json"}

    # 圧縮時間を計測
    compress_start = time.time()
    img1_base64 = compress_image(img1_path)
    img2_base64 = compress_image(img2_path)
    compress_time = time.time() - compress_start

    if not img1_base64 or not img2_base64:
        return {"error": "❌ 画像を圧縮できませんでした。"}

    data = {
        "model": "gemma3:12b",
        "prompt": prompt,
        "images": [img1_base64, img2_base64],
        "stream": False
    }

    try:
        # APIリクエスト時間を計測
        api_start = time.time()
        response = requests.post(url, headers=headers, json=data)
        api_time = time.time() - api_start
        
        if response.status_code == 200:
            result = response.json()
            return {
                "response": result.get('response', 'AIからの応答がありません。'),
                "timing": {
                    "compress_time": round(compress_time, 2),
                    "api_time": round(api_time, 2),
                    "total_time": round(compress_time + api_time, 2)
                }
            }
        else:
            print(f"⚠️ APIエラー: {response.status_code} - {response.text}")
            return {
                "error": f"❌ APIエラー: {response.status_code}",
                "timing": {
                    "compress_time": round(compress_time, 2),
                    "api_time": round(api_time, 2),
                    "total_time": round(compress_time + api_time, 2)
                }
            }
    except Exception as e:
        print(f"❌ 接続エラー: {e}")
        return {
            "error": f"❌ 接続エラー: {str(e)}",
            "timing": {
                "compress_time": round(compress_time, 2),
                "api_time": 0,
                "total_time": round(compress_time, 2)
            }
        }
    

def handle_diff_frames(result_dir):
    frame_data_path = os.path.join(result_dir, "diff_frames.json")

    if not os.path.exists(frame_data_path):
        print("⚠️ diff_frames.json が見つかりません。")
        return None

    # YOLOの出力データを読み込む
    with open(frame_data_path, "r", encoding="utf-8") as f:
        frame_data = json.load(f)

    print(f"🔁 AIによる処理を開始します。動画グループ数: {len(frame_data)}")

    for group_idx, video_group in enumerate(frame_data):  # 各グループは1つの動画に対応
        print(f"🎬 グループ {group_idx+1}/{len(frame_data)}: {len(video_group)} 枚のフレームを比較")

        for frame_idx, frame_pair in enumerate(video_group):
            org_frame_path = os.path.join(result_dir, frame_pair["org_frame"])
            compare_frame_path = os.path.join(result_dir, frame_pair["compare_frame"])

            print(f"🟦 [{group_idx+1}-{frame_idx+1}] 比較中: {frame_pair['org_frame']} vs {frame_pair['compare_frame']}")

            # AIによる画像比較
            ai_result = compare_images(
                org_frame_path,
                compare_frame_path,
                prompt="2枚の画像の違いを説明してください。"
            )

            # 結果またはエラーメッセージを取得
            diff_text_ai = ai_result.get("response") if not ai_result.get("error") else ai_result["error"]

            # データを更新
            frame_pair["diff_text_ai"] = diff_text_ai
            frame_pair["timing"] = ai_result.get("timing", {})

            # 各フレームごとにJSONを更新して保存
            with open(frame_data_path, "w", encoding="utf-8") as f:
                json.dump(frame_data, f, ensure_ascii=False, indent=2)

            time.sleep(1)  # APIを連続で呼び出しすぎないように少し待機

    print("✅ diff_frames.json に diff_text_ai を直接更新しました。")
    return frame_data_path
