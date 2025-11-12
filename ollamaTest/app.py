import os
import base64
import io
import requests
from PIL import Image
from flask import Flask, render_template, request, jsonify
import uuid
import time  # Add time library

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- IMAGE COMPRESSION FUNCTION ---
def compress_image(image_path, max_size=(512, 512), quality=80):
    """
    Compress the image to reduce size, convert to JPEG, and return as base64 string.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB to ensure consistent format
            img = img.convert("RGB")

            # Resize (keep aspect ratio)
            img.thumbnail(max_size)

            # Write compressed image to memory buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)

            # Encode to base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return img_base64
    except Exception as e:
        print(f"❌ 画像圧縮エラー: {image_path} -> {e}")  # Image compression error
        return None

# --- IMAGE COMPARISON FUNCTION ---
def compare_images(img1_path, img2_path, prompt="Compare the two images", model="gemma3:12b"):
    url = "https://lsi-dvc.aa0.netvolante.jp/ollama/api/generate"  # Ollama API endpoint
    headers = {"Content-Type": "application/json"}

    # Measure compression time
    compress_start = time.time()
    img1_base64 = compress_image(img1_path)
    img2_base64 = compress_image(img2_path)
    compress_time = time.time() - compress_start

    if not img1_base64 or not img2_base64:
        return {"error": "画像を圧縮できませんでした。"}  # Failed to compress images

    data = {
        "model": "qwen2.5vl:3b",
        "prompt": prompt,
        "images": [img1_base64, img2_base64],
        "stream": False
    }

    try:
        # Measure API request time
        api_start = time.time()
        response = requests.post(url, headers=headers, json=data)
        api_time = time.time() - api_start
        
        if response.status_code == 200:
            result = response.json()
            return {
                "response": result.get('response', 'AIからの応答がありません。'),  # No response from AI
                "timing": {
                    "compress_time": round(compress_time, 2),
                    "api_time": round(api_time, 2),
                    "total_time": round(compress_time + api_time, 2)
                }
            }
        else:
            print(f"⚠️ APIエラー: {response.status_code} - {response.text}")  # API error
            return {
                "error": f"❌ APIエラー: {response.status_code}",
                "timing": {
                    "compress_time": round(compress_time, 2),
                    "api_time": round(api_time, 2),
                    "total_time": round(compress_time + api_time, 2)
                }
            }
    except Exception as e:
        print(f"❌ 接続エラー: {e}")  # Connection error
        return {
            "error": f"❌ 接続エラー: {str(e)}",
            "timing": {
                "compress_time": round(compress_time, 2),
                "api_time": 0,
                "total_time": round(compress_time, 2)
            }
        }

# --- ROUTES ---
@app.route('/')
def index():
    # Render the main HTML upload page
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        # Check if both files are uploaded
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': '両方の画像を選択してください。'}), 400  # Please select both images
        
        image1 = request.files['image1']
        image2 = request.files['image2']
        prompt = request.form.get('prompt', 'Compare the two images')
        
        # Validate file names
        if image1.filename == '' or image2.filename == '':
            return jsonify({'error': '両方の画像を選択してください。'}), 400
        
        # Save temporary files
        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{image1.filename}")
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{image2.filename}")
        
        image1.save(img1_path)
        image2.save(img2_path)
        print("✅ 画像がアップロードされました。")  # Images uploaded successfully
        
        # Compare the two images
        result = compare_images(img1_path, img2_path, prompt)
        
        # Remove temporary files
        try:
            os.remove(img1_path)
            os.remove(img2_path)
            print("🧹 一時ファイルを削除しました。")  # Temporary files deleted
        except Exception as e:
            print(f"⚠️ 一時ファイル削除エラー: {e}")  # Failed to delete temp files
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ 処理エラー: {e}")  # Processing error
        return jsonify({'error': f'処理エラー: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the Flask app on all interfaces for local network access
    app.run(debug=True, host='0.0.0.0', port=5000)
