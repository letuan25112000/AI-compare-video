import os
import base64
import io
import requests
from PIL import Image
from flask import Flask, render_template, request, jsonify
import uuid
import time  # Thêm thư viện time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Tạo thư mục upload nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- HÀM NÉN ẢNH ---
def compress_image(image_path, max_size=(512, 512), quality=80):
    """
    Nén ảnh để giảm dung lượng, chuyển sang JPEG và trả về chuỗi base64.
    """
    try:
        with Image.open(image_path) as img:
            # Chuyển sang RGB để đảm bảo định dạng nhất quán
            img = img.convert("RGB")

            # Resize (giữ tỉ lệ)
            img.thumbnail(max_size)

            # Ghi ảnh nén vào bộ nhớ
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)

            # Encode base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return img_base64
    except Exception as e:
        print(f"❌ Lỗi khi nén ảnh {image_path}: {e}")
        return None

# --- HÀM SO SÁNH ẢNH ---
def compare_images(img1_path, img2_path, prompt="Compare the two images"):
    url = "https://lsi-dvc.aa0.netvolante.jp/ollama/api/generate"  # API Ollama
    headers = {"Content-Type": "application/json"}

    # Đếm thời gian nén ảnh
    compress_start = time.time()
    img1_base64 = compress_image(img1_path)
    img2_base64 = compress_image(img2_path)
    compress_time = time.time() - compress_start

    if not img1_base64 or not img2_base64:
        return {"error": "❌ Không thể nén ảnh."}

    data = {
        "model": "gemma3:12b",
        "prompt": prompt,
        "images": [img1_base64, img2_base64],
        "stream": False
    }

    try:
        # Đếm thời gian request API
        api_start = time.time()
        # Send POST request
        response = requests.post(url, headers=headers, json=data)
        api_time = time.time() - api_start
        
        if response.status_code == 200:
            result = response.json()
            return {
                "response": result.get('response', 'Không có phản hồi từ AI.'),
                "timing": {
                    "compress_time": round(compress_time, 2),
                    "api_time": round(api_time, 2),
                    "total_time": round(compress_time + api_time, 2)
                }
            }
        else:
            return {
                "error": f"❌ Lỗi API: {response.status_code} - {response.text}",
                "timing": {
                    "compress_time": round(compress_time, 2),
                    "api_time": round(api_time, 2),
                    "total_time": round(compress_time + api_time, 2)
                }
            }
    except Exception as e:
        return {
            "error": f"❌ Lỗi kết nối: {str(e)}",
            "timing": {
                "compress_time": round(compress_time, 2),
                "api_time": 0,
                "total_time": round(compress_time, 2)
            }
        }

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        # Kiểm tra file upload
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Vui lòng chọn cả hai ảnh'}), 400
        
        image1 = request.files['image1']
        image2 = request.files['image2']
        prompt = request.form.get('prompt', 'Compare the two images')
        
        # Kiểm tra tên file
        if image1.filename == '' or image2.filename == '':
            return jsonify({'error': 'Vui lòng chọn cả hai ảnh'}), 400
        
        # Lưu file tạm thời
        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{image1.filename}")
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{image2.filename}")
        
        image1.save(img1_path)
        image2.save(img2_path)
        
        # So sánh ảnh
        result = compare_images(img1_path, img2_path, prompt)
        
        # Xóa file tạm
        try:
            os.remove(img1_path)
            os.remove(img2_path)
        except:
            pass
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)