import os, sys

MODEL_PATH="models/best_0301.pt"

MODEL_CLASS_IDS = [
            "BT", 
            "Wifi", 
            "Cel", 
            "Hots", 
            "Bri", 
            "Dev", 
            "Home", 
            "Radio",
        ]
MODEL_CLASS_IDS_JP = [
            "ブルートゥース", 
            "Wi-Fi", 
            "セルラー", 
            "テザリング", 
            "輝度", 
            "開発", 
            "ホーム", 
            "ラジオ",
        ]
SCREEN_CLASSES = {"Home", "Radio"} 

MIN_DIFF_FRAMES = 5

FPS = 30

PROCESS_FPS = 10

# -----------------------------
# Detect base directory
# -----------------------------
if getattr(sys, 'frozen', False):
    # Running as EXE
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running from source
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Define key folders
# -----------------------------
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FFMPEG_PATH = os.path.join(BASE_DIR,"ffmpeg", "ffmpeg.exe")