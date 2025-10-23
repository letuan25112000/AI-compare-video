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