# 🤖 AI-compare-video

---

## ⚙️ インストール手順

### 1. リポジトリをクローンする

```bash
git clone https://github.com/letuan25112000/AI-compare-video.git
cd AI-compare-video
```

### 2. （任意）仮想環境を作成して有効化する

### Windows の場合

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS の場合

```bash
python -m venv venv
source venv/bin/activate
```

### 3. 依存関係をインストールする

```bash
pip install -r requirements.txt
```

### 4. 実行方法

```bash
python main.py
```

http://localhost:5000/ また　http://127.0.0.1:5000/
にアクセスしてください。

## 現場的のトレーニングデータ作成

### 1.フォルダ構成

YOLO が認識できるデータセット構成は以下のようになります：
例：dataset_example フォルダを確認してください

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   └── val/
│       ├── image3.jpg
│       ├── image4.jpg
│
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   └── val/
│       ├── image3.txt
│       ├── image4.txt
│
└── dataset.yaml
```

### 2.dataset.yaml ファイル

このファイルで、YOLO に学習用データの場所とクラス情報を教えます。

例：

```yaml
train: ./images/train
val: ./images/val

nc: 6
names: ["Bluetooth", "Wifi", "Celular", "Hotspot", "Brightness", "Development"]
```

・train: 学習用画像フォルダへのパス

・val: 検証用画像フォルダへのパス

・nc: クラス数（例: 3 クラス）

・names: クラス名のリスト

### 3. ラベルファイル（.txt）の形式

各画像に対応する .txt ファイルを同じ名前で作成します。

```
image1.jpg
image1.txt
```

.txt ファイルの中身は以下のようなフォーマットになります：

```
<class_id> <x_center> <y_center> <width> <height>
```

例：

・dataset/train/images/image_002.jpg
![alt text](ReadmeImg/image_016.png)

・dataset/train/labels/image_002.txt

```
0 0.020028 0.344936 0.030913 0.020681
2 0.061390 0.345568 0.030913 0.019418
4 0.101445 0.345481 0.029171 0.020463
5 0.143243 0.345916 0.030042 0.020463
```

クラス 0 の物体が、画像の中央(0.020028, 0.344936)、幅 0.030913、高さ 0.020681

### 4.　学習コマンドの例

```bash
yolo train model=yolov8n.pt data=dataset.yaml epochs=100 imgsz=640
```

トレーニングしたモデルを/models フォルダにコピーします。

### 5. ニャトのノート

・全体画面入れた方がいいです
例：
![alt text](ReadmeImg/image_001.png)
・アイコンや認証したい物体が全ケースを入れた方がいい
・アイアイずつをトレーニングデータにならないと思います
例：これは NG

![alt text](ReadmeImg/{2B65BB3F-A8D3-4D16-BD16-0DE4BA8DDCFD}.png)

本デモはシミュレーターで自動動作の動画は　 A、B、C 　あり、
上左が「Bluetooth」、「WIFI」、「輝度」、「開発」というアイコンです、
「WIFI」が「セルラー」や「テザリング」に切り替えられます。
想定の状況は「WIFI」から「セルラー」や「テザリング」切り替えると NG になります。
・A：普通の動作、「WIFI」がそのまま
・B：動作中に「WIFI」→「セルラー」に切り替えられる
・C：動作中に「WIFI」→「テザリング」に切り替えられる
※不具合がさせられないため、強制的に切り替えられる
→ 不具合パータンが事前に追加が必要です。

デモ内のエラー宣言箇所：
１．VideoComparator.py
２．VideoProcessor.py

```python
    def __init__(self, model_path="models/best_main.pt", conf_thresh=0.6, process_fps=3, no_error_limit=5):
        self.MODEL_CLASS_IDS = ["BT", "Wifi", "Cel", "Hots", "Bri", "Dev"]
        self.MODEL_CLASS_IDS_JP = [
            "ブルートゥース", "Wi-Fi", "セルラー",
            "テザリング", "輝度", "開発"
        ]
        self.ERROR_CLASS_ID = [2, 3]        #　★　ここがエラーアイコン対象です。
        self.CONF_THRESH = conf_thresh
        self.PROCESS_FPS = process_fps
        self.NO_ERROR_LIMIT = no_error_limit

        self.model = YOLO(model_path, task='detect')
```

エラーボタンによると実装の理論が変わる可能性もあります。