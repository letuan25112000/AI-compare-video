import shutil
from datetime import datetime
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
import subprocess
from pathlib import Path
import subprocess
import tempfile

# =====================================
# フォルダ内の古いファイルを削除
# =====================================
def clean_folder(results_dir):
    """
    指定フォルダ内のファイル・サブフォルダを全削除
    """
    results_dir = Path(results_dir)
    for f in results_dir.glob("*"):
        try:
            if f.is_file() or f.is_symlink():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        except Exception as e:
            print(f"古いファイル削除エラー: {f} → {e}")


# =====================================
# タイムスタンプ付きファイル名生成
# =====================================
def timestamp_file_path(filename, ext):
    """
    ファイル名に現在時刻（YYYYMMDD_HHMMSS）を付与して返す
    例: result_20251023_153000.xlsx
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{filename}_{timestamp}.{ext}"


# =====================================
# Excel書式設定
# =====================================
def apply_excel_format(ws):
    """
    Excel表のフォーマットを適用：
    ヘッダーのスタイル、枠線、セルの整列、背景色、自動列幅調整など
    """

    # --- ヘッダー行の書式 ---
    header_font = Font(bold=True, color="000000")
    header_fill = PatternFill(fill_type="solid", fgColor="D9D9D9")
    center_align = Alignment(horizontal="center", vertical="center")

    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_align

    # --- 枠線（細線）設定 ---
    thin_border = Border(
        left=Side(style="thin", color="000000"),
        right=Side(style="thin", color="000000"),
        top=Side(style="thin", color="000000"),
        bottom=Side(style="thin", color="000000"),
    )

    # --- 全セルに枠線と整列を適用 ---
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
        for cell in row:
            cell.border = thin_border
            if cell.row != 1:  # ヘッダーは既に中央揃え済み
                cell.alignment = Alignment(vertical="center", wrap_text=True)

    # --- 列幅を自動調整 ---
    column_widths = {}
    for row in ws.iter_rows():
        for cell in row:
            if cell.value:
                width = len(str(cell.value)) + 2
                column_widths[cell.column_letter] = max(column_widths.get(cell.column_letter, 10), width)

    for col, width in column_widths.items():
        ws.column_dimensions[col].width = width


# =====================================
# Web再生対応の動画へ変換（ffmpeg使用）
# =====================================
def make_web_ready(input_path):
    """
    OpenCVで出力したMP4ファイルを
    ブラウザで再生できる形式（H.264 + yuv420p）に変換する。
    """
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_web.mp4")

    cmd = [
        "ffmpeg",
        "-y",                # 既存ファイルを上書き
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",    # ブラウザ再生に必要
        "-movflags", "+faststart",  # ストリーミング高速開始
        str(output_path)
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return str(output_path)

# =====================================
# 動画のfps同期
# =====================================
def convert_to_30fps(video_path: str) -> str:
    """
    入力動画を30FPSに変換して一時ファイルとして保存する
    """
    temp_dir = Path(tempfile.gettempdir())
    output_path = temp_dir / f"{Path(video_path).stem}_30fps.mp4"

    # ffmpeg コマンド
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter:v", "fps=30",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-an",  # 音声を削除
        str(output_path)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(output_path)