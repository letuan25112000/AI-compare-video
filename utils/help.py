import shutil
from datetime import datetime
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
import subprocess

def clean_folder(results_dir):
    for f in results_dir.glob("*"):
        try:
            if f.is_file() or f.is_symlink():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        except Exception as e:
            print(f"削除できません {f}: {e}")

def timestamp_file_path(filename, ext):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{filename}_{timestamp}.{ext}"

def apply_excel_format(self, ws):
        """Định dạng bảng Excel: header, border, căn lề, màu nền, auto-width."""

        # --- Định dạng header ---
        header_font = Font(bold=True, color="000000")
        header_fill = PatternFill(fill_type="solid", fgColor="D9D9D9")
        center_align = Alignment(horizontal="center", vertical="center")

        # ⚠️ Sửa lỗi enumerate(ws.max_col) → range(1, ws.max_column + 1)
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align

        # --- Border mảnh ---
        thin_border = Border(
            left=Side(style="thin", color="000000"),
            right=Side(style="thin", color="000000"),
            top=Side(style="thin", color="000000"),
            bottom=Side(style="thin", color="000000"),
        )

        # --- Áp dụng border và căn lề cho toàn bộ bảng ---
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
            for cell in row:
                cell.border = thin_border
                if cell.row != 1:  # Header đã căn giữa ở trên
                    cell.alignment = Alignment(vertical="center", wrap_text=True)

        # --- Tự động điều chỉnh độ rộng cột ---
        column_widths = {}
        for row in ws.iter_rows():
            for cell in row:
                if cell.value:
                    width = len(str(cell.value)) + 2
                    column_widths[cell.column_letter] = max(column_widths.get(cell.column_letter, 10), width)

        for col, width in column_widths.items():
            ws.column_dimensions[col].width = width

import subprocess
from pathlib import Path

def make_web_ready(input_path):
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_web.mp4")
    
    cmd = [
        "ffmpeg",
        "-y",  # overwrite existing
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",  # required for browser playback
        "-movflags", "+faststart",  # enable fast start for streaming
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return str(output_path)

