import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Параметры по умолчанию
DEFAULT_FONT = "C:/Windows/Fonts/times.ttf"
DEFAULT_FONT_SIZE = 52
PROFILE_THRESHOLD = 1


def render_text_to_bmp(text: str, font_file: str = DEFAULT_FONT, size: int = DEFAULT_FONT_SIZE, output: Path = Path("phrase.bmp")):
    """
    Создаёт монохромный BMP без лишних полей из заданного текста.
    """
    font = ImageFont.truetype(font_file, size)
    # Определяем размер текста
    dummy = Image.new('L', (1, 1), 255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Рисуем текст на чистом фоне и обрезаем
    img = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=0)
    bmp = img.crop(img.getbbox()).convert('1')  # 1-bit image
    bmp.save(output, "BMP")
    print(f"BMP saved: {output}")
    return output


def compute_projections(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Получает горизонтальный и вертикальный профили проэкции чёрных пикселей.
    """
    mask = (binary == 0).astype(np.uint8)
    vert = mask.sum(axis=0)
    hor = mask.sum(axis=1)
    return vert, hor


def find_runs(profile: np.ndarray, thresh: int = PROFILE_THRESHOLD) -> list[tuple[int,int]]:
    """
    Разбивает одномерный профиль на отрезки, где значения > thresh.
    """
    segments = []
    active = False
    start = 0
    for idx, val in enumerate(profile):
        if not active and val > thresh:
            active = True
            start = idx
        elif active and val <= thresh:
            segments.append((start, idx - 1))
            active = False
    if active:
        segments.append((start, len(profile) - 1))
    return segments


def segment_letters(gray_img: np.ndarray, v_segs: list, h_segs: list) -> list[tuple[int,int,int,int]]:
    """
    Формирует прямоугольники для каждого символа по вертикальным и горизонтальным сегментам.
    """
    boxes = []
    for y0, y1 in h_segs:
        for x0, x1 in v_segs:
            boxes.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
    # Сортируем по строкам, потом по столбцам
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def save_results(
    bmp_path: Path,
    output_dir: Path,
    font_text: str,
    h_thresh: int = PROFILE_THRESHOLD,
    v_thresh: int = PROFILE_THRESHOLD
):
    """
    Вычисляет профили, сегментирует символы и сохраняет всё на диск.
    """
    output_dir.mkdir(exist_ok=True)
    gray = cv2.imread(str(bmp_path), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    vert_prof, hor_prof = compute_projections(binary)
    v_runs = find_runs(vert_prof, v_thresh)
    h_runs = find_runs(hor_prof, h_thresh)

    # Сохраняем профили
    plt.figure(); plt.bar(range(len(vert_prof)), vert_prof); plt.title('Vertical'); plt.savefig(output_dir / 'prof_vert.png'); plt.close()
    plt.figure(); plt.bar(range(len(hor_prof)), hor_prof); plt.title('Horizontal'); plt.savefig(output_dir / 'prof_hor.png'); plt.close()

    # Сегменты символов
    chars_dir = output_dir / 'chars'
    chars_dir.mkdir(exist_ok=True)
    color_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    boxes = segment_letters(gray, v_runs, h_runs)
    for idx, (x, y, w, h) in enumerate(boxes):
        # сохраняем вырез символа
        crop = gray[y:y+h, x:x+w]
        cv2.imwrite(str(chars_dir / f'char_{idx}.png'), crop)
        # рисуем рамку
        cv2.rectangle(color_vis, (x, y), (x+w, y+h), (0, 0, 255), 1)

    cv2.imwrite(str(output_dir / 'outline.png'), color_vis)
    print(f"Outputs in {output_dir}")


def main():
    phrase = "սիրում եմ քեզ"
    bmp_file = Path('phrase.bmp')
    out_folder = Path('lab6_out')

    # Генерация и сегментация
    render_text_to_bmp(phrase, output=bmp_file)
    save_results(bmp_file, out_folder, phrase)


if __name__ == '__main__':
    main()
