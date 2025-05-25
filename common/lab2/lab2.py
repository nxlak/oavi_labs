import sys
import os
import math
from pathlib import Path
from PIL import Image

# Константы по умолчанию
DEFAULT_K = 0.2
WINDOW_RADIUS = 1  # Для окна 3x3


def load_rgb_image(path: str) -> Image.Image:
    """
    Загружает изображение из файла и конвертирует в RGB.
    """
    return Image.open(path).convert('RGB')


def rgb_to_gray(img: Image.Image) -> Image.Image:
    """
    Переводит изображение в 8-битную шкалу серого по формуле Y = 0.299R + 0.587G + 0.114B.
    """
    w, h = img.size
    gray = Image.new('L', (w, h))
    src = img.load()
    dst = gray.load()

    for y in range(h):
        for x in range(w):
            r, g, b = src[x, y]
            dst[x, y] = int(0.299*r + 0.587*g + 0.114*b)
    return gray


def local_stats(gray: Image.Image, x: int, y: int, r: int) -> tuple[float, float]:
    """
    Возвращает локальное среднее и стандартное отклонение в квадратном окне радиуса r.
    """
    pixels = gray.load()
    vals = []
    w, h = gray.size
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h:
                vals.append(pixels[nx, ny])
    mean = sum(vals)/len(vals)
    var = sum((v-mean)**2 for v in vals)/len(vals)
    return mean, math.sqrt(var)


def adaptive_binarize(gray: Image.Image, k: float = DEFAULT_K) -> Image.Image:
    """
    Выполняет адаптивную бинаризацию по методу Феня и Тана.
    Порог T = local_mean - k * local_std
    """
    w, h = gray.size
    binary = Image.new('L', (w, h))
    src = gray.load()
    dst = binary.load()

    for y in range(h):
        for x in range(w):
            mean, std = local_stats(gray, x, y, WINDOW_RADIUS)
            threshold = mean - k * std
            dst[x, y] = 0 if src[x, y] < threshold else 255
    return binary


def save_image(img: Image.Image, dest: Path) -> None:
    """
    Сохраняет изображение, выводя сообщение в случае ошибки.
    """
    try:
        img.save(dest)
    except Exception as e:
        print(f"Ошибка при сохранении {dest}: {e}")


def generate_report(input_file: Path, gray_file: Path, bin_file: Path, k: float) -> None:
    """
    Генерирует Markdown отчёт по этапам обработки.
    """
    img = Image.open(input_file)
    w, h = img.size
    report = []
    report.append(f"# Отчёт: Адаптивная бинаризация (Феня и Тана)")
    report.append("## Исходное изображение")
    report.append(f"- Файл: **{input_file.name}**")
    report.append(f"- Размер: {w}×{h}")
    report.append("## Полутоновая версия")
    report.append(f"![]({gray_file.name})")
    report.append("## Бинаризация")
    report.append(f"- Метод: Fen & Tan, окно {2*WINDOW_RADIUS+1}×{2*WINDOW_RADIUS+1}")
    report.append(f"- k = {k}")
    report.append(f"![]({bin_file.name})")
    output = Path("report_lab2.md")
    output.write_text("\n".join(report), encoding='utf-8')
    print(f"Отчёт сохранён в {output}")


def main():
    if len(sys.argv) < 2:
        print("Использование: python lab2.py <путь_изображения> [k]")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Файл {path} не найден.")
        sys.exit(1)

    k = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_K

    # Обработка
    rgb = load_rgb_image(str(path))
    gray = rgb_to_gray(rgb)
    gray_path = path.with_name(path.stem + '_gray.bmp')
    save_image(gray, gray_path)

    binary = adaptive_binarize(gray, k)
    bin_path = path.with_name(path.stem + '_binarized.bmp')
    save_image(binary, bin_path)

    generate_report(path, gray_path, bin_path, k)


if __name__ == '__main__':
    main()
