import cv2
import numpy as np
from pathlib import Path

# Параметры по умолчанию
DEFAULT_THRESHOLD = 50
SOBEL_KERNEL_SIZE = 3
MORPH_KERNEL = np.ones((3, 3), dtype=np.uint8)


def load_color_image(filepath: Path) -> np.ndarray:
    """
    Загружает цветное изображение BGR через OpenCV.
    """
    img = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить {filepath}")
    return img


def to_grayscale(bgr_img: np.ndarray) -> np.ndarray:
    """
    Переводит BGR в 8-битное полутоновое изображение.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return gray


def compute_sobel(gray_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет горизонтальную (Gx) и вертикальную (Gy) градиентные составляющие.
    Возвращает нормированные (0-255) градиентные изображения.
    """
    gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
    gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)

    abs_gx = np.abs(gx)
    abs_gy = np.abs(gy)
    norm_gx = normalize_to_uint8(abs_gx)
    norm_gy = normalize_to_uint8(abs_gy)

    return norm_gx, norm_gy


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Приводит массив с плавающей точкой к диапазону [0,255] uint8.
    """
    if arr.max() == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr / arr.max()) * 255
    return scaled.astype(np.uint8)

def combine_gradients(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Суммирует модули градиентов и нормирует результат.
    """
    combined = gx.astype(np.int32) + gy.astype(np.int32)
    return normalize_to_uint8(combined)


def threshold_image(img: np.ndarray, thresh: int = DEFAULT_THRESHOLD) -> np.ndarray:
    """
    Применяет бинаризацию по заданному порогу.
    """
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return binary


def morphological_difference(gray: np.ndarray, kernel: np.ndarray = MORPH_KERNEL) -> np.ndarray:
    """
    Вычисляет разность между полутоном и его дилатацией.
    """
    dilated = cv2.dilate(gray, kernel, iterations=1)
    diff = cv2.absdiff(gray, dilated)
    return diff


def save_image(img: np.ndarray, filepath: Path) -> None:
    """
    Сохраняет numpy-изображение на диск.
    """
    cv2.imwrite(str(filepath), img)


def generate_report(
    report_path: Path,
    original: Path,
    gray: Path,
    gx: Path,
    gy: Path,
    g: Path,
    binary: Path,
    diff: Path,
    thresh: int
) -> None:
    """
    Формирует Markdown-отчёт по всем этапам.
    """
    # Определяем размер из полутонового
    import json
    # Читаем, чтобы получить размеры
    img_gray = cv2.imread(str(gray), cv2.IMREAD_GRAYSCALE)
    h, w = img_gray.shape

    lines = [
        '# Отчёт по ЛР №4: Детекция границ и морфология',
        '## 1. Исходное изображение',
        f'- Файл: **{original.name}**',
        f'- Размер: {w}×{h}',
        '## 2. Полутона',
        f'![]({gray.name})',
        '## 3. Градиенты по Собелю',
        f'- Gx: ![]({gx.name})',
        f'- Gy: ![]({gy.name})',
        '## 4. Итоговый градиент',
        f'![]({g.name})',
        '## 5. Бинаризация градиента',
        f'- Порог: {thresh}',
        f'![]({binary.name})',
        '## 6. Морфологическая разность',
        f'![]({diff.name})',
        '---',
        '**Конец отчёта**'
    ]
    report_path.write_text("\n".join(lines), encoding='utf-8')


def main():
    base = Path('image.png')
    gray_file = base.with_name('gray.jpg')
    gx_file = base.with_name('gx.jpg')
    gy_file = base.with_name('gy.jpg')
    g_file = base.with_name('g.jpg')
    binary_file = base.with_name('g_bin.jpg')
    diff_file = base.with_name('diff_morph.jpg')

    # Параметр
    thresh = DEFAULT_THRESHOLD

    # Загрузка и обработка
    color = load_color_image(base)
    gray = to_grayscale(color)
    save_image(gray, gray_file)

    gx, gy = compute_sobel(gray)
    save_image(gx, gx_file)
    save_image(gy, gy_file)

    g = combine_gradients(gx, gy)
    save_image(g, g_file)

    binary = threshold_image(g, thresh)
    save_image(binary, binary_file)

    diff = morphological_difference(gray)
    save_image(diff, diff_file)

    # Отчёт
    report_md = Path('report_lab4.md')
    generate_report(report_md, base, gray_file, gx_file, gy_file, g_file, binary_file, diff_file, thresh)
    print(f'Отчёт сохранён: {report_md}')

if __name__ == '__main__':
    main()