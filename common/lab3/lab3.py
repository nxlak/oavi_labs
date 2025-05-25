import numpy as np
from PIL import Image
from pathlib import Path

# Константы
KERNEL_SIZE = 5
THRESHOLD = 128

# Загрузить и подготовить изображение

def load_grayscale(path: Path) -> np.ndarray:
    """Чтение изображения и преобразование в матрицу серых тонов."""
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.uint8)

# Морфологические операции

def pad_image(img: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(img, pad, mode='constant', constant_values=0)


def dilate(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Дилатация с использованием структурного элемента."""
    pad = se.shape[0] // 2
    img_padded = pad_image(img, pad)
    result = np.zeros_like(img)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            window = img_padded[y:y+se.shape[0], x:x+se.shape[1]]
            result[y, x] = (window * se).max()
    return result


def erode(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Эрозия с использованием структурного элемента."""
    pad = se.shape[0] // 2
    img_padded = pad_image(img, pad)
    result = np.zeros_like(img)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            window = img_padded[y:y+se.shape[0], x:x+se.shape[1]]
            result[y, x] = window[se == 1].min()
    return result


def close(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Замыкание: дилатация, затем эрозия."""
    return erode(dilate(img, se), se)

# Прочие функции

def binarize(img: np.ndarray, thresh: int = THRESHOLD) -> np.ndarray:
    """Простая бинаризация по порогу."""
    return np.where(img > thresh, 255, 0).astype(np.uint8)


def absolute_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Модуль разности двух изображений."""
    return np.abs(a.astype(int) - b.astype(int)).astype(np.uint8)


def save_matrix_as_image(matrix: np.ndarray, path: Path) -> None:
    """Запись массива numpy как изображения."""
    Image.fromarray(matrix).save(path)


# Основной поток
if __name__ == '__main__':
    base = Path('image3.png')
    se = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.uint8)

    gray = load_grayscale(base)
    closed_img = close(gray, se)
    binary_img = binarize(closed_img)
    diff_img = absolute_difference(gray, closed_img)

    # Сохранение
    save_matrix_as_image(binary_img, base.with_name('binary3.png'))
    save_matrix_as_image(closed_img, base.with_name('closed3.png'))
    save_matrix_as_image(diff_img, base.with_name('difference3.png'))

    # Отчёт
    report = []
    h, w = gray.shape
    report.append('# Отчёт по лаб. работе №3')
    report.append(f'- Исходник: {base.name} ({w}×{h})')
    report.append('- Структурный элемент: квадрат 5×5')
    report.append('## Результаты:')
    report.append(f'![](binary3.png) бинаризация')
    report.append(f'![](closed3.png) замыкание')
    report.append(f'![](difference3.png) разность')
    Path('report_lab3.md').write_text("\n".join(report), encoding='utf-8')
    print('Готово: бинаризация, замыкание, разность, отчёт.')
