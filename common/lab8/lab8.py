import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv, hsv2rgb

# --- Настройки ---
INPUT_IMG = Path('input_image.jpg')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(exist_ok=True)
GAMMA = 0.5
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS

# --- Функции обработки ---
def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Применяет степенное преобразование яркости."""
    return np.power(image, gamma)


def extract_lbp_and_hist(channel: np.ndarray, points: int, radius: int):
    """Вычисляет карту LBP и её гистограмму плотности."""
    lbp_map = local_binary_pattern(channel, points, radius, method='uniform')
    bins = np.arange(0, points + 3)
    hist, _ = np.histogram(lbp_map, bins=bins, density=True)
    return lbp_map, hist


def save_gray_array(arr: np.ndarray, path: Path):
    """Сохраняет 2D-массив как изображение в 8-битном формате."""
    img8 = Image.fromarray((arr * 255).astype(np.uint8))
    img8.save(path)


def plot_histogram(data: np.ndarray, filename: Path, **kwargs):
    """Строит и сохраняет гистограмму."""
    plt.figure(figsize=(10, 5))
    plt.hist(data.flatten(), **kwargs)
    plt.savefig(str(filename)); plt.close()


def plot_lbp_histograms(orig_hist: np.ndarray, trans_hist: np.ndarray, filename: Path):
    """Визуализирует и сохраняет LBP-гистограммы."""
    indices = np.arange(orig_hist.size)
    plt.figure(figsize=(10,5))
    plt.bar(indices, orig_hist, alpha=0.5, label='Original')
    plt.bar(indices, trans_hist, alpha=0.5, label='Transformed')
    plt.xlabel('LBP Pattern'); plt.ylabel('Frequency'); plt.legend()
    plt.savefig(str(filename)); plt.close()

# --- Основной поток ---
# Чтение и преобразование в [0,1]
rgb_img = np.array(Image.open(INPUT_IMG).convert('RGB')) / 255.0
hsv_img = rgb2hsv(rgb_img)

# Работа с каналом яркости
v = hsv_img[..., 2]
v_gamma = adjust_gamma(v, GAMMA)

# Обратное преобразование в RGB
hsv_img[..., 2] = v_gamma
rgb_gamma = hsv2rgb(hsv_img)

# Сохранение изображений
Image.fromarray((rgb_img * 255).astype(np.uint8)).save(OUTPUT_DIR / 'original.jpg')
Image.fromarray((rgb_gamma * 255).astype(np.uint8)).save(OUTPUT_DIR / 'corrected.jpg')

# LBP на исходном и скорректированном Value
lbp_orig, hist_orig = extract_lbp_and_hist(v, LBP_POINTS, LBP_RADIUS)
lbp_corr, hist_corr = extract_lbp_and_hist(v_gamma, LBP_POINTS, LBP_RADIUS)

# Сохранение карт LBP
save_gray_array(lbp_orig / lbp_orig.max(), OUTPUT_DIR / 'lbp_orig.jpg')
save_gray_array(lbp_corr / lbp_corr.max(), OUTPUT_DIR / 'lbp_corr.jpg')

# Гистограммы яркости
plot_histogram(v, OUTPUT_DIR / 'hist_brightness.png', bins=256, range=(0,1), alpha=0.5, label='Original')
plot_histogram(v_gamma, OUTPUT_DIR / 'hist_brightness.png', bins=256, range=(0,1), alpha=0.5, label='Transformed')

# Гистограммы LBP
plot_lbp_histograms(hist_orig, hist_corr, OUTPUT_DIR / 'hist_lbp.png')

# --- Создание отчёта ---
md = []
md.append('# ЛР №8: Текстурный анализ и гамма-коррекция')
md.append('## 1. Исходное и скорректированное')
md.append('![Original](results/original.jpg)')
md.append(f'![Corrected (gamma={GAMMA})](results/corrected.jpg)')
md.append('## 2. LBP-карты')
md.append('![LBP Original](results/lbp_orig.jpg)')
md.append('![LBP Corrected](results/lbp_corr.jpg)')
md.append('## 3. Гистограммы яркости')
md.append('![Brightness Hist](results/hist_brightness.png)')
md.append('## 4. Гистограммы LBP')
md.append('![LBP Hist](results/hist_lbp.png)')
md.append('## 5. Выводы')
md.append(f'- Gamma={GAMMA} {"увеличивает" if GAMMA<1 else "уменьшает"})
