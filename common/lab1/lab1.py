from PIL import Image
import numpy as np
import math

# Конфигурация
INPUT_FILE = 'image.png'
UPSCALE_FACTOR = 3
DOWNSCALE_FACTOR = 2

# Вспомогательные функции

def load_image(path):
    """Загрузить и привести к RGB"""
    img = Image.open(path)
    return img.convert('RGB')


def img_to_array(img):
    """Конвертировать PIL Image в numpy массив"""
    return np.array(img)


def save_from_array(arr, filename):
    """Сохранить numpy массив как изображение"""
    Image.fromarray(arr.astype(np.uint8)).save(filename)

# 1. Разделение каналов

def split_rgb(img):
    arr = img_to_array(img)
    channels = {
        'R': arr[:, :, 0],
        'G': arr[:, :, 1],
        'B': arr[:, :, 2]
    }
    for key, comp in channels.items():
        blank = np.zeros_like(comp)
        if key == 'R':
            layer = np.stack([comp, blank, blank], axis=2)
        elif key == 'G':
            layer = np.stack([blank, comp, blank], axis=2)
        else:
            layer = np.stack([blank, blank, comp], axis=2)
        save_from_array(layer, f'Output_{key}.png')

# 2. Преобразование RGB в HSI

def convert_rgb_to_hsi(img):
    arr = img_to_array(img).astype('float32') / 255.0
    R, G, B = arr[...,0], arr[...,1], arr[...,2]
    I = (R + G + B) / 3.0
    min_val = np.minimum(np.minimum(R, G), B)
    S = 1 - min_val / (I + 1e-8)
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-8
    theta = np.arccos(np.clip(numerator / denominator, -1, 1))
    H = np.where(B <= G, theta, 2*math.pi - theta) / (2*math.pi)
    return H, S, I

# Сохранение HSI компонентов

def save_hsi_components(h, s, i):
    save_from_array((i * 255).round(), 'Output_Intensity.png')
    save_from_array(((1 - i) * 255).round(), 'Output_Inverted_Intensity.png')

# 3. Изменение размера без интерполяции

def scale_up(img, factor):
    w, h = img.size
    out = Image.new(img.mode, (w*factor, h*factor))
    for y in range(h*factor):
        for x in range(w*factor):
            out.putpixel((x,y), img.getpixel((x//factor, y//factor)))
    return out


def scale_down(img, factor):
    w, h = img.size
    out = Image.new(img.mode, (w//factor, h//factor))
    for y in range(h//factor):
        for x in range(w//factor):
            out.putpixel((x,y), img.getpixel((x*factor, y*factor)))
    return out

# 4. Одношаговая и двухшаговая передискретизация

def two_step_resample(img, up, down):
    temp = scale_up(img, up)
    return scale_down(temp, down)


def one_step_resample(img, ratio):
    w, h = img.size
    new_size = (int(w/ratio), int(h/ratio))
    out = Image.new(img.mode, new_size)
    for y in range(new_size[1]):
        for x in range(new_size[0]):
            sx, sy = int(x * ratio), int(y * ratio)
            out.putpixel((x,y), img.getpixel((sx, sy)))
    return out


# Основной блок
if __name__ == '__main__':
    image = load_image(INPUT_FILE)
    # Каналы RGB
    split_rgb(image)

    # HSI
    H, S, I = convert_rgb_to_hsi(image)
    save_hsi_components(H, S, I)

    # Изменение размера
    up_img = scale_up(image, UPSCALE_FACTOR)
    up_img.save('Output_Stretched.png')

    down_img = scale_down(image, DOWNSCALE_FACTOR)
    down_img.save('Output_Compressed.png')

    # Двухшаговая передискретизация
    two_img = two_step_resample(image, UPSCALE_FACTOR, DOWNSCALE_FACTOR)
    two_img.save('Output_TwoPass_Resampled.png')

    # Одношаговая передискретизация
    ratio = UPSCALE_FACTOR / DOWNSCALE_FACTOR
    one_img = one_step_resample(image, ratio)
    one_img.save('Output_OnePass_Resampled.png')

    # Генерация отчета
    report = []
    report.append('# Лабораторная работа')
    report.append('## 1. RGB компоненты')
    for ch in ['R','G','B']:
        report.append(f'- Канал {ch}: ![](Output_{ch}.png)')
    report.append('## 2. HSI')
    report.append('- Яркость: ![](Output_Intensity.png)')
    report.append('- Инверсия яркости: ![](Output_Inverted_Intensity.png)')
    report.append('## 3. Размеры и передискретизация')
    report.append(f'- Исходный размер: {image.size}')
    report.append(f'- Растянуто (x{UPSCALE_FACTOR}): ![](Output_Stretched.png) размер {up_img.size}')
    report.append(f'- Сжато (/ {DOWNSCALE_FACTOR}): ![](Output_Compressed.png) размер {down_img.size}')
    report.append(f'- Двухшаговое (M/N={ratio}): ![](Output_TwoPass_Resampled.png) размер {two_img.size}')
    report.append(f'- Одношаговое (K={ratio}): ![](Output_OnePass_Resampled.png) размер {one_img.size}')

    with open('report.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    print('Готово, файлы сохранены.')
