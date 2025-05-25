import csv
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Настройки по умолчанию
BINARY_THRESHOLD_METHOD = cv2.THRESH_BINARY + cv2.THRESH_OTSU
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}


def read_image_grayscale(image_path: Path) -> np.ndarray:
    """
    Считывает изображение и возвращает 2D-массив серого.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить {image_path}")
    return img


def threshold_image(img_gray: np.ndarray) -> np.ndarray:
    """
    Автоматическая бинаризация (Otsu) или пороговая.
    """
    _, img_bin = cv2.threshold(img_gray, 0, 255, BINARY_THRESHOLD_METHOD)
    return img_bin


def compute_features(img_bin: np.ndarray) -> dict:
    """
    Извлекает признаки из бинарного символа:
    - массы по четырем четвертям
    - относительные массы
    - центр масс и нормированные координаты
    - моменты инерции и нормированные
    - профили (горизонтальный/вертикальный)
    """
    mask = (img_bin == 0).astype(np.uint8)
    h, w = mask.shape
    total = mask.sum()

    mid_h, mid_w = h // 2, w // 2
    quarters = [mask[0:mid_h, 0:mid_w],  # верхний левый
                mask[0:mid_h, mid_w:w],  # верхний правый
                mask[mid_h:h, 0:mid_w],  # нижний левый
                mask[mid_h:h, mid_w:w]]  # нижний правый

    masses = [q.sum() for q in quarters]
    rel_masses = [m / q.size if q.size else 0 for m, q in zip(masses, quarters)]

    ys, xs = np.nonzero(mask)
    if total > 0:
        c_x, c_y = xs.mean(), ys.mean()
    else:
        c_x = c_y = 0.0

    norm_cx, norm_cy = (c_x / w if w else 0), (c_y / h if h else 0)

    I_x = ((ys - c_y) ** 2).sum()
    I_y = ((xs - c_x) ** 2).sum()
    max_dim = max(w, h) or 1
    norm_Ix = I_x / (total * max_dim**2) if total else 0
    norm_Iy = I_y / (total * max_dim**2) if total else 0

    profile_x = mask.sum(axis=0)
    profile_y = mask.sum(axis=1)

    return {
        'mass': masses,
        'rel_mass': rel_masses,
        'center': (c_x, c_y),
        'norm_center': (norm_cx, norm_cy),
        'I': (I_x, I_y),
        'norm_I': (norm_Ix, norm_Iy),
        'profile_x': profile_x,
        'profile_y': profile_y
    }


def plot_and_save_profile(values: np.ndarray, title: str, out_path: Path) -> None:
    """
    Строит столбчатую диаграмму и сохраняет её.
    """
    plt.figure()
    plt.bar(np.arange(values.size), values)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Count')
    plt.savefig(str(out_path))
    plt.close()


def export_features(
    folder: Path,
    features_list: list[dict],
    names: list[str],
    csv_path: Path,
    profiles_dir: Path
) -> None:
    """
    Сохраняет CSV с основными признаками и профили как PNG.
    """
    profiles_dir.mkdir(exist_ok=True)
    header = [
        'symbol',
        'mass_q1','mass_q2','mass_q3','mass_q4',
        'rel_q1','rel_q2','rel_q3','rel_q4',
        'cx','cy','norm_cx','norm_cy',
        'Ix','Iy','norm_Ix','norm_Iy'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        for name, feats in zip(names, features_list):
            row = [name] + feats['mass'] + feats['rel_mass'] \
                + list(feats['center']) + list(feats['norm_center']) \
                + list(feats['I']) + list(feats['norm_I'])
            writer.writerow(row)

            # Профили X/Y
            plot_and_save_profile(
                feats['profile_x'],
                f"Profile X: {name}",
                profiles_dir / f"{name}_x.png"
            )
            plot_and_save_profile(
                feats['profile_y'],
                f"Profile Y: {name}",
                profiles_dir / f"{name}_y.png"
            )


def process_folder(input_folder: Path) -> None:
    """
    Основной поток: обходит папку, собирает признаки и сохраняет результаты.
    """
    img_paths = [p for p in input_folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    names, feats_list = [], []
    for img_path in img_paths:
        gray = read_image_grayscale(img_path)
        bin_img = threshold_image(gray)
        feats = compute_features(bin_img)
        names.append(img_path.stem)
        feats_list.append(feats)

    csv_file = Path('symbol_features.csv')
    profiles_folder = Path('profiles')
    export_features(input_folder, feats_list, names, csv_file, profiles_folder)


def main():
    symbols_dir = Path('symbols')
    if not symbols_dir.exists():
        print(f"Директория не найдена: {symbols_dir}")
        return
    process_folder(symbols_dir)
    # Отчёт
    report = [
        '# Отчёт по ЛР №5: Признаки символов',
        f'- Папка: {symbols_dir}',
        f'- Файлов: {len(list(symbols_dir.iterdir()))}',
        '- CSV: symbol_features.csv',
        '- Профили в папке profiles',
        '---',
        '**Готово**'
    ]
    Path('report_lab5.md').write_text("\n".join(report), encoding='utf-8')
    print('Отчёт сохранён: report_lab5.md')

if __name__ == '__main__':
    main()
