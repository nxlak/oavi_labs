import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.spatial import distance

# --- Feature Extraction ---
def extract_symbol_metrics(binary_image: np.ndarray) -> Dict[str, float]:
    """
    Извлекает базовые метрики символа из бинарного изображения.
    Возвращает словарь с массами по четвертям, центроидом и моментами инерции.
    """
    mask = (binary_image == 0).astype(np.uint8)
    height, width = mask.shape
    total = mask.sum()

    # Разбиение на четверти
    mid_h, mid_w = height // 2, width // 2
    quarters = [mask[0:mid_h, 0:mid_w],
                mask[0:mid_h, mid_w:width],
                mask[mid_h:height, 0:mid_w],
                mask[mid_h:height, mid_w:width]]
    masses = [q.sum() for q in quarters]

    # Центр масс
    ys, xs = np.nonzero(mask)
    cx = xs.mean() if total else 0.0
    cy = ys.mean() if total else 0.0

    # Моменты инерции
    Ix = ((ys - cy) ** 2).sum()
    Iy = ((xs - cx) ** 2).sum()

    return {
        'm1': masses[0], 'm2': masses[1], 'm3': masses[2], 'm4': masses[3],
        'cx': cx, 'cy': cy,
        'Ix': Ix, 'Iy': Iy
    }

# --- Reference Data Handling ---
def load_reference_vectors(csv_file: Path, feature_keys: List[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Считывает CSV референсных признаков и возвращает нормированные векторы,
    а также min/max для последующей нормализации.
    """
    raw_vectors: Dict[str, np.ndarray] = {}
    with csv_file.open(encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            name = row['symbol_name']
            vec = np.array([float(row[k]) for k in feature_keys])
            raw_vectors[name] = vec

    mat = np.vstack(list(raw_vectors.values()))
    mins = mat.min(axis=0)
    maxs = mat.max(axis=0)

    norm_vectors = {name: (vec - mins) / (maxs - mins + 1e-8) for name, vec in raw_vectors.items()}
    return norm_vectors, mins, maxs

# --- Classification ---
def classify_folder(
    images_dir: Path,
    ref_vectors: Dict[str, np.ndarray],
    mins: np.ndarray,
    maxs: np.ndarray,
    keys: List[str],
    top_k: int = 5
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """
    Для каждого файла PNG в папке, бинаризует, вычисляет метрики,
    нормирует и сравнивает с референсом, возвращает топ-K гипотез.
    """
    results = []
    for img_path in sorted(images_dir.glob('*.png')):
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        feats = extract_symbol_metrics(binary)
        vec = np.array([feats[k] for k in keys])
        norm_vec = (vec - mins) / (maxs - mins + 1e-8)

        scores = []
        for name, ref_vec in ref_vectors.items():
            d = distance.euclidean(norm_vec, ref_vec)
            sim = 1.0 / (1.0 + d)
            scores.append((name, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        results.append((img_path.name, scores[:top_k]))
    return results

# --- Evaluation ---
def evaluate_results(
    classified: List[Tuple[str, List[Tuple[str, float]]]],
    ground_truth: str
) -> Tuple[str, int, float]:
    """
    Формирует распознанную строку по лучшим гипотезам,
    вычисляет число ошибок и точность.
    """
    recognized = ''.join([hyps[0][0] for _, hyps in classified])
    errors = sum(a != b for a, b in zip(recognized, ground_truth))
    errors += abs(len(recognized) - len(ground_truth))
    accuracy = (len(ground_truth) - errors) / len(ground_truth) * 100
    return recognized, errors, accuracy

# --- Main ---
def main():
    ref_csv = Path('symbol_features.csv')
    feature_keys = ['m1','m2','m3','m4','cx','cy','Ix','Iy']

    ref_vecs, mins, maxs = load_reference_vectors(ref_csv, feature_keys)

    chars_folder = Path('chars')
    hypotheses = classify_folder(chars_folder, ref_vecs, mins, maxs, feature_keys, top_k=5)

    out_txt = Path('lab7_hypotheses.txt')
    with out_txt.open('w', encoding='utf-8') as f:
        for idx, (fname, hyps) in enumerate(hypotheses, 1):
            f.write(f"{idx}: {hyps}\n")

    truth = "սիրում եմ քեզ"
    rec, errs, acc = evaluate_results(hypotheses, truth)
    print(f"Recognized: {rec}")
    print(f"Errors: {errs}, Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    main()
