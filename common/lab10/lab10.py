import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

# --- Анализ одного файла ---
def extract_audio_features(path: Path, label: str) -> dict:
    """
    Считает спектрограмму, определяет крайние частоты,
    основной тон и ключевые форманты.
    """
    y, sr = librosa.load(str(path), sr=None)

    # Строим и сохраняем спектрограмму
    S = np.abs(librosa.stft(y, window='hann'))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(10,6))
    librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {label}')
    spec_file = Path(f'{label}_spec.png')
    plt.savefig(spec_file)
    plt.close()

    # Усреднённая амплитуда по частоте
    freq = librosa.fft_frequencies(sr=sr)
    avg_amp = S.mean(axis=1)

    # Определяем порог 10% от максимума
    thresh = 0.1 * avg_amp.max()
    active = freq[avg_amp > thresh]
    min_f = active.min() if active.size else 0.0
    max_f = active.max() if active.size else 0.0

    # Фундаментальная частота через пики в фрейме с наименьшей плоскостностью
    sfm = librosa.feature.spectral_flatness(y=y).flatten()
    frame_idx = sfm.argmin()
    frame_spec = S[:, frame_idx]
    peaks, _ = find_peaks(frame_spec, height=0.1 * frame_spec.max())
    peak_freqs = freq[peaks]
    peak_vals = frame_spec[peaks]
    fund = peak_freqs[peak_vals.argmax()] if peak_freqs.size else 0.0

    # Три наибольших форманты по средней амплитуде
    peaks2, _ = find_peaks(avg_amp, distance=5)
    top3 = peaks2[np.argsort(avg_amp[peaks2])[-3:]]
    formants = sorted(freq[top3].tolist())

    return {
        'min_freq': min_f,
        'max_freq': max_f,
        'fundamental': fund,
        'formants': formants,
        'spec_image': spec_file
    }

# --- Основной блок ---
def main():
    sounds = {
        'A': Path('1.wav'),
        'I': Path('2.wav'),
        'bark': Path('3.wav')
    }
    results = {}
    for label, file in sounds.items():
        results[label] = extract_audio_features(file, label)

    # Пишем Markdown отчёт
    report = ['# Lab 10: Voice Analysis', '']
    for label, feats in results.items():
        report.append(f'## Sound "{label}"')
        report.append(f'- Min freq: {feats["min_freq"]:.2f} Hz')
        report.append(f'- Max freq: {feats["max_freq"]:.2f} Hz')
        report.append(f'- Fundamental: {feats["fundamental"]:.2f} Hz')
        forms = ', '.join(f'{f:.2f}' for f in feats['formants'])
        report.append(f'- Top 3 formants: {forms} Hz')
        report.append(f'![Spectrogram]({feats["spec_image"]})')
        report.append('')

    Path('lab_10_report.md').write_text('\n'.join(report), encoding='utf-8')
    print('Report saved: lab_10_report.md')

if __name__ == '__main__':
    main()
