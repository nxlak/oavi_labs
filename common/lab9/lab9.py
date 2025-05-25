import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
from scipy.signal import savgol_filter, wiener
import soundfile as sf


class NoiseAnalyzer:
    def __init__(self, audio_path: Path):
        self.audio_path = audio_path
        self.fs, data = wavfile.read(str(audio_path))
        audio = data.mean(axis=1) if hasattr(data, 'ndim') and data.ndim > 1 else data
        self.waveform = audio.astype(float) / np.max(np.abs(audio))
        self.duration = len(self.waveform) / self.fs
        self.noise_rms = None
        self.filtered = {}
        self.energy_info = {}

    def run(self):
        self._spectrogram(self.waveform, 'spec_original.png', 'Original Spectrogram')
        self._compute_noise_rms()
        self._filter_methods()
        self._energy_analysis()
        self._export_audio_versions()
        self._write_report()

    def _spectrogram(self, sig, filename, title=None):
        f, t, Sxx = signal.spectrogram(sig, self.fs, window='hann', nperseg=1024, noverlap=512)
        plt.figure(figsize=(10,6))
        plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading='gouraud')
        plt.yscale('log')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        if title:
            plt.title(title)
        plt.colorbar(label='Intensity [dB]')
        plt.savefig(filename)
        plt.close()

    def _compute_noise_rms(self, frac=0.1):
        n = int(self.fs*frac)
        self.noise_rms = np.sqrt(np.mean(self.waveform[:n]**2))

    def _filter_methods(self):
        # Savitzky-Golay
        sg = savgol_filter(self.waveform, 101, 3)
        self.filtered['savitzky'] = sg
        self._spectrogram(sg, 'spec_savitzky.png', 'Savitzky-Golay')

        # Wiener
        wi = wiener(self.waveform)
        self.filtered['wiener'] = wi
        self._spectrogram(wi, 'spec_wiener.png', 'Wiener')

        # Low-pass
        b, a = signal.butter(4, 1000/(self.fs/2), 'low')
        lp = signal.filtfilt(b, a, self.waveform)
        self.filtered['lowpass'] = lp
        self._spectrogram(lp, 'spec_lowpass.png', 'Low-pass')

    def _energy_analysis(self, win=0.1, band=(40,50)):
        step = int(win*self.fs)
        segs = len(self.waveform)//step
        energies = [np.sum(self.waveform[i*step:(i+1)*step]**2) for i in range(segs)]
        top = np.argsort(energies)[-3:][::-1]
        self.energy_info['times'] = [i*win for i in top]

        # Band energy
        f, t, Sxx = signal.spectrogram(self.waveform, self.fs)
        mask = (f>=band[0]) & (f<=band[1])
        be = Sxx[mask].sum(axis=0)
        plt.figure(figsize=(10,4))
        plt.plot(t, be)
        plt.xlabel('Time [s]')
        plt.ylabel(f'Energy {band[0]}-{band[1]} Hz')
        plt.title('Band Energy')
        plt.savefig('energy_band.png')
        plt.close()
        self.energy_info['band_plot'] = 'energy_band.png'

    def _export_audio_versions(self):
        for key, sig in self.filtered.items():
            sf.write(f'{self.audio_path.stem}_{key}.wav', sig, self.fs)

    def _write_report(self):
        lines = [
            '# Lab 9: Noise Analysis',
            '## Audio Info',
            f'- File: {self.audio_path.name}',
            f'- Sample rate: {self.fs} Hz',
            f'- Duration: {self.duration:.2f} s',
            f'- Noise RMS: {self.noise_rms:.4f}',
            '## Spectrograms',
            '![Original](spec_original.png)',
            '|Filter|Spectrogram|',
            '|---|---|',
            f'|Savitzky-Golay|![](spec_savitzky.png)|',
            f'|Wiener|![](spec_wiener.png)|',
            f'|Low-pass|![ ](spec_lowpass.png)|',
            '## Energy Peaks',
            f'- Peak times: {self.energy_info["times"]} s',
            '![Band Energy](energy_band.png)',
            '## Conclusions',
            '- Savitzky-Golay: preserves waveform.',
            '- Wiener: effective against noise.',
            '- Low-pass: removes HF noise.'
        ]
        Path('report_lab9.md').write_text('\n'.join(lines), encoding='utf-8')
        print('Report: report_lab9.md')

if __name__ == '__main__':
    analyzer = NoiseAnalyzer(Path('sound.wav'))
    analyzer.run()