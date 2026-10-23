# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# Mains (power-line) interference removal without a spectral hole.
# ---------------------------------------------------------
"""Spectrum interpolation (Leske & Dalal 2019, NeuroImage 189).

A notch filter removes a band around the mains frequency whether or not there
is interference in it, cutting a hole into genuine EEG. Interpolation only
replaces the *excess* power at the line with the level of the neighbouring
frequencies, so it removes a line where there is one and leaves the signal
unchanged where there is none -- one recipe (50 and 60 Hz) fits corpora from
50 Hz and 60 Hz countries alike. Evidence: notebooks/tuh_preprocessing_analysis.ipynb
(Section 8). numpy + scipy only, so the data-prep container can import it flat.
"""

import numpy as np

DEFAULT_HALF_WIDTH_HZ = 1.5
DEFAULT_FLANK_HZ = 3.0
DEFAULT_WINDOW_SEC = 10.0


def spectrum_interpolate(x: np.ndarray, fs: float, f0: float = 60.0, half: float = DEFAULT_HALF_WIDTH_HZ,
                         flank: float = DEFAULT_FLANK_HZ,
                         win_sec: float = DEFAULT_WINDOW_SEC) -> np.ndarray:
    """Remove a line at ``f0`` without a spectral hole (Leske & Dalal 2019).

    Short-time Fourier transform with 10 s Hann windows and 50% overlap (a
    constant-overlap-add pair, so an untouched signal is reconstructed
    exactly); in every window the magnitude of the bins within ``half`` Hz of
    ``f0`` is set from the power of the ``flank`` Hz on either side, their
    phase kept; inverse transform. The flank level is the median power divided
    by ln 2: for noise-like EEG the power of a bin is exponentially distributed,
    whose median is ln 2 times its mean, so this is an unbiased power estimate
    (a mean *magnitude* would be 4/pi, ~1 dB, too low and cut a shallow hole)
    that an unrelated narrow peak inside the flanks cannot pull up.
    1-D or ``[C, T]`` input, same shape out.
    """
    from scipy.signal import istft, stft
    n = int(round(win_sec * fs))
    if x.shape[-1] < n:
        return x
    f, _, z = stft(x, fs=fs, window="hann", nperseg=n, noverlap=n // 2, axis=-1)
    band = np.abs(f - f0) <= half
    neigh = (np.abs(f - f0) > half) & (np.abs(f - f0) <= half + flank)
    power = np.median(np.abs(z[..., neigh, :]) ** 2, axis=-2, keepdims=True) / np.log(2)
    mag = np.sqrt(power)
    z[..., band, :] = mag * np.exp(1j * np.angle(z[..., band, :]))
    _, y = istft(z, fs=fs, window="hann", nperseg=n, noverlap=n // 2, input_onesided=True)
    return y[..., :x.shape[-1]]
