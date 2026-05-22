"""
filters.py — Filtros de señal ECG + Baseline Removal
======================================================
Todos los filtros retornan np.ndarray del mismo tamaño que la entrada.
"""

import numpy as np
from scipy.signal import (
    butter, sosfiltfilt, cheby2,
    firwin, lfilter, medfilt,
)
from scipy.interpolate import UnivariateSpline

try:
    import pywt
    WAVELET_OK = True
except ImportError:
    WAVELET_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# FILTROS DE SEÑAL
# ══════════════════════════════════════════════════════════════════════════════

def filtro_chebyshev2_hp(signal: np.ndarray, fs: float, fc: float = 0.5) -> np.ndarray:
    """Chebyshev tipo II pasa-altas — elimina baseline wander de baja frecuencia."""
    nyq = fs / 2.0
    wp = fc / nyq
    try:
        sos = cheby2(6, 40, wp, btype="high", output="sos")
        return sosfiltfilt(sos, signal)
    except Exception:
        sos = butter(4, fc / nyq, btype="high", output="sos")
        return sosfiltfilt(sos, signal)


def filtro_butterworth_bp(signal: np.ndarray, fs: float,
                           fl: float = 0.5, fh: float = 40.0) -> np.ndarray:
    """Butterworth pasa-banda 0.5–40 Hz."""
    nyq = fs / 2.0
    fh = min(fh, nyq * 0.95)
    sos = butter(4, [fl / nyq, fh / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def filtro_fir_kaiser_bp(signal: np.ndarray, fs: float,
                          fl: float = 0.5, fh: float = 40.0) -> np.ndarray:
    """FIR con ventana Kaiser — fase lineal, preserva morfología del QRS."""
    nyq = fs / 2.0
    fh = min(fh, nyq * 0.95)
    num_taps = int(fs * 0.5) | 1
    num_taps = max(num_taps, 101)
    taps = firwin(num_taps, [fl / nyq, fh / nyq],
                  window=("kaiser", 8.6), pass_zero=False)
    delay = (len(taps) - 1) // 2
    padded = np.pad(signal, (0, delay), mode="edge")
    filtered = lfilter(taps, 1.0, padded)
    return filtered[delay:]


def filtro_mediana(signal: np.ndarray, fs: float,
                   window_ms: float = 200.0) -> np.ndarray:
    """Filtro de mediana — suprime picos de artefacto."""
    window = int(fs * window_ms / 1000.0) | 1
    window = max(window, 3)
    return medfilt(signal, kernel_size=window)


def filtro_wavelet(signal: np.ndarray, fs: float,
                   wavelet: str = "db6", level: int = 8) -> np.ndarray:
    """
    Umbralización wavelet soft (Donoho-Johnstone).
    Elimina ruido de alta frecuencia preservando morfología.
    """
    if not WAVELET_OK:
        return signal.copy()
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    level = min(level, max_level)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(max(len(signal), 1)))
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, threshold, mode="soft"))
    reconstructed = pywt.waverec(coeffs_thresh, wavelet)
    return reconstructed[: len(signal)]


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

def baseline_removal_morfologico(signal: np.ndarray, fs: float,
                                  window_ms: float = 200.0) -> np.ndarray:
    """
    Estimación de línea base por apertura morfológica (erosión + dilatación).

    La apertura morfológica con una ventana ~200 ms captura la tendencia lenta
    del baseline sin distorsionar el complejo QRS (duración ~80–120 ms).
    El baseline estimado se resta de la señal original.

    Parámetros
    ----------
    signal     : señal ECG (mV)
    fs         : frecuencia de muestreo (Hz)
    window_ms  : ancho de la ventana estructural (ms). 200 ms es estándar AAMI.

    Retorna
    -------
    baseline : tendencia estimada (mV) — restar de signal para obtener señal limpia.
    """
    w = int(fs * window_ms / 1000.0) | 1
    w = max(w, 3)

    # Erosión: mínimo local en ventana deslizante
    from scipy.ndimage import minimum_filter1d, maximum_filter1d
    eroded = minimum_filter1d(signal, size=w, mode="nearest")
    # Dilatación de la erosión = apertura
    opened = maximum_filter1d(eroded, size=w, mode="nearest")

    # Segunda pasada (cierre) para suavizar
    dilated = maximum_filter1d(opened, size=w, mode="nearest")
    closed  = minimum_filter1d(dilated, size=w, mode="nearest")

    baseline = (opened + closed) / 2.0
    return baseline


def baseline_removal_polinomial(signal: np.ndarray, order: int = 3) -> np.ndarray:
    """
    Estimación de línea base por ajuste polinomial de mínimos cuadrados.

    Ajusta un polinomio de grado `order` a la señal completa. Funciona bien
    para tendencias lentas y suaves. Órdenes bajos (1–3) modelan drift lineal
    o cuadrático; órdenes mayores capturan variaciones más complejas pero
    pueden causar efecto Runge en los extremos.

    Parámetros
    ----------
    signal : señal ECG (mV)
    order  : grado del polinomio (1 = lineal, 3 = cúbico recomendado)

    Retorna
    -------
    baseline : tendencia polinomial estimada (mV).
    """
    n = len(signal)
    t = np.linspace(0, 1, n)
    coef = np.polyfit(t, signal, order)
    baseline = np.polyval(coef, t)
    return baseline


def baseline_removal_spline(signal: np.ndarray, n_knots: int = 15) -> np.ndarray:
    """
    Estimación de línea base por spline cúbico suavizado (smoothing spline).

    Coloca `n_knots` nudos equiespaciados y ajusta un spline cúbico con
    mínima curvatura (penalización de segunda derivada). Captura variaciones
    de baseline más flexibles que el polinomio, sin los artefactos de borde.

    El factor de suavizado `s` se calibra automáticamente como
    s = n * std(signal)^2 / 4, priorizando suavidad sobre ajuste exacto.

    Parámetros
    ----------
    signal  : señal ECG (mV)
    n_knots : número de nudos internos (5–50). Más nudos → baseline más flexible.

    Retorna
    -------
    baseline : spline estimado (mV).
    """
    n = len(signal)
    t = np.linspace(0, 1, n)

    # Suavizado automático: queremos capturar tendencias lentas, no los QRS
    s_factor = n * (np.std(signal) ** 2) / 4.0

    try:
        spl = UnivariateSpline(t, signal, k=3, s=s_factor)
        baseline = spl(t)
    except Exception:
        # Fallback: spline sin suavizado con pocos nodos
        knots = np.linspace(0.05, 0.95, max(n_knots, 4))
        try:
            spl = UnivariateSpline(t, signal, k=3, t=knots)
            baseline = spl(t)
        except Exception:
            baseline = np.zeros_like(signal)

    return baseline
