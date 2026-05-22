"""
ECG MIMIC — Visualizador Interactivo de Señales
================================================
Streamlit app: sube archivos .hea + .dat, aplica filtros,
baseline removal y visualiza con Plotly + ecg-plot clínico.

Mejoras v2:
  - Carga de carpeta con múltiples pacientes + selector
  - Pestaña de comparativa con superposición de filtros activables

Mejora v3:
  - Super Resolución ECG con modelo CECGSR real (carga de pesos .pt)
"""

import io
import json
import os
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from threading import RLock

import matplotlib
matplotlib.use("Agg")          # modo sin ventana — obligatorio en Streamlit
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from scipy.signal import resample

try:
    import ecg_plot
    ECG_PLOT_OK = True
except ImportError:
    ECG_PLOT_OK = False

try:
    import neurokit2 as nk
    import pandas as pd
    NK_OK = True
except ImportError:
    NK_OK = False

from filters import (
    filtro_chebyshev2_hp,
    filtro_butterworth_bp,
    filtro_fir_kaiser_bp,
    filtro_mediana,
    filtro_wavelet,
    baseline_removal_morfologico,
    baseline_removal_polinomial,
    baseline_removal_spline,
)

# Lock global para operaciones Matplotlib (evita race conditions con usuarios concurrentes)
_mpl_lock = RLock()


# ══════════════════════════════════════════════════════════════════════════════
# MODELO CECGSR (debe coincidir con el entrenamiento)
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
    def forward(self, x):
        return x + self.net(x)

class CECGSR(nn.Module):
    """
    Entrada:  (batch, 1, N)  — señal LR
    Salida:   (batch, 1, N)  — señal SR reconstruida
    """
    def __init__(self, n_res_blocks=8, channels=64):
        super().__init__()
        self.entrada = nn.Conv1d(1, channels, kernel_size=9, padding=4)
        self.res_blocks = nn.Sequential(*[ResBlock1D(channels) for _ in range(n_res_blocks)])
        self.salida = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, 1, kernel_size=9, padding=4),
        )
    def forward(self, x):
        feat = self.entrada(x)
        out  = self.res_blocks(feat)
        return self.salida(out + feat)


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DEL MODELO (cacheado)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def cargar_modelo_cecgsr(ruta_weights="cecgsr_weights.pt"):
    """Carga el modelo entrenado o devuelve None si no existe."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = CECGSR(n_res_blocks=8, channels=64).to(device)
    if os.path.exists(ruta_weights):
        try:
            state_dict = torch.load(ruta_weights, map_location=device)
            modelo.load_state_dict(state_dict)
            modelo.eval()
            st.success(f"✅ Modelo CECGSR cargado desde {ruta_weights} en {device}")
            return modelo, device
        except Exception as e:
            st.warning(f"⚠️ Error al cargar el modelo: {e}")
            return None, device
    else:
        st.info(f"ℹ️ Archivo {ruta_weights} no encontrado. Se usará simulación.")
        return None, device

# Cargamos el modelo al inicio (se cachea)
modelo_cecgsr, device_cecgsr = cargar_modelo_cecgsr()


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ECG MIMIC Visualizer",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 Visualizador ECG MIMIC")
st.caption(
    "Carga registros MIMIC (.hea + .dat) · Filtros digitales · "
    "Baseline removal · Vista clínica (ecg-plot) · "
    "Super Resolución CECGSR · Análisis HRV con NeuroKit2"
)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE CARGA
# ══════════════════════════════════════════════════════════════════════════════

def cargar_registro(hea_bytes, dat_bytes, hea_name, fs_out, duracion):
    """Lee un registro MIMIC y retorna (record_id, fs, signals_dict)."""
    try:
        import wfdb
    except ImportError:
        st.error("wfdb no instalado. Revisa requirements.txt.")
        return None, None, None

    stem = Path(hea_name).stem
    with tempfile.TemporaryDirectory() as tmpdir:
        hea_path = os.path.join(tmpdir, stem + ".hea")
        dat_path = os.path.join(tmpdir, stem + ".dat")
        with open(hea_path, "wb") as f:
            f.write(hea_bytes)
        with open(dat_path, "wb") as f:
            f.write(dat_bytes)

        try:
            rec = wfdb.rdrecord(os.path.join(tmpdir, stem))
        except Exception as e:
            st.error(f"Error leyendo el registro: {e}")
            return None, None, None

        fs    = rec.fs
        leads = rec.sig_name
        try:
            signals = rec.p_signal
        except Exception:
            signals = rec.d_signal.astype(float)

        if signals is None or len(signals) == 0:
            st.error("El registro no contiene señales válidas.")
            return None, None, None

        signals = signals[: int(fs * duracion), :]

        factor  = max(1, int(fs / fs_out))
        signals = signals[::factor, :]
        fs_use  = fs / factor

        signals_dict = {}
        for i, lead in enumerate(leads):
            sig = signals[:, i].copy()
            sig = np.where(np.isnan(sig), 0.0, sig)
            signals_dict[lead] = sig

        return stem, float(fs_use), signals_dict


def cargar_registro_desde_archivos(uploaded_files):
    grupos = {}
    for f in uploaded_files:
        stem = Path(f.name).stem
        ext  = Path(f.name).suffix.lower()
        if ext not in (".hea", ".dat"):
            continue
        if stem not in grupos:
            grupos[stem] = {}
        grupos[stem][ext] = f

    pacientes = {}
    omitidos  = []
    for stem, archivos in grupos.items():
        if ".hea" in archivos and ".dat" in archivos:
            pacientes[stem] = (
                archivos[".hea"].read(),
                archivos[".dat"].read(),
                archivos[".hea"].name,
            )
        else:
            omitidos.append(stem)

    if omitidos:
        st.warning(
            f"⚠️ {len(omitidos)} registro(s) sin par completo ignorados: "
            f"{', '.join(omitidos[:5])}{'…' if len(omitidos) > 5 else ''}"
        )
    return pacientes


@st.cache_data(show_spinner=False)
def cargar_db_desde_zip(zip_bytes):
    pacientes = {}
    sin_par   = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        nombres = zf.namelist()

        indice = {}
        for nombre in nombres:
            p   = Path(nombre)
            ext = p.suffix.lower()
            if ext not in (".hea", ".dat"):
                continue
            clave = str(p.parent / p.stem)
            if clave not in indice:
                indice[clave] = {}
            indice[clave][ext] = nombre

        for clave, archivos in sorted(indice.items()):
            if ".hea" in archivos and ".dat" in archivos:
                hea_bytes = zf.read(archivos[".hea"])
                dat_bytes = zf.read(archivos[".dat"])
                stem      = Path(archivos[".hea"]).stem
                pacientes[clave] = (hea_bytes, dat_bytes, stem)
            else:
                sin_par.append(Path(clave).name)

    resumen = f"{len(pacientes)} registros encontrados"
    if sin_par:
        resumen += f" · {len(sin_par)} sin par ignorados"

    carpetas = Counter()
    for k in pacientes:
        partes = Path(k).parts
        carpetas[partes[0] if len(partes) > 1 else "(raíz)"] += 1

    return pacientes, resumen, dict(carpetas)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — CONTROLES
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("📁 Registro MIMIC")
    modo_carga = st.radio(
        "Modo de carga",
        [
            "Par individual (.hea + .dat)",
            "Múltiples archivos",
            "Base de datos completa (.zip)",
        ],
        index=0,
    )

    if modo_carga == "Par individual (.hea + .dat)":
        hea_file  = st.file_uploader("Archivo .hea", type=["hea"])
        dat_file  = st.file_uploader("Archivo .dat", type=["dat"])
        multi_files = None
        zip_file    = None

    elif modo_carga == "Múltiples archivos":
        st.caption("Selecciona todos los `.hea` y `.dat` de tus pacientes a la vez.")
        multi_files = st.file_uploader(
            "Archivos de pacientes",
            type=["hea", "dat"],
            accept_multiple_files=True,
        )
        hea_file = None
        dat_file = None
        zip_file = None

    else:  # ZIP
        st.caption(
            "Comprime tu carpeta MIMIC en un `.zip` (conservando subcarpetas) "
            "y súbela aquí. La app detectará todos los pares `.hea` + `.dat` automáticamente."
        )
        with st.expander("💡 ¿Cómo crear el ZIP?", expanded=False):
            st.markdown(
                "**En Windows:** clic derecho en la carpeta → *Comprimir en archivo ZIP*  \n"
                "**En Mac/Linux:** `zip -r mimic_db.zip ./carpeta_mimic/`  \n\n"
                "El ZIP puede contener subcarpetas (p00/p000020/…), "
                "la app las escanea recursivamente."
            )
        zip_file = st.file_uploader(
            "Base de datos MIMIC (.zip)",
            type=["zip"],
        )
        hea_file    = None
        dat_file    = None
        multi_files = None

    st.divider()

    st.subheader("📊 Señal")
    fs_out   = st.slider("Frecuencia de muestreo de salida (Hz)", 100, 500, 250, step=50)
    duracion = st.slider("Duración a mostrar (s)", 1, 30, 10)

    st.divider()

    st.subheader("📉 Baseline Removal")
    baseline_sel = st.selectbox(
        "Método",
        ["Ninguno", "Morfológico (apertura)", "Polinomial", "Spline cúbico"],
    )
    poly_order   = st.slider("Orden del polinomio", 1, 10, 3) if baseline_sel == "Polinomial" else 3
    spline_knots = st.slider("Nudos (knots)", 5, 50, 15)      if baseline_sel == "Spline cúbico" else 15
    show_baseline_overlay = st.checkbox("Mostrar línea base en gráfica", value=True)

    st.divider()

    st.subheader("🔬 Filtros de señal")
    filtros_sel = st.multiselect(
        "Filtros a aplicar (sobre señal corregida)",
        ["Chebyshev II HP", "Butterworth BP", "FIR Kaiser BP", "Mediana"],
        default=["Butterworth BP"],
    )
    wavelet_on = st.checkbox("Wavelet (umbralización suave)", value=False)

    st.divider()

    st.subheader("🩺 Vista clínica (ecg-plot)")
    if not ECG_PLOT_OK:
        st.warning("ecg-plot no instalado. Añade `ecg-plot` a requirements.txt.")
    ecg_modo = st.radio(
        "Modo",
        ["Derivación única", "Multi-derivación"],
        horizontal=False,
    )
    ecg_columnas = st.slider("Columnas (multi-derivación)", 1, 4, 2) if ecg_modo == "Multi-derivación" else 2
    ecg_dpi      = st.slider("DPI exportación PNG", 72, 300, 150, step=50)
    ecg_usar_corregida = st.checkbox("Usar señal con baseline removed", value=True)

    st.divider()

    st.subheader("🧠 NeuroKit2")
    if not NK_OK:
        st.warning("neurokit2 no instalado. Añade `neurokit2` a requirements.txt.")
    nk_metodo = st.selectbox(
        "Detector de picos R",
        ["neurokit", "pantompkins1985", "hamilton2002", "elgendi2010", "engzeemod2012"],
        key="nk_metodo_sb",
    )
    nk_dur_sb       = st.slider("Duración análisis (s)", 5, 30, 15, key="nk_dur_sb")
    nk_usar_filtrada = st.checkbox("Usar señal con baseline removed", value=True, key="nk_filtrada_sb")


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE PROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def aplicar_baseline(sig, fs, metodo, poly_order, spline_knots):
    if metodo == "Morfológico (apertura)":
        baseline = baseline_removal_morfologico(sig, fs)
    elif metodo == "Polinomial":
        baseline = baseline_removal_polinomial(sig, poly_order)
    elif metodo == "Spline cúbico":
        baseline = baseline_removal_spline(sig, spline_knots)
    else:
        return sig.copy(), np.zeros_like(sig)
    return sig - baseline, baseline


def aplicar_filtros(sig, fs, filtros_sel, wavelet_on):
    mapa = {
        "Chebyshev II HP": filtro_chebyshev2_hp,
        "Butterworth BP":  filtro_butterworth_bp,
        "FIR Kaiser BP":   filtro_fir_kaiser_bp,
        "Mediana":         filtro_mediana,
    }
    resultado = {}
    for nombre in filtros_sel:
        try:
            resultado[nombre] = mapa[nombre](sig, fs)
        except Exception as e:
            resultado[nombre] = sig.copy()
            st.warning(f"Filtro '{nombre}' falló: {e}")
    if wavelet_on:
        try:
            resultado["Wavelet"] = filtro_wavelet(sig, fs)
        except Exception as e:
            st.warning(f"Wavelet falló: {e}")
    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#7F77DD", "#639922", "#BA7517"]


def graficar_plotly(tiempo, sig_original, señales_filtradas,
                    sig_corregida, baseline, baseline_sel,
                    show_baseline_overlay, lead_name, fs):
    subplot_titles = ["Original"]
    if baseline_sel != "Ninguno":
        subplot_titles.append(f"Baseline Removal — {baseline_sel}")
    subplot_titles.extend(list(señales_filtradas.keys()))

    n_rows = len(subplot_titles)
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
    )

    # Original + baseline
    fig.add_trace(go.Scatter(
        x=tiempo, y=sig_original, name="Original",
        line=dict(width=1, color="#888780")), row=1, col=1)

    if baseline_sel != "Ninguno" and show_baseline_overlay:
        fig.add_trace(go.Scatter(
            x=tiempo, y=baseline, name="Línea base",
            line=dict(width=1.5, color="#E24B4A", dash="dash")), row=1, col=1)

    row = 2
    if baseline_sel != "Ninguno":
        fig.add_trace(go.Scatter(
            x=tiempo, y=sig_corregida,
            name=f"Corregida ({baseline_sel})",
            line=dict(width=1, color="#1D9E75")), row=row, col=1)
        row += 1

    for i, (nombre, sig_f) in enumerate(señales_filtradas.items()):
        fig.add_trace(go.Scatter(
            x=tiempo, y=sig_f, name=nombre,
            line=dict(width=1, color=PALETTE[i % len(PALETTE)])),
            row=row, col=1)
        row += 1

    fig.update_layout(
        height=240 * n_rows,
        title=f"Derivación: {lead_name}  |  fs = {fs:.0f} Hz",
        showlegend=True,
        margin=dict(l=50, r=20, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    for i in range(1, n_rows + 1):
        fig.update_yaxes(title_text="mV", row=i, col=1,
                         gridcolor="#f0f0f0", zerolinecolor="#ddd")
    fig.update_xaxes(title_text="Tiempo (s)", row=n_rows, col=1)
    return fig


def graficar_comparativa(tiempo, sig_original, sig_corregida,
                          señales_filtradas, baseline_sel, lead_name, fs,
                          capas_activas):
    fig = go.Figure()
    colores = {
        "Original":  "#888780",
        "Corregida": "#1D9E75",
    }
    for i, nombre in enumerate(señales_filtradas):
        colores[nombre] = PALETTE[i % len(PALETTE)]

    if capas_activas.get("Original", True):
        fig.add_trace(go.Scatter(
            x=tiempo, y=sig_original,
            name="Original",
            line=dict(width=1.5, color=colores["Original"]),
            opacity=0.85,
        ))

    if baseline_sel != "Ninguno" and capas_activas.get("Corregida", True):
        fig.add_trace(go.Scatter(
            x=tiempo, y=sig_corregida,
            name=f"Corregida ({baseline_sel})",
            line=dict(width=1.5, color=colores["Corregida"]),
            opacity=0.85,
        ))

    for nombre, sig_f in señales_filtradas.items():
        if capas_activas.get(nombre, True):
            fig.add_trace(go.Scatter(
                x=tiempo, y=sig_f,
                name=nombre,
                line=dict(width=1.5, color=colores.get(nombre, "#999")),
                opacity=0.85,
            ))

    fig.update_layout(
        height=480,
        title=f"Comparativa superpuesta — Derivación: {lead_name}  |  fs = {fs:.0f} Hz",
        showlegend=True,
        margin=dict(l=50, r=20, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Tiempo (s)",
        yaxis_title="mV",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#f0f0f0", zerolinecolor="#ddd")
    fig.update_yaxes(gridcolor="#f0f0f0", zerolinecolor="#ddd")
    return fig


def _construir_matriz_ecg(signals_dict, fs, baseline_sel,
                           poly_order, spline_knots, usar_corregida):
    all_leads = list(signals_dict.keys())
    n_min = min(len(signals_dict[l]) for l in all_leads)
    matrix = []
    for lead_name in all_leads:
        s = signals_dict[lead_name][:n_min]
        if usar_corregida and baseline_sel != "Ninguno":
            s_corr, _ = aplicar_baseline(s, fs, baseline_sel, poly_order, spline_knots)
        else:
            s_corr = s
        matrix.append(s_corr)
    return np.array(matrix), all_leads


def render_ecg_plot_unica(sig, fs, lead_name, record_id, dpi):
    with _mpl_lock:
        plt.close("all")
        ecg_plot.plot_1(sig, sample_rate=int(fs),
                        title=f"{lead_name}  |  {record_id}")
        fig = plt.gcf()
        fig.set_size_inches(16, 3.5)
        fig.patch.set_facecolor("white")
        st.pyplot(fig, clear_figure=False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        plt.close("all")
        buf.seek(0)
        return buf.getvalue()


def render_ecg_plot_multi(matrix, lead_names, fs, record_id, columnas, dpi):
    with _mpl_lock:
        plt.close("all")
        ecg_plot.plot(
            matrix,
            sample_rate=int(fs),
            title=f"ECG multi-derivación  |  {record_id}",
            lead_index=lead_names,
            columns=columnas,
        )
        fig = plt.gcf()
        n_leads = len(lead_names)
        rows_per_col = -(-n_leads // columnas)
        fig.set_size_inches(7 * columnas, 2.5 * rows_per_col + 1)
        fig.patch.set_facecolor("white")
        st.pyplot(fig, clear_figure=False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        plt.close("all")
        buf.seek(0)
        return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# RESOLUCIÓN DE ARCHIVOS CARGADOS
# ══════════════════════════════════════════════════════════════════════════════

pacientes_disponibles = {}
record_id    = None
fs           = None
signals_dict = None
paciente_sel = None

if modo_carga == "Par individual (.hea + .dat)":
    if hea_file is not None and dat_file is not None:
        pacientes_disponibles["__single__"] = (
            hea_file.read(), dat_file.read(), hea_file.name
        )
        paciente_sel = "__single__"

elif modo_carga == "Múltiples archivos":
    if multi_files:
        pacientes_disponibles = cargar_registro_desde_archivos(multi_files)
        if pacientes_disponibles:
            stems = sorted(pacientes_disponibles.keys())
            with st.sidebar:
                st.divider()
                st.subheader("👤 Paciente")
                paciente_sel = st.selectbox(
                    f"Selecciona ({len(stems)} encontrados)",
                    stems,
                )

else:  # ZIP
    if zip_file is not None:
        with st.spinner("🗜️ Descomprimiendo y escaneando la base de datos…"):
            zip_bytes = zip_file.read()
            pacientes_disponibles, resumen, carpetas_resumen = cargar_db_desde_zip(zip_bytes)

        if not pacientes_disponibles:
            st.error("No se encontraron pares `.hea` + `.dat` dentro del ZIP.")
        else:
            with st.expander(f"📊 Base de datos cargada — {resumen}", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("Total de registros", len(pacientes_disponibles))
                c2.metric("Subcarpetas", len(carpetas_resumen))
                if carpetas_resumen:
                    st.markdown("**Registros por subcarpeta:**")
                    cols_res = st.columns(min(len(carpetas_resumen), 4))
                    for i, (carpeta, count) in enumerate(sorted(carpetas_resumen.items())):
                        cols_res[i % len(cols_res)].metric(f"📁 {carpeta}", f"{count} reg.")

            stems = sorted(pacientes_disponibles.keys())
            with st.sidebar:
                st.divider()
                st.subheader("👤 Paciente")
                st.caption(f"📂 **{len(stems)}** registros · `{zip_file.name}`")

                buscar = st.text_input(
                    "🔎 Buscar paciente",
                    placeholder="Filtra por nombre o ruta…",
                )
                stems_filtrados = (
                    [s for s in stems if buscar.lower() in s.lower()]
                    if buscar else stems
                )

                if not stems_filtrados:
                    st.warning("Ningún registro coincide con la búsqueda.")
                else:
                    paciente_sel = st.selectbox(
                        f"{len(stems_filtrados)} mostrados",
                        stems_filtrados,
                        format_func=lambda x: Path(x).name,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# CARGA Y PROCESAMIENTO DEL PACIENTE SELECCIONADO
# ══════════════════════════════════════════════════════════════════════════════

if paciente_sel and paciente_sel in pacientes_disponibles:
    hea_b, dat_b, hea_n = pacientes_disponibles[paciente_sel]

    with st.spinner("Cargando registro..."):
        record_id, fs, signals_dict = cargar_registro(
            hea_b, dat_b, hea_n, fs_out, duracion
        )

if signals_dict:
    st.success(
        f"✅ **{record_id}** cargado — "
        f"{len(signals_dict)} derivaciones · fs = {fs:.0f} Hz · "
        f"{duracion} s"
    )

    leads_disponibles = list(signals_dict.keys())
    lead_sel = st.selectbox("Derivación activa (para análisis de filtros)", leads_disponibles)

    sig    = signals_dict[lead_sel]
    tiempo = np.arange(len(sig)) / fs

    # Baseline removal
    sig_corregida, baseline = aplicar_baseline(
        sig, fs, baseline_sel, poly_order, spline_knots
    )

    # Filtros
    sig_para_filtrar  = sig_corregida if baseline_sel != "Ninguno" else sig
    señales_filtradas = aplicar_filtros(sig_para_filtrar, fs, filtros_sel, wavelet_on)

    # TABS
    tab_plotly, tab_comp, tab_clinica, tab_stats, tab_export, tab_sr, tab_nk = st.tabs([
        "📈 Análisis de filtros",
        "🔀 Comparativa",
        "🩺 Vista clínica",
        "📊 Estadísticas",
        "💾 Exportar",
        "🔬 Super Resolución ECG",
        "🧠 NeuroKit2",
    ])

    # Tab 1
    with tab_plotly:
        fig_plotly = graficar_plotly(
            tiempo, sig, señales_filtradas, sig_corregida, baseline,
            baseline_sel, show_baseline_overlay, lead_sel, fs,
        )
        st.plotly_chart(fig_plotly, use_container_width=True)

    # Tab 2
    with tab_comp:
        st.markdown("#### 🔀 Superposición de señales")
        st.caption(
            "Activa o desactiva cada capa con los interruptores. "
            "Todas las señales se muestran en el mismo eje para comparación directa."
        )

        capas = ["Original"]
        if baseline_sel != "Ninguno":
            capas.append("Corregida")
        capas.extend(list(señales_filtradas.keys()))

        colores_capa = {
            "Original":  "#888780",
            "Corregida": "#1D9E75",
        }
        for i, nombre in enumerate(señales_filtradas):
            colores_capa[nombre] = PALETTE[i % len(PALETTE)]

        st.markdown("**Capas visibles:**")
        n_cols = min(len(capas), 4)
        cols_toggle = st.columns(n_cols)
        capas_activas = {}

        for idx, capa in enumerate(capas):
            color_hex = colores_capa.get(capa, "#999")
            with cols_toggle[idx % n_cols]:
                st.markdown(
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:{color_hex};border-radius:3px;margin-right:5px;'
                    f'vertical-align:middle;"></span>',
                    unsafe_allow_html=True,
                )
                capas_activas[capa] = st.toggle(
                    capa if capa != "Corregida" else f"Corregida ({baseline_sel})",
                    value=True,
                    key=f"toggle_{capa}",
                )

        st.divider()

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            zoom_inicio = st.number_input(
                "Zoom — inicio (s)", min_value=0.0,
                max_value=float(tiempo[-1]), value=0.0, step=0.5,
            )
        with col_opt2:
            zoom_fin = st.number_input(
                "Zoom — fin (s)", min_value=0.0,
                max_value=float(tiempo[-1]), value=float(tiempo[-1]), step=0.5,
            )
        with col_opt3:
            opacidad = st.slider("Opacidad global", 0.2, 1.0, 0.85, step=0.05)

        mask = (tiempo >= zoom_inicio) & (tiempo <= zoom_fin)
        t_zoom  = tiempo[mask]

        def _recortar(arr):
            return arr[mask] if len(arr) == len(tiempo) else arr

        sig_orig_zoom  = _recortar(sig)
        sig_corr_zoom  = _recortar(sig_corregida)
        filtradas_zoom = {k: _recortar(v) for k, v in señales_filtradas.items()}

        fig_comp = go.Figure()

        if capas_activas.get("Original", True):
            fig_comp.add_trace(go.Scatter(
                x=t_zoom, y=sig_orig_zoom,
                name="Original",
                line=dict(width=1.8, color=colores_capa["Original"]),
                opacity=opacidad,
            ))

        if baseline_sel != "Ninguno" and capas_activas.get("Corregida", True):
            fig_comp.add_trace(go.Scatter(
                x=t_zoom, y=sig_corr_zoom,
                name=f"Corregida ({baseline_sel})",
                line=dict(width=1.8, color=colores_capa["Corregida"]),
                opacity=opacidad,
            ))

        for nombre, sig_f in filtradas_zoom.items():
            if capas_activas.get(nombre, True):
                fig_comp.add_trace(go.Scatter(
                    x=t_zoom, y=sig_f,
                    name=nombre,
                    line=dict(width=1.8, color=colores_capa.get(nombre, "#999")),
                    opacity=opacidad,
                ))

        n_activas = sum(capas_activas.values())
        fig_comp.update_layout(
            height=500,
            title=(
                f"Comparativa — {lead_sel}  |  fs = {fs:.0f} Hz  |  "
                f"{n_activas} de {len(capas)} capas activas"
            ),
            showlegend=True,
            margin=dict(l=50, r=20, t=70, b=50),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Tiempo (s)",
            yaxis_title="Amplitud (mV)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.04,
                xanchor="left",
                x=0,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#ddd",
                borderwidth=1,
            ),
            hovermode="x unified",
        )
        fig_comp.update_xaxes(gridcolor="#f0f0f0", zerolinecolor="#ccc", showgrid=True)
        fig_comp.update_yaxes(gridcolor="#f0f0f0", zerolinecolor="#ccc", showgrid=True)

        st.plotly_chart(fig_comp, use_container_width=True)

        if n_activas >= 2 and señales_filtradas:
            st.markdown("**Diferencia RMS entre señales activas (ventana visible)**")
            rms_vals = {}
            if capas_activas.get("Original", True):
                rms_vals["Original"] = float(np.sqrt(np.mean(sig_orig_zoom**2)))
            if baseline_sel != "Ninguno" and capas_activas.get("Corregida", True):
                rms_vals[f"Corregida"] = float(np.sqrt(np.mean(sig_corr_zoom**2)))
            for nombre, sig_f in filtradas_zoom.items():
                if capas_activas.get(nombre, True):
                    rms_vals[nombre] = float(np.sqrt(np.mean(sig_f**2)))
            cols_rms = st.columns(len(rms_vals))
            for col, (nombre, rms) in zip(cols_rms, rms_vals.items()):
                col.metric(nombre, f"{rms:.4f} mV")

    # Tab 3
    with tab_clinica:
        if not ECG_PLOT_OK:
            st.error("ecg-plot no está instalado. Añade `ecg-plot>=0.2.8` a requirements.txt y redespliega.")
            st.stop()

        st.caption(
            "Vista con cuadrícula milimetrada estándar. "
            "La señal mostrada tiene baseline removal aplicado si está activo en el sidebar."
        )

        if ecg_modo == "Derivación única":
            sig_ecg = sig_corregida if (ecg_usar_corregida and baseline_sel != "Ninguno") else sig
            st.markdown(f"**Derivación:** {lead_sel}")
            with st.spinner("Generando vista clínica..."):
                png_bytes = render_ecg_plot_unica(
                    sig_ecg, fs, lead_sel, record_id, ecg_dpi
                )
            st.download_button(
                f"⬇️ Descargar PNG — {lead_sel}",
                data=png_bytes,
                file_name=f"{record_id}_{lead_sel}_ecg.png",
                mime="image/png",
            )
        else:
            matrix, lead_names = _construir_matriz_ecg(
                signals_dict, fs, baseline_sel if ecg_usar_corregida else "Ninguno",
                poly_order, spline_knots, ecg_usar_corregida,
            )
            st.markdown(
                f"**{len(lead_names)} derivaciones** · "
                f"{ecg_columnas} columna{'s' if ecg_columnas > 1 else ''}"
            )
            with st.spinner("Generando vista multi-derivación..."):
                png_bytes = render_ecg_plot_multi(
                    matrix, lead_names, fs, record_id, ecg_columnas, ecg_dpi
                )
            st.download_button(
                f"⬇️ Descargar PNG — {len(lead_names)} derivaciones",
                data=png_bytes,
                file_name=f"{record_id}_multi_ecg.png",
                mime="image/png",
            )

    # Tab 4
    with tab_stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Muestras", f"{len(sig):,}")
        col2.metric("Duración", f"{len(sig)/fs:.2f} s")
        col3.metric("Amplitud máx.", f"{np.max(np.abs(sig)):.4f} mV")
        col4.metric("RMS original", f"{np.sqrt(np.mean(sig**2)):.4f} mV")

        if baseline_sel != "Ninguno":
            st.divider()
            c1, c2, c3 = st.columns(3)
            rms_orig = np.sqrt(np.mean(sig**2))
            rms_corr = np.sqrt(np.mean(sig_corregida**2))
            rms_base = np.sqrt(np.mean(baseline**2))
            reduccion = (1 - rms_corr / max(rms_orig, 1e-9)) * 100
            c1.metric("RMS señal corregida", f"{rms_corr:.4f} mV")
            c2.metric("RMS línea base", f"{rms_base:.4f} mV")
            c3.metric("Reducción baseline", f"{reduccion:.1f} %")

        if señales_filtradas:
            st.divider()
            st.markdown("**RMS por filtro aplicado**")
            cols = st.columns(len(señales_filtradas))
            for col, (nombre, sig_f) in zip(cols, señales_filtradas.items()):
                col.metric(nombre, f"{np.sqrt(np.mean(sig_f**2)):.4f} mV")

    # Tab 6: Super Resolución ECG (con modelo real)
    with tab_sr:
        st.markdown("### 🔬 Super Resolución de Señales ECG (CECGSR)")
        st.caption(
            "Aplica filtros clásicos como preprocesamiento y usa el modelo CECGSR entrenado para mejorar la señal. "
            "Compara la señal filtrada (LR) contra la señal de alta resolución reconstruida (SR) "
            "con métricas SNR, RMSE y correlación."
        )

        # ── Imports SR ────────────────────────────────────────────────────
        import math
        from scipy.signal import butter, sosfiltfilt, welch
        from scipy.ndimage import uniform_filter1d

        # ── Degradación realista con nivel de ruido controlable ───────────
        def degradar_senal(sig: np.ndarray, fs: float,
                           factor: int = 2,
                           snr_objetivo_db: float = 30.0) -> np.ndarray:
            """
            Degradación realista de dos etapas:
            1. Pérdida de resolución temporal (downsample/upsample con AA)
            2. Ruido aditivo gaussiano calibrado al SNR objetivo
            Factor por defecto = 2 (25 Hz si fs=250) para no destruir la señal.
            """
            n   = len(sig)
            rng = np.random.default_rng(42)
            nyq = fs / 2.0

            # Anti-aliasing conservador antes del downsample
            fc_aa = min((fs / factor) * 0.40, nyq * 0.85)
            sos_aa = butter(6, fc_aa / nyq, btype="low", output="sos")
            sig_aa = sosfiltfilt(sos_aa, sig)

            # Downsample → upsample
            n_down = max(n // factor, 32)
            sig_lr = resample(resample(sig_aa, n_down), n)

            # Ruido AWGN calibrado
            pot_sig   = np.mean(sig_lr ** 2) + 1e-12
            sigma     = math.sqrt(pot_sig / (10 ** (snr_objetivo_db / 10)))
            sig_lr   += rng.normal(0, sigma, n)
            return sig_lr

        # ── Wiener espectral MMSE (estima H/N desde la señal) ────────────
        def sr_wiener_espectral(sig_lr: np.ndarray, fs: float) -> np.ndarray:
            """
            Filtro de Wiener en dominio frecuencial (MMSE):
            H_opt(f) = Pss(f) / (Pss(f) + Pnn(f))
            - Pss estimado como la parte 'suave' del periodograma (mediana)
            - Pnn estimado desde los coeficientes de alta frecuencia
            La reconstrucción conserva la fase exacta de la LR.
            """
            n      = len(sig_lr)
            SIG_LR = np.fft.rfft(sig_lr, n=n)
            Pyy    = np.abs(SIG_LR) ** 2

            # Estimación de Pss: suavizar el espectro (mínimo en ventana de 1/8 del espectro)
            k      = max(len(Pyy) // 8, 3)
            Pss    = uniform_filter1d(Pyy, size=k)

            # Estimación de Pnn: mediana del cuarto superior del espectro (zona ruidosa)
            q3     = int(len(Pyy) * 0.75)
            Pnn    = max(np.median(Pyy[q3:]), 1e-12)

            # Ganancia de Wiener (acotada a [0, 1])
            H      = Pss / (Pss + Pnn)
            H      = np.clip(H, 0.0, 1.0)

            # Aplicar filtro manteniendo fase
            SIG_SR = SIG_LR * H
            sig_sr = np.fft.irfft(SIG_SR, n=n)

            # Post-filtro Butterworth suave (40 Hz) para eliminar artefactos espectrales
            nyq = fs / 2.0
            fc  = min(40.0, nyq * 0.85)
            sos = butter(4, fc / nyq, btype="low", output="sos")
            return sosfiltfilt(sos, sig_sr)

        # ── Reconstrucción iterativa (Papoulis-Gerchberg) ─────────────────
        def sr_iterativo(sig_lr: np.ndarray, fs: float,
                         n_iter: int = 8) -> np.ndarray:
            """
            Algoritmo iterativo de proyecciones alternadas:
            Itera entre dominio temporal (mantener consistencia con LR)
            y dominio frecuencial (proyectar sobre banda ECG 0.5–40 Hz).
            Converge en ~8 iteraciones mejorando SNR ~3–6 dB sobre la entrada.
            """
            nyq    = fs / 2.0
            fl, fh = 0.5 / nyq, min(40.0 / nyq, 0.92)
            sos    = butter(6, [fl, fh], btype="band", output="sos")

            # Punto de partida: Wiener espectral
            est = sr_wiener_espectral(sig_lr, fs)

            # Estimar sigma del ruido: std de la parte fuera de banda de la LR
            sigma_n = np.std(sig_lr - sosfiltfilt(sos, sig_lr)) + 1e-8

            for _ in range(n_iter):
                # Paso A — proyección frecuencial: forza la estimación a la banda ECG
                est_band = sosfiltfilt(sos, est)

                # Paso B — proyección temporal: incorpora datos de la LR con peso
                # adaptativo.
                # CORRECCIÓN: la mezcla debe ser proporcional a la "limpieza local"
                # de la LR. Usamos la suavidad local (varianza alta = zona de QRS
                # o artefacto → confiar más en el estimado filtrado).
                suavidad_lr  = uniform_filter1d(sig_lr ** 2, size=max(int(fs*0.02)|1,3))
                suavidad_est = uniform_filter1d(est_band ** 2, size=max(int(fs*0.02)|1,3))
                # alpha: peso para el estimado filtrado (0=todo LR, 1=todo estimado)
                # En zonas de alta energía (QRS) confiamos más en el estimado filtrado
                # porque el ruido relativo es menor allí.
                snr_local = suavidad_est / (np.abs(sig_lr - est_band) ** 2 + sigma_n ** 2)
                alpha     = np.clip(snr_local / (snr_local + 1.0), 0.3, 0.95)
                est       = alpha * est_band + (1.0 - alpha) * sig_lr

            return sosfiltfilt(sos, est)

        # ── Ensemble ponderado por suavidad espectral ─────────────────────
        def sr_ensemble(sig_lr: np.ndarray, fs: float) -> np.ndarray:
            """
            Combina Wiener + Iterativo + Wavelet con pesos por calidad espectral.

            Criterio de peso: cuánto de la energía del resultado está dentro
            de la banda ECG (0.5–40 Hz) vs fuera de ella.
            Un buen reconstructor tiene alta energía en banda y baja fuera → ratio alto.
            Esto es independiente de la LR ruidosa (evita el sesgo anterior).
            """
            nyq    = fs / 2.0
            fl, fh = 0.5 / nyq, min(40.0 / nyq, 0.92)
            sos    = butter(4, [fl, fh], btype="band", output="sos")

            s1 = sr_wiener_espectral(sig_lr, fs)
            s2 = sr_iterativo(sig_lr, fs)
            try:
                s3 = filtro_wavelet(sig_lr, fs)
            except Exception:
                s3 = s1.copy()

            def calidad_espectral(s: np.ndarray) -> float:
                """Ratio energía en banda / energía total (mayor = mejor)."""
                e_total = np.sum(s ** 2) + 1e-12
                e_banda = np.sum(sosfiltfilt(sos, s) ** 2) + 1e-12
                return float(e_banda / e_total)

            q1, q2, q3 = calidad_espectral(s1), calidad_espectral(s2), calidad_espectral(s3)
            total = q1 + q2 + q3
            return (s1 * q1 + s2 * q2 + s3 * q3) / total

        # ── Modelo CECGSR real (ventanas solapadas con Hann) ──────────────
        def aplicar_modelo_real(sig_lr: np.ndarray, modelo, device) -> np.ndarray:
            SEG  = 512
            HOP  = SEG // 2
            mean = sig_lr.mean()
            std  = sig_lr.std() + 1e-8
            norm = (sig_lr - mean) / std
            n    = len(norm)
            out  = np.zeros(n, dtype=np.float64)
            cnt  = np.zeros(n, dtype=np.float64)
            win  = np.hanning(SEG)
            for start in range(0, max(n - SEG + 1, 1), HOP):
                end = start + SEG
                if end > n:
                    break
                seg = norm[start:end].astype(np.float32)
                x   = torch.tensor(seg).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = modelo(x).squeeze().cpu().numpy()
                out[start:end] += pred * win
                cnt[start:end] += win
            cnt = np.where(cnt < 1e-6, 1.0, cnt)
            return (out / cnt) * std + mean

        # ── Métricas: SNR / RMSE / PRD / PSNR / Correlación ─────────────
        def calcular_metricas(original: np.ndarray,
                              reconstruida: np.ndarray) -> dict:
            """
            PRD (Percent Root-mean-square Difference) es el estándar AAMI EC57.
            Antes de medir se resta la media (elimina sesgo DC que infla PRD).
            """
            n   = min(len(original), len(reconstruida))
            s   = (original[:n]      - original[:n].mean()).astype(float)
            r   = (reconstruida[:n]  - reconstruida[:n].mean()).astype(float)
            err = s - r
            pot_s = np.sum(s ** 2) + 1e-12
            pot_e = np.sum(err ** 2) + 1e-12
            snr   = 10 * np.log10(pot_s / pot_e)
            rmse  = float(np.sqrt(np.mean(err ** 2)))
            prd   = float(np.sqrt(pot_e / pot_s) * 100)
            peak  = float(np.max(np.abs(s)))
            psnr  = float(20 * np.log10(peak / (rmse + 1e-12))) if peak > 0 else 0.0
            corr  = float(np.corrcoef(s, r)[0, 1]) if (
                np.std(s) > 1e-8 and np.std(r) > 1e-8) else 0.0
            return {
                "SNR (dB)":    round(snr,  3),
                "RMSE (mV)":   round(rmse, 6),
                "PRD (%)":     round(prd,  3),
                "PSNR (dB)":   round(psnr, 3),
                "Correlación": round(corr, 6),
            }

        # ── Estado de sesión para muestras guardadas ──
        if "sr_muestras" not in st.session_state:
            st.session_state["sr_muestras"] = {}
        if "sr_sel_idx" not in st.session_state:
            st.session_state["sr_sel_idx"] = None

        # Layout principal
        col_muestras, col_centro, col_inference = st.columns([1, 2.4, 2.4])

        with col_muestras:
            st.markdown("**Muestras**")
            muestras = st.session_state["sr_muestras"]
            if not muestras:
                st.caption("Sin muestras aún.\nGuarda una desde la tabla de métricas.")
            else:
                for idx, (nombre, _) in enumerate(muestras.items()):
                    selected = (st.session_state["sr_sel_idx"] == nombre)
                    btn_style = "primary" if selected else "secondary"
                    if st.button(
                        f"📌 {nombre}",
                        key=f"sr_btn_{nombre}",
                        use_container_width=True,
                        type=btn_style,
                    ):
                        st.session_state["sr_sel_idx"] = nombre
                        st.rerun()

            st.divider()
            st.markdown("**Modelos**")
            modelo_sel = st.radio(
                "",
                [
                    "Ensemble (Wiener+Iter+Wavelet)",
                    "Wiener Espectral (MMSE)",
                    "Iterativo (Papoulis-Gerchberg)",
                    "Wavelet SR",
                    "CECGSR (real)",
                ],
                key="sr_modelo",
                label_visibility="collapsed",
            )

        with col_centro:
            st.markdown("**Imagen de entrada (LR)**")
            c1, c2, c3 = st.columns(3)
            with c1:
                sr_filtro_pre = st.selectbox(
                    "Prefiltro",
                    ["Butterworth BP", "FIR Kaiser BP", "Wavelet", "Ninguno"],
                    key="sr_pre",
                )
            with c2:
                sr_factor_deg = st.slider(
                    "Factor degradación", 2, 6, 2, key="sr_deg",
                    help="Factor=2 pierde el 50% de muestras. Factor=4 destruye el 75%.")
            with c3:
                snr_deg_target = st.slider(
                    "SNR degradación (dB)", 10, 40, 30, key="sr_snr_deg",
                    help="30 dB = ruido leve realista. 10 dB = señal muy degradada.")

        with col_inference:
            st.markdown("**Inferencia (SR)**")
            c4, c5, c6 = st.columns(3)
            with c4:
                sr_lead = st.selectbox("Derivación SR", leads_disponibles, key="sr_lead")
            with c5:
                sr_seg_dur = st.slider("Segmento (s)", 1, min(10, duracion), 5, key="sr_seg")
            with c6:
                sr_n_iter = st.slider(
                    "Iteraciones (SR iterativo)", 4, 20, 8, key="sr_niter",
                    help="Más iteraciones = mejor convergencia, más lento.")

        # ── Procesamiento ─────────────────────────────────────────────────
        sig_sr_orig = signals_dict[sr_lead]
        n_seg       = int(sr_seg_dur * fs)
        sig_seg_raw = sig_sr_orig[:n_seg]

        # PASO 1 — Baseline removal (referencia limpia para métricas)
        sig_ref, _ = aplicar_baseline(sig_seg_raw, fs, "Morfológico (apertura)", 3, 15)

        # PASO 2 — Prefiltro sobre la referencia limpia
        if sr_filtro_pre == "Butterworth BP":
            sig_pre = filtro_butterworth_bp(sig_ref, fs)
        elif sr_filtro_pre == "FIR Kaiser BP":
            sig_pre = filtro_fir_kaiser_bp(sig_ref, fs)
        elif sr_filtro_pre == "Wavelet":
            sig_pre = filtro_wavelet(sig_ref, fs)
        else:
            sig_pre = sig_ref.copy()

        # PASO 3 — Degradar → LR
        sig_lr = degradar_senal(sig_pre, fs,
                                factor=sr_factor_deg,
                                snr_objetivo_db=float(snr_deg_target))

        # PASO 4 — Reconstrucción SR
        with st.spinner("Aplicando modelo SR…"):
            try:
                if modelo_sel == "Wiener Espectral (MMSE)":
                    sig_sr = sr_wiener_espectral(sig_lr, fs)
                elif modelo_sel == "Iterativo (Papoulis-Gerchberg)":
                    sig_sr = sr_iterativo(sig_lr, fs, n_iter=sr_n_iter)
                elif modelo_sel == "Ensemble (Wiener+Iter+Wavelet)":
                    sig_sr = sr_ensemble(sig_lr, fs)
                elif modelo_sel == "Wavelet SR":
                    sig_sr = filtro_wavelet(sig_lr, fs)
                else:  # CECGSR real
                    if modelo_cecgsr is not None:
                        sig_sr = aplicar_modelo_real(sig_lr, modelo_cecgsr, device_cecgsr)
                    else:
                        st.info("Modelo CECGSR no cargado — usando Ensemble como fallback.")
                        sig_sr = sr_ensemble(sig_lr, fs)
            except Exception as e:
                st.error(f"Error en SR: {e}")
                sig_sr = sig_lr.copy()

        # Referencia para métricas = señal preprocesada (sin degradación)
        sig_seg = sig_pre

        t_seg = np.arange(len(sig_seg)) / fs

        def _fig_sr_simple(t, y, titulo, color):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t, y=y, mode="lines",
                line=dict(width=1.4, color=color), name=titulo,
            ))
            fig.update_layout(
                height=230,
                margin=dict(l=40, r=10, t=36, b=36),
                title=dict(text=titulo, font=dict(size=13)),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
                xaxis=dict(title="Tiempo (s)", gridcolor="#f0f0f0"),
                yaxis=dict(title="mV", gridcolor="#f0f0f0"),
            )
            return fig

        with col_centro:
            fig_lr = _fig_sr_simple(t_seg, sig_lr, f"Señal LR — prefiltro: {sr_filtro_pre}", "#555e6e")
            st.plotly_chart(fig_lr, use_container_width=True, key="sr_fig_lr")

        with col_inference:
            fig_sr = _fig_sr_simple(t_seg, sig_sr, f"Señal SR — {modelo_sel}", "#378ADD")
            st.plotly_chart(fig_sr, use_container_width=True, key="sr_fig_sr")

        # ── Métricas ───────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### 📐 Métricas de calidad (referencia: señal preprocesada)")

        metricas_lr = calcular_metricas(sig_seg, sig_lr)
        metricas_sr = calcular_metricas(sig_seg, sig_sr)

        # Cabecera
        hc = st.columns([1.6, 1, 1, 1, 1, 1])
        hc[0].markdown("**Señal**")
        hc[1].markdown("**SNR (dB) ↑**")
        hc[2].markdown("**RMSE (mV) ↓**")
        hc[3].markdown("**PRD (%) ↓**")
        hc[4].markdown("**PSNR (dB) ↑**")
        hc[5].markdown("**Correlación ↑**")

        # Fila LR
        r1 = st.columns([1.6, 1, 1, 1, 1, 1])
        r1[0].markdown("🔴 LR (degradada)")
        r1[1].metric("", f"{metricas_lr['SNR (dB)']:.2f}")
        r1[2].metric("", f"{metricas_lr['RMSE (mV)']:.5f}")
        r1[3].metric("", f"{metricas_lr['PRD (%)']:.2f}")
        r1[4].metric("", f"{metricas_lr['PSNR (dB)']:.2f}")
        r1[5].metric("", f"{metricas_lr['Correlación']:.4f}")

        # Fila SR con deltas
        r2 = st.columns([1.6, 1, 1, 1, 1, 1])
        r2[0].markdown(f"🔵 **SR** ({modelo_sel})")
        d_snr  = metricas_sr["SNR (dB)"]   - metricas_lr["SNR (dB)"]
        d_rmse = metricas_sr["RMSE (mV)"]  - metricas_lr["RMSE (mV)"]
        d_prd  = metricas_sr["PRD (%)"]    - metricas_lr["PRD (%)"]
        d_psnr = metricas_sr["PSNR (dB)"]  - metricas_lr["PSNR (dB)"]
        d_corr = metricas_sr["Correlación"]- metricas_lr["Correlación"]
        r2[1].metric("", f"{metricas_sr['SNR (dB)']:.2f}",   delta=f"{d_snr:+.2f} dB")
        r2[2].metric("", f"{metricas_sr['RMSE (mV)']:.5f}",  delta=f"{d_rmse:+.5f}")
        r2[3].metric("", f"{metricas_sr['PRD (%)']:.2f}",    delta=f"{d_prd:+.2f} %",
                     delta_color="inverse")
        r2[4].metric("", f"{metricas_sr['PSNR (dB)']:.2f}",  delta=f"{d_psnr:+.2f} dB")
        r2[5].metric("", f"{metricas_sr['Correlación']:.4f}", delta=f"{d_corr:+.4f}")

        # Interpretación automática
        with st.expander("💡 Interpretación de métricas", expanded=False):
            st.markdown("""
| Métrica | Qué mide | Valores de referencia (ECG) |
|---|---|---|
| **SNR** | Potencia señal / potencia ruido | > 20 dB = aceptable · > 30 dB = bueno |
| **RMSE** | Error cuadrático medio (mV) | < 0.05 mV = clínicamente aceptable |
| **PRD** | % de distorsión respecto a la señal original | < 9% = excelente · < 14% = bueno (AAMI) |
| **PSNR** | Relación pico-señal/ruido (dB) | > 30 dB = buena calidad |
| **Correlación** | Similitud morfológica (0–1) | > 0.99 = excelente · > 0.95 = aceptable |

**PRD** es el estándar de la norma AAMI EC57 para compresión/reconstrucción ECG.
            """)


        # Superposición LR vs SR
        with st.expander("📊 Comparativa LR vs SR superpuesta", expanded=False):
            fig_comp_sr = go.Figure()
            fig_comp_sr.add_trace(go.Scatter(
                x=t_seg, y=sig_seg, name="Original",
                line=dict(width=1, color="#888780", dash="dot"),
            ))
            fig_comp_sr.add_trace(go.Scatter(
                x=t_seg, y=sig_lr, name="LR (degradada)",
                line=dict(width=1.5, color="#D85A30"),
            ))
            fig_comp_sr.add_trace(go.Scatter(
                x=t_seg, y=sig_sr, name=f"SR ({modelo_sel})",
                line=dict(width=1.5, color="#378ADD"),
            ))
            fig_comp_sr.update_layout(
                height=320,
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis_title="Tiempo (s)",
                yaxis_title="Amplitud (mV)",
                legend=dict(orientation="h", y=1.08),
                margin=dict(l=50, r=20, t=40, b=40),
                hovermode="x unified",
            )
            fig_comp_sr.update_xaxes(gridcolor="#f0f0f0")
            fig_comp_sr.update_yaxes(gridcolor="#f0f0f0")
            st.plotly_chart(fig_comp_sr, use_container_width=True)

        # Guardar / Eliminar muestras
        st.divider()
        col_save_label, col_save_btn, col_del_btn, _ = st.columns([2, 1, 1, 3])
        with col_save_label:
            nombre_muestra = st.text_input(
                "Nombre de muestra",
                value=f"{record_id}_{sr_lead}_{modelo_sel.split()[0]}",
                key="sr_nombre",
                label_visibility="collapsed",
                placeholder="Nombre de muestra…",
            )
        with col_save_btn:
            if st.button("💾 Save", type="primary", use_container_width=True, key="sr_save"):
                if nombre_muestra.strip():
                    st.session_state["sr_muestras"][nombre_muestra.strip()] = {
                        "record_id":  record_id,
                        "lead":       sr_lead,
                        "modelo":     modelo_sel,
                        "prefiltro":  sr_filtro_pre,
                        "metricas_lr":  metricas_lr,
                        "metricas_sr":  metricas_sr,
                        "sig_lr":     sig_lr.tolist(),
                        "sig_sr":     sig_sr.tolist(),
                    }
                    st.session_state["sr_sel_idx"] = nombre_muestra.strip()
                    st.success(f"✅ Muestra '{nombre_muestra.strip()}' guardada.")
                    st.rerun()
                else:
                    st.warning("Ingresa un nombre para la muestra.")
        with col_del_btn:
            if st.button("🗑️ Delete", type="secondary", use_container_width=True, key="sr_del"):
                sel = st.session_state.get("sr_sel_idx")
                if sel and sel in st.session_state["sr_muestras"]:
                    del st.session_state["sr_muestras"][sel]
                    st.session_state["sr_sel_idx"] = None
                    st.success(f"Muestra '{sel}' eliminada.")
                    st.rerun()
                else:
                    st.warning("Selecciona una muestra de la lista para eliminar.")

        # Detalle de muestra seleccionada
        sel_nombre = st.session_state.get("sr_sel_idx")
        if sel_nombre and sel_nombre in st.session_state["sr_muestras"]:
            datos = st.session_state["sr_muestras"][sel_nombre]
            with st.expander(f"📌 Muestra seleccionada: {sel_nombre}", expanded=True):
                sc1, sc2, sc3 = st.columns(3)
                sc1.markdown(f"**Registro:** {datos['record_id']}")
                sc2.markdown(f"**Derivación:** {datos['lead']}")
                sc3.markdown(f"**Modelo:** {datos['modelo']}")
                sc1.markdown(f"**Prefiltro:** {datos['prefiltro']}")
                sc2.metric("SNR SR", f"{datos['metricas_sr']['SNR (dB)']:.2f} dB")
                sc3.metric("RMSE SR", f"{datos['metricas_sr']['RMSE (mV)']:.5f} mV")

    # Tab 5: Exportar
    with tab_export:
        st.markdown("#### JSON para el visualizador web")
        export_todas = st.checkbox("Incluir todas las derivaciones", value=False)

        if st.button("Generar JSON"):
            def to_list(arr):
                return np.round(arr.astype(float), 5).tolist()

            leads_export = leads_disponibles if export_todas else [lead_sel]

            leads_json = []
            for ln in leads_export:
                s = signals_dict[ln]
                s_corr, bl = aplicar_baseline(s, fs, baseline_sel, poly_order, spline_knots)
                s_filtrar  = s_corr if baseline_sel != "Ninguno" else s
                filtros_ln = aplicar_filtros(s_filtrar, fs, filtros_sel, wavelet_on)
                leads_json.append({
                    "lead":             ln,
                    "original":         to_list(s),
                    "baseline_removed": to_list(s_corr),
                    "baseline":         to_list(bl),
                    **{k: to_list(v) for k, v in filtros_ln.items()},
                })

            export = {
                "patients": [{"id": record_id, "age": 0, "sex": "?", "diag": "ECG"}],
                "signals": {
                    record_id: {"fs": fs, "leads": leads_json}
                },
            }
            json_str = json.dumps(export, separators=(",", ":"))
            st.download_button(
                "⬇️ Descargar signals.json",
                data=json_str,
                file_name="signals.json",
                mime="application/json",
            )
            st.caption(
                f"{'Todas las derivaciones' if export_todas else f'Derivación: {lead_sel}'} · "
                f"{len(json_str)/1e3:.1f} KB"
            )

    # ── Tab 7: NeuroKit2 ──────────────────────────────────────────────────
    with tab_nk:
        if not NK_OK:
            st.error(
                "neurokit2 no está instalado. "
                "Añade `neurokit2>=0.2.7` a requirements.txt y redespliega."
            )
            st.stop()

        st.markdown("### 🧠 Análisis ECG con NeuroKit2")
        st.caption(
            "Detección automática de picos R · Segmentación de ondas P/Q/R/S/T · "
            "Intervalos RR · Métricas de variabilidad de la frecuencia cardiaca (HRV)"
        )

        # ── Controles locales ──────────────────────────────────────────────
        nk_col1, nk_col2 = st.columns(2)
        with nk_col1:
            nk_lead = st.selectbox("Derivación", leads_disponibles, key="nk_lead_tab")
        with nk_col2:
            nk_dur = st.slider(
                "Duración (s)", 5, min(60, duracion), min(nk_dur_sb, duracion), key="nk_dur_tab"
            )

        # ── Señal de entrada ──────────────────────────────────────────────
        sig_nk_base = signals_dict[nk_lead]
        if nk_usar_filtrada and baseline_sel != "Ninguno":
            sig_nk_base, _ = aplicar_baseline(sig_nk_base, fs, baseline_sel, poly_order, spline_knots)

        n_nk   = min(int(nk_dur * fs), len(sig_nk_base))
        sig_nk = sig_nk_base[:n_nk].copy()
        t_nk   = np.arange(n_nk) / fs

        # ── Procesamiento ─────────────────────────────────────────────────
        with st.spinner("Procesando señal con NeuroKit2…"):
            try:
                sig_nk_clean = nk.ecg_clean(sig_nk, sampling_rate=int(fs), method="neurokit")

                _, rpeaks_info = nk.ecg_peaks(
                    sig_nk_clean, sampling_rate=int(fs), method=nk_metodo
                )
                rpeaks = rpeaks_info["ECG_R_Peaks"]

                try:
                    _, waves_info = nk.ecg_delineate(
                        sig_nk_clean, rpeaks_info,
                        sampling_rate=int(fs), method="dwt",
                    )
                    waves_ok = True
                except Exception:
                    waves_ok = False

                hrv_ok   = False
                hrv_time = {}
                if len(rpeaks) >= 4:
                    try:
                        hrv_df   = nk.hrv_time(rpeaks_info, sampling_rate=int(fs), show=False)
                        hrv_time = hrv_df.to_dict(orient="records")[0]
                        hrv_ok   = True
                    except Exception:
                        pass

                nk_error = None
            except Exception as e:
                nk_error = str(e)

        if nk_error:
            st.error(f"Error en NeuroKit2: {nk_error}")
            st.stop()

        # ── Métricas clave ────────────────────────────────────────────────
        rr_intervals = np.diff(rpeaks) / fs * 1000  # ms
        fc_bpm = len(rpeaks) / nk_dur * 60 if nk_dur > 0 else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Picos R detectados", f"{len(rpeaks)}")
        m2.metric("FC estimada (bpm)",  f"{fc_bpm:.1f}")
        m3.metric("RR medio (ms)",      f"{rr_intervals.mean():.1f}" if len(rr_intervals) else "—")
        m4.metric("RR std (ms)",        f"{rr_intervals.std():.1f}"  if len(rr_intervals) else "—")
        m5.metric("Método detector",    nk_metodo)

        st.divider()

        # ── Gráfica principal: ECG + picos R + ondas ──────────────────────
        fig_nk = go.Figure()

        fig_nk.add_trace(go.Scatter(
            x=t_nk, y=sig_nk_clean,
            mode="lines", name="ECG limpio",
            line=dict(color="#378ADD", width=1.2),
        ))
        fig_nk.add_trace(go.Scatter(
            x=t_nk[rpeaks], y=sig_nk_clean[rpeaks],
            mode="markers", name="Picos R",
            marker=dict(color="#E63946", size=8, symbol="triangle-up"),
        ))

        if waves_ok:
            wave_cfg = {
                "ECG_P_Peaks": ("Onda P", "#2ECC71", "circle",       6),
                "ECG_Q_Peaks": ("Onda Q", "#F39C12", "triangle-down",6),
                "ECG_S_Peaks": ("Onda S", "#9B59B6", "triangle-down",6),
                "ECG_T_Peaks": ("Onda T", "#1ABC9C", "diamond",      7),
            }
            for key, (label, color, symbol, sz) in wave_cfg.items():
                if key in waves_info:
                    idxs = np.array(waves_info[key])
                    idxs = idxs[~np.isnan(idxs)].astype(int)
                    idxs = idxs[(idxs >= 0) & (idxs < len(sig_nk_clean))]
                    if len(idxs):
                        fig_nk.add_trace(go.Scatter(
                            x=t_nk[idxs], y=sig_nk_clean[idxs],
                            mode="markers", name=label,
                            marker=dict(color=color, size=sz, symbol=symbol),
                        ))

        fig_nk.update_layout(
            height=360,
            plot_bgcolor="white",
            paper_bgcolor="white",
            title=dict(text=f"ECG procesado · {nk_lead} · detector: {nk_metodo}", font=dict(size=13)),
            xaxis=dict(title="Tiempo (s)", gridcolor="#f0f0f0"),
            yaxis=dict(title="Amplitud (mV)", gridcolor="#f0f0f0"),
            legend=dict(orientation="h", y=1.08, font=dict(size=11)),
            margin=dict(l=50, r=20, t=60, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig_nk, use_container_width=True, key="nk_fig_main")

        # ── Tacograma (intervalos RR) ─────────────────────────────────────
        if len(rr_intervals) >= 3:
            with st.expander("📊 Tacograma — intervalos RR", expanded=True):
                fig_rr = go.Figure()
                fig_rr.add_trace(go.Scatter(
                    x=np.arange(1, len(rr_intervals) + 1),
                    y=rr_intervals,
                    mode="lines+markers",
                    line=dict(color="#378ADD", width=1.5),
                    marker=dict(size=5),
                    name="RR (ms)",
                ))
                fig_rr.add_hline(
                    y=rr_intervals.mean(), line_dash="dash",
                    line_color="#E63946",
                    annotation_text=f"Media: {rr_intervals.mean():.1f} ms",
                )
                fig_rr.update_layout(
                    height=230,
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Latido #", gridcolor="#f0f0f0"),
                    yaxis=dict(title="RR (ms)",  gridcolor="#f0f0f0"),
                    showlegend=False,
                    margin=dict(l=50, r=20, t=20, b=40),
                )
                st.plotly_chart(fig_rr, use_container_width=True, key="nk_fig_rr")

        # ── HRV dominio temporal ──────────────────────────────────────────
        if hrv_ok and hrv_time:
            st.divider()
            st.markdown("#### 💓 Métricas HRV (dominio temporal)")

            hrv_campos = {
                "HRV_MeanNN": ("RR medio (ms)",  "ms", "Duración media de intervalos RR"),
                "HRV_SDNN":   ("SDNN (ms)",       "ms", "Desviación estándar de NN — variabilidad global"),
                "HRV_RMSSD":  ("RMSSD (ms)",      "ms", "Raíz cuadrática de diferencias sucesivas — tono vagal"),
                "HRV_pNN50":  ("pNN50 (%)",        "%",  "% intervalos NN con diferencia >50 ms"),
                "HRV_SDSD":   ("SDSD (ms)",        "ms", "Std de diferencias sucesivas NN"),
                "HRV_MadNN":  ("MadNN (ms)",       "ms", "Desviación absoluta mediana de NN"),
            }

            hrv_cols = st.columns(3)
            for i, (k, (label, unit, tip)) in enumerate(hrv_campos.items()):
                if k in hrv_time and hrv_time[k] is not None:
                    try:
                        hrv_cols[i % 3].metric(label, f"{float(hrv_time[k]):.2f} {unit}", help=tip)
                    except Exception:
                        pass

            with st.expander("📋 Tabla HRV completa", expanded=False):
                df_hrv = pd.DataFrame([
                    {"Métrica": k, "Valor": round(float(v), 4)}
                    for k, v in hrv_time.items()
                    if v is not None
                ])
                st.dataframe(df_hrv, use_container_width=True, hide_index=True)

        # ── Latido promedio (template) ────────────────────────────────────
        if len(rpeaks) >= 5:
            with st.expander("💗 Latido promedio (template)", expanded=False):
                try:
                    pre_ms, post_ms = 300, 500
                    pre  = int(pre_ms  / 1000 * fs)
                    post = int(post_ms / 1000 * fs)
                    beats = [
                        sig_nk_clean[r - pre: r + post]
                        for r in rpeaks
                        if r - pre >= 0 and r + post < len(sig_nk_clean)
                    ]
                    if beats:
                        t_beat   = np.linspace(-pre_ms, post_ms, pre + post) / 1000
                        beat_arr = np.array(beats)
                        mean_b   = beat_arr.mean(axis=0)
                        std_b    = beat_arr.std(axis=0)

                        fig_beat = go.Figure()
                        fig_beat.add_trace(go.Scatter(
                            x=np.concatenate([t_beat, t_beat[::-1]]),
                            y=np.concatenate([mean_b + std_b, (mean_b - std_b)[::-1]]),
                            fill="toself",
                            fillcolor="rgba(55,138,221,0.15)",
                            line=dict(color="rgba(255,255,255,0)"),
                            showlegend=False,
                            name="±1 SD",
                        ))
                        fig_beat.add_trace(go.Scatter(
                            x=t_beat, y=mean_b,
                            mode="lines",
                            name=f"Media ({len(beats)} latidos)",
                            line=dict(color="#378ADD", width=2),
                        ))
                        fig_beat.add_vline(
                            x=0, line_dash="dash", line_color="#E63946",
                            annotation_text="R",
                        )
                        fig_beat.update_layout(
                            height=260,
                            plot_bgcolor="white", paper_bgcolor="white",
                            xaxis=dict(title="Tiempo rel. al pico R (s)", gridcolor="#f0f0f0"),
                            yaxis=dict(title="Amplitud (mV)", gridcolor="#f0f0f0"),
                            margin=dict(l=50, r=20, t=20, b=40),
                            showlegend=True,
                        )
                        st.plotly_chart(fig_beat, use_container_width=True, key="nk_fig_beat")
                        st.caption(
                            f"Banda sombreada = ±1 desviación estándar · "
                            f"basado en {len(beats)} latidos."
                        )
                except Exception as e:
                    st.warning(f"No se pudo calcular el latido promedio: {e}")

# Estado vacío
else:
    st.info(
        "👈 **Par individual:** sube un `.hea` y su `.dat` en el panel lateral.  \n"
        "👈 **Múltiples archivos:** selecciona todos los `.hea` y `.dat` a la vez.  \n"
        "👈 **Base de datos completa:** comprime tu carpeta MIMIC en un `.zip` y súbela."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        with st.expander("¿Cómo obtener archivos MIMIC?"):
            st.markdown("""
1. Regístrate en [PhysioNet](https://physionet.org) y completa el curso de ética CITI.
2. Accede a **MIMIC-IV-ECG** o **MIMIC-III Waveform Database**.
3. Descarga pares `.hea` + `.dat` de cualquier registro.
4. Sube ambos archivos aquí.

Descarga directa con `wfdb`:
```python
import wfdb
wfdb.dl_database(
    'mimic3wdb/matched/p00/p000020',
    './datos',
    records=['3141595_0010']
)
                        """)

    with col_b:
         with st.expander("¿Qué hace cada módulo?"):
            st.markdown("""

Pestaña	Qué muestra
📈 Análisis de filtros	Gráficas Plotly apiladas: original, baseline corregida y cada filtro
🔀 Comparativa	Todas las señales superpuestas con toggles para activar/desactivar cada capa
🩺 Vista clínica	Cuadrícula milimetrada estándar vía ecg-plot (exportable en PNG)
📊 Estadísticas	RMS, amplitud, reducción de baseline por filtro
💾 Exportar	JSON compatible con el visualizador web original
🔬 Super Resolución ECG	Mejora la señal usando modelo CECGSR (real) o métodos clásicos
🧠 NeuroKit2	Detección R, ondas P/Q/S/T, tacograma, HRV y latido promedio
Carga de carpeta:

Selecciona múltiples archivos .hea + .dat a la vez

La app detecta automáticamente los pares por nombre

Usa el selector de paciente en el sidebar para cambiar de registro

Flujo recomendado:

Ajusta baseline removal hasta que la línea base se aplane

Aplica los filtros deseados sobre la señal ya corregida

Compara visualmente en la pestaña Comparativa con los toggles

Verifica la morfología en la vista clínica

Exporta el JSON o el PNG según necesites

Prueba la Super Resolución con tu modelo CECGSR
""")