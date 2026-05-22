"""
ECG MIMIC — Visualizador Interactivo de Señales
================================================
Streamlit app: sube archivos .hea + .dat, aplica filtros,
baseline removal y visualiza con Plotly + ecg-plot clínico.

Mejoras v2:
  - Carga de carpeta con múltiples pacientes + selector
  - Pestaña de comparativa con superposición de filtros activables
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

try:
    import ecg_plot
    ECG_PLOT_OK = True
except ImportError:
    ECG_PLOT_OK = False

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
    "Baseline removal · Vista clínica estándar (ecg-plot)"
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
    """
    Recibe lista de UploadedFile (.hea y .dat mezclados).
    Retorna dict { stem: (hea_bytes, dat_bytes, hea_name) }.
    """
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
    """
    Descomprime un ZIP con estructura de carpetas MIMIC y extrae todos los
    pares .hea + .dat encontrados.
    Retorna (dict { clave_relativa: (hea_bytes, dat_bytes, stem) }, resumen_str)
    """
    pacientes = {}
    sin_par   = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        nombres = zf.namelist()

        # Indexar por (directorio_dentro_zip, stem) → {ext: nombre_completo}
        indice = {}
        for nombre in nombres:
            p   = Path(nombre)
            ext = p.suffix.lower()
            if ext not in (".hea", ".dat"):
                continue
            clave = str(p.parent / p.stem)   # ruta relativa sin extensión
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

    # Agrupar por subcarpeta de primer nivel para resumen
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

    # ── Modo de carga ──────────────────────────────────────────────────────
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
        st.caption(
            "Selecciona todos los `.hea` y `.dat` de tus pacientes a la vez."
        )
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

    # ── Preprocesamiento ───────────────────────────────────────────────────
    st.subheader("📊 Señal")
    fs_out   = st.slider("Frecuencia de muestreo de salida (Hz)", 100, 500, 250, step=50)
    duracion = st.slider("Duración a mostrar (s)", 1, 30, 10)

    st.divider()

    # ── Baseline removal ───────────────────────────────────────────────────
    st.subheader("📉 Baseline Removal")
    baseline_sel = st.selectbox(
        "Método",
        ["Ninguno", "Morfológico (apertura)", "Polinomial", "Spline cúbico"],
    )
    poly_order   = st.slider("Orden del polinomio", 1, 10, 3) if baseline_sel == "Polinomial" else 3
    spline_knots = st.slider("Nudos (knots)", 5, 50, 15)      if baseline_sel == "Spline cúbico" else 15
    show_baseline_overlay = st.checkbox("Mostrar línea base en gráfica", value=True)

    st.divider()

    # ── Filtros ────────────────────────────────────────────────────────────
    st.subheader("🔬 Filtros de señal")
    filtros_sel = st.multiselect(
        "Filtros a aplicar (sobre señal corregida)",
        ["Chebyshev II HP", "Butterworth BP", "FIR Kaiser BP", "Mediana"],
        default=["Butterworth BP"],
    )
    wavelet_on = st.checkbox("Wavelet (umbralización suave)", value=False)

    st.divider()

    # ── Vista clínica ──────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE PROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def aplicar_baseline(sig, fs, metodo, poly_order, spline_knots):
    """Retorna (señal_corregida, baseline_estimada)."""
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
    """Retorna dict {nombre_filtro: señal_filtrada}."""
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
    """Figura Plotly apilada: original → corregida → filtros."""
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

    # Fila 1 — Original (+ baseline overlay)
    fig.add_trace(go.Scatter(
        x=tiempo, y=sig_original, name="Original",
        line=dict(width=1, color="#888780")), row=1, col=1)

    if baseline_sel != "Ninguno" and show_baseline_overlay:
        fig.add_trace(go.Scatter(
            x=tiempo, y=baseline, name="Línea base",
            line=dict(width=1.5, color="#E24B4A", dash="dash")), row=1, col=1)

    # Fila 2 — Señal corregida
    row = 2
    if baseline_sel != "Ninguno":
        fig.add_trace(go.Scatter(
            x=tiempo, y=sig_corregida,
            name=f"Corregida ({baseline_sel})",
            line=dict(width=1, color="#1D9E75")), row=row, col=1)
        row += 1

    # Filas siguientes — Filtros
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
    """
    Figura Plotly con TODAS las señales superpuestas en un solo eje.
    capas_activas: dict {nombre_capa: bool}
    """
    fig = go.Figure()

    colores = {
        "Original":  "#888780",
        "Corregida": "#1D9E75",
    }
    for i, nombre in enumerate(señales_filtradas):
        colores[nombre] = PALETTE[i % len(PALETTE)]

    # Original
    if capas_activas.get("Original", True):
        fig.add_trace(go.Scatter(
            x=tiempo, y=sig_original,
            name="Original",
            line=dict(width=1.5, color=colores["Original"]),
            opacity=0.85,
        ))

    # Corregida
    if baseline_sel != "Ninguno" and capas_activas.get("Corregida", True):
        fig.add_trace(go.Scatter(
            x=tiempo, y=sig_corregida,
            name=f"Corregida ({baseline_sel})",
            line=dict(width=1.5, color=colores["Corregida"]),
            opacity=0.85,
        ))

    # Filtros
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
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

# ══════════════════════════════════════════════════════════════════════════════
# RESOLUCIÓN DE ARCHIVOS CARGADOS
# ══════════════════════════════════════════════════════════════════════════════

pacientes_disponibles = {}   # { clave: (hea_bytes, dat_bytes, stem_name) }
record_id    = None
fs           = None
signals_dict = None
paciente_sel = None

# ── Modo 1: Par individual ─────────────────────────────────────────────────
if modo_carga == "Par individual (.hea + .dat)":
    if hea_file is not None and dat_file is not None:
        pacientes_disponibles["__single__"] = (
            hea_file.read(), dat_file.read(), hea_file.name
        )
        paciente_sel = "__single__"

# ── Modo 2: Múltiples archivos ─────────────────────────────────────────────
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
        else:
            st.warning(
                "No se encontraron pares `.hea` + `.dat` completos. "
                "Asegúrate de subir ambos archivos para cada paciente."
            )

# ── Modo 3: ZIP con la DB completa ────────────────────────────────────────
else:
    if zip_file is not None:
        with st.spinner("🗜️ Descomprimiendo y escaneando la base de datos…"):
            zip_bytes = zip_file.read()
            pacientes_disponibles, resumen, carpetas_resumen = cargar_db_desde_zip(zip_bytes)

        if not pacientes_disponibles:
            st.error("No se encontraron pares `.hea` + `.dat` dentro del ZIP.")
        else:
            # ── Resumen de la DB en el área principal ──────────────────
            with st.expander(f"📊 Base de datos cargada — {resumen}", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("Total de registros", len(pacientes_disponibles))
                c2.metric("Subcarpetas", len(carpetas_resumen))
                if carpetas_resumen:
                    st.markdown("**Registros por subcarpeta:**")
                    cols_res = st.columns(min(len(carpetas_resumen), 4))
                    for i, (carpeta, count) in enumerate(sorted(carpetas_resumen.items())):
                        cols_res[i % len(cols_res)].metric(f"📁 {carpeta}", f"{count} reg.")

            # ── Selector de paciente en sidebar ───────────────────────
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

    # ── Baseline removal ───────────────────────────────────────────────────
    sig_corregida, baseline = aplicar_baseline(
        sig, fs, baseline_sel, poly_order, spline_knots
    )

    # ── Filtros ────────────────────────────────────────────────────────────
    sig_para_filtrar  = sig_corregida if baseline_sel != "Ninguno" else sig
    señales_filtradas = aplicar_filtros(sig_para_filtrar, fs, filtros_sel, wavelet_on)

    # ══════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════
    tab_plotly, tab_comp, tab_clinica, tab_stats, tab_export = st.tabs([
        "📈 Análisis de filtros",
        "🔀 Comparativa",
        "🩺 Vista clínica",
        "📊 Estadísticas",
        "💾 Exportar",
    ])

    # ── Tab 1: Plotly apilado ──────────────────────────────────────────────
    with tab_plotly:
        fig_plotly = graficar_plotly(
            tiempo, sig, señales_filtradas, sig_corregida, baseline,
            baseline_sel, show_baseline_overlay, lead_sel, fs,
        )
        st.plotly_chart(fig_plotly, use_container_width=True)

    # ── Tab 2: Comparativa superpuesta ────────────────────────────────────
    with tab_comp:
        st.markdown("#### 🔀 Superposición de señales")
        st.caption(
            "Activa o desactiva cada capa con los interruptores. "
            "Todas las señales se muestran en el mismo eje para comparación directa."
        )

        # Construir lista de capas disponibles
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

        # ── Controles de activación ────────────────────────────────────
        st.markdown("**Capas visibles:**")
        n_cols = min(len(capas), 4)
        cols_toggle = st.columns(n_cols)
        capas_activas = {}

        for idx, capa in enumerate(capas):
            color_hex = colores_capa.get(capa, "#999")
            with cols_toggle[idx % n_cols]:
                # Pequeño cuadrado de color como indicador visual
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

        # ── Opciones adicionales ───────────────────────────────────────
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

        # Aplicar zoom al tiempo
        mask = (tiempo >= zoom_inicio) & (tiempo <= zoom_fin)
        t_zoom  = tiempo[mask]

        def _recortar(arr):
            return arr[mask] if len(arr) == len(tiempo) else arr

        sig_orig_zoom  = _recortar(sig)
        sig_corr_zoom  = _recortar(sig_corregida)
        filtradas_zoom = {k: _recortar(v) for k, v in señales_filtradas.items()}

        # ── Figura comparativa ─────────────────────────────────────────
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

        # ── Tabla de diferencias ───────────────────────────────────────
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

    # ── Tab 3: ecg-plot ────────────────────────────────────────────────────
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

        else:  # Multi-derivación
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

    # ── Tab 4: Estadísticas ────────────────────────────────────────────────
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

    # ── Tab 5: Exportar ────────────────────────────────────────────────────
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

# ── Estado vacío ───────────────────────────────────────────────────────────────
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
```
            """)

    with col_b:
        with st.expander("¿Qué hace cada módulo?"):
            st.markdown("""
| Pestaña | Qué muestra |
|---|---|
| 📈 Análisis de filtros | Gráficas Plotly apiladas: original, baseline corregida y cada filtro |
| 🔀 Comparativa | Todas las señales superpuestas con toggles para activar/desactivar cada capa |
| 🩺 Vista clínica | Cuadrícula milimetrada estándar vía **ecg-plot** (exportable en PNG) |
| 📊 Estadísticas | RMS, amplitud, reducción de baseline por filtro |
| 💾 Exportar | JSON compatible con el visualizador web original |

**Carga de carpeta:**
- Selecciona múltiples archivos `.hea` + `.dat` a la vez
- La app detecta automáticamente los pares por nombre
- Usa el selector de paciente en el sidebar para cambiar de registro

**Flujo recomendado:**
1. Ajusta baseline removal hasta que la línea base se aplane
2. Aplica los filtros deseados sobre la señal ya corregida
3. Compara visualmente en la pestaña **Comparativa** con los toggles
4. Verifica la morfología en la vista clínica
5. Exporta el JSON o el PNG según necesites
            """)