<<<<<<< HEAD
# 🫀 ECG MIMIC Visualizador Interactivo

App web para visualizar y filtrar señales ECG del dataset MIMIC, desplegada en Streamlit Cloud.

## ✨ Funciones

| Categoría | Opciones |
|---|---|
| **Filtros de señal** | Chebyshev II HP · Butterworth BP · FIR Kaiser BP · Mediana · Wavelet |
| **Baseline Removal** | Morfológico (apertura) · Polinomial · Spline cúbico |
| **Visualización** | Interactiva con Plotly, múltiples derivaciones |
| **Exportación** | JSON compatible con el visualizador web original |

## 🚀 Despliegue en Streamlit Cloud (paso a paso)

### 1. Sube el código a GitHub

```bash
# Crea un repositorio nuevo en github.com, luego:
git init
git add .
git commit -m "ECG MIMIC Visualizer"
git remote add origin https://github.com/TU_USUARIO/ecg-visualizer.git
git push -u origin main
```

### 2. Despliega en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con tu cuenta de GitHub
3. Haz clic en **"New app"**
4. Selecciona tu repositorio y asegúrate de que el **Main file path** sea `app.py`
5. Haz clic en **"Deploy!"**

En 2–3 minutos tendrás una URL pública del tipo:
```
https://tu-usuario-ecg-visualizer-app-xxxx.streamlit.app
```

## 📁 Estructura del proyecto

```
ecg-visualizer/
├── app.py              # App principal Streamlit
├── filters.py          # Módulo de filtros + baseline removal
├── requirements.txt    # Dependencias Python
└── README.md
```

## 🧪 Uso local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📥 Datos MIMIC

1. Regístrate en [PhysioNet](https://physionet.org) y completa el curso de ética CITI
2. Accede a **MIMIC-IV-ECG** o **MIMIC-III Waveform Database**
3. Descarga pares `.hea` + `.dat`

O directamente con Python:
```python
import wfdb
wfdb.dl_database('mimic3wdb/matched/p00/p000020', './datos', records=['3141595_0010'])
```

## 📉 Métodos de Baseline Removal

### Morfológico (apertura)
Aplica erosión + dilatación con ventana de 200 ms. Captura la tendencia lenta sin distorsionar el QRS. Es el método más robusto para señales con artefactos de movimiento.

### Polinomial
Ajusta un polinomio de grado N por mínimos cuadrados. Ideal para drift lineal o cuadrático. Órdenes 2–4 son los más comunes.

### Spline cúbico
Spline suavizado con nudos equiespaciados. Más flexible que el polinomio, captura variaciones no lineales del baseline sin seguir los complejos QRS.
=======
# ecg-visualizer
ECG MIMIC Visualizer is an interactive web application built with Streamlit for visualizing and processing ECG signals from the MIMIC database. It supports advanced filtering techniques, baseline drift removal, and multi-lead visualization using Plotly, enabling efficient exploration and analysis of biomedical signals.
>>>>>>> 80f6d6e327609dc6e9f2594c87ce53ddfef1b69c
