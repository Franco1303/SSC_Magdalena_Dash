import pathlib
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import base64, io as _io
try:
    import rasterio
    import rasterio.mask
    from rasterio.features import geometry_mask
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
import io 
from dash import ctx
from dash import ALL
import re
import glob
import os
from glob import glob  # ← reemplaza:  import glob

# ─────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────
#df = pd.read_csv("puntos_finales2.csv")
#df = pd.read_csv("puntos_unificados.csv", sep=";")
df = pd.read_csv('puntos_alternativos.csv')
df["reflectance_date"] = pd.to_datetime(df["reflectance_date"])
df["scc_date"]         = pd.to_datetime(df["scc_date"])
df["km_label"]         = "Km " + df["km"].astype(str)

BANDAS  = ["aerosol", "blue", "green", "red", "rojo 1", "rojo 2",
           "rojo 3", "NIR", "rojo 4", "SWIR1", "SWIR2"]
INDICES = ["RANS", "VNES", "NDTI", "NIR/RED"]
SSCS =["SSC", "SSC2", "SSC4"]
KMS_ALL = sorted(df["km"].unique().tolist())

KM_COLORS = {
    0:  "#4db6ff",   # azul claro
    1:  "#9be564",   # verde lima suave
    3:  "#36d399",   # turquesa
    5:  "#f38ba8",   # rosado coral
    7:  "#74c7ec",   # celeste suave
    11: "#f6c177",   # arena/naranja suave
    14: "#bb86fc",   # violeta claro
    17: "#ff9e64",   # naranja
    18: "#ff6b6b",   # rojo coral
    19: "#c792ea",   # púrpura suave
}

MODEL_DEFS = {
    "lineal": ("Regresión Lineal", None),
}

#------------------------------------------
#Caudal y TSS de calamar y campo 
#-------------------------------------------

Q= pd.read_csv(r"Q_MEDIA_D@29037020.data", sep="|" )
QG= pd.read_excel(r"caudal_ganara.xlsx")
tss= pd.read_csv(r"TR_KT_D_QS_D@29037020.data", sep="|" )

Q['Fecha']= pd.to_datetime(Q['Fecha'],format="%Y-%m-%d %H:%M:%S")
QG['Fecha']= pd.to_datetime(QG['Fecha'])
tss['Fecha']= pd.to_datetime(tss['Fecha'],format="%Y-%m-%d %H:%M:%S")

merged_data= pd.merge(Q, tss, on='Fecha', how='inner')

# ─────────────────────────────────────────
# CARGA DE PERFILES DE CAMPO
# ─────────────────────────────────────────
PROFILES_BASE = pathlib.Path("DATOS_FRANCISCO") 

def load_all_profiles():
    records = []
    if not PROFILES_BASE.exists():
        return pd.DataFrame(columns=["km", "+m", "depth", "ssc", "fecha"])
    for month_folder in sorted(PROFILES_BASE.iterdir()):
        if not month_folder.is_dir():
            continue
        for csv_file in sorted(month_folder.iterdir()):
            if csv_file.suffix.lower() != ".csv":
                continue
            try:
                try:
                    df_raw = pd.read_csv(csv_file, sep=";", encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df_raw = pd.read_csv(csv_file, sep=";", encoding="latin-1")
                df_raw.columns = ["km", "+m", "depth", "ssc"]
                fecha = datetime.strptime(csv_file.stem, "%d%m%Y")
                df_raw["fecha"] = fecha
                records.append(df_raw)
            except Exception:
                continue
    if not records:
        return pd.DataFrame(columns=["km", "+m", "depth", "ssc", "fecha"])
    return pd.concat(records, ignore_index=True)

df_profiles = load_all_profiles()



# ─────────────────────────────────────────
# DATOS HIDROLÓGICOS — CALAMAR Y BARRANQUILLA
# ─────────────────────────────────────────
def load_hydro():
    try:
        Q_cal = pd.read_csv("Q_MEDIA_D@29037020.data", sep="|")
        Q_cal.columns = ["Fecha", "Q_calamar"]
        Q_cal["Fecha"] = pd.to_datetime(Q_cal["Fecha"])
    except Exception:
        Q_cal = pd.DataFrame(columns=["Fecha", "Q_calamar"])
    try:
        TSS_cal = pd.read_csv("TR_KT_D_QS_D@29037020.data", sep="|")
        TSS_cal.columns = ["Fecha", "TSS_calamar"]
        TSS_cal["Fecha"] = pd.to_datetime(TSS_cal["Fecha"])
    except Exception:
        TSS_cal = pd.DataFrame(columns=["Fecha", "TSS_calamar"])
    try:
        Q_baq = pd.read_excel("caudal_ganara.xlsx")
        Q_baq.columns = ["Fecha", "Q_barranquilla"]
        Q_baq["Fecha"] = pd.to_datetime(Q_baq["Fecha"])
    except Exception:
        Q_baq = pd.DataFrame(columns=["Fecha", "Q_barranquilla"])
    merged = Q_cal.merge(TSS_cal, on="Fecha", how="outer")
    merged = merged.merge(Q_baq, on="Fecha", how="outer")
    merged = merged.sort_values("Fecha").reset_index(drop=True)
    km19_ssc = df[df["km"] == 19][["scc_date","SSC"]].rename(columns={"scc_date":"Fecha"})
    if not km19_ssc.empty and not Q_baq.empty:
        tss_baq = km19_ssc.merge(Q_baq, on="Fecha", how="inner")
        tss_baq["TSS_barranquilla"] = tss_baq["SSC"] * tss_baq["Q_barranquilla"] * 0.0864 / 1000
    else:
        tss_baq = pd.DataFrame(columns=["Fecha","SSC","Q_barranquilla","TSS_barranquilla"])
    return Q_cal, TSS_cal, Q_baq, merged, tss_baq

Q_cal, TSS_cal, Q_baq, df_hydro, df_tss_baq = load_hydro()
HYDRO_YEAR_MIN = int(df_hydro["Fecha"].dt.year.min()) if not df_hydro.empty else 1972
HYDRO_YEAR_MAX = int(df_hydro["Fecha"].dt.year.max()) if not df_hydro.empty else 2026

# ─────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────
FONT_SANS  = "'Sora', 'DM Sans', sans-serif"
FONT_MONO  = "'JetBrains Mono', 'Fira Code', monospace"
FONT_SERIF = "'Playfair Display', Georgia, serif"

C_BG       = "#F8F9FB"
C_WHITE    = "#FFFFFF"
C_BORDER   = "#E5E9F0"
C_BORDER2  = "#D0D7E3"

C_TEXT     = "#1A2033"
C_MUTED    = "#6B7694"
C_LABEL    = "#3D4F7C"

C_ACCENT   = "#2563EB"   # blue-600
C_ACCENT_L = "#EFF6FF"   # blue-50
C_ACCENT_D = "#1D4ED8"

C_GREEN    = "#16A34A"
C_GREEN_L  = "#F0FDF4"
C_AMBER    = "#D97706"
C_RED      = "#DC2626"
C_RED_L    = "#FEF2F2"
C_PURPLE   = "#7C3AED"

SHADOW_SM  = "0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)"
SHADOW_MD  = "0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -1px rgba(0,0,0,0.04)"
SHADOW_LG  = "0 10px 15px -3px rgba(0,0,0,0.07), 0 4px 6px -2px rgba(0,0,0,0.03)"

# ─────────────────────────────────────────
# FUENTES BASE
# ─────────────────────────────────────────
FONT_BODY  = FONT_SANS
FONT_TITLE = FONT_SERIF

# ─────────────────────────────────────────
# ALIASES DE COLORES
# ─────────────────────────────────────────
COLOR_BG      = C_BG

COLOR_CARD    = C_WHITE
COLOR_CARD2   = "#FCFDFF"

COLOR_BORDER  = C_BORDER
COLOR_BORDER2 = C_BORDER2

COLOR_TEXT    = C_TEXT
COLOR_MUTED   = C_MUTED
COLOR_LABEL   = C_LABEL

COLOR_ACCENT  = C_ACCENT
COLOR_ACCENT2 = C_ACCENT_D
COLOR_ACCENT_L = C_ACCENT_L

COLOR_GREEN   = C_GREEN
COLOR_RED     = C_RED
COLOR_AMBER   = C_AMBER
COLOR_PURPLE  = C_PURPLE
COLOR_INPUT_BG = "#FFFFFF"
COLOR_INPUT_BORDER = C_BORDER
COLOR_INPUT_TEXT = C_TEXT
COLOR_INPUT_PLACEHOLDER = C_MUTED


COLOR_INPUT_BG = "#FFFFFF"
COLOR_INPUT_BORDER = C_BORDER
COLOR_INPUT_TEXT = C_TEXT
COLOR_INPUT_PLACEHOLDER = C_MUTED

COLOR_TEXT_INPUT = COLOR_TEXT
COLOR_INPUT_BD   = COLOR_BORDER
INTRO_YOUTUBE_EMBED_SRC = "https://www.youtube.com/embed/uzEXjveDNGw"

CARD = {
    "backgroundColor": COLOR_CARD,
    "borderRadius": "12px",
    "padding": "28px 32px",
    "marginBottom": "24px",
    "boxShadow": SHADOW_MD,
    "border": f"1px solid {COLOR_BORDER}",
}

CARD2 = {
    "backgroundColor": COLOR_CARD2,
    "borderRadius": "12px",
    "padding": "20px 24px",
    "marginBottom": "16px",
    "boxShadow": SHADOW_SM,
    "border": f"1px solid {COLOR_BORDER}",
}

TAB_STYLE = {
    "fontFamily": FONT_BODY, "fontSize": "13px", "fontWeight": "500",
    "color": COLOR_MUTED, "backgroundColor": COLOR_BG, "border": "none",
    "borderBottom": f"2px solid {COLOR_BORDER}", "padding": "11px 20px",
    "letterSpacing": "0.04em", "textTransform": "uppercase",
}
TAB_SELECTED = {
    **TAB_STYLE, "color": COLOR_ACCENT2,
    "borderBottom": f"3px solid {COLOR_ACCENT2}",
    "backgroundColor": COLOR_CARD,
    "fontWeight": "700",
}

def section_title(text, subtitle=None):
    els = [html.H3(text, style={"fontFamily": FONT_TITLE, "fontSize": "20px",
                                 "color": COLOR_TEXT, "marginBottom": "6px", "fontWeight": "700"})]
    if subtitle:
        els.append(html.P(subtitle, style={"fontFamily": FONT_BODY, "color": COLOR_MUTED,
                                            "fontSize": "14px", "marginTop": "0", "marginBottom": "18px"}))
    return html.Div(els)

def stat_card(label, value, unit=""):
    return html.Div([
        html.P(label, style={"fontFamily": FONT_BODY, "fontSize": "12px", "color": COLOR_MUTED,
                              "margin": "0 0 4px 0", "textTransform": "uppercase", "letterSpacing": "0.06em"}),
        html.Div([
            html.Span(value, style={"fontFamily": FONT_TITLE, "fontSize": "28px",
                                     "fontWeight": "700", "color": COLOR_ACCENT}),
            html.Span(f" {unit}", style={"fontSize": "13px", "color": COLOR_MUTED}),
        ])
    ], style={**CARD, "padding": "20px 24px", "textAlign": "center", "marginBottom": "0"})
    
    
    
# ── Configuración de carpeta ─────────────────────────────────────────────────
CARPETA_IMAGENES = "fotografias_sentinel_2"   # ← cambia si tu carpeta tiene otro nombre
BANDA_NIR        = 8                        # banda NIR por defecto (Sentinel-2 B8)
FACTOR_ESCALA    = 10000                    # DN → reflectancia
 
 
# ── Helper: leer TIFs de la carpeta y extraer fechas ────────────────────────
def listar_imagenes():
    """
    Devuelve lista de dicts ordenados por fecha:
      [{"path": "...", "fecha": datetime, "label": "27 Ene 2018"}, ...]
    El nombre esperado tiene la fecha en la 2ª posición separada por '_':
      S2_2018-01-27_20180127T153609_...tif
    """
    patron_fecha = re.compile(r"(\d{4}-\d{2}-\d{2})")
    archivos = sorted(
        glob(os.path.join(CARPETA_IMAGENES, "*.tif")) +
        glob(os.path.join(CARPETA_IMAGENES, "*.tiff"))
    )
    resultado = []
    for ruta in archivos:
        nombre = os.path.basename(ruta)
        m = patron_fecha.search(nombre)
        if m:
            fecha = datetime.strptime(m.group(1), "%Y-%m-%d")
            label = fecha.strftime("%-d %b %Y")   # ej. "27 Ene 2018"
        else:
            fecha = datetime.min
            label = nombre  # fallback: mostrar nombre completo
        resultado.append({"path": ruta, "fecha": fecha, "label": label})
    return sorted(resultado, key=lambda x: x["fecha"])

# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Merriweather:wght@700&family=JetBrains+Mono:wght@400;500&display=swap"
    ],
    suppress_callback_exceptions=True,
)
app.title = "CSS Magdalena — EDA"

# ── Component CSS overrides ──
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Light dropdown menu */
            .Select-control {
                background-color: #ffffff !important;
                border-color: #d0d7e3 !important;
                color: #1a2033 !important;
                box-shadow: none !important;
            }
            .Select-value-label, .Select-placeholder, .Select--single > .Select-control .Select-value {
                color: #1a2033 !important;
            }
            .Select-input input {
                color: #1a2033 !important;
            }
            .Select-menu-outer {
                background-color: #ffffff !important;
                border-color: #d0d7e3 !important;
                color: #1a2033 !important;
                box-shadow: 0 8px 18px rgba(26,32,51,0.12) !important;
                z-index: 9999 !important;
            }
            .Select-option {
                background-color: #ffffff !important;
                color: #1a2033 !important;
            }
            .Select-option:hover, .Select-option.is-focused {
                background-color: #eff6ff !important;
                color: #1d4ed8 !important;
            }
            .Select-option.is-selected {
                background-color: #dbeafe !important;
                color: #1d4ed8 !important;
            }
            .Select-arrow-zone .Select-arrow {
                border-top-color: #6b7694 !important;
            }
            .Select-clear-zone {
                color: #6b7694 !important;
            }
            .Select--multi .Select-value {
                background-color: #eff6ff !important;
                border-color: #bfdbfe !important;
                color: #1d4ed8 !important;
            }
            .Select--multi .Select-value-icon {
                border-right-color: #bfdbfe !important;
                color: #1d4ed8 !important;
            }
            .Select--multi .Select-value-icon:hover {
                background-color: #dc2626 !important;
                color: #fff !important;
            }
            .VirtualizedSelectOption {
                background-color: #ffffff !important;
                color: #1a2033 !important;
            }
            .VirtualizedSelectFocusedOption {
                background-color: #eff6ff !important;
                color: #1d4ed8 !important;
            }
            /* Tooltip slider */
            .rc-slider-tooltip-inner {
                background-color: #ffffff !important;
                color: #1d4ed8 !important;
                border: 1px solid #d0d7e3 !important;
                box-shadow: 0 4px 12px rgba(26,32,51,0.12) !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

server = app.server

app.layout = html.Div(style={"backgroundColor": COLOR_BG, "minHeight": "100vh",
                               "fontFamily": FONT_BODY, "color": COLOR_TEXT}, children=[

    # Store con el dataset filtrado
    dcc.Store(id="store-df-filtered"),

    # ── HEADER ──
    html.Div(style={
        "backgroundColor": COLOR_CARD,
        "borderBottom": f"1px solid {COLOR_BORDER}",
        "padding": "0 48px", "display": "flex", "alignItems": "center",
        "gap": "16px", "height": "64px",
        "boxShadow": SHADOW_SM,
        "backdropFilter": "blur(8px)",
    }, children=[
        html.Div([
        html.Img(src="/assets/satelite.png", 
                style={"width": "44px", "height": "44px"})
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span("CSS Magdalena", style={"fontFamily": FONT_TITLE, "fontSize": "17px",
                                               "color": COLOR_TEXT, "fontWeight": "700"}),
            html.Span(" — Análisis Exploratorio de Datos",
                      style={"fontFamily": FONT_BODY, "fontSize": "14px", "color": COLOR_MUTED}),
        ]),
        html.Div("Universidad del Norte · 2025–2026",
                 style={"marginLeft": "auto", "fontSize": "12px",
                        "color": COLOR_MUTED, "letterSpacing": "0.04em"}),
    ]),

    # ── TABS ──
    html.Div(style={"padding": "0 48px"}, children=[
        dcc.Tabs(id="tabs", value="intro", style={"borderBottom": f"1px solid {COLOR_BORDER}"},
                 children=[
            dcc.Tab(label="Introducción",  value="intro",        style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Contexto",      value="contexto",     style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Problema",      value="problema",     style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Objetivo",      value="objetivo",     style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Marco Teórico", value="marco",        style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="EDA",           value="eda",          style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Modelo",        value="modelo",       style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Aplicación",    value="aplicacion",   style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Conclusiones",  value="conclusiones", style=TAB_STYLE, selected_style=TAB_SELECTED),
        ]),
        html.Div(id="tab-content", style={"padding": "36px 0 60px 0"}),
    ]),
])

# ═══════════════════════════════════════════
# PESTAÑAS 
# ═══════════════════════════════════════════

def tab_intro():
    return html.Div([
        html.Div(style={**CARD,
                        "background": f"linear-gradient(135deg, {COLOR_ACCENT}10 0%, {COLOR_CARD} 60%)",
                        "borderLeft": f"4px solid {COLOR_ACCENT}", "padding": "36px 40px"}, children=[
            html.H1("Estimación de Concentración de Sedimentos en Suspensión en el Río Magdalena "
                    "mediante Imágenes Satelitales Sentinel-2",
                    style={"fontFamily": FONT_TITLE, "fontSize": "26px", "color": COLOR_TEXT,
                           "lineHeight": "1.4", "marginBottom": "20px"}),
            html.P("Este Dash presenta un resumen general de mi tesis de pregrado de Geología en la Universidad del Norte,"
                   "que se enfoca en el desarrollo de un modelo empirico para estimar la concentración de sedimentos en suspensión (SSC) "
                   "en el tramo final del rio Magdalena a partir de reflectancia superficial del agua obtenida a partir de imagenes satelitales Sentinel 2 del programa Copernicus de "
                   "la Agencia Espacial Europea (ESA). Para llevar a cabo este estudio es necesario el uso de mediciones in situ de SSC que deben ser unidas con la reflectancia "
                   "reportada por imagenes satelitales contemporaneas. El dataset final con el que se realizara el modelo esta compuesto entonces por aquellos puntos de las campañas de campo "
                   "para los cuales fue posible obtener reflectancia de Sentinel-2 aplicando criterios de control de calidad rigurosos. ",
                   style={"fontSize": "15px", "color": COLOR_TEXT, "lineHeight": "1.8",
                          "maxWidth": "820px", "marginBottom": "28px"}),
            html.P("Los datos de campo fueron tomados con un perfilador LISST (laser in situ scattering and transmissometer), que permite obtener perfiles verticales de SSC. Estos fueron tomados cada dos semanas en el periodo Junio 2025 - Marzo 2026 "
                   "y a continuación se presentan los puntos finales que pudieron ser unidos con reflectancia de Sentinel-2 en un intervalo de tolerancia de 1 dia de diferencia entre la medición in situ y captura de la imagen.",
                   style={"fontSize": "15px", "color": COLOR_TEXT, "lineHeight": "1.8",
                          "maxWidth": "820px", "marginBottom": "28px"}),
            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                stat_card("Observaciones", str(len(df)), "puntos"),
                stat_card("Período", "Jun 2025 – Mar 2026", ""),
                stat_card("Rango CSS", f"{int(df['SSC'].min())}–{int(df['SSC'].max())}", "mg/L"),
                stat_card("Estaciones", str(df["km"].nunique()), "km"),
            ]),
        ]),
        html.Div(style={**CARD}, children=[
            section_title("Estructura del dashboard"),
            html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "16px"},
                     children=[
                html.Div([
                    html.Div(t, style={"fontWeight": "700", "color": COLOR_ACCENT,
                                       "fontSize": "13px", "marginBottom": "4px"}),
                    html.Div(d, style={"fontSize": "13px", "color": COLOR_MUTED, "lineHeight": "1.6"}),
                ], style={**CARD, "marginBottom": "0", "padding": "16px 20px"})
                for t, d in [
                    ("Introducción", "Presentación general del proyecto y resumen estadístico del dataset."),
                    ("Contexto",     "Descripción del área de estudio y relevancia del río Magdalena."),
                    ("Problema",     "Planteamiento del problema de investigación."),
                    ("Objetivo",     "Objetivo general y específicos del estudio."),
                    ("Marco Teórico","Fundamentos de teledetección de sedimentos y Sentinel-2."),
                    ("EDA",          "Análisis exploratorio interactivo con filtro global por estación."),
                ]
            ]),
        ]),
        html.Div(style={**CARD}, children=[
            section_title("Video de presentación",
                          "Resumen grabado del flujo del dashboard y de las decisiones principales del análisis"),
            html.Div(style={
                "border": f"1px solid {COLOR_BORDER}",
                "borderRadius": "10px",
                "overflow": "hidden",
                "backgroundColor": COLOR_CARD2,
                "boxShadow": SHADOW_SM,
            }, children=[
                html.Iframe(
                    src=INTRO_YOUTUBE_EMBED_SRC,
                    title="Video de presentación",
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share",
                    style={
                        "display": "block",
                        "width": "100%",
                        "aspectRatio": "16 / 9",
                        "border": "0",
                        "backgroundColor": COLOR_BG,
                    },
                ),
            ]),
        ]),
        # ── Contacto ──
        html.Div(style={**CARD, "borderLeft": f"4px solid {COLOR_ACCENT}"}, children=[
            section_title("Autor"),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "24px",
                            "flexWrap": "wrap"}, children=[

                # Avatar iniciales
                html.Div("FM", style={
                    "width": "56px", "height": "56px", "borderRadius": "50%",
                    "background": f"{COLOR_ACCENT}20", "color": COLOR_ACCENT,
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontSize": "18px", "fontWeight": "700", "flexShrink": "0",
                }),

                # Nombre y título
                html.Div([
                    html.P("Francisco Javier Morales Carroll",
                           style={"fontSize": "16px", "fontWeight": "700",
                                  "color": COLOR_TEXT, "margin": "0 0 2px"}),
                    html.P("Estudiante de Geología · Universidad del Norte",
                           style={"fontSize": "13px", "color": COLOR_MUTED, "margin": "0"}),
                ]),

                # Links
                html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                                "marginLeft": "auto"}, children=[
                    html.A("GitHub", href="https://github.com/Franco1303", target="_blank",
                           style={"fontSize": "13px", "padding": "6px 14px",
                                  "borderRadius": "6px", "border": f"0.5px solid {COLOR_BORDER}",
                                  "color": COLOR_TEXT, "textDecoration": "none",
                                  "background": COLOR_CARD}),
                    html.A("LinkedIn", href="www.linkedin.com/in/francisco-morales-4715092a0", target="_blank",
                           style={"fontSize": "13px", "padding": "6px 14px",
                                  "borderRadius": "6px", "border": f"0.5px solid {COLOR_BORDER}",
                                  "color": COLOR_TEXT, "textDecoration": "none",
                                  "background": COLOR_CARD}),
                    html.A("Correo", href="mailto:fcarroll@uninorte.edu.co", target="_blank",
                           style={"fontSize": "13px", "padding": "6px 14px",
                                  "borderRadius": "6px", "border": f"0.5px solid {COLOR_BORDER}",
                                  "color": COLOR_TEXT, "textDecoration": "none",
                                  "background": COLOR_CARD}),
                    html.A("Universidad del Norte", href="https://www.uninorte.edu.co",
                           target="_blank",
                           style={"fontSize": "13px", "padding": "6px 14px",
                                  "borderRadius": "6px",
                                  "border": f"0.5px solid {COLOR_ACCENT}",
                                  "color": COLOR_ACCENT, "textDecoration": "none",
                                  "background": f"{COLOR_ACCENT}10"}),
                ]),
            ]),
        ]),
    ])


def tab_contexto():
    fig = go.Figure(go.Scattermapbox(
        lat=[11.10354,11.102,11.09628,11.09018,11.0755,11.0585,11.0428,
             11.0249,11.0011327,10.9919,10.9782,10.9637,10.9546],
        lon=[-74.8516,-74.8513,-74.8497,-74.8492,-74.8456,-74.8387,-74.8195,
             -74.7911,-74.7660752,-74.7611,-74.7579,-74.7569,-74.7562],
        mode="markers+lines", marker=dict(size=12, color=COLOR_ACCENT),
        text=["Km 0 +250","Km 0 +500","Km 1", "Km 1 + 900","Km 3 +500","Km 5 +500", "Km 7 +900", "Km 11 +200","Km 14 +800", "Km 17 +600", "Km 18 + 200", "Km 19 +800", "Km 19 +940"], hoverinfo="text",
    ))
    fig.update_layout(mapbox=dict(style="carto-positron", center=dict(lat=11.10, lon=-74.85), zoom=12),
                      margin=dict(l=0,r=0,t=0,b=0), height=500, paper_bgcolor=COLOR_CARD)
    
    fig2 = go.Figure(go.Scattermapbox(
        lat = [10.2422934, 	10.30],
        lon = [-74.9138168, -74.95],
        mode="markers", marker=dict(size=12, color=COLOR_ACCENT),
        text=["Calamar, IDEAM (20037020)", "INKORA K-7 (29037360)"], hoverinfo="text",
    ))
    fig2.update_layout(mapbox=dict(style="carto-positron", center=dict(lat=10.2422934, lon=-74.9138168), zoom=9),
                      margin=dict(l=0,r=0,t=0,b=0), height=500, paper_bgcolor=COLOR_CARD)
    
    return html.Div([
        html.Div(style={**CARD}, children=[
            section_title("Área de estudio", "Tramo estuarino del río Magdalena, Barranquilla, Colombia"),
            html.P("El rio magdalena es el ecosistema fluvial con la mayor área y extensión en el país," 
                    "cubriendo un área de 257,438 km2 que representa el 24%" "del territorio nacional, su cuenca " 
                    "esta cateterizada por alta actividad tectónica, altas pendientes que exceden los 45° y tiene" 
                    "como principales tributarios El rio Cauca, Sogamoso, San Jorge y Cesar (Restrepo et al.," 
                    "2006). Este representa al mayor contribuyente de sedimentos en el caribe con una descarga" 
                    "de 144 x 106 t yr-1 (Higgins et al., 2016) y de es uno de los principales influyentes en los" 
                    "cambios morfodinámicos del Caribe Colombiano siendo este considerado la principal fuente" 
                    "de sedimentos de las playas de la costa norte del caribe colombiano (Restrepo et al., 2006).",
                   style={"fontSize": "14.5px", "color": COLOR_TEXT, "lineHeight": "1.8",
                          "maxWidth": "820px", "marginBottom": "20px"}),
            html.P("A continuación se presentan las estaciones en las que se llevaron a cabo las mediciones originales de campo y nombradas por su distancia de la desembocadura.",
                   style={"fontSize": "14.5px", "color": COLOR_TEXT, "lineHeight": "1.8",
                          "maxWidth": "820px", "marginBottom": "20px"}),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ]),
        html.Div(style={**CARD}, children=[
            section_title("Estaciones de muestreo"),
            html.P("las estaciones del kilometro 5 y 7 fueron descartadas del analisis final por estar fuertemente afectadas por actividades de dragado "
                   "que introducen una mayor incertidumbre sobre su uso para entrenar el modelo. Las estaciones 0, 1 y 3 tambien introducen este tipo de ruido en menor medida.",
                   style={"fontSize": "14.5px", "color": COLOR_TEXT, "lineHeight": "1.8",
                          "maxWidth": "820px", "marginBottom": "20px"}),
            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
                html.Div([
                    html.Div(f"Km {km}", style={"fontWeight": "700",
                                                 "color": KM_COLORS.get(km, COLOR_ACCENT), "fontSize": "16px"}),
                    html.Div(f"{len(df[df['km']==km])} obs.", style={"fontSize": "13px", "color": COLOR_MUTED}),
                ], style={**CARD, "marginBottom": "0", "padding": "16px 24px",
                           "borderTop": f"3px solid {KM_COLORS.get(km, COLOR_ACCENT)}"})
                for km in sorted(df["km"].unique())
            ]),
        ]),
        
         html.Div(style={**CARD}, children=[
            section_title("Calamar", "Estación de monitoreo del Instituto de Hidrología, Meteorología y Estudios Ambientales (IDEAM)"),
            html.P("La estación de monitoreo hidrologico del IDEAM ubicada a unos 100 Km de Barranquilla en calamar es una fuente de datos adicionales para el análisis "
                   "propuesto en este trabajo. En esta no se mide directamente la SSC pero si varibales fuertemente relacionadas como el "
                   "caudal (Q), y la Carga solida total (TSS) de las cuales puede derivarse una concentración equivalente de promedio diario "
                   "e intengrada en sección (SSC = TSS / Q) que puede ser comparada con las mediciones de campo y estimaciones de Sentinel-2 para evaluar "
                   "su posible uso como fuente de datos adicionales para la calibración del modelo. Por otro lado, parte del caudal de Calamar se pierde al "
                   "salir por el canal del dique, por esto para evaluar este efecto se usaran caudales de la estación Inkora del Ideam que monitorea el caudal de este corredor fluvial.",
                   style={"fontSize": "14.5px", "color": COLOR_TEXT, "lineHeight": "1.8",
                          "maxWidth": "820px", "marginBottom": "20px"}),
            html.P("A continuación se presenta la ubicación de esta estación:",
                   style={"fontSize": "14.5px", "color": COLOR_TEXT, "lineHeight": "1.8",
                          "maxWidth": "820px", "marginBottom": "20px"}),
            dcc.Graph(figure=fig2, config={"displayModeBar": False}),
        ]),
    ])


def tab_problema():
    return html.Div([
        html.Div(style={**CARD, "borderLeft": "4px solid #e07b2a"}, children=[
            section_title("Planteamiento del problema"),
            html.P("El monitoreo de la CSS en ríos de gran caudal como el Magdalena representa un desafío "
                   "logístico y económico considerable. Los métodos tradicionales requieren campañas de campo "
                   "intensivas con equipos especializados como el perfilador LISST.",
                   style={"fontSize": "14.5px", "color": COLOR_TEXT, "lineHeight": "1.8", "marginBottom": "16px"}),
            html.P("La teledetección satelital con Sentinel-2 ofrece una alternativa de bajo costo con "
                   "cobertura sistemática. Sin embargo, en entornos estuarinos la estimación de CSS es "
                   "compleja por la interferencia de otros constituyentes ópticos y los efectos de marea.",
                   style={"fontSize": "14.5px", "color": COLOR_TEXT, "lineHeight": "1.8"}),
        ]),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
            html.Div(style={**CARD}, children=[
                html.H4("Limitaciones del monitoreo tradicional",
                        style={"fontFamily": FONT_TITLE, "fontSize": "15px", "color": COLOR_TEXT, "marginBottom": "14px"}),
                html.Ul([html.Li(t, style={"fontSize": "14px", "color": COLOR_TEXT, "marginBottom": "8px", "lineHeight": "1.6"})
                         for t in ["Alta demanda de recursos para campañas de campo",
                                   "Cobertura temporal limitada a fechas de muestreo",
                                   "Variabilidad espacial difícil de capturar puntualmente",
                                   "Influencia de dragados en zonas del canal navegable"]], style={"paddingLeft": "18px"}),
            ]),
            html.Div(style={**CARD}, children=[
                html.H4("Potencial de la teledetección",
                        style={"fontFamily": FONT_TITLE, "fontSize": "15px", "color": COLOR_TEXT, "marginBottom": "14px"}),
                html.Ul([html.Li(t, style={"fontSize": "14px", "color": COLOR_TEXT, "marginBottom": "8px", "lineHeight": "1.6"})
                         for t in ["Revisita cada 5 días con Sentinel-2",
                                   "Cobertura espacial continua del tramo fluvial",
                                   "Datos gratuitos accesibles mediante Google Earth Engine",
                                   "Posibilidad de reconstrucción histórica de series de CSS"]], style={"paddingLeft": "18px"}),
            ]),
        ]),
    ])


def tab_objetivo():
    return html.Div([
        html.Div(style={**CARD, "borderLeft": f"4px solid {COLOR_ACCENT}"}, children=[
            section_title("Objetivo general"),
            html.P("Estimar la concentración superficial de sedimento en suspensión (SSC) en el sector fluvial entre Calamar y Bocas de Ceniza (bajo río Magdalena) "
                   "mediante un modelo empírico derivado de variables espectrales satelitales e información hidrosedimentológica in situ, orientado a caracterizar su "
                   "variabilidad espaciotemporal.",
                   style={"fontSize": "17px", "color": COLOR_TEXT, "lineHeight": "1.9",
                          "maxWidth": "800px", "fontWeight": "400"}),
        ]),
        html.Div(style={**CARD}, children=[
            section_title("Objetivos específicos"),
            html.Div(style={"display": "flex", "flexDirection": "column", "gap": "12px"}, children=[
                html.Div(style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}, children=[
                    html.Div(str(i+1), style={
                        "minWidth": "32px", "height": "32px", "borderRadius": "50%",
                        "backgroundColor": COLOR_ACCENT, "color": "white",
                        "display": "flex", "alignItems": "center", "justifyContent": "center",
                        "fontWeight": "700", "fontSize": "14px", "marginTop": "2px",
                    }),
                    html.P(t, style={"fontSize": "14.5px", "color": COLOR_TEXT, "lineHeight": "1.7", "margin": "0"}),
                ])
                for i, t in enumerate([
                    "Caracterizar la respuesta espectral del agua asociada a diferentes concentraciones de sedimento en suspensión, "
                    "utilizando bandas del visible, NIR y SWIR (e índices espectrales derivados), en el sector fluvial entre Calamar y "
                    "Bocas de Ceniza (bajo río Magdalena).",
                    "Calibrar y validar un modelo empírico de estimación de SSC a partir de variables espectrales satelitales, "
                    "empleando información hidrosedimentológica in situ para el sector fluvial entre Calamar y Bocas de Ceniza (bajo río Magdalena).",
                    "Cuantificar la variabilidad espaciotemporal de la SSC superficial en el sector fluvial entre Calamar y Bocas de Ceniza, a partir "
                    "de la serie satelital estimada, incluyendo estacionalidad, eventos extremos y tendencias.",
                ])
            ]),
        ]),
    ])


def tab_marco():
    return html.Div([
        html.Div(style={**CARD}, children=[
            section_title("Teledetección de sedimentos en suspensión", "Fundamentos físicos y estado del arte"),
            html.P("La estimación de CSS mediante teledetección se basa en la relación entre la reflectancia "
                "espectral del agua y la concentración de partículas en suspensión. Los sedimentos aumentan "
                "la reflectancia en las bandas roja y NIR al incrementar la retrodispersión.",
                style={"fontSize": "14.5px", "color": COLOR_TEXT, "lineHeight": "1.8"}),
        ]),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[

            # ── Gráfica espectral ──
            html.Div(style={**CARD}, children=[
                html.H4("Sentinel-2 MSI", style={"fontFamily": FONT_TITLE, "fontSize": "15px",
                                                "color": COLOR_TEXT, "marginBottom": "4px"}),
                html.P("Haz clic en una banda para ver detalles",
                    style={"fontSize": "12px", "color": COLOR_MUTED, "marginBottom": "12px"}),
 
                html.Div(style={"position": "relative", "width": "120%", "height": "260px"}, children=[
                    html.Div(id="bands-overlay", style={"position": "absolute", "top": "0",
                                                        "left": "0", "width": "100%", "height": "100%"}),
                    # Etiquetas eje X
                    html.Div(style={"position": "absolute", "bottom": "0", "left": "0",
                                    "width": "100%", "height": "24px"}, children=[
                        html.Span(str(nm), style={
                            "position": "absolute",
                            "left": f"{(nm - 400) / 2000 * 100}%",
                            "transform": "translateX(-50%)",
                            "fontSize": "11px", "color": COLOR_MUTED,
                        }) for nm in [400, 600, 800, 1000, 1400, 1800, 2200]
                    ]),
                ]),


                html.Div(id="band-info",
                        style={"background": COLOR_BG, "borderRadius": "8px",
                                "padding": "14px 18px", "marginTop": "12px",
                                "border": f"0.5px solid {COLOR_BORDER}", "minHeight": "70px"},
                        children=html.P("Selecciona una banda del espectro",
                                        style={"fontSize": "13px", "color": COLOR_MUTED, "margin": "0"})),
            ]),

            # ── Índices espectrales ──
            html.Div(style={**CARD}, children=[
                html.H4("Índices espectrales evaluados",
                        style={"fontFamily": FONT_TITLE, "fontSize": "15px",
                            "color": COLOR_TEXT, "marginBottom": "14px"}),
                html.Div([
                    html.Div([
                        html.Div(n, style={"fontWeight": "700", "color": COLOR_ACCENT,
                                        "fontSize": "14px", "marginBottom": "4px"}),
                        html.Div(f, style={"fontFamily": "monospace", "fontSize": "12px",
                                        "color": COLOR_MUTED, "marginBottom": "4px"}),
                        html.Div(d, style={"fontSize": "13px", "color": COLOR_TEXT, "lineHeight": "1.5"}),
                    ], style={"marginBottom": "16px", "paddingBottom": "16px",
                            "borderBottom": f"1px solid {COLOR_BORDER}"})
                    for n, f, d in [
                        ("RANS",    "(Red+NIR)/(Red+NIR+Blue+Green+SWIR1+SWIR2)",      "Índice normalizado para sedimentos"),
                        ("VNES",    "(Red+RE1+NIR)/(Blue+Green+Red+NIR+SWIR1+SWIR2)",  "Variante extendida con red edge"),
                        ("NDTI",    "(Red−Green)/(Red+Green)",                          "Índice de turbidez normalizado"),
                        ("NIR/RED", "(NIR)/(Red)",                                      "Relación simple entre NIR y rojo"),
                    ]
                ]),
            ]),
        ]),
html.Div(style={**CARD}, children=[
    section_title("Fórmulas derivadas", "Navega entre las variables calculadas"),

    # Store índice actual
    dcc.Store(id="formula-idx", data=0),

    # Tarjeta
    html.Div(style={"position": "relative"}, children=[

        # Tag
        html.Div(id="formula-tag", style={"marginBottom": "12px"}),

        # Título y subtítulo
        html.P(id="formula-title",
               style={"fontSize": "15px", "fontWeight": "500",
                      "color": COLOR_TEXT, "margin": "0 0 4px"}),
        html.P(id="formula-sub",
               style={"fontSize": "12px", "color": COLOR_MUTED, "margin": "0 0 4px"}),

        # Fórmula
        html.Div(id="formula-box",
                 style={"fontFamily": "monospace", "fontSize": "15px",
                        "backgroundColor": f"{COLOR_ACCENT}10",
                        "border": f"1px solid {COLOR_ACCENT}30",
                        "borderRadius": "6px", "padding": "14px 20px",
                        "color": COLOR_TEXT, "margin": "16px 0"}),

        # Descripción
        html.P(id="formula-desc",
               style={"fontSize": "13px", "color": COLOR_MUTED,
                      "lineHeight": "1.7", "margin": "0"}),

        # Navegación
        html.Div(style={"display": "flex", "alignItems": "center",
                        "justifyContent": "space-between", "marginTop": "20px"}, children=[
            html.Button("←", id="formula-prev", n_clicks=0,
                        style={"background": "none", "border": f"0.5px solid {COLOR_BORDER}",
                               "borderRadius": "6px", "padding": "6px 14px",
                               "fontSize": "18px", "cursor": "pointer", "color": COLOR_TEXT}),
            html.Span(id="formula-counter",
                      style={"fontSize": "12px", "color": COLOR_MUTED}),
            html.Button("→", id="formula-next", n_clicks=0,
                        style={"background": "none", "border": f"0.5px solid {COLOR_BORDER}",
                               "borderRadius": "6px", "padding": "6px 14px",
                               "fontSize": "18px", "cursor": "pointer", "color": COLOR_TEXT}),
        ]),
    ]),
]),
    ])


def tab_eda():
    return html.Div([

        # ── FILTRO GENERAL (sticky) ──
        html.Div(id = "filter.card", style={**CARD,"borderLeft": f"4px solid {COLOR_ACCENT}", "position": "sticky",
                         "top": "0", "zIndex": "100", "padding": "20px 32px", "transition": "padding 0.25s ease, box-shadow 0.25s ease",
        "borderBottom": "1px solid #eee",}, children=[
            section_title("Filtro global por estación",
                          "Selecciona las estaciones por kilómetros que deseas incluir en todo el análisis exploratorio, aquellas entre el Km 1 al 11 pueden introducir ruido por dragados y cambios rapidos en las condiciones hidrodinamicas afectando negativamente las correlaciones"),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px", "flexWrap": "wrap"}, children=[
                dcc.Checklist(
                    id="km-filter",
                    options=[{"label": html.Span(f" Km {k}",
                               style={"color": KM_COLORS.get(k, COLOR_ACCENT), "fontWeight": "700",
                                      "marginRight": "8px"}), "value": k} for k in KMS_ALL],
                    value=KMS_ALL,
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    style={"fontSize": "14px"},
                ),
                html.Div(id="km-filter-count",
                         style={"fontSize": "12px", "color": COLOR_MUTED,
                                "marginLeft": "auto", "fontStyle": "italic"}),
            ]),
        ]),

        # ── Estadísticas descriptivas ──
        # ── Layout ──
html.Div(style={**CARD}, children=[
    section_title("Estadísticas descriptivas", "Resumen del subconjunto seleccionado"),

    html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px",
                    "marginBottom": "16px", "flexWrap": "wrap"}, children=[

        # Pills grupo
        html.Div(id="group-pills", style={"display": "flex", "gap": "8px"}, children=[
            html.Button("Bandas",  id="pill-bandas",  n_clicks=0,
                        style={"fontSize":"11px","padding":"3px 10px","borderRadius":"6px",
                               "border":"0.5px solid","cursor":"pointer"}),
            html.Button("Índices", id="pill-indices", n_clicks=0,
                        style={"fontSize":"11px","padding":"3px 10px","borderRadius":"6px",
                               "border":"0.5px solid","cursor":"pointer"}),
            html.Button("SSC",     id="pill-ssc",     n_clicks=0,
                        style={"fontSize":"11px","padding":"3px 10px","borderRadius":"6px",
                               "border":"0.5px solid","cursor":"pointer"}),
        ]),

        # Dropdown variable
        dcc.Dropdown(
            id="stats-var-dropdown",
            options=[{"label": v, "value": v} for v in BANDAS],
            value=SSCS[1],
            clearable=False,
            style={
            "backgroundColor": COLOR_INPUT_BG,
            "color": COLOR_TEXT_INPUT,
            "border": f"1px solid {COLOR_INPUT_BD}",
            "borderRadius": "10px",
}
        ),
    ]),

    html.Div(id="stats-table"),
]),

# Store para grupo activo
dcc.Store(id="stats-group", data="bandas"),
        
        # ── Perfiles de campo ──
        html.Div(style={**CARD}, children=[
            section_title("Perfiles de concentración de campo",
                          "Perfiles verticales de SSC medidos con LISST — selecciona fecha, Km y transecto"),
            html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                            "alignItems": "center", "flexWrap": "wrap"}, children=[
                html.Label("Fecha:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600"}),
                dcc.Dropdown(id="profile-fecha", options=[], placeholder="Selecciona una fecha",
                                clearable=False, style={
                                                            "backgroundColor": COLOR_INPUT_BG,
                                                            "color": COLOR_TEXT_INPUT,
                                                            "border": f"1px solid {COLOR_INPUT_BD}",
                                                            "borderRadius": "10px",
                                                        }),
                html.Label("Km:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600", "marginLeft": "8px"}),
                dcc.Dropdown(id="profile-km", options=[], placeholder="Km",
                             clearable=False, style={
                                                        "backgroundColor": COLOR_INPUT_BG,
                                                        "color": COLOR_TEXT_INPUT,
                                                        "border": f"1px solid {COLOR_INPUT_BD}",
                                                        "borderRadius": "10px",
                                                    }),
                html.Label("+m:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600", "marginLeft": "8px"}),
                dcc.Dropdown(id="profile-pm", options=[], placeholder="+m",
                             clearable=False, style={
                                                        "backgroundColor": COLOR_INPUT_BG,
                                                        "color": COLOR_TEXT_INPUT,
                                                        "border": f"1px solid {COLOR_INPUT_BD}",
                                                        "borderRadius": "10px",
                                                    }),
            ]),
            html.Div(id="profile-stats", style={"marginBottom": "12px"}),
            dcc.Graph(id="profile-plot", config={"displayModeBar": False}, style={"height": "480px"}),
        ]),

        # ── Distribución ──
        html.Div(style={**CARD}, children=[
            section_title("Distribución de CSS por estación",
                          "Histograma y boxplot de concentración de sedimentos en suspensión"),
            html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px", "alignItems": "center"}, children=[
                html.Label("Variable:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600"}),
                dcc.Dropdown(id="dist-variable",
                             options=[{"label": v, "value": v} for v in ["SSC"] + BANDAS + INDICES],
                             value="SSC", clearable=False, style={
                                                                    "backgroundColor": COLOR_INPUT_BG,
                                                                    "color": COLOR_TEXT_INPUT,
                                                                    "border": f"1px solid {COLOR_INPUT_BD}",
                                                                    "borderRadius": "10px",
                                                                }),
            ]),
            dcc.Graph(id="dist-plot", config={"displayModeBar": False}),
        ]),

        # ── Series de tiempo ──
        html.Div(style={**CARD}, children=[
            section_title("Series de tiempo", "Evolución temporal de CSS y reflectancia por estación"),
            html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                            "alignItems": "center", "flexWrap": "wrap"}, children=[
                html.Label("Variable:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600"}),
                dcc.Dropdown(id="ts-variable",
                             options=[{"label": v, "value": v} for v in ["SSC"] + BANDAS + INDICES],
                             value="SSC", clearable=False, style={"width": "200px", "fontSize": "13px"}),
            ]),
            dcc.Graph(id="ts-plot", config={"displayModeBar": False}),
        ]),

        # ── Scatter ──
        html.Div(style={**CARD}, children=[
            section_title("Relación reflectancia / CSS", "Scatter con ajuste de regresión y estadísticos"),
            html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                            "alignItems": "center", "flexWrap": "wrap"}, children=[
                html.Label("Variable X:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600"}),
                dcc.Dropdown(id="scatter-x",
                             options=[{"label": v, "value": v} for v in BANDAS + INDICES],
                             value="red", clearable=False, style={
                                                                    "backgroundColor": COLOR_INPUT_BG,
                                                                    "color": COLOR_TEXT_INPUT,
                                                                    "border": f"1px solid {COLOR_INPUT_BD}",
                                                                    "borderRadius": "10px",
                                                                }),
                dcc.Dropdown(id="scatter-y",
                             options=[{"label": v, "value": v} for v in SSCS],
                             value="SSC", clearable=False, style={
                                                                    "backgroundColor": COLOR_INPUT_BG,
                                                                    "color": COLOR_TEXT_INPUT,
                                                                    "border": f"1px solid {COLOR_INPUT_BD}",
                                                                    "borderRadius": "10px",
                                                                }),
                html.Label("Transformación Y:", style={"fontSize": "13px", "color": COLOR_MUTED,
                                                        "fontWeight": "600", "marginLeft": "16px"}),
                dcc.RadioItems(id="scatter-transform",
                               options=[{"label": " CSS", "value": "linear"},{"label": " ln(CSS)", "value": "log"}],
                               value="log", inline=True, style={"fontSize": "13px"}),
                html.Label("Color por:", style={"fontSize": "13px", "color": COLOR_MUTED,
                                                 "fontWeight": "600", "marginLeft": "16px"}),
                dcc.RadioItems(id="scatter-color",
                               options=[{"label": " Km", "value": "km"},{"label": " CSS", "value": "CSS"}, {"label": " Ninguno", "value": "none"}],
                               value="km", inline=True, style={"fontSize": "13px"}),
                html.Label("Ajuste:", style={"fontSize": "13px", "color": COLOR_MUTED,
                                                 "fontWeight": "600", "marginLeft": "16px"}),
                dcc.RadioItems(id="scatter-ajuste",
                               options=[{"label": "Lineal", "value": "lineal"},{"label": "Potencial", "value": "potencial"}],
                               value="lineal", inline=True, style={"fontSize": "13px"}),
            ]),
            dcc.Graph(id="scatter-plot", config={"displayModeBar": False}),
            html.Div(id="scatter-stats", style={"marginTop": "8px"}),
        ]),
                # ── Ranking de correlaciones ──
        html.Div(style={**CARD}, children=[
            section_title("Ranking de correlaciones con CSS",
                          "Correlación de Pearson entre cada banda/índice y CSS, ordenado por valor absoluto"),
            html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px", "alignItems": "center"}, children=[
                html.Label("Transformación CSS:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600"}),
                dcc.RadioItems(id="corrbar-transform",
                               options=[{"label": " CSS", "value": "linear"},{"label": " ln(CSS)", "value": "log"}],
                               value="log", inline=True, style={"fontSize": "13px"}),
            ]),
            dcc.Graph(id="corrbar-plot", config={"displayModeBar": False}),
        ]),
        
        # ── Firmas espectrales ──
        html.Div(style={**CARD}, children=[
            section_title("Firmas espectrales", "Reflectancia por banda para cada observación, coloreada por CSS"),
            html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px", "alignItems": "center"}, children=[
                html.Label("Estación (km):", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600"}),
                dcc.Dropdown(id="spec-km", value="all", clearable=False,
                             style={
                                    "backgroundColor": COLOR_INPUT_BG,
                                    "color": COLOR_TEXT_INPUT,
                                    "border": f"1px solid {COLOR_INPUT_BD}",
                                    "borderRadius": "10px",
                                }),
            ]),
            dcc.Graph(id="spec-plot", config={"displayModeBar": False}, style={"height": "560px"}),
        ]),
        
        
        # ── Hidrología Calamar & Barranquilla ──
        html.Div(style={**CARD}, children=[
            section_title("Hidrología — Calamar y Barranquilla",
                          "Series de tiempo, estacionalidad y relaciones entre caudal y transporte de sedimentos"),

            # Sub-tabs hidro
            dcc.Tabs(id="hydro-tabs", value="hydro-ts", style={"marginBottom": "20px"},
                     children=[
                dcc.Tab(label="Series de tiempo",   value="hydro-ts",   style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="Estacionalidad",     value="hydro-seas", style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="Q vs TSS Calamar",   value="hydro-qtss", style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="Q Calamar vs Q Baq", value="hydro-qq",   style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="Q calamar - Q Inkoras", value="hydro-qincora", style=TAB_STYLE, selected_style=TAB_SELECTED),
                dcc.Tab(label="TSS Barranquilla",   value="hydro-tss",  style=TAB_STYLE, selected_style=TAB_SELECTED),
            ]),

            # Slider de años
            html.Div(style={"marginBottom": "20px"}, children=[
                html.Label("Intervalo de años:",
                           style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600",
                                  "marginBottom": "8px", "display": "block"}),
                dcc.RangeSlider(
                    id="hydro-year-slider",
                    min=HYDRO_YEAR_MIN, max=HYDRO_YEAR_MAX,
                    step=1,
                    value=[2022, HYDRO_YEAR_MAX],
                    marks={y: str(y) for y in range(HYDRO_YEAR_MIN, HYDRO_YEAR_MAX+1, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]),

            html.Div(id="hydro-content"),
        ]),

        # ── Correlación ──
        html.Div(style={**CARD}, children=[
            section_title("Matriz de correlación",
                          "Correlación de Pearson entre bandas espectrales, índices y CSS"),
            html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px", "alignItems": "center"}, children=[
                html.Label("Transformación CSS:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600"}),
                dcc.RadioItems(id="corr-transform",
                               options=[{"label": " CSS", "value": "linear"},{"label": " ln(CSS)", "value": "log"}],
                               value="log", inline=True, style={"fontSize": "13px"}),
            ]),
            dcc.Graph(id="corr-plot", config={"displayModeBar": False}),
        ]),
        


        # ── Mapa de calor espacio-temporal ──
        html.Div(style={**CARD}, children=[
            section_title("Mapa de calor espacio-temporal",
                          "CSS promedio por estación (km) y fecha de imagen — dataset matcheado"),
            html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px", "alignItems": "center"}, children=[
                html.Label("Variable:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600"}),
                dcc.Dropdown(id="heatmap-var",
                             options=[{"label": v, "value": v} for v in ["SSC"] + BANDAS + INDICES],
                             value="SSC", clearable=False, style={
                                                                "backgroundColor": COLOR_INPUT_BG,
                                                                "color": COLOR_TEXT_INPUT,
                                                                "border": f"1px solid {COLOR_INPUT_BD}",
                                                                "borderRadius": "10px",
                                                            }),
            ]),
            dcc.Graph(id="heatmap-plot", config={"displayModeBar": False}),
        ]),

        # ── Climatograma CSS ──
        html.Div(style={**CARD}, children=[
            section_title("Climatograma de CSS",
                          "Distribución mensual de CSS en el período de estudio — dataset matcheado"),
            dcc.Graph(id="climo-plot", config={"displayModeBar": False}),
        ]),
    ])
    
    


def tab_conclusiones():

    p1 = (
        "El proceso de control de calidad permitió consolidar entre 27 y 39 observaciones "
        "de calibración distribuidas en los kilómetros 0, 1, 3, 14, 17, 18 y 19, alcanzando "
        "coeficientes de determinación entre 0.58 y 0.73. Las bandas Red, NIR y rojo 3 "
        "presentaron las correlaciones más altas con la concentración de sedimentos suspendidos "
        "(SSC), resultado consistente con lo reportado en la literatura para sistemas fluviales "
        "turbios. Entre los modelos evaluados, la regresión potencial log-log basada en la banda "
        "NIR mostró el mejor equilibrio entre simplicidad y desempeño, con métricas satisfactorias "
        "tanto en calibración como en validación LOOCV. Los modelos de aprendizaje automático "
        "como Random Forest y Gradient Boosting mostraron potencial, aunque el tamaño actual del "
        "dataset limita su capacidad de generalización."
    )

    p2 = (
        "Con aproximadamente 30 observaciones, el conjunto de datos resulta adecuado para "
        "aplicar regresiones potenciales y modelos múltiples validados mediante LOOCV, mientras "
        "que una ampliación futura de la base de datos permitiría implementar enfoques de machine "
        "learning de manera más robusta. El modelo desarrollado busca ofrecer una alternativa "
        "frente a la limitada disponibilidad de datos in situ en el río Magdalena; sin embargo, "
        "esta misma limitación condiciona su alcance y capacidad predictiva. Además, actividades "
        "como los dragados en estaciones cercanas a la desembocadura pueden introducir ruido "
        "adicional en la señal espectral y deben considerarse durante la interpretación de los "
        "resultados y en futuros procesos de calibración."
    )

    return html.Div([
        html.Div(
            style={**CARD, "borderLeft": f"4px solid {COLOR_ACCENT}"},
            children=[

                section_title("Conclusiones del análisis exploratorio"),

                html.Div(
                    style={
                        **CARD2,
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "22px",
                    },
                    children=[

                        html.P(
                            p1,
                            style={
                                "fontSize": "15px",
                                "color": COLOR_TEXT,
                                "lineHeight": "1.9",
                                "margin": "0",
                                "textAlign": "justify",
                            }
                        ),

                        html.P(
                            p2,
                            style={
                                "fontSize": "15px",
                                "color": COLOR_TEXT,
                                "lineHeight": "1.9",
                                "margin": "0",
                                "textAlign": "justify",
                            }
                        ),

                    ]
                ),
            ]
        ),


        html.Div(style={**CARD}, children=[
            section_title("Referencias"),
            html.Ul([html.Li(r, style={"fontSize": "13.5px", "color": COLOR_TEXT,
                                        "marginBottom": "8px", "lineHeight": "1.6"})
                     for r in ["Qiu, Z., Liu, D., Duan, M., Chen, P., Yang, C., Li, K., & Duan, H. (2024). Four-decades of sediment transport variations in the Yellow River on the Loess Plateau using Landsat imagery. Remote Sensing of Environment, 306. https://doi.org/10.1016/j.rse.2024.114147",
                               "Qiu, Z., Liu, D., Yan, N., Yang, C., Chen, P., Zhang, C., & Duan, H. (2024). Improving the observations of suspended sediment concentrations in rivers from Landsat to Sentinel-2 imagery. International Journal of Applied Earth Observation and Geoinformation, 134. https://doi.org/10.1016/j.jag.2024.104209",
                               "Restrepo, J. D., Zapata, P., Díaz, J. M., Garzón-Ferreira, J., & García, C. B. (2006). Fluvial fluxes into the Caribbean Sea and their impact on coastal ecosystems: The Magdalena River, Colombia. Global and Planetary Change, 50(1–2), 33–49. https://doi.org/10.1016/j.gloplacha.2005.09.002",
                               "Yepez, S., Laraque, A., Martinez, J. M., De Sa, J., Carrera, J. M., Castellanos, B., Gallay, M., & Lopez, J. L. (2018). Retrieval of suspended sediment concentrations using Landsat-8 OLI satellite images in the Orinoco River (Venezuela). Comptes Rendus - Geoscience, 350(1–2), 20–30. https://doi.org/10.1016/j.crte.2017.08.004"]],
                    style={"paddingLeft": "18px"}),
        ]),
    ])


# ═══════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    return {"intro": tab_intro, "contexto": tab_contexto, "problema": tab_problema,
            "objetivo": tab_objetivo, "marco": tab_marco, "eda": tab_eda,
            "modelo":       tab_modelo,
            "aplicacion":  tab_aplicacion,
            "conclusiones": tab_conclusiones}.get(tab, tab_intro)()


# ── Store: filtra el dataset según km seleccionados ──
@app.callback(
    Output("store-df-filtered", "data"),
    Output("km-filter-count", "children"),
    Input("km-filter", "value"),
)
def update_store(kms_sel):
    kms_sel = kms_sel or KMS_ALL
    filtered = df[df["km"].isin(kms_sel)]
    label = f"{len(filtered)} observaciones · {len(kms_sel)} estación(es) seleccionada(s)"
    return filtered.to_json(date_format="iso", orient="split"), label


# ── Estadísticas ──

#Guardar grupo activo al hacer click en pill
@app.callback(
    Output("stats-group", "data"),
    Input("pill-bandas",  "n_clicks"),
    Input("pill-indices", "n_clicks"),
    Input("pill-ssc",     "n_clicks"),
    prevent_initial_call=True,
)
def set_group(b, i, s):
    return ctx.triggered_id.replace("pill-", "")


#Actualizar opciones del dropdown según grupo
@app.callback(
    Output("stats-var-dropdown", "options"),
    Output("stats-var-dropdown", "value"),
    Input("stats-group", "data"),
)
def update_dropdown(group):
    mapping = {"bandas": BANDAS, "indices": INDICES, "ssc": SSCS}
    vars_ = mapping[group]
    opts  = [{"label": v, "value": v} for v in vars_]
    return opts, vars_[0]


#
@app.callback(
    Output("stats-table", "children"),
    Input("stats-var-dropdown", "value"),
    Input("store-df-filtered",  "data"),
)
def update_stats(variable, data):
    if not data or not variable:
        return []
    dff   = pd.read_json(io.StringIO(data), orient="split")
    if variable not in dff.columns:
        return html.P("Variable no disponible", style={"color": COLOR_MUTED, "fontSize": "13px"})

    s = dff[variable].describe()
    stats = {
        "Media":     round(s["mean"], 4),
        "Desv. Est.": round(s["std"],  4),
        "Mín.":      round(s["min"],  4),
        "Mediana":   round(dff[variable].median(), 4),
        "Máx.":      round(s["max"],  4),
    }

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Variable", style={"textAlign":"left","padding":"8px 14px","fontSize":"12px",
                                       "color":COLOR_MUTED,"borderBottom":f"2px solid {COLOR_BORDER}"}),
            *[html.Th(k, style={"textAlign":"right","padding":"8px 14px","fontSize":"12px",
                                "color":COLOR_MUTED,"borderBottom":f"2px solid {COLOR_BORDER}"})
              for k in stats],
        ])),
        html.Tbody([
            html.Tr([
                html.Td(variable, style={"padding":"7px 14px","fontSize":"13px",
                                         "fontWeight":"600","color":COLOR_ACCENT,
                                         "borderBottom":f"1px solid {COLOR_BORDER}"}),
                *[html.Td(str(v), style={"padding":"7px 14px","fontSize":"13px",
                                          "textAlign":"right","borderBottom":f"1px solid {COLOR_BORDER}"})
                  for v in stats.values()]
            ])
        ])
    ], style={"width":"100%","borderCollapse":"collapse"})
    
    
# ── Distribución ──
@app.callback(Output("dist-plot","figure"),
              Input("dist-variable","value"), Input("store-df-filtered","data"))
def update_dist(var, data):
    if not data: return go.Figure()
    dff = pd.read_json(io.StringIO(data), orient="split")
    if var not in dff.columns: return go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Histograma por Km","Boxplot por Km"))
    for km in sorted(dff["km"].unique()):
        sub = dff[dff["km"]==km]
        fig.add_trace(go.Histogram(x=sub[var], name=f"Km {km}", marker_color=COLOR_ACCENT, opacity=0.75, nbinsx=12), row=1,col=1)
        fig.add_trace(go.Box(y=sub[var], name=f"Km {km}", marker_color=COLOR_ACCENT, boxmean=True, showlegend=False), row=1,col=2)
    fig.update_layout(barmode="overlay", height=380, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                      font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
                      legend=dict(orientation="h",y=-0.15), margin=dict(l=40,r=20,t=40,b=40))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor=COLOR_BORDER)
    return fig


# ── Series de tiempo ──
@app.callback(Output("ts-plot","figure"),
              Input("ts-variable","value"), Input("store-df-filtered","data"))
def update_ts(var, data):
    if not data: return go.Figure()
    dff = pd.read_json(io.StringIO(data), orient="split")
    dff["reflectance_date"] = pd.to_datetime(dff["reflectance_date"])
    if var not in dff.columns: return go.Figure()
    fig = go.Figure()
    for km in sorted(dff["km"].unique()):
        sub = dff[dff["km"]==km].sort_values("reflectance_date"); color = KM_COLORS.get(km, COLOR_ACCENT)
        fig.add_trace(go.Scatter(x=sub["reflectance_date"], y=sub[var], mode="lines+markers",
                                 name=f"Km {km}", line=dict(color=color,width=2), marker=dict(size=7,color=color)))
    fig.update_layout(height=360, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                      font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
                      xaxis_title="Fecha", yaxis_title=var,
                      legend=dict(orientation="h",y=-0.2), margin=dict(l=50,r=20,t=20,b=50))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor=COLOR_BORDER)
    return fig


# ── Scatter ──
@app.callback(Output("scatter-plot","figure"), Output("scatter-stats","children"),
              Input("scatter-x","value"), Input("scatter-transform","value"),
              Input("scatter-color","value"), Input("store-df-filtered","data"), Input("scatter-y","value"),
              Input("scatter-ajuste", "value"))
              
def update_scatter(x_var, transform, color_by, data, y_var, ajuste):
    if not data: return go.Figure(), ""
    dff = pd.read_json(io.StringIO(data), orient="split")
    if x_var not in dff.columns or y_var not in dff.columns: return go.Figure(), ""
    x = dff[x_var]
    y_raw = dff[y_var]

    # Transformación seleccionada por el usuario
    y = np.log(y_raw) if transform == "log" else y_raw
    y_label = "ln(CSS)" if transform == "log" else "CSS (mg/L)"

    fig = go.Figure()
    
    if ajuste == "lineal":
        r, p = pearsonr(x, y); r2 = r**2
        m, b = np.polyfit(x, y, 1); x_line = np.linspace(x.min(), x.max(), 200)
        y_line = m * x_line + b
        eq_text = f"y = {m:.4f}x + {b:.4f}"

    elif ajuste == "potencial":
        
        mask = (x > 0) & (y_raw > 0)
        x_fit = x[mask]
        y_fit = y_raw[mask]

        logx = np.log(x_fit)
        logy = np.log(y_fit)

        r, p = pearsonr(logx, logy)
        r2 = r**2

        b_exp, loga = np.polyfit(logx, logy, 1)
        a = np.exp(loga)

        x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
        y_line = a * (x_line ** b_exp)

        # si estás en modo log, graficar log(y)
        if transform == "log":
            y_line = np.log(y_line)

        eq_text = f"y = {a:.4f}x^{b_exp:.4f}"
    if color_by == "km":
        for km in sorted(dff["km"].unique()):
            sub = dff[dff["km"]==km]; y_sub = np.log(sub["SSC"]) if transform=="log" else sub["SSC"]
            fig.add_trace(go.Scatter(x=sub[x_var], y=y_sub, mode="markers", name=f"Km {km}",
                                     marker=dict(size=9, color=KM_COLORS.get(km,COLOR_ACCENT),
                                                 line=dict(width=1,color="white"))))
    elif color_by == "CSS":
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Datos",
            marker=dict(
                size=9,
                color=dff[y_var],
                colorscale="Inferno",
                colorbar=dict(title="CSS"),
                showscale=True,
                line=dict(width=1, color="white")
            )
        ))
        
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Datos",
                                 marker=dict(size=9,color=COLOR_ACCENT,line=dict(width=1,color="white"))))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Regresión",
                             line=dict(color="#c0392b",width=2,dash="dash")))
    fig.update_layout(height=400, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                      font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
                      xaxis_title=x_var, yaxis_title=y_label,
                      legend=dict(orientation="h",y=-0.2), margin=dict(l=50,r=20,t=20,b=50))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor=COLOR_BORDER)
    p_text = "< 0.0001" if p<0.0001 else f"{p:.4f}"
    stats_div = html.Div(style={"display":"flex","gap":"24px","flexWrap":"wrap","marginTop":"8px"}, children=[
        html.Span(eq_text, style={"fontFamily":"monospace","fontSize":"13px","color":COLOR_TEXT,
                   "backgroundColor":f"{COLOR_ACCENT}10","padding":"4px 10px","borderRadius":"4px"}),
        html.Span(f"R² = {r2:.3f}", style={"fontFamily":"monospace","fontSize":"13px","color":COLOR_ACCENT,
                   "fontWeight":"700","padding":"4px 10px","borderRadius":"4px","backgroundColor":f"{COLOR_ACCENT}10"}),
        html.Span(f"p = {p_text}", style={"fontFamily":"monospace","fontSize":"13px","color":COLOR_TEXT,
                   "padding":"4px 10px","borderRadius":"4px","backgroundColor":f"{COLOR_ACCENT}10"}),
        html.Span(f"n = {len(dff)}", style={"fontFamily":"monospace","fontSize":"13px","color":COLOR_MUTED,
                   "padding":"4px 10px","borderRadius":"4px","backgroundColor":f"{COLOR_BORDER}"}),
    ])
    return fig, stats_div


# ── Firmas espectrales: opciones de km según filtro ──
@app.callback(Output("spec-km","options"), Output("spec-km","value"),
              Input("store-df-filtered","data"), State("spec-km","value"))
def update_spec_km_options(data, current_val):
    if not data: return [{"label":"Todas","value":"all"}], "all"
    dff = pd.read_json(io.StringIO(data), orient="split")
    kms = sorted(dff["km"].unique())
    opts = [{"label":"Todas","value":"all"}] + [{"label":f"Km {k}","value":k} for k in kms]
    val = current_val if (current_val=="all" or current_val in kms) else "all"
    return opts, val


# ── Firmas espectrales ──
@app.callback(Output("spec-plot","figure"),
              Input("spec-km","value"), Input("store-df-filtered","data"))
def update_spec(km_sel, data):
    if not data: return go.Figure()
    dff = pd.read_json(io.StringIO(data), orient="split")
    BAND_NAMES = ["aerosol","blue","green","red","rojo 1","rojo 2","rojo 3","NIR","rojo 4","SWIR1","SWIR2"]
    WL_REAL    = [443.9,496.6,560,664.5,703.9,740.2,782.5,835.1,864.8,1613.7,2202.4]
    SWIR1_real=(1550,1700); SWIR2_real=(2140,2290); S1_fict=(0,150); S2_fict=(170,320)
    def to_fict(wl):
        if SWIR1_real[0]<=wl<=SWIR1_real[1]: return S1_fict[0]+(wl-SWIR1_real[0])
        if SWIR2_real[0]<=wl<=SWIR2_real[1]: return S2_fict[0]+(wl-SWIR2_real[0])
        return None
    sub = dff if km_sel=="all" else dff[dff["km"]==km_sel]
    sub = sub.sort_values("SSC").reset_index(drop=True)
    if sub.empty or not all(b in sub.columns for b in BAND_NAMES): return go.Figure()
    ssc_min,ssc_max = sub["SSC"].min(),sub["SSC"].max()
    def ssc_color(ssc):
        t=(ssc-ssc_min)/(ssc_max-ssc_min+1e-9); return f"rgb(255,{int(165*(1-t))},0)"
    WL_VIS=[WL_REAL[i] for i in range(9)]; BAND_VIS=[BAND_NAMES[i] for i in range(9)]
    fig=make_subplots(rows=1,cols=2,column_widths=[0.73,0.27],shared_yaxes=True,
                      horizontal_spacing=0.04,subplot_titles=["Visible / NIR (400–950 nm)","SWIR"])
    for x0,x1,col in [(458,523,"rgba(32,32,229,0.12)"),(543,578,"rgba(0,200,0,0.12)"),
                       (650,680,"rgba(228,0,0,0.12)"),(785,900,"rgba(230,192,4,0.12)")]:
        fig.add_shape(type="rect",x0=x0,x1=x1,y0=0,y1=1,yref="paper",fillcolor=col,line_width=0,row=1,col=1)
    for xf0,xf1 in [S1_fict,S2_fict]:
        fig.add_shape(type="rect",x0=xf0,x1=xf1,y0=0,y1=1,yref="paper",
                      fillcolor="rgba(139,69,19,0.12)",line_width=0,row=1,col=2)
    fig.add_shape(type="line",x0=(S1_fict[1]+S2_fict[0])/2,x1=(S1_fict[1]+S2_fict[0])/2,
                  y0=0,y1=1,yref="paper",line=dict(color="gray",width=1,dash="dash"),row=1,col=2)
    for i,row_data in sub.iterrows():
        color=ssc_color(row_data["SSC"]); date_str=str(row_data["reflectance_date"])[:10]
        hover=f"SSC: {row_data['SSC']:.1f} mg/L<br>Fecha: {date_str}<br>Km: {row_data['km']}"
        fig.add_trace(go.Scatter(x=WL_VIS,y=[row_data[b] for b in BAND_VIS],mode="lines+markers",
                                 line=dict(color=color,width=2),
                                 marker=dict(size=7,color=color,line=dict(width=0.5,color="white")),
                                 hovertemplate=hover+"<extra></extra>",showlegend=False),row=1,col=1)
        for si in [9,10]:
            wl_f=to_fict(WL_REAL[si])
            if wl_f:
                fig.add_trace(go.Scatter(x=[wl_f],y=[row_data[BAND_NAMES[si]]],mode="markers",
                                         marker=dict(size=9,color=color,line=dict(width=0.5,color="white")),
                                         hovertemplate=f"{BAND_NAMES[si]} ({WL_REAL[si]:.0f} nm)<br>"+hover+"<extra></extra>",
                                         showlegend=False),row=1,col=2)
    fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",showlegend=False,hoverinfo="skip",
                             marker=dict(colorscale=[[0,"rgb(255,165,0)"],[1,"rgb(255,0,0)"]],
                                         cmin=ssc_min,cmax=ssc_max,color=[ssc_min],showscale=True,
                                         colorbar=dict(title=dict(text="CSS (mg/L)",side="right"),
                                                       thickness=14,len=0.7,tickfont=dict(size=11)))))
    tick_real=[1614,1650,2202,2250]
    tick_fict=[to_fict(t) for t in tick_real if to_fict(t) is not None]
    tick_lbl=[str(t) for t in tick_real if to_fict(t) is not None]
    all_refl=[row_data[b] for b in BAND_NAMES if b in sub.columns
              for row_data in [sub.iloc[i] for i in range(len(sub))]]
    all_refl=[v for v in all_refl if pd.notna(v)]
    y_min,y_max=max(0,min(all_refl)*0.90),max(all_refl)*1.08
    fig.update_xaxes(title_text="Longitud de onda (nm)",showgrid=False,range=[400,950],row=1,col=1)
    fig.update_xaxes(tickvals=tick_fict,ticktext=tick_lbl,showgrid=False,range=[0,320],row=1,col=2)
    fig.update_yaxes(title_text="Reflectancia (sr⁻¹)",gridcolor=COLOR_BORDER,range=[y_min,y_max],row=1,col=1)
    fig.update_yaxes(showgrid=True,gridcolor=COLOR_BORDER,range=[y_min,y_max],row=1,col=2)
    fig.update_layout(height=560,paper_bgcolor=COLOR_CARD,plot_bgcolor=COLOR_BG,
                      font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
                      margin=dict(l=60,r=80,t=40,b=50),hovermode="closest")
    return fig


# ── Correlación ──
@app.callback(Output("corr-plot","figure"),
              Input("corr-transform","value"), Input("store-df-filtered","data"))
def update_corr(transform, data):
    if not data: return go.Figure()
    dff = pd.read_json(io.StringIO(data), orient="split")
    cols = BANDAS + INDICES + ["SSC"]
    cols = [c for c in cols if c in dff.columns]
    data_c = dff[cols].copy()
    if transform=="log": data_c["SSC"] = np.log(data_c["SSC"])
    cols_label = [c if c!="SSC" else ("ln(CSS)" if transform=="log" else "CSS") for c in cols]
    corr = data_c.corr()
    fig = go.Figure(go.Heatmap(z=corr.values,x=cols_label,y=cols_label,colorscale="RdBu",
                               zmid=0,zmin=-1,zmax=1,text=np.round(corr.values,2),
                               texttemplate="%{text}",textfont={"size":10},hoverongaps=False))
    fig.update_layout(height=480,paper_bgcolor=COLOR_CARD,plot_bgcolor=COLOR_CARD,
                      font=dict(family=FONT_BODY,size=11,color=COLOR_TEXT),
                      margin=dict(l=80,r=20,t=20,b=80),xaxis=dict(tickangle=-45))
    return fig


# ── Perfiles: poblar fechas ──
@app.callback(
    Output("profile-fecha", "options"),
    Output("profile-fecha", "value"),
    Input("tabs", "value")
)
def populate_fechas(tab):
    if tab != "eda" or df_profiles.empty:
        return [], None
    fechas = sorted(df_profiles["fecha"].unique())
    options = [{"label": pd.Timestamp(f).strftime("%d/%m/%Y"), "value": str(f)} for f in fechas]
    return options, str(fechas[0]) if fechas else None


# ── Perfiles: poblar km ──
@app.callback(
    Output("profile-km", "options"),
    Output("profile-km", "value"),
    Input("profile-fecha", "value")
)
def populate_kms(fecha_str):
    if not fecha_str or df_profiles.empty:
        return [], None
    sub = df_profiles[df_profiles["fecha"] == pd.Timestamp(fecha_str)]
    kms = sorted(sub["km"].unique())
    return [{"label": f"Km {k}", "value": k} for k in kms], (kms[0] if kms else None)


# ── Perfiles: poblar +m ──
@app.callback(
    Output("profile-pm", "options"),
    Output("profile-pm", "value"),
    Input("profile-fecha", "value"),
    Input("profile-km", "value")
)
def populate_pm(fecha_str, km_val):
    if not fecha_str or km_val is None or df_profiles.empty:
        return [], None
    sub = df_profiles[(df_profiles["fecha"] == pd.Timestamp(fecha_str)) & (df_profiles["km"] == km_val)]
    pms = sorted(sub["+m"].unique())
    return [{"label": f"+{p} m", "value": p} for p in pms], (pms[0] if pms else None)


# ── Perfiles: graficar ──
@app.callback(
    Output("profile-plot", "figure"),
    Output("profile-stats", "children"),
    Input("profile-fecha", "value"),
    Input("profile-km", "value"),
    Input("profile-pm", "value")
)
def update_profile(fecha_str, km_val, pm_val):
    empty = go.Figure()
    empty.update_layout(paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                        height=480, margin=dict(l=60, r=20, t=30, b=50))
    if not fecha_str or km_val is None or pm_val is None or df_profiles.empty:
        return empty, ""
    sub = df_profiles[(df_profiles["fecha"] == pd.Timestamp(fecha_str)) &
                      (df_profiles["km"] == km_val) &
                      (df_profiles["+m"] == pm_val)].sort_values("depth", ascending=False)
    if sub.empty:
        return empty, ""
    color = KM_COLORS.get(km_val, COLOR_ACCENT)
    ssc_4   = sub[sub["depth"] <= 4]["ssc"].mean()
    ssc_7   = sub[sub["depth"] <= 7]["ssc"].mean()
    ssc_tot = sub["ssc"].mean()
    fecha_l = pd.Timestamp(fecha_str).strftime("%d/%m/%Y")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["ssc"], y=sub["depth"], mode="lines+markers",
                             line=dict(color=color, width=2.5),
                             marker=dict(size=6, color=color, line=dict(width=1, color="white")),
                             hovertemplate="Prof: %{y:.2f} m<br>SSC: %{x:.1f} mg/L<extra></extra>"))
    for d_ref, dash_ref in [(4, "dash"), (7, "dot")]:
        fig.add_shape(type="line", x0=sub["ssc"].min()*0.95, x1=sub["ssc"].max()*1.05,
                      y0=d_ref, y1=d_ref, line=dict(color="gray", width=1, dash=dash_ref))
        fig.add_annotation(x=sub["ssc"].max()*1.04, y=d_ref, text=f"{d_ref} m",
                           showarrow=False, font=dict(size=10, color="gray"), xanchor="right")
    fig.update_layout(height=480, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                      font=dict(family=FONT_BODY, size=12, color=COLOR_TEXT),
                      xaxis=dict(title="SSC (mg/L)", showgrid=True, gridcolor=COLOR_BORDER),
                      yaxis=dict(title="Profundidad (m)", autorange="reversed", showgrid=True, gridcolor=COLOR_BORDER),
                      margin=dict(l=60, r=30, t=40, b=50), hovermode="y unified",
                      title=dict(text=f"Perfil SSC — Km {km_val}, +{pm_val} m | {fecha_l}",
                                 font=dict(family=FONT_TITLE, size=14, color=COLOR_TEXT), x=0.5))
    stats = html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
        html.Div([html.Div("Promedio total", style={"fontSize":"11px","color":COLOR_MUTED,"textTransform":"uppercase","letterSpacing":"0.05em"}),
                  html.Div(f"{ssc_tot:.1f} mg/L", style={"fontSize":"18px","fontWeight":"700","color":COLOR_ACCENT,"fontFamily":FONT_TITLE})],
                 style={**CARD, "padding":"12px 20px", "marginBottom":"0"}),
        html.Div([html.Div("Promedio 0–4 m", style={"fontSize":"11px","color":COLOR_MUTED,"textTransform":"uppercase","letterSpacing":"0.05em"}),
                  html.Div(f"{ssc_4:.1f} mg/L", style={"fontSize":"18px","fontWeight":"700","color":color,"fontFamily":FONT_TITLE})],
                 style={**CARD, "padding":"12px 20px", "marginBottom":"0"}),
        html.Div([html.Div("Promedio 0–7 m", style={"fontSize":"11px","color":COLOR_MUTED,"textTransform":"uppercase","letterSpacing":"0.05em"}),
                  html.Div(f"{ssc_7:.1f} mg/L", style={"fontSize":"18px","fontWeight":"700","color":color,"fontFamily":FONT_TITLE})],
                 style={**CARD, "padding":"12px 20px", "marginBottom":"0"}),
        html.Div([html.Div("N mediciones", style={"fontSize":"11px","color":COLOR_MUTED,"textTransform":"uppercase","letterSpacing":"0.05em"}),
                  html.Div(str(len(sub)), style={"fontSize":"18px","fontWeight":"700","color":COLOR_MUTED,"fontFamily":FONT_TITLE})],
                 style={**CARD, "padding":"12px 20px", "marginBottom":"0"}),
    ])
    return fig, stats

# ── Hidrología: contenido según sub-tab y slider ──
@app.callback(
    Output("hydro-content", "children"),
    Input("hydro-tabs", "value"),
    Input("hydro-year-slider", "value"),
)
def update_hydro(subtab, year_range):
    y0, y1 = year_range
    date0 = pd.Timestamp(f"{y0}-01-01")
    date1 = pd.Timestamp(f"{y1}-12-31")

    def filter_df(d, col="Fecha"):
        return d[(d[col] >= date0) & (d[col] <= date1)]

    empty_fig = go.Figure()
    empty_fig.update_layout(paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                             height=420, margin=dict(l=50,r=20,t=30,b=50),
                             annotations=[dict(text="Sin datos para el período seleccionado",
                                               xref="paper", yref="paper", x=0.5, y=0.5,
                                               showarrow=False, font=dict(size=14, color=COLOR_MUTED))])

    # ── Serie de tiempo ──
    if subtab == "hydro-ts":
        Qf   = filter_df(Q_cal)
        TSSf = filter_df(TSS_cal)
        QGf  = filter_df(Q_baq)
        merged = Qf.merge(TSSf, on="Fecha", how="inner").dropna(subset=["Q_calamar","TSS_calamar"])
        merged ['ssc_derived'] = ((merged["TSS_calamar"]*(1000000/86400)) / merged["Q_calamar"])*(1000000/1000)  # mg/L
        merged ['Q_sinincora'] = merged['Q_calamar']-(0.080649*merged['Q_calamar']- 126.19)
        print("Qf vacío:", Qf.empty)
        print("TSSf vacío:", TSSf.empty)
        print("merged vacío:", merged.empty)
        print("merged filas:", len(merged))
        
        fig  = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                             subplot_titles=["Caudal (m³/s)", "TSS Calamar (Kt/día)", "SSC Derivado (mg/L)"])
        if not Qf.empty:
            fig.add_trace(go.Scatter(x=Qf["Fecha"], y=Qf["Q_calamar"], mode="lines",
                                     name="Q Calamar", line=dict(color=COLOR_ACCENT, width=1.5)), row=1,col=1)
            
        if not QGf.empty:
            fig.add_trace(go.Scatter(x=QGf["Fecha"], y=QGf["Q_barranquilla"], mode="markers+lines",
                                     name="Q Barranquilla", line=dict(color="#e07b2a", width=2),
                                     marker=dict(size=7)), row=1,col=1)
            fig.add_trace(go.Scatter(x=merged["Fecha"], y=merged["Q_sinincora"], mode="lines",
                            name="Q calamar - Q Incora derivado", 
                            line=dict(color="red", width=2.5)),  # ← gordo y rojo
                row=1, col=1)
        if not TSSf.empty:
            fig.add_trace(go.Scatter(x=merged["Fecha"], y=TSSf["TSS_calamar"], mode="lines",
                                     name="TSS Calamar", line=dict(color="#c0392b", width=1.5)), row=2,col=1)
        if not merged.empty:
            fig.add_trace(go.Scatter(x=merged["Fecha"], y=merged["ssc_derived"], mode="lines",
                                     name="SSC Derivado", line=dict(color="#4bb929", width=1.5)), row=3,col=1)
    
        fig.update_layout(height=520, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                          font=dict(family=FONT_BODY, size=12, color=COLOR_TEXT),
                          legend=dict(orientation="h", y=-0.08),
                          margin=dict(l=60,r=20,t=40,b=50))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor=COLOR_BORDER)

        # Estadísticas rápidas
        stats_rows = []
        for label, ser, unit in [
            ("Q Calamar",      Qf["Q_calamar"]       if not Qf.empty   else pd.Series(dtype=float), "m³/s"),
            ("TSS Calamar",    TSSf["TSS_calamar"]    if not TSSf.empty else pd.Series(dtype=float), "Kt/día"),
            ("Q Barranquilla", QGf["Q_barranquilla"]  if not QGf.empty  else pd.Series(dtype=float), "m³/s"),
            ("SSC Derivado",   merged["ssc_derived"] if not merged.empty else pd.Series(dtype=float), "mg/L"),
        ]:
            if ser.empty or ser.isna().all():
                continue
            stats_rows.append(html.Div([
                html.Div(label, style={"fontSize":"11px","color":COLOR_MUTED,
                                       "textTransform":"uppercase","letterSpacing":"0.05em"}),
                html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap","marginTop":"4px"}, children=[
                    html.Span(f"Media: {ser.mean():.1f} {unit}",
                              style={"fontSize":"13px","color":COLOR_TEXT}),
                    html.Span(f"Mín: {ser.min():.1f}",
                              style={"fontSize":"13px","color":COLOR_MUTED}),
                    html.Span(f"Máx: {ser.max():.1f}",
                              style={"fontSize":"13px","color":COLOR_MUTED}),
                    html.Span(f"n={len(ser.dropna())}",
                              style={"fontSize":"13px","color":COLOR_MUTED}),
                ]),
            ], style={**CARD, "padding":"12px 20px","marginBottom":"8px"}))

        return html.Div([dcc.Graph(figure=fig, config={"displayModeBar":False}),
                         html.Div(stats_rows, style={"marginTop":"16px"})])

    # ── Estacionalidad ──
    elif subtab == "hydro-seas":
        Qf   = filter_df(Q_cal)
        TSSf = filter_df(TSS_cal)
        if Qf.empty and TSSf.empty:
            return dcc.Graph(figure=empty_fig, config={"displayModeBar":False})
        Qf   = Qf.copy();   Qf["mes"]   = Qf["Fecha"].dt.month
        TSSf = TSSf.copy(); TSSf["mes"] = TSSf["Fecha"].dt.month
        meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Caudal Calamar (m³/s)", "TSS Calamar (Kt/día)"])
        for mes_num in range(1, 13):
            Qm   = Qf[Qf["mes"]==mes_num]["Q_calamar"]
            TSSm = TSSf[TSSf["mes"]==mes_num]["TSS_calamar"]
            if not Qm.empty:
                fig.add_trace(go.Box(y=Qm, name=meses[mes_num-1], marker_color=COLOR_ACCENT,
                                     showlegend=False), row=1,col=1)
            if not TSSm.empty:
                fig.add_trace(go.Box(y=TSSm, name=meses[mes_num-1], marker_color="#c0392b",
                                     showlegend=False), row=1,col=2)
        fig.update_layout(height=440, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                          font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
                          margin=dict(l=60,r=20,t=40,b=50))
        fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor=COLOR_BORDER)
        return dcc.Graph(figure=fig, config={"displayModeBar":False})

    # ── Q vs TSS Calamar ──
    elif subtab == "hydro-qtss":
        merged = filter_df(df_hydro).dropna(subset=["Q_calamar","TSS_calamar"])
        if merged.empty:
            return dcc.Graph(figure=empty_fig, config={"displayModeBar":False})
        x = merged["Q_calamar"]; y = merged["TSS_calamar"]
        r, p = pearsonr(x, y); r2 = r**2
        m_coef, b_coef = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                                 marker=dict(size=5, color=COLOR_ACCENT, opacity=0.5,
                                             line=dict(width=0)),
                                 name="Observaciones",
                                 hovertemplate="Q: %{x:.0f} m³/s<br>TSS: %{y:.1f} Kt/día<extra></extra>"))
        fig.add_trace(go.Scatter(x=x_line, y=m_coef*x_line+b_coef, mode="lines",
                                 line=dict(color="#c0392b",width=2,dash="dash"), name="Regresión lineal"))
        p_text = "< 0.0001" if p < 0.0001 else f"{p:.4f}"
        fig.update_layout(height=440, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                          font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
                          xaxis_title="Q Calamar (m³/s)", yaxis_title="TSS Calamar (Kt/día)",
                          margin=dict(l=60,r=20,t=20,b=50))
        fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor=COLOR_BORDER)
        stats = html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap","marginTop":"8px"}, children=[
            html.Span(f"y = {m_coef:.4f}x + {b_coef:.2f}",
                      style={"fontFamily":"monospace","fontSize":"13px","backgroundColor":f"{COLOR_ACCENT}10",
                             "padding":"4px 10px","borderRadius":"4px","color":COLOR_TEXT}),
            html.Span(f"R² = {r2:.3f}",
                      style={"fontFamily":"monospace","fontSize":"13px","color":COLOR_ACCENT,"fontWeight":"700",
                             "backgroundColor":f"{COLOR_ACCENT}10","padding":"4px 10px","borderRadius":"4px"}),
            html.Span(f"p = {p_text}",
                      style={"fontFamily":"monospace","fontSize":"13px","backgroundColor":f"{COLOR_ACCENT}10",
                             "padding":"4px 10px","borderRadius":"4px","color":COLOR_TEXT}),
            html.Span(f"n = {len(merged)}",
                      style={"fontFamily":"monospace","fontSize":"13px","backgroundColor":f"{COLOR_BORDER}",
                             "padding":"4px 10px","borderRadius":"4px","color":COLOR_MUTED}),
        ])
        return html.Div([dcc.Graph(figure=fig, config={"displayModeBar":False}), stats])
    
        # ── Q Calamar vs Q Barranquilla ──
    elif subtab == "hydro-qq":
        Qf  = filter_df(Q_cal)
        QGf = filter_df(Q_baq)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Series superpuestas", "Scatter Q Calamar vs Q Barranquilla"])
        # Series
        if not Qf.empty:
            fig.add_trace(go.Scatter(x=Qf["Fecha"], y=Qf["Q_calamar"], mode="lines",
                                     name="Q Calamar", line=dict(color=COLOR_ACCENT,width=1.5)), row=1,col=1)
        if not QGf.empty:
            fig.add_trace(go.Scatter(x=QGf["Fecha"], y=QGf["Q_barranquilla"], mode="markers+lines",
                                     name="Q Barranquilla", line=dict(color="#e07b2a",width=2),
                                     marker=dict(size=7)), row=1,col=1)
        # Scatter — solo período coincidente
        coincident = Qf.merge(QGf, on="Fecha", how="inner")
        if not coincident.empty:
            x2 = coincident["Q_calamar"]; y2 = coincident["Q_barranquilla"]
            r2, p2 = pearsonr(x2, y2)
            m2, b2 = np.polyfit(x2, y2, 1)
            x_line2 = np.linspace(x2.min(), x2.max(), 200)
            fig.add_trace(go.Scatter(x=x2, y=y2, mode="markers",
                                     marker=dict(size=8, color=COLOR_ACCENT,
                                                 line=dict(width=1,color="white")),
                                     name="Coincidentes",
                                     hovertemplate="Q Cal: %{x:.0f}<br>Q Baq: %{y:.0f}<extra></extra>"),
                          row=1,col=2)
            fig.add_trace(go.Scatter(x=x_line2, y=m2*x_line2+b2, mode="lines",
                                     line=dict(color="#c0392b",width=2,dash="dash"),
                                     name="Regresión", showlegend=False), row=1,col=2)
            p_text2 = "< 0.0001" if p2 < 0.0001 else f"{p2:.4f}"
            b2_sign = "+" if b2 >= 0 else "-"

            fig.add_annotation(
                x=0.97, y=0.05, xref="x2 domain", yref="y2 domain",
                text=(f"y = {m2:.4f}x {b2_sign} {abs(b2):.2f}<br>"
                    f"R² = {r2**2:.3f}   p = {p_text2}   n = {len(coincident)}"),
                showarrow=False,
                font=dict(size=11, color=COLOR_ACCENT),
                bgcolor="rgba(26,107,154,0.08)",
                borderpad=6,
                align="left",
            )
        fig.update_layout(height=460, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                          font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
                          legend=dict(orientation="h",y=-0.12),
                          margin=dict(l=60,r=20,t=40,b=60))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor=COLOR_BORDER)
        fig.update_xaxes(title_text="Q Calamar (m³/s)", row=1, col=2)
        fig.update_yaxes(title_text="Q Barranquilla (m³/s)", row=1, col=2)
        return dcc.Graph(figure=fig, config={"displayModeBar":False})
    
            # ── Q Calamar - Incora vs Q Barranquilla ──
    elif subtab == "hydro-qincora":
        Qf  = filter_df(Q_cal)
        QGf = filter_df(Q_baq)

        merged = Qf.merge(QGf, on="Fecha", how="inner").dropna(subset=["Q_calamar","Q_barranquilla"])
        merged["Q_sinincora"] = merged["Q_calamar"] - (0.080649 * merged["Q_calamar"] - 126.19)
        merged = merged.dropna(subset=["Q_sinincora","Q_barranquilla"])

        if merged.empty:
            return dcc.Graph(figure=empty_fig, config={"displayModeBar":False})

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Series superpuestas", "Scatter Q Sinincora vs Q Barranquilla"])

        # ── Col 1: series ──
        if not Qf.empty:
            fig.add_trace(go.Scatter(x=Qf["Fecha"], y=Qf["Q_calamar"], mode="lines",
                                    name="Q Calamar", line=dict(color=COLOR_ACCENT, width=1.5)), row=1, col=1)
        if not QGf.empty:
            fig.add_trace(go.Scatter(x=QGf["Fecha"], y=QGf["Q_barranquilla"], mode="markers+lines",
                                    name="Q Barranquilla", line=dict(color="#e07b2a", width=2),
                                    marker=dict(size=7)), row=1, col=1)
        fig.add_trace(go.Scatter(x=merged["Fecha"], y=merged["Q_sinincora"], mode="lines",
                                name="Q Sinincora", line=dict(color="#c0392b", width=1.5)), row=1, col=1)

        # ── Col 2: scatter ──
        x2 = merged["Q_sinincora"]; y2 = merged["Q_barranquilla"]
        r, p = pearsonr(x2, y2)
        m, b = np.polyfit(x2, y2, 1)
        x_line = np.linspace(x2.min(), x2.max(), 200)

        fig.add_trace(go.Scatter(x=x2, y=y2, mode="markers",
                                marker=dict(size=5, color=COLOR_ACCENT, opacity=0.5,
                                            line=dict(width=0)),
                                name="Coincidentes",
                                hovertemplate="Q Sin: %{x:.0f} m³/s<br>Q Baq: %{y:.0f} m³/s<extra></extra>"),
                    row=1, col=2)
        fig.add_trace(go.Scatter(x=x_line, y=m*x_line+b, mode="lines",
                                line=dict(color="#c0392b", width=2, dash="dash"),
                                name="Regresión", showlegend=False), row=1, col=2)

        p_text = "< 0.0001" if p < 0.0001 else f"{p:.4f}"
        b_sign = "+" if b >= 0 else "-"

        fig.add_annotation(
            x=0.97, y=0.05, xref="x2 domain", yref="y2 domain",
            text=(f"y = {m:.4f}x {b_sign} {abs(b):.2f}<br>"
                f"R² = {r**2:.3f}   p = {p_text}   n = {len(merged)}"),
            showarrow=False,
            font=dict(size=11, color=COLOR_ACCENT),
            bgcolor="rgba(26,107,154,0.08)",
            borderpad=6,
            align="left",
        )

        fig.update_layout(height=460, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                        font=dict(family=FONT_BODY, size=12, color=COLOR_TEXT),
                        legend=dict(orientation="h", y=-0.12),
                        margin=dict(l=60, r=20, t=40, b=60))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor=COLOR_BORDER)
        fig.update_xaxes(title_text="Q Sin Inkora (m³/s)", row=1, col=2)
        fig.update_yaxes(title_text="Q Barranquilla (m³/s)", row=1, col=2)

        return dcc.Graph(figure=fig, config={"displayModeBar":False})
    
    elif subtab == "hydro-qq":
        Qf  = filter_df(Q_cal)
        QGf = filter_df(Q_baq)
        incor = merged()
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Series superpuestas", "Scatter Q Calamar vs Q Barranquilla"])
        # Series
        if not Qf.empty:
            fig.add_trace(go.Scatter(x=Qf["Fecha"], y=Qf["Q_calamar"], mode="lines",
                                     name="Q Calamar", line=dict(color=COLOR_ACCENT,width=1.5)), row=1,col=1)
        if not QGf.empty:
            fig.add_trace(go.Scatter(x=QGf["Fecha"], y=QGf["Q_barranquilla"], mode="markers+lines",
                                     name="Q Barranquilla", line=dict(color="#e07b2a",width=2),
                                     marker=dict(size=7)), row=1,col=1)
        # Scatter — solo período coincidente
        coincident = Qf.merge(QGf, on="Fecha", how="inner")
        if not coincident.empty:
            x2 = coincident["Q_calamar"]; y2 = coincident["Q_barranquilla"]
            r2, p2 = pearsonr(x2, y2)
            m2, b2 = np.polyfit(x2, y2, 1)
            x_line2 = np.linspace(x2.min(), x2.max(), 200)
            fig.add_trace(go.Scatter(x=x2, y=y2, mode="markers",
                                     marker=dict(size=8, color=COLOR_ACCENT,
                                                 line=dict(width=1,color="white")),
                                     name="Coincidentes",
                                     hovertemplate="Q Cal: %{x:.0f}<br>Q Baq: %{y:.0f}<extra></extra>"),
                          row=1,col=2)
            fig.add_trace(go.Scatter(x=x_line2, y=m2*x_line2+b2, mode="lines",
                                     line=dict(color="#c0392b",width=2,dash="dash"),
                                     name="Regresión", showlegend=False), row=1,col=2)
            p_text2 = "< 0.0001" if p2 < 0.0001 else f"{p2:.4f}"
            fig.add_annotation(x=0.97, y=0.05, xref="x2 domain", yref="y2 domain",
                               text=f"R²={r2**2:.3f}  p={p_text2}  n={len(coincident)}",
                               showarrow=False, font=dict(size=11,color=COLOR_ACCENT),
                               bgcolor="rgba(26,107,154,0.08)", borderpad=4)
        fig.update_layout(height=460, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                          font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
                          legend=dict(orientation="h",y=-0.12),
                          margin=dict(l=60,r=20,t=40,b=60))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor=COLOR_BORDER)
        fig.update_xaxes(title_text="Q Calamar (m³/s)", row=1, col=2)
        fig.update_yaxes(title_text="Q Barranquilla (m³/s)", row=1, col=2)
        return dcc.Graph(figure=fig, config={"displayModeBar":False})
 
    # ── TSS Barranquilla km17 ──
    elif subtab == "hydro-tss":
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["TSS estimado Km 19 vs TSS Calamar",
                                            "Comparación directa (período coincidente)"])
        TSSf = filter_df(TSS_cal)
        if not TSSf.empty:
            fig.add_trace(go.Scatter(x=TSSf["Fecha"], y=TSSf["TSS_calamar"], mode="lines",
                                     name="TSS Calamar", line=dict(color=COLOR_ACCENT,width=1.5)), row=1,col=1)
        if not df_tss_baq.empty:
            fig.add_trace(go.Scatter(x=df_tss_baq["Fecha"], y=df_tss_baq["TSS_barranquilla"],
                                     mode="markers", name="TSS Km 19 (est.)",
                                     marker=dict(size=9,color="#e07b2a",
                                                 line=dict(width=1,color="white")),
                                     hovertemplate="Fecha: %{x}<br>TSS Baq: %{y:.2f} Kt/día<extra></extra>"),
                          row=1,col=1)
            # Scatter comparativo si hay coincidencia
            coincident2 = df_tss_baq.merge(TSSf.rename(columns={"TSS_calamar":"TSS_cal"}),
                                            on="Fecha", how="inner")
            if not coincident2.empty:
                x2 = coincident2["TSS_cal"]; y2 = coincident2["TSS_barranquilla"]
                r2, p2 = pearsonr(x2, y2)
                m2, b2 = np.polyfit(x2, y2, 1)
                x_line2 = np.linspace(x2.min(), x2.max(), 200)
                fig.add_trace(go.Scatter(x=x2, y=y2, mode="markers",
                                        marker=dict(size=8, color=COLOR_ACCENT,
                                                    line=dict(width=1,color="white")),
                                        name="Coincidentes",
                                        hovertemplate="TSS Cal: %{x:.1f}<br>TSS Baq: %{y:.2f}<extra></extra>"),
                            row=1,col=2)
                fig.add_trace(go.Scatter(x=x_line2, y=m2*x_line2+b2, mode="lines",
                                        line=dict(color="#c0392b",width=2,dash="dash"),
                                        name="Regresión", showlegend=False), row=1,col=2)
                p_text2 = "< 0.0001" if p2 < 0.0001 else f"{p2:.4f}"
                fig.add_annotation(x=0.97, y=0.05, xref="x2 domain", yref="y2 domain",
                                text=f"R²={r2**2:.3f}  p={p_text2}  n={len(coincident2)}",
                                showarrow=False, font=dict(size=11,color=COLOR_ACCENT),
                                bgcolor="rgba(26,107,154,0.08)", borderpad=4)
            fig.update_layout(height=460, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                            font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
                            legend=dict(orientation="h",y=-0.12),
                            margin=dict(l=60,r=20,t=40,b=60))
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor=COLOR_BORDER)
            fig.update_xaxes(title_text="TSS Calamar (Kt/día)", row=1, col=2)
            fig.update_yaxes(title_text="TSS Km 19 estimado (Kt/día)", row=1, col=2)
            return dcc.Graph(figure=fig, config={"displayModeBar":False})
    
        if df_tss_baq.empty:
            return dcc.Graph(figure=empty_fig, config={"displayModeBar":False})
        fig.update_layout(height=460, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
                          font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
                          legend=dict(orientation="h",y=-0.1),
                          margin=dict(l=60,r=20,t=40,b=60))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor=COLOR_BORDER)
        fig.update_xaxes(title_text="TSS Calamar (Kt/día)", row=1,col=2)
        fig.update_yaxes(title_text="TSS Km17 estimado (Kt/día)", row=1,col=2)
        note = html.Div(
            "⚠ El TSS de Barranquilla es una estimación puntual basada en CSS superficial del Km 17 "
            "y el caudal medido en Barranquilla. Sólo coincide con las fechas de campañas de campo "
            "que tienen imagen Sentinel-2 disponible.",
            style={"fontSize":"12px","color":COLOR_MUTED,"fontStyle":"italic",
                   "marginTop":"8px","padding":"8px 16px",
                   "backgroundColor":f"{COLOR_ACCENT}08","borderRadius":"6px"}
        )
        return html.Div([dcc.Graph(figure=fig, config={"displayModeBar":False}), note])
 
    return html.Div()

# ── Ranking de correlaciones ──
@app.callback(Output("corrbar-plot","figure"),
              Input("corrbar-transform","value"),
              Input("store-df-filtered","data"))
def update_corrbar(transform, data):
    if not data: return go.Figure()
    dff = pd.read_json(io.StringIO(data), orient="split")
    cols = BANDAS + INDICES
    cols = [c for c in cols if c in dff.columns]
    if "SSC" not in dff.columns: return go.Figure()
    y_css = np.log(dff["SSC"]) if transform == "log" else dff["SSC"]
    y_label = "ln(CSS)" if transform == "log" else "CSS"

    results = []
    for c in cols:
        if dff[c].isna().all(): continue
        try:
            r, p = pearsonr(dff[c].dropna(), y_css[dff[c].notna()])
            results.append({"variable": c, "r": r, "r_abs": abs(r), "p": p})
        except Exception:
            continue

    res = pd.DataFrame(results).sort_values("r_abs", ascending=True)

    # Color: verde si positivo, rojo si negativo
    colors = [
        "#2eaa6b" if r >= 0 else "#c0392b"
        for r in res["r"]
    ]

    fig = go.Figure(go.Bar(
        x=res["r"],
        y=res["variable"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"r={r:.3f}  p={'<0.001' if p<0.001 else f'{p:.3f}'}"
              for r, p in zip(res["r"], res["p"])],
        textposition="outside",
        hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>",
    ))

    fig.add_vline(x=0, line=dict(color=COLOR_TEXT, width=1, dash="dash"))
    fig.add_vline(x=0.7,  line=dict(color="#2eaa6b", width=1, dash="dot"), opacity=0.5)
    fig.add_vline(x=-0.7, line=dict(color="#c0392b", width=1, dash="dot"), opacity=0.5)
    fig.add_vline(x=-0.7, line=dict(color="#c0392b", width=1, dash="dot"), opacity=0.5)

    fig.add_annotation(x=0.72, y=1.02, xref="x", yref="paper",
                       text="|r|=0.7", showarrow=False,
                       font=dict(size=10, color="#2eaa6b"), xanchor="left")
    fig.add_annotation(x=-0.72, y=1.02, xref="x", yref="paper",
                       text="|r|=0.7", showarrow=False,
                       font=dict(size=10, color="#c0392b"), xanchor="right")

    fig.update_layout(
        height=max(320, len(res) * 32 + 80),
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor=COLOR_BG,
        font=dict(family=FONT_BODY, size=12, color=COLOR_TEXT),
        xaxis=dict(title=f"Correlación de Pearson con {y_label}",
                   range=[-1.15, 1.15], showgrid=True, gridcolor=COLOR_BORDER,
                   zeroline=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=80, r=120, t=30, b=50),
        showlegend=False,
    )
    return fig


# ── Mapa de calor espacio-temporal ──
@app.callback(Output("heatmap-plot","figure"),
              Input("heatmap-var","value"),
              Input("store-df-filtered","data"))
def update_heatmap(var, data):
    if not data: return go.Figure()
    dff = pd.read_json(io.StringIO(data), orient="split")
    dff["reflectance_date"] = pd.to_datetime(dff["reflectance_date"])
    if var not in dff.columns: return go.Figure()

    # Pivot: filas = km, columnas = fecha, valores = media de var
    pivot = (dff.groupby(["km", dff["reflectance_date"].dt.strftime("%Y-%m-%d")])[var]
               .mean()
               .reset_index()
               .pivot(index="km", columns="reflectance_date", values=var))

    # Ordenar km de mayor a menor (desembocadura abajo)
    pivot = pivot.sort_index(ascending=False)

    # Etiquetas km con color
    y_labels = [f"Km {k}" for k in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=y_labels,
        colorscale="YlOrRd",
        colorbar=dict(title=dict(text=var, side="right"),
                      thickness=14, tickfont=dict(size=11)),
        hovertemplate="Fecha: %{x}<br>%{y}<br>" + var + ": %{z:.1f}<extra></extra>",
        xgap=2, ygap=2,
    ))

    fig.update_layout(
        height=max(280, len(pivot) * 60 + 100),
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor=COLOR_CARD,
        font=dict(family=FONT_BODY, size=12, color=COLOR_TEXT),
        xaxis=dict(title="Fecha de imagen", tickangle=-45, showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=80, r=60, t=20, b=80),
    )
    return fig


# ── Climatograma CSS ──
@app.callback(Output("climo-plot","figure"),
              Input("store-df-filtered","data"))
def update_climo(data):
    if not data: return go.Figure()
    dff = pd.read_json(io.StringIO(data), orient="split")
    dff["reflectance_date"] = pd.to_datetime(dff["reflectance_date"])
    if "SSC" not in dff.columns: return go.Figure()

    dff["mes"] = dff["reflectance_date"].dt.month
    meses_label = ["Ene","Feb","Mar","Abr","May","Jun",
                   "Jul","Ago","Sep","Oct","Nov","Dic"]

    fig = go.Figure()

    # Boxplot por mes con puntos superpuestos coloreados por km
    for mes_num in range(1, 13):
        sub = dff[dff["mes"] == mes_num]
        if sub.empty: continue
        fig.add_trace(go.Box(
            y=sub["SSC"],
            x=[meses_label[mes_num-1]] * len(sub),
            name=meses_label[mes_num-1],
            marker=dict(color=COLOR_ACCENT, opacity=0.4, size=5),
            line=dict(color=COLOR_ACCENT),
            boxmean=True,
            showlegend=False,
            hoverinfo="skip",
        ))

    # Puntos individuales coloreados por km encima
    for km in sorted(dff["km"].unique()):
        sub_km = dff[dff["km"] == km]
        fig.add_trace(go.Scatter(
            x=[meses_label[m-1] for m in sub_km["mes"]],
            y=sub_km["SSC"],
            mode="markers",
            name=f"Km {km}",
            marker=dict(size=8, color=KM_COLORS.get(km, COLOR_ACCENT),
                        line=dict(width=1, color="white"), opacity=0.85),
            hovertemplate=f"Km {km}<br>Mes: %{{x}}<br>CSS: %{{y:.1f}} mg/L<extra></extra>",
        ))

    # Línea de media mensual
    monthly_mean = (dff.groupby("mes")["SSC"].mean()
                       .reindex(range(1, 13)))
    fig.add_trace(go.Scatter(
        x=[meses_label[m-1] for m in monthly_mean.index if not pd.isna(monthly_mean[m])],
        y=[v for v in monthly_mean.values if not pd.isna(v)],
        mode="lines+markers",
        name="Media mensual",
        line=dict(color=COLOR_TEXT, width=2, dash="dash"),
        marker=dict(size=7, color=COLOR_TEXT),
        hovertemplate="Media %{x}: %{y:.1f} mg/L<extra></extra>",
    ))

    fig.update_layout(
        height=420,
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor=COLOR_BG,
        font=dict(family=FONT_BODY, size=12, color=COLOR_TEXT),
        xaxis=dict(title="Mes", categoryorder="array",
                   categoryarray=meses_label, showgrid=False),
        yaxis=dict(title="CSS (mg/L)", gridcolor=COLOR_BORDER),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=60, r=20, t=20, b=80),
        boxmode="overlay",
    )
    return fig

# Callback — solo JS, sin CSS externo
app.clientside_callback(
    """
    function() {
        const card = document.getElementById('filter-card');
        const subtitle = card.querySelector('p:first-child');  // ajusta al selector de tu section_title
        if (!card) return '';

        window.addEventListener('scroll', function() {
            if (window.scrollY > 10) {
                card.style.padding = '10px 32px';
                card.style.boxShadow = '0 2px 12px rgba(0,0,0,0.08)';
                if (subtitle) {
                    subtitle.style.maxHeight = '0';
                    subtitle.style.opacity = '0';
                    subtitle.style.overflow = 'hidden';
                    subtitle.style.transition = 'max-height 0.25s ease, opacity 0.2s ease';
                    subtitle.style.marginBottom = '0';
                }
            } else {
                card.style.padding = '20px 32px';
                card.style.boxShadow = 'none';
                if (subtitle) {
                    subtitle.style.maxHeight = '60px';
                    subtitle.style.opacity = '1';
                    subtitle.style.marginBottom = '';
                }
            }
        }, { passive: true });

        return '';
    }
    """,
    Output("filter-card", "data-scroll"),
    Input("filter-card", "id"),
)


FORMULAS = [
    {
        "tag": "Sedimentos",
        "tag_bg": f"{COLOR_ACCENT}15", "tag_color": COLOR_ACCENT,
        "title": "Transporte de sedimentos en suspensión (TSS)",
        "sub": "Conversión de concentración y caudal a flujo másico diario",
        "formula": "TSS [ton/día] = SSC [mg/L] × Q [m³/s] × 0.0864",
        "desc": "El factor 0.0864 convierte mg/L × m³/s a toneladas por día. "
                "Expresa cuántas toneladas de sedimento pasan por la sección cada día.",
    },
    {
        "tag": "Derivado",
        "tag_bg": "#E1F5EE", "tag_color": "#085041",
        "title": "SSC derivado de TSS y caudal",
        "sub": "Estimación inversa de concentración a partir de datos hidrológicos",
        "formula": "SSC [mg/L] = ( TSS [Kt/día] × 10⁶ / 86400 ) / Q [m³/s] × 10⁶ / 10³",
        "desc": "Permite estimar la concentración de sedimentos cuando se dispone de TSS "
                "en Calamar y caudal simultáneo. Sirve como referencia independiente para "
                "validar estimaciones satelitales.",
    },
    {
        "tag": "Hidrología",
        "tag_bg": "#FAEEDA", "tag_color": "#633806",
        "title": "Caudal aguas abajo del Canal del Dique",
        "sub": "Corrección por pérdida de caudal hacia el Canal del Dique",
        "formula": "Q_sinincora [m³/s] = Q_calamar − ( 0.080649 × Q_calamar − 126.19 )",
        "desc": "La estación Calamar se ubica aguas arriba de la bifurcación con el Canal del Dique — "
                "canal artificial que desvía parte del caudal hacia la bahía de Cartagena. "
                "Esta fórmula empírica descuenta ese caudal derivado, estimando el caudal real "
                "que continúa por el Magdalena hacia Barranquilla.",
    },
]

@app.callback(
    Output("formula-idx", "data"),
    Input("formula-prev", "n_clicks"),
    Input("formula-next", "n_clicks"),
    State("formula-idx", "data"),
    prevent_initial_call=True,
)
def nav_formula(prev, nxt, idx):
    triggered = ctx.triggered_id
    if triggered == "formula-prev":
        return max(0, idx - 1)
    return min(len(FORMULAS) - 1, idx + 1)


@app.callback(
    Output("formula-tag",     "children"),
    Output("formula-title",   "children"),
    Output("formula-sub",     "children"),
    Output("formula-box",     "children"),
    Output("formula-desc",    "children"),
    Output("formula-counter", "children"),
    Input("formula-idx", "data"),
)
def render_formula(idx):
    f = FORMULAS[idx]
    tag = html.Span(f["tag"], style={
        "fontSize": "11px", "fontWeight": "500",
        "padding": "3px 10px", "borderRadius": "6px",
        "background": f["tag_bg"], "color": f["tag_color"],
    })
    return tag, f["title"], f["sub"], f["formula"], f["desc"], f"{idx+1} / {len(FORMULAS)}"


BANDS_S2 = [
    {"name": "Aerosol (B1)",     "lambda": 443,  "res": "60 m", "width": 20,  "color": "#7F77DD",
     "desc": "Detección de aerosoles costeros. Útil para corrección atmosférica."},
    {"name": "Blue (B2)",        "lambda": 492,  "res": "10 m", "width": 66,  "color": "#378ADD",
     "desc": "Alta reflectancia en agua clara. Sensible a sedimentos finos en suspensión."},
    {"name": "Green (B3)",       "lambda": 560,  "res": "10 m", "width": 36,  "color": "#639922",
     "desc": "Pico de reflectancia del agua. Correlaciona bien con concentración de sedimentos."},
    {"name": "Red (B4)",         "lambda": 665,  "res": "10 m", "width": 31,  "color": "#E24B4A",
     "desc": "Fuerte absorción en agua con sedimentos. Clave para índices como NDTI."},
    {"name": "Red Edge 1 (B5)", "lambda": 704,  "res": "20 m", "width": 15,  "color": "#D85A30",
     "desc": "Transición rojo-NIR. Sensible a fitoplancton y material orgánico."},
    {"name": "Red Edge 2 (B6)", "lambda": 740,  "res": "20 m", "width": 15,  "color": "#BA7517",
     "desc": "Complementa B5 para análisis de vegetación acuática."},
    {"name": "Red Edge 3 (B7)", "lambda": 783,  "res": "20 m", "width": 20,  "color": "#854F0B",
     "desc": "Permite separar sedimentos de clorofila."},
    {"name": "NIR (B8)",         "lambda": 833,  "res": "10 m", "width": 106, "color": "#3C3489",
     "desc": "Muy alta reflectancia en agua turbia con sedimentos gruesos."},
    {"name": "Red Edge 4 (B8A)","lambda": 865,  "res": "20 m", "width": 21,  "color": "#534AB7",
     "desc": "Alta sensibilidad a SSC en aguas turbias."},
    {"name": "SWIR1 (B11)",      "lambda": 1614, "res": "20 m", "width": 91,  "color": "#0F6E56",
     "desc": "Penetra neblina leve. Discrimina humedad en sedimentos."},
    {"name": "SWIR2 (B12)",      "lambda": 2202, "res": "20 m", "width": 175, "color": "#085041",
     "desc": "Alta absorción en agua. Útil para mapeo de sedimentos costeros."},
]

MIN_NM, MAX_NM = 400, 2400

def nm_to_pct(nm):
    return (nm - MIN_NM) / (MAX_NM - MIN_NM) * 100

@app.callback(
    Output("bands-overlay", "children"),
    Input("tabs", "value")
)
def render_bands(_):
    bars = []
    for i, b in enumerate(BANDS_S2):
        left  = nm_to_pct(b["lambda"] - b["width"] / 2)
        width = max(nm_to_pct(b["width"]), 0.8)
        height = {"10 m": 100, "20 m": 76, "60 m": 52}.get(b["res"], 76)
        bars.append(
            html.Div(
                id={"type": "band-bar", "index": i},
                n_clicks=0,
                title=b["name"],
                style={
                    "position": "absolute", "left": f"{left}%",
                    "width": f"{width}%", "bottom": "24px",
                    "height": f"{height}px", "background": b["color"],
                    "opacity": "0.82", "borderRadius": "4px",
                    "cursor": "pointer", "border": "1.5px solid transparent",
                },
                children=html.Div(str(b["lambda"]),
                                  style={"position": "absolute", "top": "-18px",
                                         "left": "50%", "transform": "translateX(-50%)",
                                         "fontSize": "9px", "whiteSpace": "nowrap",
                                         "color": COLOR_MUTED, "pointerEvents": "none"})
            )
        )
    return bars


@app.callback(
    Output("band-info", "children"),
    Input({"type": "band-bar", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def show_band_info(clicks):
    if not any(clicks):
        return ""
    i = next(j for j, c in enumerate(clicks) if c and c == max(c2 for c2 in clicks if c2))
    b = BANDS_S2[i]
    res_styles = {
        "10 m":  {"bg": "#E6F1FB", "color": "#0C447C"},
        "20 m":  {"bg": "#E1F5EE", "color": "#085041"},
        "60 m":  {"bg": "#FAEEDA", "color": "#633806"},
    }
    rs = res_styles.get(b["res"], {})
    return html.Div([
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px",
                        "marginBottom": "8px"}, children=[
            html.Div(style={"width": "12px", "height": "12px", "borderRadius": "3px",
                            "background": b["color"], "flexShrink": "0"}),
            html.Span(b["name"], style={"fontSize": "14px", "fontWeight": "600",
                                        "color": COLOR_TEXT}),
            html.Span(b["res"], style={"fontSize": "11px", "fontWeight": "500",
                                       "padding": "2px 8px", "borderRadius": "6px",
                                       "background": rs["bg"], "color": rs["color"]}),
            html.Span(f"λ = {b['lambda']} nm",
                      style={"fontSize": "12px", "color": COLOR_MUTED, "marginLeft": "auto"}),
        ]),
        html.P(b["desc"], style={"fontSize": "13px", "color": COLOR_MUTED,
                                  "margin": "0", "lineHeight": "1.6"}),
    ])
    
# ─────────────────────────────────────────
# CARGA DE MODELOS PRE-ENTRENADOS (.pkl)
# ─────────────────────────────────────────
import pickle as _pickle

def load_model_results():
    """Carga results.pkl y models.pkl generados por train_models.py."""
    try:
        with open("models/results.pkl","rb") as f:
            res = _pickle.load(f)
        with open("models/models.pkl","rb") as f:
            mdls = _pickle.load(f)
        return res, mdls
    except FileNotFoundError:
        print("⚠ models/results.pkl no encontrado. Corre train_models.py primero.")
        return None, None

PKL_RESULTS, PKL_MODELS = load_model_results()
PKL_OK = PKL_RESULTS is not None

# Fallback calibration loader (still needed for map inference)
def load_calib():
    try:
        dc = pd.read_csv("puntos_finales_calamar.csv")
        dc["reflectance_date"] = pd.to_datetime(dc["reflectance_date"], errors="coerce")
        # Split estaciones vs Calamar
        dc_sta = dc[dc["km"] != "Calamar"].copy()
        dc_cal = dc[dc["km"] == "Calamar"].copy()
        dc_sta["km"] = pd.to_numeric(dc_sta["km"], errors="coerce")
        dc_sta = dc_sta.dropna(subset=["NIR","SSC"])
        return dc_sta, dc_cal
    except Exception as e:
        print("load_calib error:", e)
        return pd.DataFrame(), pd.DataFrame()

df_calib, df_calamar = load_calib()

# Pre-compute base linear model coefficients for display
_NIR_MODELS = {}
if not df_calib.empty:
    _X = df_calib[["NIR"]].values
    _y = df_calib["SSC"].values
    _lr = LinearRegression().fit(_X, _y)
    _NIR_MODELS["lineal"] = (_lr.coef_[0], _lr.intercept_,
                              r2_score(_y, _lr.predict(_X)),
                              np.sqrt(mean_squared_error(_y, _lr.predict(_X))))

# ─────────────────────────────────────────
# HELPERS ESTÉTICOS Y MODELOS LINEALES
# ─────────────────────────────────────────
ACCENT2   = "#2eaa6b"
ACCENT3   = "#e07b2a"
ACCENT4   = "#7b52ab"
LINEAR_MODEL_PALETTE = [COLOR_ACCENT, ACCENT2, ACCENT3, ACCENT4, COLOR_GREEN, COLOR_AMBER]
LINEAR_MODEL_LIMIT = 6

def _bias(y_true, y_pred):
    return float(np.mean(np.array(y_pred) - np.array(y_true)))

def _metrics_dict(y_true, y_pred, prefix=""):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    mape_val = 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return {
        f"{prefix}r2": r2_score(y_true, y_pred),
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{prefix}mape": mape_val,
        f"{prefix}bias": _bias(y_true, y_pred),
    }

def _loocv_linear_predict(X, y):
    yhat = np.zeros(len(y), dtype=float)
    loo = LeaveOneOut()
    for tr_idx, te_idx in loo.split(X):
        mdl = LinearRegression()
        mdl.fit(X[tr_idx], y[tr_idx])
        yhat[te_idx] = mdl.predict(X[te_idx])
    return yhat

def _linear_key(features):
    return "lin_" + "_".join(re.sub(r"[^a-zA-Z0-9]+", "_", f).strip("_").lower() for f in features)

def _linear_label(features):
    return "Lineal: " + " + ".join(features)

def _linear_equation(model, features):
    parts = []
    for coef, feat in zip(model.coef_, features):
        sign = "+" if coef >= 0 else "-"
        parts.append(f" {sign} {abs(coef):.4f}·{feat}")
    return f"SSC = {model.intercept_:.2f}" + "".join(parts)

def _build_linear_model_results():
    if df_calib.empty:
        return {}, {}, []

    train = df_calib.copy()
    train["km_num"] = pd.to_numeric(train["km"], errors="coerce")
    kms_train = PKL_RESULTS.get("kms_train", [14, 17, 18, 19]) if PKL_OK else [14, 17, 18, 19]
    train = train[train["km_num"].isin(kms_train)].copy()

    selection = PKL_RESULTS.get("selection_log", []) if PKL_OK else []
    combos = []
    for row in sorted(selection, key=lambda r: r.get("r2_loo", -np.inf), reverse=True):
        feats = row.get("features", [])
        if feats and all(f in train.columns for f in feats) and feats not in combos:
            combos.append(feats)
        if len(combos) >= LINEAR_MODEL_LIMIT:
            break
    if ["NIR"] not in combos and "NIR" in train.columns:
        combos.append(["NIR"])

    results = {}
    labels = {}
    colors = {}
    for i, feats in enumerate(combos):
        model_df = train.dropna(subset=["SSC"] + feats).copy()
        if len(model_df) < 3:
            continue
        X = model_df[feats].values
        y = model_df["SSC"].values
        kms = model_df["km_num"].values
        mdl = LinearRegression().fit(X, y)
        yhat_cal = mdl.predict(X)
        yhat_loo = _loocv_linear_predict(X, y)
        key = _linear_key(feats)
        result = {
            "label": _linear_label(feats),
            "features": feats,
            "equation": _linear_equation(mdl, feats),
            "y_true": y.tolist(),
            "yhat_cal": yhat_cal.tolist(),
            "yhat_loo": yhat_loo.tolist(),
            "kms": kms.tolist(),
            "n_train": len(model_df),
            **_metrics_dict(y, yhat_cal, "cal_"),
            **_metrics_dict(y, yhat_loo, "loo_"),
        }
        results[key] = result
        labels[key] = result["label"]
        colors[key] = LINEAR_MODEL_PALETTE[i % len(LINEAR_MODEL_PALETTE)]
    return results, labels, colors

LINEAR_RESULTS, MODEL_LABELS, MODEL_COLS = _build_linear_model_results()
DEFAULT_LINEAR_MODEL = next(iter(LINEAR_RESULTS), "lineal")
DEFAULT_LINEAR_RESULT = LINEAR_RESULTS.get(DEFAULT_LINEAR_MODEL, {})
LINEAR_TRAIN_N = DEFAULT_LINEAR_RESULT.get("n_train", 0)
LINEAR_SSC_MIN = min(DEFAULT_LINEAR_RESULT.get("y_true", [])) if DEFAULT_LINEAR_RESULT.get("y_true") else None
LINEAR_SSC_MAX = max(DEFAULT_LINEAR_RESULT.get("y_true", [])) if DEFAULT_LINEAR_RESULT.get("y_true") else None

def metric_chip(label, value, color="#58a6ff", sub=None):
    children = [
        html.Div(label, style={"fontSize":"10px","color":COLOR_MUTED,
                               "textTransform":"uppercase","letterSpacing":"0.08em",
                               "marginBottom":"6px","fontFamily":FONT_BODY}),
        html.Div(value, style={"fontSize":"24px","fontWeight":"700",
                               "color":color,"fontFamily":FONT_MONO,
                               "letterSpacing":"-0.02em"}),
    ]
    if sub:
        children.append(html.Div(sub, style={"fontSize":"10px","color":COLOR_MUTED,
                                              "marginTop":"4px"}))
    return html.Div(children, style={
        "background": COLOR_CARD,
        "borderRadius":"10px","padding":"16px 20px",
        "border":f"1px solid {COLOR_BORDER}","minWidth":"130px","textAlign":"center",
        "boxShadow":f"0 0 0 1px {color}18, {SHADOW_SM}",
    })

# ═══════════════════════════════════════════════════════
# PESTAÑA MODELO
# ═══════════════════════════════════════════════════════
def tab_modelo():
    return html.Div([

        # Header card
        html.Div(style={**CARD, "borderLeft":f"4px solid {COLOR_ACCENT}",
                        "background":f"linear-gradient(135deg,{COLOR_ACCENT}0d 0%,{COLOR_CARD} 55%)"},
                 children=[
            section_title("Comparación de regresiones lineales",
                          "Evaluación de combinaciones de bandas e índices con regresión lineal y validación LOOCV"),
            html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","marginTop":"8px"},
                     children=[
                metric_chip("Calibración (n)", str(LINEAR_TRAIN_N or len(df_calib)), COLOR_ACCENT),
                metric_chip("Familia", "Regresión lineal", ACCENT2),
                metric_chip("Rango SSC",
                            f"{int(LINEAR_SSC_MIN) if LINEAR_SSC_MIN is not None else '–'}–"
                            f"{int(LINEAR_SSC_MAX) if LINEAR_SSC_MAX is not None else '–'} mg/L",
                            ACCENT3),
            ]),
        ]),

        # ── Selector de modelo + características ──
        html.Div(style={**CARD}, children=[
            section_title("Configuración de la regresión"),
            html.Div(style={"display":"flex","gap":"24px","flexWrap":"wrap","alignItems":"flex-end"},
                     children=[
                html.Div([
                    html.Label("Modelo lineal:", style={"fontSize":"13px","color":COLOR_MUTED,"fontWeight":"600"}),
                    dcc.Dropdown(
                        id="model-selector",
                        options=[{"label":v,"value":k} for k,v in MODEL_LABELS.items()],
                        value=DEFAULT_LINEAR_MODEL, clearable=False,
                        style={
                                "backgroundColor": COLOR_INPUT_BG,
                                "color": COLOR_TEXT_INPUT,
                                "border": f"1px solid {COLOR_INPUT_BD}",
                                "borderRadius": "10px",
                            },
                    ),
                ]),
                html.Div([
                    html.Label("Variable Y:", style={"fontSize":"13px","color":COLOR_MUTED,"fontWeight":"600"}),
                    dcc.RadioItems(id="model-y-var",
                        options=[{"label":" SSC (mg/L)","value":"SSC"}],
                        value="SSC", inline=True, style={"fontSize":"13px"}),
                ]),
                html.Div([
                    dcc.RadioItems(id="model-include-cal",
                        options=[],
                        value="no", inline=True, style={"fontSize":"13px"}),
                ]),
            ]),
            html.Div(id="model-equation-box",
                     style={"marginTop":"16px","fontFamily":"monospace","fontSize":"14px",
                            "background":f"{COLOR_ACCENT}0d","border":f"1px solid {COLOR_ACCENT}33",
                            "borderRadius":"8px","padding":"12px 20px","color":COLOR_TEXT}),
        ]),

        # ── Métricas ──
        html.Div(id="model-metrics-row",
                 style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"24px"}),

        # ── Gráficas ──
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"20px"}, children=[
            html.Div(style={**CARD}, children=[
                section_title("SSC medido vs modelado",
                              "Línea 1:1 ideal — puntos sobre la línea = subestimación"),
                dcc.Graph(id="model-1to1", config={"displayModeBar":False}, style={"height":"400px"}),
            ]),
            html.Div(style={**CARD}, children=[
                section_title("Residuos",
                              "SSC medido − SSC modelado por observación"),
                dcc.Graph(id="model-residuals", config={"displayModeBar":False}, style={"height":"400px"}),
            ]),
        ]),

        # ── Ratio por estación ──
        html.Div(style={**CARD}, children=[
            section_title("Desempeño por estación",
                          "Ratio SSC_med / SSC_mod — rango aceptable ±20% (verde) y ±30% (amarillo)"),
            dcc.Graph(id="model-ratio-km", config={"displayModeBar":False}, style={"height":"440px"}),
        ]),

        # ── LOOCV ──
        html.Div(style={**CARD, "borderLeft":f"4px solid {ACCENT2}"}, children=[
            section_title("Validación cruzada — Leave-One-Out (LOOCV)",
                          "Cada punto es validado usando el modelo entrenado con los n−1 restantes"),
            html.Div(id="loocv-metrics-row",
                     style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"16px"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"20px"}, children=[
                dcc.Graph(id="loocv-1to1", config={"displayModeBar":False}, style={"height":"380px"}),
                dcc.Graph(id="loocv-error-dist", config={"displayModeBar":False}, style={"height":"380px"}),
            ]),
        ]),

        # ── Comparación de modelos ──
        html.Div(style={**CARD}, children=[
            section_title("Ranking de regresiones lineales",
                          "Comparación de desempeño entre combinaciones de predictores lineales"),
            dcc.Graph(id="model-comparison", config={"displayModeBar":False}, style={"height":"380px"}),
        ]),

    ])


# ═══════════════════════════════════════════════════════
# PESTAÑA APLICACIÓN
# ═══════════════════════════════════════════════════════
def tab_aplicacion():
    # Load virtual stations data if available
    try:
        _df_vs = pd.read_csv("series_estaciones_virtuales.csv")
        _df_vs["fecha"] = pd.to_datetime(_df_vs["fecha"])
        estaciones_opts = [{"label":"Todas","value":"all"}] + \
            [{"label":f"E{e}","value":e} for e in sorted(_df_vs["estacion"].unique())]
    except Exception:
        estaciones_opts = [{"label":"Sin datos","value":"all"}]

    return html.Div([

        # Header
        html.Div(style={**CARD,
                        "borderLeft":f"4px solid {ACCENT2}",
                        "background":f"linear-gradient(135deg,{ACCENT2}0d 0%,{COLOR_CARD} 55%)"},
                 children=[
            section_title("Aplicación del modelo — Estaciones virtuales",
                          "Series de SSC reconstruidas 2019–2025 · Estaciones E1–E16 · Río Magdalena km 11–19"),
            html.Div(id="app-model-badge",
                     style={"marginTop":"12px","display":"flex","alignItems":"center","gap":"10px"}),
        ]),

        # Sub-tabs aplicación
        dcc.Tabs(id="app-tabs", value="app-estmap",
                 style={"marginBottom":"20px","borderBottom":f"1px solid {COLOR_BORDER}"},
                 children=[
            dcc.Tab(label="Estaciones virtuales", value="app-estmap",  style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Serie anual",          value="app-anual",   style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Serie mensual",        value="app-mensual", style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Media móvil espacial", value="app-movil",   style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Ciclo multianual",     value="app-ciclo",   style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Mapa SSC",             value="app-mapa",    style=TAB_STYLE, selected_style=TAB_SELECTED),
        ]),

        # ── Filtros comunes ──
        html.Div(id="app-filters", style={**CARD,"padding":"18px 28px"}, children=[
            html.Div(style={"display":"flex","gap":"20px","flexWrap":"wrap","alignItems":"flex-end"}, children=[
                html.Div([
                    html.Label("Estaciones:",
                               style={"fontSize":"13px","color":COLOR_MUTED,"fontWeight":"600"}),
                    dcc.Dropdown(id="app-estaciones",
                                 options=estaciones_opts,
                                 value=["all"], multi=True,
                                 clearable=False,
                                 style={
                                    "backgroundColor": COLOR_INPUT_BG,
                                    "color": COLOR_TEXT_INPUT,
                                    "border": f"1px solid {COLOR_INPUT_BD}",
                                    "borderRadius": "10px",
                                }),
                ]),
                html.Div([
                    html.Label("Ventana media móvil espacial (nro estaciones):",
                               style={"fontSize":"13px","color":COLOR_MUTED,"fontWeight":"600"}),
                    dcc.Slider(id="app-window", min=2, max=6, step=1, value=3,
                               marks={i:str(i) for i in range(2,7)},
                               tooltip={"placement":"bottom","always_visible":True}),
                ], style={"minWidth":"260px"}),
                html.Div([
                    html.Label("Media móvil temporal (meses, 0=ninguna):",
                               style={"fontSize":"13px","color":COLOR_MUTED,"fontWeight":"600"}),
                    dcc.Slider(id="app-ma-window", min=0, max=12, step=1, value=0,
                               marks={0:"Sin MA",3:"3",6:"6",12:"12"},
                               tooltip={"placement":"bottom","always_visible":True}),
                ], style={"minWidth":"260px"}),
                html.Div([
                    html.Label("Comparar con Calamar:",
                               style={"fontSize":"13px","color":COLOR_MUTED,"fontWeight":"600"}),
                    dcc.RadioItems(id="app-show-calamar",
                        options=[{"label":" Sí","value":"si"},{"label":" No","value":"no"}],
                        value="si", inline=True, style={"fontSize":"13px"}),
                ]),
            ]),
            html.Div(style={"marginTop":"20px","borderTop":f"1px solid {COLOR_BORDER}","paddingTop":"16px"},
                     children=[
                html.Label("Intervalo de años:",
                           style={"fontSize":"13px","color":COLOR_MUTED,"fontWeight":"600",
                                  "marginBottom":"8px","display":"block"}),
                dcc.RangeSlider(
                    id="app-year-slider",
                    min=2018, max=2026, step=1,
                    value=[2019, 2026],
                    marks={y: {"label":str(y),
                               "style":{"color":COLOR_MUTED,"fontSize":"11px"}}
                           for y in range(2018, 2027)},
                    tooltip={"placement":"bottom","always_visible":True},
                ),
            ]),
        ]),

        html.Div(id="app-content"),

        # ── LAYOUT ───────────────────────────────────────────────────────────────────
        #
        # Reemplaza el bloque:
        #   html.Div(id="app-mapa-section", style={**CARD}, children=[...])
        #
        # por este:
        
        html.Div(id="app-mapa-section", style={**CARD}, children=[
        
            section_title(
                "Mapa de SSC modelada",
                "Navega entre las imágenes Sentinel-2 disponibles para visualizar la "
                "concentración de sedimentos en suspensión"
            ),
        
            # ── Controles de navegación ──────────────────────────────────────────────
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "center",
                    "marginBottom": "16px", "flexWrap": "wrap"},
                children=[
        
                    # Botón ◀ Anterior
                    html.Button(
                        "◀ Anterior",
                        id="btn-img-prev",
                        n_clicks=0,
                        style={
                            "padding": "8px 18px", "fontSize": "13px", "fontWeight": "600",
                            "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}",
                            "background": COLOR_INPUT_BG, "color": COLOR_TEXT,
                            "cursor": "pointer",
                        },
                    ),
        
                    # Etiqueta de fecha activa
                    html.Div(
                        id="mapa-fecha-label",
                        style={
                            "flex": "1", "textAlign": "center", "fontSize": "15px",
                            "fontWeight": "700", "color": COLOR_ACCENT,
                            "minWidth": "160px",
                        },
                        children="—",
                    ),
        
                    # Botón Siguiente ▶
                    html.Button(
                        "Siguiente ▶",
                        id="btn-img-next",
                        n_clicks=0,
                        style={
                            "padding": "8px 18px", "fontSize": "13px", "fontWeight": "600",
                            "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}",
                            "background": COLOR_INPUT_BG, "color": COLOR_TEXT,
                            "cursor": "pointer",
                        },
                    ),
        
                    # Contador  "3 / 8"
                    html.Div(
                        id="mapa-img-counter",
                        style={"fontSize": "12px", "color": COLOR_MUTED, "minWidth": "50px",
                            "textAlign": "right"},
                        children="",
                    ),
                ],
            ),
        
            # ── Store: índice de imagen actual ───────────────────────────────────────
            dcc.Store(id="store-img-index", data=0),
        
            # ── Disclaimer ───────────────────────────────────────────────────────────
            html.Div(
                style={
                    "backgroundColor": f"{COLOR_ACCENT}0d",
                    "border": f"1px solid {COLOR_ACCENT}33",
                    "borderRadius": "8px", "padding": "12px 18px",
                    "marginBottom": "16px", "display": "flex",
                    "gap": "10px", "alignItems": "flex-start",
                },
                children=[
                    html.Span("💡", style={"fontSize": "16px", "flexShrink": "0",
                                        "marginTop": "1px"}),
                    html.Div([
                        html.Span("Imágenes disponibles: ",
                                style={"fontWeight": "600", "color": COLOR_ACCENT,
                                        "fontSize": "13px"}),
                        html.Span(
                            f"Las imágenes Sentinel-2 están en la carpeta "
                            f"'{CARPETA_IMAGENES}/'. Corresponden al tramo Km 11–19 "
                            f"del río Magdalena. Usa los botones para navegar entre fechas.",
                            style={"fontSize": "13px", "color": COLOR_TEXT,
                                "lineHeight": "1.6"},
                        ),
                    ]),
                ],
            ),
        
            # ── Barra de estado ──────────────────────────────────────────────────────
            html.Div(
                id="mapa-output",
                style={
                    "minHeight": "44px", "borderRadius": "8px", "padding": "10px 14px",
                    "background": COLOR_BG, "border": f"1px solid {COLOR_BORDER}",
                    "fontSize": "13px", "color": COLOR_MUTED, "marginBottom": "16px",
                },
                children="Selecciona una imagen con los botones para generar el mapa de SSC.",
            ),
        
            # ── Mapa ─────────────────────────────────────────────────────────────────
            dcc.Graph(
                id="mapa-ssc",
                config={"displayModeBar": True},
                style={"height": "560px"},
            ),
        ])

    ])


# ═══════════════════════════════════════════════════════
# CALLBACKS — MODELO
# ═══════════════════════════════════════════════════════

def _build_model(model_name, X_tr, y_tr):
    """Fit and return a simple linear model for NIR map inference."""
    m = LinearRegression()
    m.fit(X_tr, y_tr)
    return m, None

def _mape(y, yhat):
    mask = np.array(y) != 0
    return 100 * np.mean(np.abs((np.array(y)[mask] - np.array(yhat)[mask]) / np.array(y)[mask]))


@app.callback(
    Output("model-equation-box","children"),
    Output("model-metrics-row","children"),
    Output("model-1to1","figure"),
    Output("model-residuals","figure"),
    Output("model-ratio-km","figure"),
    Output("loocv-metrics-row","children"),
    Output("loocv-1to1","figure"),
    Output("loocv-error-dist","figure"),
    Output("model-comparison","figure"),
    Input("model-selector","value"),
    Input("model-y-var","value"),
    Input("model-include-cal","value"),
)
def update_model(model_name, y_var, include_cal):
    # ── Layout helpers ──
    _layout = dict(
        paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
        font=dict(family=FONT_BODY, size=12, color=COLOR_TEXT),
        xaxis=dict(showgrid=True, gridcolor=COLOR_BORDER, zeroline=False,
                   linecolor=COLOR_BORDER, tickfont=dict(color=COLOR_MUTED)),
        yaxis=dict(showgrid=True, gridcolor=COLOR_BORDER, zeroline=False,
                   linecolor=COLOR_BORDER, tickfont=dict(color=COLOR_MUTED)),
        legend=dict(orientation="h", y=-0.22, font=dict(color=COLOR_TEXT),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=56,r=20,t=30,b=70),
    )

    empty = go.Figure().update_layout(**_layout, height=380,
        annotations=[dict(text="Sin modelos lineales disponibles",
                          x=0.5,y=0.5,xref="paper",yref="paper",
                          showarrow=False,font=dict(size=14,color=COLOR_MUTED))])

    if not LINEAR_RESULTS:
        return ("Sin modelos lineales disponibles",
                [], empty, empty, empty, [], empty, empty, empty)

    model_name = model_name if model_name in LINEAR_RESULTS else DEFAULT_LINEAR_MODEL
    res   = LINEAR_RESULTS.get(model_name)
    if res is None:
        return ("Modelo lineal no disponible", [], empty, empty, empty, [], empty, empty, empty)

    col   = MODEL_COLS.get(model_name, COLOR_ACCENT2)
    y_arr = np.array(res["y_true"])
    yhat  = np.array(res["yhat_cal"])
    yloo  = np.array(res["yhat_loo"])
    kms   = np.array(res["kms"])

    # ── Equation ──
    eq_text = html.Span([
        html.Span("f(x)  =  ", style={"color":COLOR_MUTED,"fontSize":"13px"}),
        html.Span(res["equation"], style={"color":COLOR_ACCENT2,"fontFamily":FONT_MONO,"fontSize":"14px"}),
    ])

    # ── Calibration metrics ──
    metrics = [
        metric_chip("R² Cal",   f"{res['cal_r2']:.3f}",  col),
        metric_chip("RMSE Cal", f"{res['cal_rmse']:.1f}", col, "mg/L"),
        metric_chip("MAPE Cal", f"{res['cal_mape']:.1f}", col, "%"),
        metric_chip("Bias",     f"{res['cal_bias']:+.1f}",col, "mg/L"),
        metric_chip("n",        str(res["n_train"]), COLOR_MUTED),
    ]

    # ── 1:1 scatter ──
    ssc_min = min(y_arr.min(), yhat.min()) * 0.95
    ssc_max = max(y_arr.max(), yhat.max()) * 1.05
    fig_11  = go.Figure()
    fig_11.add_trace(go.Scatter(x=[ssc_min,ssc_max], y=[ssc_min,ssc_max],
        mode="lines", line=dict(color=COLOR_MUTED,dash="dash",width=1.5),
        name="1:1", showlegend=True))
    for km in sorted(set(kms)):
        mask = kms == km
        kc   = KM_COLORS.get(int(km), COLOR_ACCENT)
        fig_11.add_trace(go.Scatter(
            x=y_arr[mask], y=yhat[mask], mode="markers",
            name=f"Km {int(km)}",
            marker=dict(size=10, color=kc, line=dict(width=1.5,color=COLOR_CARD),
                        opacity=0.9),
            hovertemplate="SSC_med: %{x:.1f}<br>SSC_mod: %{y:.1f}<extra></extra>"))
    fig_11.update_layout(**_layout, height=400,
        xaxis_title="SSC medido (mg/L)", yaxis_title="SSC modelado (mg/L)")

    # ── Residuals ──
    res_arr = y_arr - yhat
    fig_res = go.Figure()
    for km in sorted(set(kms)):
        mask = kms == km
        kc   = KM_COLORS.get(int(km), COLOR_ACCENT)
        fig_res.add_trace(go.Bar(
            x=np.where(mask)[0].tolist(), y=res_arr[mask].tolist(),
            name=f"Km {int(km)}", marker_color=kc, opacity=0.85))
    fig_res.add_hline(y=0, line=dict(color=COLOR_MUTED,dash="dash",width=1.5))
    fig_res.update_layout(**_layout, height=400, barmode="overlay",
        xaxis_title="Observación", yaxis_title="Residuo SSC_med − SSC_mod (mg/L)")

    # ── Ratio by station ──
    ratio  = y_arr / np.where(yhat==0, np.nan, yhat)
    km_lbl = [f"Km {int(k)}" for k in kms]
    uniq   = sorted(set(km_lbl), key=lambda x: int(x.split()[1]))
    fig_ratio = go.Figure()
    fig_ratio.add_hrect(y0=0.8,y1=1.2, fillcolor="rgba(46,170,107,0.08)",line_width=0)
    fig_ratio.add_hrect(y0=0.7,y1=1.3, fillcolor="rgba(230,180,0,0.06)",line_width=0)
    fig_ratio.add_hline(y=1.0,line=dict(color=COLOR_MUTED,dash="dash",width=1.5))
    for kl in uniq:
        mask  = np.array(km_lbl)==kl
        rv    = ratio[mask]
        kv    = kms[mask][0]
        kc    = KM_COLORS.get(int(kv), COLOR_ACCENT)
        n     = mask.sum()
        med   = np.median(rv)
        p25,p75 = np.percentile(rv,25), np.percentile(rv,75)
        fig_ratio.add_trace(go.Scatter(x=[kl]*n, y=rv.tolist(), mode="markers",
            marker=dict(size=9,color="#4c8bc7",opacity=0.55,
                        line=dict(width=0)), showlegend=False,
            hovertemplate=f"{kl}<br>Ratio: %{{y:.2f}}<extra></extra>"))
        fig_ratio.add_trace(go.Scatter(x=[kl,kl], y=[p25,p75], mode="lines",
            line=dict(color=kc,width=6), showlegend=False))
        fig_ratio.add_trace(go.Scatter(x=[kl], y=[med], mode="markers",
            marker=dict(size=14,color=kc,symbol="circle",
                        line=dict(width=2,color=COLOR_CARD)),
            showlegend=False,
            hovertemplate=f"{kl}<br>n={n}<br>Med: {med:.2f}<br>IQR: {p25:.2f}–{p75:.2f}<extra></extra>"))
        fig_ratio.add_annotation(x=kl,y=p75+0.05,text=f"n={n}",
            showarrow=False,font=dict(size=9,color=COLOR_MUTED))
        layout_ratio = _layout.copy()

        layout_ratio["yaxis"] = dict(
            title="SSC_med / SSC_mod",
            gridcolor=COLOR_BORDER,
            range=[0.4, max(ratio[~np.isnan(ratio)]) * 1.15]
        )

        fig_ratio.update_layout(
            **layout_ratio,
            height=440,
            xaxis_title="Estación"
        )

    # ── LOOCV metrics ──
    loocv_metrics = [
        metric_chip("R² LOOCV",   f"{res['loo_r2']:.3f}",   COLOR_GREEN),
        metric_chip("RMSE LOOCV", f"{res['loo_rmse']:.1f}",  COLOR_GREEN, "mg/L"),
        metric_chip("MAPE LOOCV", f"{res['loo_mape']:.1f}",  COLOR_GREEN, "%"),
        metric_chip("Bias LOOCV", f"{res['loo_bias']:+.1f}", COLOR_GREEN, "mg/L"),
    ]

    # ── LOOCV 1:1 ──
    fig_loo = go.Figure()
    fig_loo.add_trace(go.Scatter(x=[ssc_min,ssc_max],y=[ssc_min,ssc_max],
        mode="lines",line=dict(color=COLOR_MUTED,dash="dash",width=1.5),showlegend=False))
    fig_loo.add_trace(go.Scatter(x=y_arr.tolist(),y=yloo.tolist(),mode="markers",
        marker=dict(size=10,color=COLOR_GREEN,line=dict(width=1.5,color=COLOR_CARD),opacity=0.9),
        name="LOOCV",
        hovertemplate="SSC_med: %{x:.1f}<br>SSC_loo: %{y:.1f}<extra></extra>"))
    fig_loo.update_layout(**_layout,height=380,
        xaxis_title="SSC medido (mg/L)",yaxis_title="SSC predicho LOOCV (mg/L)")

    # ── Error distribution ──
    errors = y_arr - yloo
    fig_err = go.Figure()
    fig_err.add_trace(go.Histogram(x=errors.tolist(),nbinsx=12,
        marker=dict(color=COLOR_GREEN,opacity=0.75,
                    line=dict(color=COLOR_BORDER,width=1)),
        name="Error LOOCV"))
    fig_err.add_vline(x=0,line=dict(color=COLOR_MUTED,dash="dash",width=1.5))
    fig_err.add_vline(x=float(np.mean(errors)),
        line=dict(color=COLOR_GREEN,dash="dot",width=1.5),
        annotation_text=f"μ={np.mean(errors):.1f}",
        annotation_position="top right",
        annotation_font=dict(color=COLOR_GREEN,size=11))
    fig_err.update_layout(**_layout,height=380,
        xaxis_title="Error SSC_med − SSC_loo (mg/L)",yaxis_title="Frecuencia")

    # ── Linear model comparison ──
    fig_comp = go.Figure()
    for key, lin_res in LINEAR_RESULTS.items():
        lbl = lin_res["label"].replace("Lineal: ", "")
        mc = MODEL_COLS.get(key, COLOR_ACCENT2)
        fig_comp.add_trace(go.Bar(
            name=lbl,
            x=["R²_cal","R²_loo","RMSE_cal/500","RMSE_loo/500"],
            y=[lin_res["cal_r2"],
               lin_res["loo_r2"],
               lin_res["cal_rmse"]/500,
               lin_res["loo_rmse"]/500],
            marker=dict(color=mc,opacity=0.85,
                        line=dict(color=COLOR_BORDER,width=1)),
            hovertemplate=(f"<b>{lin_res['label']}</b><br>"
                           f"Predictores: {' + '.join(lin_res['features'])}<br>"
                           f"R²_cal: {lin_res['cal_r2']:.3f}<br>"
                           f"R²_loo: {lin_res['loo_r2']:.3f}<br>"
                           f"RMSE_loo: {lin_res['loo_rmse']:.1f} mg/L<extra></extra>")))
    layout_comp = _layout.copy()

    layout_comp["xaxis"] = dict(
        ticktext=["R² Cal","R² LOOCV","RMSE Cal (÷500)","RMSE LOOCV (÷500)"],
        tickvals=["R²_cal","R²_loo","RMSE_cal/500","RMSE_loo/500"],
        showgrid=False
    )

    fig_comp.update_layout(
        **layout_comp,
        barmode="group",
        height=380,
        yaxis_title="Valor (RMSE normalizado)"
    )

    return eq_text, metrics, fig_11, fig_res, fig_ratio, loocv_metrics, fig_loo, fig_err, fig_comp



# ═══════════════════════════════════════════════════════
# CALLBACKS — APLICACIÓN
# ═══════════════════════════════════════════════════════

def _load_vs():
    """Load virtual stations CSV."""
    try:
        d = pd.read_csv("series_estaciones_virtuales.csv")
        d["fecha"] = pd.to_datetime(d["fecha"])
        return d
    except Exception:
        return pd.DataFrame()

def _ssc_from_nir(nir_arr, model_name="lineal"):
    """Apply linear model SSC = m*NIR + b."""
    if df_calib.empty:
        return nir_arr * 0
    X = df_calib[["NIR"]].values; y = df_calib["SSC"].values
    mdl, sc = _build_model(model_name, X, y)
    flat = nir_arr.flatten().reshape(-1,1)
    flat_p = sc.transform(flat) if sc else flat
    return mdl.predict(flat_p).reshape(nir_arr.shape)


@app.callback(Output("app-content","children"),
              Input("app-tabs","value"),
              Input("app-estaciones","value"),
              Input("app-window","value"),
              Input("app-show-calamar","value"),
              Input("app-year-slider","value"),
              Input("app-ma-window","value"))
def update_app(subtab, estaciones, window, show_cal, year_range, ma_window):
    df_vs = _load_vs()
    empty_note = html.Div("Archivo series_estaciones_virtuales.csv no encontrado.",
                          style={"color":COLOR_MUTED,"fontSize":"13px","padding":"20px"})
    if df_vs.empty:
        return empty_note

    # Filter estaciones
    all_est = sorted(df_vs["estacion"].unique())
    if "all" in (estaciones or []) or not estaciones:
        sel_est = all_est
    else:
        sel_est = [e for e in estaciones if e != "all"]

    # Year filter
    y0, y1 = (year_range or [2018, 2026])
    df_vs = df_vs[(df_vs["fecha"].dt.year >= y0) & (df_vs["fecha"].dt.year <= y1)]
    df_f  = df_vs[df_vs["estacion"].isin(sel_est)]

    # Calamar from hydro — also filtered by year range
    calamar_ssc = None
    if show_cal == "si":
        try:
            mg_c = Q_cal.merge(TSS_cal, on="Fecha", how="inner").dropna()
            mg_c["ssc_cal"] = ((mg_c["TSS_calamar"]*(1e6/86400)) / mg_c["Q_calamar"]) * (1e6/1e3)
            mg_c = mg_c[(mg_c["Fecha"].dt.year >= y0) & (mg_c["Fecha"].dt.year <= y1)]
            mg_c["fecha"]   = mg_c["Fecha"].dt.to_period("M").dt.to_timestamp()
            calamar_ssc = mg_c.groupby("fecha")["ssc_cal"].mean().reset_index()
        except Exception:
            calamar_ssc = None

    ma_win = int(ma_window) if ma_window and int(ma_window) > 0 else 0

    colors_e = px.colors.qualitative.Set2
    def ecol(est): return colors_e[all_est.index(est) % len(colors_e)]

    # ── A: Serie anual ──
    if subtab == "app-anual":
        df_yr = df_f.copy()
        df_yr["year"] = df_yr["fecha"].dt.year
        annual = df_yr.groupby("year")["SSC"].agg(["mean","std"]).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(annual["year"])+list(annual["year"][::-1]),
            y=list(annual["mean"]+annual["std"])+list((annual["mean"]-annual["std"])[::-1]),
            fill="toself", fillcolor="rgba(26,107,154,0.12)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=annual["year"], y=annual["mean"], mode="lines+markers",
            name="Estaciones virtuales (media ± SD)",
            line=dict(color=COLOR_ACCENT,width=2.5),
            marker=dict(size=9,color=COLOR_ACCENT),
            hovertemplate="Año: %{x}<br>SSC: %{y:.1f} mg/L<extra></extra>"))
        # Trend
        if len(annual) > 2:
            m_t, b_t = np.polyfit(annual["year"], annual["mean"], 1)
            xl_t = np.array([annual["year"].min(), annual["year"].max()])
            fig.add_trace(go.Scatter(x=xl_t, y=m_t*xl_t+b_t, mode="lines",
                line=dict(color=COLOR_ACCENT,dash="dash",width=1.5),
                name=f"Tendencia ({m_t:+.1f} mg/L/año)"))
        if calamar_ssc is not None and len(calamar_ssc) > 0:
            cal_yr = calamar_ssc.copy()
            cal_yr["year"] = cal_yr["fecha"].dt.year
            cal_a = cal_yr.groupby("year")["ssc_cal"].mean().reset_index()
            fig.add_trace(go.Scatter(x=cal_a["year"], y=cal_a["ssc_cal"],
                mode="lines+markers", name="Calamar (IDEAM)",
                line=dict(color="#c0392b",width=2.5), marker=dict(size=9,symbol="square")))
            if len(cal_a) > 2:
                mc, bc = np.polyfit(cal_a["year"], cal_a["ssc_cal"], 1)
                xl_c = np.array([cal_a["year"].min(), cal_a["year"].max()])
                fig.add_trace(go.Scatter(x=xl_c, y=mc*xl_c+bc, mode="lines",
                    line=dict(color="#c0392b",dash="dash",width=1.5),
                    name=f"Tendencia Calamar ({mc:+.1f} mg/L/año)"))
        fig.update_layout(height=440, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
            title=dict(text=f"Serie anual de SSC  ({y0}–{y1})",
                       font=dict(color=COLOR_TEXT,size=14)),
            xaxis=dict(title="Año", showgrid=True, gridcolor=COLOR_BORDER,
                       zeroline=False, linecolor=COLOR_BORDER, tickfont=dict(color=COLOR_MUTED)),
            yaxis=dict(title="SSC (mg/L)", gridcolor=COLOR_BORDER,
                       zeroline=False, linecolor=COLOR_BORDER, tickfont=dict(color=COLOR_MUTED)),
            font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
            legend=dict(orientation="h",y=-0.2,bgcolor="rgba(0,0,0,0)",
                        font=dict(color=COLOR_TEXT)),
            margin=dict(l=60,r=20,t=50,b=80))
        return html.Div(style={**CARD}, children=[
            section_title("Serie anual de SSC","Media anual ± desviación estándar"),
            dcc.Graph(figure=fig, config={"displayModeBar":False})])

    # ── B: Serie mensual ──
    elif subtab == "app-mensual":
        df_m = df_f.copy()
        df_m["mes"] = df_m["fecha"].dt.to_period("M").dt.to_timestamp()
        monthly = df_m.groupby(["estacion","mes"])["SSC"].mean().reset_index()

        fig = go.Figure()
        for est in sel_est:
            sub = monthly[monthly["estacion"]==est].sort_values("mes")
            y_vals = sub["SSC"]
            if ma_win > 1:
                y_vals_raw = y_vals.values
                y_ma = pd.Series(y_vals_raw).rolling(window=ma_win, center=True, min_periods=1).mean()
                # Plot raw as faded
                fig.add_trace(go.Scatter(
                    x=sub["mes"], y=y_vals_raw, mode="lines",
                    name=f"E{est} (bruto)", line=dict(color=ecol(est), width=1, dash="dot"),
                    opacity=0.35, showlegend=False,
                    hovertemplate=f"E{est} bruto<br>%{{x|%Y-%m}}<br>SSC: %{{y:.1f}} mg/L<extra></extra>"))
                fig.add_trace(go.Scatter(
                    x=sub["mes"], y=y_ma.values, mode="lines+markers",
                    name=f"E{est} (MA {ma_win}m)", line=dict(color=ecol(est),width=2.5),
                    marker=dict(size=5,color=ecol(est)),
                    hovertemplate=f"E{est} MA{ma_win}m<br>%{{x|%Y-%m}}<br>SSC: %{{y:.1f}} mg/L<extra></extra>"))
            else:
                fig.add_trace(go.Scatter(
                    x=sub["mes"], y=y_vals, mode="lines+markers",
                    name=f"E{est}", line=dict(color=ecol(est),width=2),
                    marker=dict(size=6,color=ecol(est)),
                    hovertemplate=f"E{est}<br>%{{x|%Y-%m}}<br>SSC: %{{y:.1f}} mg/L<extra></extra>"))
        if calamar_ssc is not None:
            cal_y = calamar_ssc["ssc_cal"].values
            if ma_win > 1:
                cal_ma = pd.Series(cal_y).rolling(window=ma_win, center=True, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=calamar_ssc["fecha"], y=cal_y,
                    mode="lines", name="Calamar bruto",
                    line=dict(color="#c0392b",dash="dot",width=1), opacity=0.35, showlegend=False))
                fig.add_trace(go.Scatter(x=calamar_ssc["fecha"], y=cal_ma.values,
                    mode="lines", name=f"Calamar MA {ma_win}m",
                    line=dict(color="#c0392b",width=2.5)))
            else:
                fig.add_trace(go.Scatter(x=calamar_ssc["fecha"], y=cal_y,
                    mode="lines", name="Calamar (IDEAM)",
                    line=dict(color="#c0392b",dash="dot",width=2)))
        fig.update_layout(height=460, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
            title=dict(text=f"Serie mensual SSC — ({y0}–{y1}){f'  ·  MA {ma_win} meses' if ma_win>1 else ''}",font=dict(color=COLOR_TEXT,size=14)),
            xaxis=dict(title="Fecha",showgrid=True,gridcolor=COLOR_BORDER,zeroline=False,linecolor=COLOR_BORDER,tickfont=dict(color=COLOR_MUTED)),
            yaxis=dict(title="SSC (mg/L)",gridcolor=COLOR_BORDER,zeroline=False,linecolor=COLOR_BORDER,tickfont=dict(color=COLOR_MUTED)),
            font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
            legend=dict(orientation="h",y=-0.22,bgcolor="rgba(0,0,0,0)",font=dict(color=COLOR_TEXT)),
            margin=dict(l=60,r=20,t=50,b=80))
        return html.Div(style={**CARD}, children=[
            section_title("Serie mensual de SSC","Promedio mensual por estación"),
            dcc.Graph(figure=fig, config={"displayModeBar":False})])

    # ── C: Media móvil espacial ──
    elif subtab == "app-movil":
        df_m2 = df_f.copy()
        df_m2["mes"] = df_m2["fecha"].dt.to_period("M").dt.to_timestamp()
        monthly2 = df_m2.groupby(["estacion","mes"])["SSC"].mean().reset_index()

        fig = go.Figure()
        for est in sel_est:
            neighbors = [e for e in all_est
                         if abs(all_est.index(e)-all_est.index(est)) <= window//2]
            nb_data = monthly2[monthly2["estacion"].isin(neighbors)]
            mm = nb_data.groupby("mes")["SSC"].mean().reset_index().sort_values("mes")
            nn = f"E{est}, MM espacial (E{min(neighbors)}–E{max(neighbors)})"
            y_vals = mm["SSC"].values
            if ma_win > 1:
                y_ma = pd.Series(y_vals).rolling(window=ma_win, center=True, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=mm["mes"], y=y_vals, mode="lines",
                    name=f"{nn} bruto", line=dict(color=ecol(est), width=1, dash="dot"),
                    opacity=0.35, showlegend=False))
                fig.add_trace(go.Scatter(
                    x=mm["mes"], y=y_ma.values, mode="lines+markers",
                    name=f"{nn} MA{ma_win}m", line=dict(color=ecol(est),width=2.5),
                    marker=dict(size=5,color=ecol(est))))
            else:
                fig.add_trace(go.Scatter(
                    x=mm["mes"], y=y_vals, mode="lines+markers",
                    name=nn, line=dict(color=ecol(est),width=2),
                    marker=dict(size=6,color=ecol(est)),
                    hovertemplate=f"{nn}<br>%{{x|%Y-%m}}<br>SSC: %{{y:.1f}} mg/L<extra></extra>"))
        if calamar_ssc is not None:
            cal_y = calamar_ssc["ssc_cal"].values
            if ma_win > 1:
                cal_ma = pd.Series(cal_y).rolling(window=ma_win, center=True, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=calamar_ssc["fecha"], y=cal_ma.values,
                    mode="lines", name=f"Calamar MA {ma_win}m",
                    line=dict(color="#c0392b",width=2.5)))
            else:
                fig.add_trace(go.Scatter(x=calamar_ssc["fecha"], y=cal_y,
                    mode="lines", name="Calamar (IDEAM)",
                    line=dict(color="#c0392b",dash="dot",width=2)))
        fig.update_layout(height=460, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
            title=dict(text=f"Media móvil espacial — ventana {window}  ({y0}–{y1})",font=dict(color=COLOR_TEXT,size=14)),
            xaxis=dict(title="Fecha",showgrid=True,gridcolor=COLOR_BORDER,zeroline=False,linecolor=COLOR_BORDER,tickfont=dict(color=COLOR_MUTED)),
            yaxis=dict(title="SSC (mg/L)",gridcolor=COLOR_BORDER,zeroline=False,linecolor=COLOR_BORDER,tickfont=dict(color=COLOR_MUTED)),
            font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
            legend=dict(orientation="h",y=-0.22,bgcolor="rgba(0,0,0,0)",font=dict(color=COLOR_TEXT)),
            margin=dict(l=60,r=20,t=50,b=80))
        return html.Div(style={**CARD}, children=[
            section_title("Media móvil espacial","Promedio de estaciones vecinas en cada fecha"),
            dcc.Graph(figure=fig, config={"displayModeBar":False})])

    # ── D: Ciclo multianual ──
    elif subtab == "app-ciclo":
        df_c = df_f.copy()
        df_c["mes_n"] = df_c["fecha"].dt.month
        meses_lbl = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

        fig = go.Figure()
        for est in sel_est:
            sub = df_c[df_c["estacion"]==est].groupby("mes_n")["SSC"].agg(
                mean="mean", se=lambda x: x.std()/np.sqrt(len(x))).reset_index()
            fig.add_trace(go.Scatter(
                x=[meses_lbl[m-1] for m in sub["mes_n"]],
                y=sub["mean"],
                error_y=dict(type="data", array=sub["se"].values, visible=True, thickness=1.5),
                mode="lines+markers", name=f"E{est}",
                line=dict(color=ecol(est),width=2),
                marker=dict(size=8,color=ecol(est))))
        if calamar_ssc is not None and len(calamar_ssc) > 0:
            calamar_ssc["mes_n"] = calamar_ssc["fecha"].dt.month
            cal_c = calamar_ssc.groupby("mes_n")["ssc_cal"].agg(
                mean="mean", se=lambda x: x.std()/np.sqrt(len(x))).reset_index()
            fig.add_trace(go.Scatter(
                x=[meses_lbl[m-1] for m in cal_c["mes_n"]],
                y=cal_c["mean"],
                error_y=dict(type="data", array=cal_c["se"].values, visible=True, thickness=1.5),
                mode="lines+markers", name="Calamar (IDEAM)",
                line=dict(color="#c0392b",dash="dash",width=2),
                marker=dict(size=8,symbol="square"), yaxis="y2"))
            fig.update_layout(yaxis2=dict(title="SSC Calamar (mg/L)", overlaying="y",
                                          side="right", showgrid=False))
        fig.update_layout(height=480, paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_BG,
            title=dict(text=f"Ciclo multianual promedio ± 1 SE  ({y0}–{y1})",font=dict(color=COLOR_TEXT,size=14)),
            xaxis=dict(title="Mes",categoryorder="array",categoryarray=meses_lbl,showgrid=True,gridcolor=COLOR_BORDER,zeroline=False,linecolor=COLOR_BORDER,tickfont=dict(color=COLOR_MUTED)),
            yaxis=dict(title="SSC estaciones virtuales (mg/L)",gridcolor=COLOR_BORDER,zeroline=False,linecolor=COLOR_BORDER,tickfont=dict(color=COLOR_MUTED)),
            font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
            legend=dict(orientation="h",y=-0.22,bgcolor="rgba(0,0,0,0)",font=dict(color=COLOR_TEXT)),
            margin=dict(l=60,r=80,t=50,b=80))
        return html.Div(style={**CARD}, children=[
            section_title("Ciclo multianual de SSC","Media mensual climatológica ± 1 error estándar"),
            dcc.Graph(figure=fig, config={"displayModeBar":False})])

    # ── E: Mapa de estaciones virtuales ──
    elif subtab == "app-estmap":
        try:
            df_ev = pd.read_csv("Estaciones_virtuales.csv")
        except Exception:
            return html.Div("Archivo Estaciones_virtuales.csv no encontrado.",
                            style={"color":COLOR_MUTED,"fontSize":"13px","padding":"20px"})

        fig_map = go.Figure()
        # Sampling stations (from the main dataset)
        sampling_lats = [11.10354,11.102,11.09628,11.09018,11.0755,11.0585,11.0428,
                         11.0249,11.0011327,10.9919,10.9782,10.9637,10.9546]
        sampling_lons = [-74.8516,-74.8513,-74.8497,-74.8492,-74.8456,-74.8387,-74.8195,
                         -74.7911,-74.7660752,-74.7611,-74.7579,-74.7569,-74.7562]
        sampling_names = ["Km 0 +250","Km 0 +500","Km 1","Km 1+900","Km 3+500","Km 5+500",
                          "Km 7+900","Km 11+200","Km 14+800","Km 17+600","Km 18+200","Km 19+800","Km 19+940"]

        # Virtual stations
        fig_map.add_trace(go.Scattermapbox(
            lat=df_ev["lat"].tolist(),
            lon=df_ev["lon"].tolist(),
            mode="markers",
            marker=dict(size=12, color=ACCENT2, opacity=0.85),
            text=[f"E{int(row['id'])}" for _, row in df_ev.iterrows()],
            hovertemplate="<b>%{text}</b><br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>",
            name="Estaciones virtuales",
        ))
        # Sampling stations
        fig_map.add_trace(go.Scattermapbox(
            lat=sampling_lats,
            lon=sampling_lons,
            mode="markers+text",
            marker=dict(size=14, color=COLOR_ACCENT, opacity=1.0, symbol="circle"),
            text=sampling_names,
            textposition="top right",
            hovertemplate="<b>%{text}</b><br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>",
            name="Estaciones de campo",
            textfont=dict(size=9, color=COLOR_TEXT),
        ))

        center_lat = (min(df_ev["lat"].tolist() + sampling_lats) + max(df_ev["lat"].tolist() + sampling_lats)) / 2
        center_lon = (min(df_ev["lon"].tolist() + sampling_lons) + max(df_ev["lon"].tolist() + sampling_lons)) / 2

        fig_map.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11,
            ),
            height=580,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=COLOR_CARD,
            legend=dict(
                orientation="v", x=0.01, y=0.99,
                bgcolor="rgba(15,27,43,0.85)",
                bordercolor=COLOR_BORDER,
                borderwidth=1,
                font=dict(color=COLOR_TEXT, size=12),
            ),
        )

        return html.Div(style={**CARD}, children=[
            section_title("Estaciones virtuales de SSC reconstruida",
                          f"Ubicación de las {len(df_ev)} estaciones virtuales (verde) y estaciones de campo (azul)"),
            html.P(f"Se reconstruyeron series temporales de SSC en {len(df_ev)} estaciones virtuales "
                   "distribuidas a lo largo del tramo Km 11–19 del río Magdalena, utilizando el modelo "
                   "empírico calibrado y las imágenes Sentinel-2 disponibles (2019–2025).",
                   style={"fontSize":"14px","color":COLOR_TEXT,"lineHeight":"1.7",
                          "maxWidth":"820px","marginBottom":"20px"}),
            dcc.Graph(figure=fig_map, config={"displayModeBar":False}),
            html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","marginTop":"16px"}, children=[
                html.Div([
                    html.Div("Estaciones virtuales", style={"fontSize":"11px","color":COLOR_MUTED,"textTransform":"uppercase","letterSpacing":"0.06em"}),
                    html.Div(str(len(df_ev)), style={"fontSize":"28px","fontWeight":"700","color":ACCENT2,"fontFamily":FONT_TITLE}),
                ], style={**CARD,"padding":"16px 24px","marginBottom":"0"}),
                html.Div([
                    html.Div("Rango latitud", style={"fontSize":"11px","color":COLOR_MUTED,"textTransform":"uppercase","letterSpacing":"0.06em"}),
                    html.Div(f"{df_ev['lat'].min():.4f}° – {df_ev['lat'].max():.4f}°",
                             style={"fontSize":"16px","fontWeight":"700","color":COLOR_TEXT,"fontFamily":FONT_TITLE}),
                ], style={**CARD,"padding":"16px 24px","marginBottom":"0"}),
                html.Div([
                    html.Div("Rango longitud", style={"fontSize":"11px","color":COLOR_MUTED,"textTransform":"uppercase","letterSpacing":"0.06em"}),
                    html.Div(f"{df_ev['lon'].min():.4f}° – {df_ev['lon'].max():.4f}°",
                             style={"fontSize":"16px","fontWeight":"700","color":COLOR_TEXT,"fontFamily":FONT_TITLE}),
                ], style={**CARD,"padding":"16px 24px","marginBottom":"0"}),
            ]),
        ])

    return html.Div()


# 1) Actualiza el índice al pulsar los botones
@app.callback(
    Output("store-img-index", "data"),
    Input("btn-img-prev", "n_clicks"),
    Input("btn-img-next", "n_clicks"),
    State("store-img-index", "data"),
    prevent_initial_call=True,
)
def navegar_imagen(prev_clicks, next_clicks, idx_actual):
    imagenes = listar_imagenes()
    if not imagenes:
        return 0
 
    triggered = dash.callback_context.triggered[0]["prop_id"]
    if "prev" in triggered:
        nuevo_idx = (idx_actual - 1) % len(imagenes)
    else:
        nuevo_idx = (idx_actual + 1) % len(imagenes)
    return nuevo_idx
 
 
# 2) Renderiza el mapa para el índice actual
@app.callback(
    Output("mapa-output",     "children"),
    Output("mapa-ssc",        "figure"),
    Output("mapa-fecha-label","children"),
    Output("mapa-img-counter","children"),
    Input("store-img-index",  "data"),
)
def update_mapa(idx):
    # ── figura vacía de fallback ─────────────────────────────────────────────
    def fig_vacia(mensaje="Sin imagen"):
        f = go.Figure()
        f.update_layout(
            paper_bgcolor=COLOR_CARD, height=560,
            mapbox=dict(style="carto-darkmatter",
                        center=dict(lat=11.0, lon=-74.85), zoom=11),
            margin=dict(l=0, r=0, t=0, b=0),
            annotations=[dict(
                text=mensaje, xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=COLOR_MUTED),
            )],
        )
        return f
 
    imagenes = listar_imagenes()
 
    if not imagenes:
        return (
            f"⚠️ No se encontraron imágenes TIF en '{CARPETA_IMAGENES}/'.",
            fig_vacia("No hay imágenes disponibles"),
            "—",
            "",
        )
 
    # Índice seguro
    idx = max(0, min(idx or 0, len(imagenes) - 1))
    img      = imagenes[idx]
    ruta     = img["path"]
    label    = img["label"]
    counter  = f"{idx + 1} / {len(imagenes)}"
 
    if not HAS_RASTERIO:
        return (
            "❌ rasterio no está instalado. Ejecuta: pip install rasterio",
            fig_vacia(),
            label,
            counter,
        )
 
    try:
        with rasterio.open(ruta) as src:
            nir = src.read(BANDA_NIR).astype(float)
            bounds = src.bounds
            sf = FACTOR_ESCALA
            nir = nir / sf
            nir[nir <= 0] = np.nan
 
            # Máscara de agua con NDWI (banda 3 = verde)
            try:
                green = src.read(3).astype(float) / sf
                ndwi  = (green - nir) / (green + nir + 1e-9)
                water_mask = ndwi > 0.0
            except Exception:
                water_mask = np.ones(nir.shape, dtype=bool)
 
            nir_water = nir.copy()
            nir_water[~water_mask] = np.nan
 
        # Modelo de calibración
        if df_calib.empty:
            return (
                "❌ Sin datos de calibración para aplicar el modelo.",
                fig_vacia(),
                label,
                counter,
            )
 
        X_c = df_calib[["NIR"]].values
        y_c = df_calib["SSC"].values
        lr    = LinearRegression().fit(X_c, y_c)
        m_val = lr.coef_[0]
        b_val = lr.intercept_
 
        ssc_map = m_val * nir_water + b_val
        ssc_map[ssc_map < 0] = np.nan
 
        height, width = ssc_map.shape
        lon_arr = np.linspace(bounds.left,  bounds.right,  width)
        lat_arr = np.linspace(bounds.top,   bounds.bottom, height)
 
        # Downsample para rendimiento
        step     = max(1, min(height, width) // 300)
        rows_idx = np.arange(0, height, step)
        cols_idx = np.arange(0, width,  step)
        lats_grid = lat_arr[rows_idx]
        lons_grid = lon_arr[cols_idx]
        lon_mesh, lat_mesh = np.meshgrid(lons_grid, lats_grid)
        z_mesh = ssc_map[np.ix_(rows_idx, cols_idx)]
 
        valid    = ~np.isnan(z_mesh)
        lat_flat = lat_mesh[valid].flatten()
        lon_flat = lon_mesh[valid].flatten()
        z_flat   = z_mesh[valid].flatten()
 
        center_lat = float(np.nanmean(lat_flat)) if len(lat_flat) > 0 else 11.0
        center_lon = float(np.nanmean(lon_flat)) if len(lon_flat) > 0 else -74.85
 
        fig = go.Figure(go.Scattermapbox(
            lat=lat_flat.tolist(),
            lon=lon_flat.tolist(),
            mode="markers",
            marker=dict(
                size=4,
                color=z_flat.tolist(),
                colorscale="YlOrRd",
                cmin=float(np.nanpercentile(ssc_map, 10)),
                cmax=float(np.nanpercentile(ssc_map, 90)),
                colorbar=dict(
                    title=dict(text="SSC (mg/L)", side="right"),
                    thickness=16,
                    tickfont=dict(size=11, color=COLOR_TEXT),
                ),
                opacity=0.85,
            ),
            hovertemplate=(
                "Lon: %{lon:.4f}<br>Lat: %{lat:.4f}<br>"
                "SSC: %{marker.color:.0f} mg/L<extra></extra>"
            ),
        ))
        fig.update_layout(
            height=560,
            paper_bgcolor=COLOR_CARD,
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=12,
            ),
            font=dict(family=FONT_BODY, size=12, color=COLOR_TEXT),
            title=dict(
                text=f"SSC modelada (mg/L) — Río Magdalena · {label}",
                font=dict(color=COLOR_TEXT, size=14),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
 
        info = (
            f"✅ {label} — banda NIR {BANDA_NIR} | escala 1/{sf:.0f} | "
            f"SSC: {np.nanmin(ssc_map):.0f}–{np.nanmax(ssc_map):.0f} mg/L | "
            f"píxeles agua: {np.sum(~np.isnan(ssc_map)):,}"
        )
        return info, fig, label, counter
 
    except Exception as e:
        return (
            f"❌ Error procesando {os.path.basename(ruta)}: {str(e)}",
            fig_vacia("Error al procesar la imagen"),
            label,
            counter,
        )

@app.callback(
    Output("app-model-badge","children"),
    Input("model-selector","value"),
)
def update_app_model_badge(model_name):
    model_name = model_name if model_name in MODEL_LABELS else DEFAULT_LINEAR_MODEL
    label = MODEL_LABELS.get(model_name, "Regresión lineal")
    col   = MODEL_COLS.get(model_name, COLOR_ACCENT)
    return [
        html.Span("Modelo utilizado:",
                  style={"fontSize":"12px","color":COLOR_MUTED,"fontWeight":"600",
                         "textTransform":"uppercase","letterSpacing":"0.06em"}),
        html.Span(label,
                  style={"fontSize":"13px","fontWeight":"700","color":col,
                         "backgroundColor":f"{col}18","padding":"4px 14px",
                         "borderRadius":"20px","border":f"1px solid {col}44"}),
    ]


if __name__ == "__main__":
    app.run(debug=True)