import pathlib
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
import io 
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
    0:  "#1a6b9a",
    1:  "#c5e71c",
    3: "#1aff00",
    14: "#6a00ed",
    11: "#2eaa6b",
    17: "#e07b2a",
    18: "#c0392b",
    19: "#7b52ab",
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
FONT_BODY  = "'Lato', sans-serif"
FONT_TITLE = "'Merriweather', serif"
COLOR_BG     = "#f7f8fa"
COLOR_CARD   = "#ffffff"
COLOR_ACCENT = "#1a6b9a"
COLOR_TEXT   = "#1c2331"
COLOR_MUTED  = "#6b7280"
COLOR_BORDER = "#e2e6ea"

CARD = {
    "backgroundColor": COLOR_CARD,
    "borderRadius": "10px",
    "padding": "28px 32px",
    "marginBottom": "24px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.07)",
    "border": f"1px solid {COLOR_BORDER}",
}

TAB_STYLE = {
    "fontFamily": FONT_BODY, "fontSize": "13.5px", "fontWeight": "600",
    "color": COLOR_MUTED, "backgroundColor": COLOR_BG, "border": "none",
    "borderBottom": f"2px solid {COLOR_BORDER}", "padding": "12px 22px",
    "letterSpacing": "0.03em",
}
TAB_SELECTED = {
    **TAB_STYLE, "color": COLOR_ACCENT,
    "borderBottom": f"2px solid {COLOR_ACCENT}", "backgroundColor": COLOR_CARD,
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

# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Lato:wght@300;400;600;700&display=swap"
    ],
    suppress_callback_exceptions=True,
)
app.title = "CSS Magdalena — EDA"

app.layout = html.Div(style={"backgroundColor": COLOR_BG, "minHeight": "100vh",
                               "fontFamily": FONT_BODY}, children=[

    # Store con el dataset filtrado
    dcc.Store(id="store-df-filtered"),

    # ── HEADER ──
    html.Div(style={
        "backgroundColor": COLOR_CARD, "borderBottom": f"1px solid {COLOR_BORDER}",
        "padding": "0 48px", "display": "flex", "alignItems": "center",
        "gap": "16px", "height": "64px", "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
    }, children=[
        html.Div("🌊", style={"fontSize": "24px"}),
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
        lat = [10.2422934],
        lon = [-74.9138168],
        mode="markers+lines", marker=dict(size=12, color=COLOR_ACCENT),
        text=["Calamar, IDEAM (20037020)"], hoverinfo="text",
    ))
    fig2.update_layout(mapbox=dict(style="carto-positron", center=dict(lat=10.2422934, lon=-74.9138168), zoom=8.5),
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
                   "su posible uso como fuente de datos adicionales para la calibración del modelo.",
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
                   style={"fontSize": "15px", "color": COLOR_TEXT, "lineHeight": "1.8",
                          "maxWidth": "800px", "fontWeight": "600"}),
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
            html.Div(style={**CARD}, children=[
                html.H4("Sentinel-2 MSI", style={"fontFamily": FONT_TITLE, "fontSize": "15px",
                                                   "color": COLOR_TEXT, "marginBottom": "14px"}),
                html.Table([
                    html.Thead(html.Tr([html.Th(h, style={"textAlign": "left", "padding": "6px 12px",
                               "fontSize": "12px", "color": COLOR_MUTED,
                               "borderBottom": f"1px solid {COLOR_BORDER}"}) for h in ["Banda","λ central (nm)","Resolución"]])),
                    html.Tbody([html.Tr([html.Td(v, style={"padding": "6px 12px", "fontSize": "13px"}) for v in row])
                                for row in [("Blue (B2)","492","10 m"),("Green (B3)","560","10 m"),
                                            ("Red (B4)","665","10 m"),("Red Edge 1 (B5)","704","20 m"),
                                            ("NIR (B8)","833","10 m"),("SWIR1 (B11)","1614","20 m"),
                                            ("SWIR2 (B12)","2202","20 m")]]),
                ], style={"width": "100%", "borderCollapse": "collapse"}),
            ]),
            html.Div(style={**CARD}, children=[
                html.H4("Índices espectrales evaluados",
                        style={"fontFamily": FONT_TITLE, "fontSize": "15px", "color": COLOR_TEXT, "marginBottom": "14px"}),
                html.Div([
                    html.Div([
                        html.Div(n, style={"fontWeight": "700", "color": COLOR_ACCENT, "fontSize": "14px", "marginBottom": "4px"}),
                        html.Div(f, style={"fontFamily": "monospace", "fontSize": "12px", "color": COLOR_MUTED, "marginBottom": "4px"}),
                        html.Div(d, style={"fontSize": "13px", "color": COLOR_TEXT, "lineHeight": "1.5"}),
                    ], style={"marginBottom": "16px", "paddingBottom": "16px", "borderBottom": f"1px solid {COLOR_BORDER}"})
                    for n, f, d in [
                        ("RANS", "(Red+NIR)/(Red+NIR+Blue+Green+SWIR1+SWIR2)", "Índice normalizado para sedimentos"),
                        ("VNES", "(Red+RE1+NIR)/(Blue+Green+Red+NIR+SWIR1+SWIR2)", "Variante extendida con red edge"),
                        ("NDTI", "(Red−Green)/(Red+Green)", "Índice de turbidez normalizado"),
                        ("NIR/RED", "(NIR)/(Red)", "Relación simple entre NIR y rojo"),
                    ]
                ]),
            ]),
        ]),
        html.Div(style={**CARD}, children=[
            section_title("Transporte de sedimentos en suspensión (TSS)"),
            html.Div("TSS [ton/día] = CSS [mg/L] × Q [m³/s] × 0.0864",
                     style={"fontFamily": "monospace", "fontSize": "15px",
                            "backgroundColor": f"{COLOR_ACCENT}10", "border": f"1px solid {COLOR_ACCENT}30",
                            "borderRadius": "6px", "padding": "14px 20px", "color": COLOR_TEXT, "marginBottom": "12px"}),
        ]),
        html.Div(style={**CARD}, children=[
            section_title("SSC derivado de TSS y Caudal"),
            html.Div("SSC [mg/l] = TSS / Q ",
                     style={"fontFamily": "monospace", "fontSize": "15px",
                            "backgroundColor": f"{COLOR_ACCENT}10", "border": f"1px solid {COLOR_ACCENT}30",
                            "borderRadius": "6px", "padding": "14px 20px", "color": COLOR_TEXT, "marginBottom": "12px"}),
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
        html.Div(style={**CARD}, children=[
            section_title("Estadísticas descriptivas", "Resumen del subconjunto seleccionado"),
            html.Div(id="stats-table"),
        ]),
        
        # ── Perfiles de campo ──
        html.Div(style={**CARD}, children=[
            section_title("Perfiles de concentración de campo",
                          "Perfiles verticales de SSC medidos con LISST — selecciona fecha, Km y transecto"),
            html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                            "alignItems": "center", "flexWrap": "wrap"}, children=[
                html.Label("Fecha:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600"}),
                dcc.Dropdown(id="profile-fecha", options=[], value = "11/06/2025", placeholder="Selecciona una fecha",
                             clearable=False, style={"width": "180px", "fontSize": "13px"}),
                html.Label("Km:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600", "marginLeft": "8px"}),
                dcc.Dropdown(id="profile-km", options=[], placeholder="Km",
                             clearable=False, style={"width": "110px", "fontSize": "13px"}),
                html.Label("+m:", style={"fontSize": "13px", "color": COLOR_MUTED, "fontWeight": "600", "marginLeft": "8px"}),
                dcc.Dropdown(id="profile-pm", options=[], placeholder="+m",
                             clearable=False, style={"width": "110px", "fontSize": "13px"}),
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
                             value="SSC", clearable=False, style={"width": "200px", "fontSize": "13px"}),
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
                             value="red", clearable=False, style={"width": "160px", "fontSize": "13px"}),
                dcc.Dropdown(id="scatter-y",
                             options=[{"label": v, "value": v} for v in SSCS],
                             value="SSC", clearable=False, style={"width": "160px", "fontSize": "13px"}),
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
                             style={"width": "160px", "fontSize": "13px"}),
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
                    value=[2010, HYDRO_YEAR_MAX],
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
                             value="SSC", clearable=False, style={"width": "180px", "fontSize": "13px"}),
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
    return html.Div([
        html.Div(style={**CARD, "borderLeft": f"4px solid {COLOR_ACCENT}"}, children=[
            section_title("Conclusiones del análisis exploratorio"),
            html.Div(style={"display": "flex", "flexDirection": "column", "gap": "16px"}, children=[
                html.Div(style={"display": "flex", "gap": "16px"}, children=[
                    html.Div(n, style={"fontFamily": FONT_TITLE, "fontSize": "22px",
                                       "color": f"{COLOR_ACCENT}50", "fontWeight": "700", "minWidth": "40px"}),
                    html.Div([html.Strong(t+": ", style={"color": COLOR_TEXT, "fontSize": "14.5px"}),
                              html.Span(d, style={"fontSize": "14.5px", "color": COLOR_TEXT, "lineHeight": "1.7"})]),
                ])
                for n, t, d in [
                    ("01","Dataset final de calibración",
                     "El proceso de control de calidad consolidó entre 27 y 39 observaciones con R² 0.58–0.73 en km 0, 1, 3, 14, 17, 18 y 19. Aunque podrian ser mas si finalmente los puntos de Calamar ofrecen resultados favorables."),
                    ("02","Bandas más informativas",
                     "Las bandas Red, NIR y rojo 3 mostraron las correlaciones más altas con SSC, consistente con la literatura."),
                    ("03","Perspectivas de modelado",
                     "Con 30 puntos el dataset es apto para regresión potencial log-log y regresión múltiple, validadas con LOOCV. De ampliarse podria aplicarse algoritmos de machine learning como Random Forest."),
                    ("04", "Limitaciones",
                     "Un modelo de este tipo busca ofrecer una alternativa ante la falta importante de datos in situ, pero su desarrollo esta tambien limitado por esta problematica")
                ]
            ]),
        ]),
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
@app.callback(Output("stats-table", "children"), Input("store-df-filtered", "data"))
def update_stats(data):
    if not data:
        return []
    dff = pd.read_json(io.StringIO(data), orient="split")
    cols = ["SSC"] + BANDAS + INDICES
    cols = [c for c in cols if c in dff.columns]
    stats = dff[cols].describe().T[["mean", "std", "min", "50%", "max"]].round(4)
    stats.columns = ["Media", "Desv. Est.", "Mín.", "Mediana", "Máx."]
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Variable", style={"textAlign":"left","padding":"8px 14px","fontSize":"12px",
                                       "color":COLOR_MUTED,"borderBottom":f"2px solid {COLOR_BORDER}"}),
            *[html.Th(c, style={"textAlign":"right","padding":"8px 14px","fontSize":"12px",
                                "color":COLOR_MUTED,"borderBottom":f"2px solid {COLOR_BORDER}"}) for c in stats.columns],
        ])),
        html.Tbody([
            html.Tr([
                html.Td(idx, style={"padding":"7px 14px","fontSize":"13px","fontWeight":"600",
                                    "color":COLOR_ACCENT,"borderBottom":f"1px solid {COLOR_BORDER}"}),
                *[html.Td(f"{v:.4f}", style={"padding":"7px 14px","fontSize":"13px","textAlign":"right",
                                              "borderBottom":f"1px solid {COLOR_BORDER}"}) for v in row]
            ], style={"backgroundColor": COLOR_CARD if i%2==0 else f"{COLOR_ACCENT}05"})
            for i, (idx, row) in enumerate(stats.iterrows())
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
        sub = dff[dff["km"]==km]; color = KM_COLORS.get(km, COLOR_ACCENT)
        fig.add_trace(go.Histogram(x=sub[var], name=f"Km {km}", marker_color=color, opacity=0.75, nbinsx=12), row=1,col=1)
        fig.add_trace(go.Box(y=sub[var], name=f"Km {km}", marker_color=color, boxmean=True, showlegend=False), row=1,col=2)
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
@app.callback(Output("profile-fecha","options"), Input("tabs","value"))
def populate_fechas(tab):
    if tab!="eda" or df_profiles.empty: return []
    fechas = sorted(df_profiles["fecha"].unique())
    return [{"label":pd.Timestamp(f).strftime("%d/%m/%Y"),"value":str(f)} for f in fechas]

# ── Perfiles: poblar km ──
@app.callback(Output("profile-km","options"), Output("profile-km","value"),
              Input("profile-fecha","value"))
def populate_kms(fecha_str):
    if not fecha_str or df_profiles.empty: return [], None
    sub = df_profiles[df_profiles["fecha"]==pd.Timestamp(fecha_str)]
    kms = sorted(sub["km"].unique())
    return [{"label":f"Km {k}","value":k} for k in kms], (kms[0] if kms else None)

# ── Perfiles: poblar +m ──
@app.callback(Output("profile-pm","options"), Output("profile-pm","value"),
              Input("profile-fecha","value"), Input("profile-km","value"))
def populate_pm(fecha_str, km_val):
    if not fecha_str or km_val is None or df_profiles.empty: return [], None
    sub = df_profiles[(df_profiles["fecha"]==pd.Timestamp(fecha_str))&(df_profiles["km"]==km_val)]
    pms = sorted(sub["+m"].unique())
    return [{"label":f"+{p} m","value":p} for p in pms], (pms[0] if pms else None)

# ── Perfiles: graficar ──
@app.callback(Output("profile-plot","figure"), Output("profile-stats","children"),
              Input("profile-fecha","value"), Input("profile-km","value"), Input("profile-pm","value"))
def update_profile(fecha_str, km_val, pm_val):
    empty=go.Figure()
    empty.update_layout(paper_bgcolor=COLOR_CARD,plot_bgcolor=COLOR_BG,
                        height=480,margin=dict(l=60,r=20,t=30,b=50))
    if not fecha_str or km_val is None or pm_val is None or df_profiles.empty: return empty,""
    sub = df_profiles[(df_profiles["fecha"]==pd.Timestamp(fecha_str))&
                      (df_profiles["km"]==km_val)&(df_profiles["+m"]==pm_val)].sort_values("depth",ascending=False)
    if sub.empty: return empty,""
    color=KM_COLORS.get(km_val,COLOR_ACCENT)
    ssc_4=sub[sub["depth"]<=4]["ssc"].mean(); ssc_7=sub[sub["depth"]<=7]["ssc"].mean()
    ssc_tot=sub["ssc"].mean(); fecha_l=pd.Timestamp(fecha_str).strftime("%d/%m/%Y")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=sub["ssc"],y=sub["depth"],mode="lines+markers",
                             line=dict(color=color,width=2.5),
                             marker=dict(size=6,color=color,line=dict(width=1,color="white")),
                             hovertemplate="Prof: %{y:.2f} m<br>SSC: %{x:.1f} mg/L<extra></extra>"))
    for d_ref,dash_ref in [(4,"dash"),(7,"dot")]:
        fig.add_shape(type="line",x0=sub["ssc"].min()*0.95,x1=sub["ssc"].max()*1.05,
                      y0=d_ref,y1=d_ref,line=dict(color="gray",width=1,dash=dash_ref))
        fig.add_annotation(x=sub["ssc"].max()*1.04,y=d_ref,text=f"{d_ref} m",
                           showarrow=False,font=dict(size=10,color="gray"),xanchor="right")
    fig.update_layout(height=480,paper_bgcolor=COLOR_CARD,plot_bgcolor=COLOR_BG,
                      font=dict(family=FONT_BODY,size=12,color=COLOR_TEXT),
                      xaxis=dict(title="SSC (mg/L)",showgrid=True,gridcolor=COLOR_BORDER),
                      yaxis=dict(title="Profundidad (m)",autorange="reversed",showgrid=True,gridcolor=COLOR_BORDER),
                      margin=dict(l=60,r=30,t=40,b=50),hovermode="y unified",
                      title=dict(text=f"Perfil SSC — Km {km_val}, +{pm_val} m | {fecha_l}",
                                 font=dict(family=FONT_TITLE,size=14,color=COLOR_TEXT),x=0.5))
    stats=html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap"},children=[
        html.Div([html.Div("Promedio total",style={"fontSize":"11px","color":COLOR_MUTED,"textTransform":"uppercase","letterSpacing":"0.05em"}),
                  html.Div(f"{ssc_tot:.1f} mg/L",style={"fontSize":"18px","fontWeight":"700","color":COLOR_ACCENT,"fontFamily":FONT_TITLE})],
                 style={**CARD,"padding":"12px 20px","marginBottom":"0"}),
        html.Div([html.Div("Promedio 0–4 m",style={"fontSize":"11px","color":COLOR_MUTED,"textTransform":"uppercase","letterSpacing":"0.05em"}),
                  html.Div(f"{ssc_4:.1f} mg/L",style={"fontSize":"18px","fontWeight":"700","color":color,"fontFamily":FONT_TITLE})],
                 style={**CARD,"padding":"12px 20px","marginBottom":"0"}),
        html.Div([html.Div("Promedio 0–7 m",style={"fontSize":"11px","color":COLOR_MUTED,"textTransform":"uppercase","letterSpacing":"0.05em"}),
                  html.Div(f"{ssc_7:.1f} mg/L",style={"fontSize":"18px","fontWeight":"700","color":color,"fontFamily":FONT_TITLE})],
                 style={**CARD,"padding":"12px 20px","marginBottom":"0"}),
        html.Div([html.Div("N mediciones",style={"fontSize":"11px","color":COLOR_MUTED,"textTransform":"uppercase","letterSpacing":"0.05em"}),
                  html.Div(str(len(sub)),style={"fontSize":"18px","fontWeight":"700","color":COLOR_MUTED,"fontFamily":FONT_TITLE})],
                 style={**CARD,"padding":"12px 20px","marginBottom":"0"}),
    ])
    return fig,stats


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
        
        fig  = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                             subplot_titles=["Caudal (m³/s)", "TSS Calamar (Kt/día)", "SSC Derivado (mg/L)"])
        if not Qf.empty:
            fig.add_trace(go.Scatter(x=Qf["Fecha"], y=Qf["Q_calamar"], mode="lines",
                                     name="Q Calamar", line=dict(color=COLOR_ACCENT, width=1.5)), row=1,col=1)
        if not QGf.empty:
            fig.add_trace(go.Scatter(x=QGf["Fecha"], y=QGf["Q_barranquilla"], mode="markers+lines",
                                     name="Q Barranquilla", line=dict(color="#e07b2a", width=2),
                                     marker=dict(size=7)), row=1,col=1)
        if not TSSf.empty:
            fig.add_trace(go.Scatter(x=TSSf["Fecha"], y=TSSf["TSS_calamar"], mode="lines",
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

if __name__ == "__main__":
    app.run(debug=True)
