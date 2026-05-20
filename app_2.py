"""
CSS Magdalena — Dashboard de Tesis
app.py  (versión rediseñada: estética limpia, pestaña Modelo simplificada)
"""

import io, os, pickle, warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut

import dash
from dash import dcc, html, Input, Output, State, ctx, ALL
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# DATOS PRINCIPALES
# ─────────────────────────────────────────────────────────────
try:
    df = pd.read_csv("puntos_alternativos.csv")
except FileNotFoundError:
    df = pd.DataFrame(columns=["km","SSC","reflectance_date","scc_date",
                                "aerosol","blue","green","red","rojo 1","rojo 2",
                                "rojo 3","NIR","rojo 4","SWIR1","SWIR2",
                                "RANS","VNES","NDTI","NIR/RED","SSC2","SSC4"])

df["reflectance_date"] = pd.to_datetime(df.get("reflectance_date"), errors="coerce")
df["scc_date"]         = pd.to_datetime(df.get("scc_date"), errors="coerce")
df["km_label"]         = "Km " + df["km"].astype(str)

BANDAS  = ["aerosol","blue","green","red","rojo 1","rojo 2",
           "rojo 3","NIR","rojo 4","SWIR1","SWIR2"]
INDICES = ["RANS","VNES","NDTI","NIR/RED"]
SSCS    = ["SSC","SSC2","SSC4"]
KMS_ALL = sorted(df["km"].unique().tolist()) if not df.empty else []

KM_COLORS = {
    0:"#60A5FA", 1:"#34D399", 3:"#A78BFA",
    5:"#F472B6", 7:"#FBBF24", 11:"#38BDF8",
    14:"#4ADE80",17:"#FB923C",18:"#E879F9",19:"#F87171",
}

# ─────────────────────────────────────────────────────────────
# CARGA DE MODELOS
# ─────────────────────────────────────────────────────────────
def _load_pkl():
    paths = ["models/results.pkl","results.pkl"]
    for p in paths:
        try:
            with open(p,"rb") as f: res = pickle.load(f)
            with open(p.replace("results","models"),"rb") as f: mdls = pickle.load(f)
            return res, mdls
        except Exception:
            continue
    return None, None

PKL_RESULTS, PKL_MODELS = _load_pkl()
PKL_OK = PKL_RESULTS is not None

# Candidate single-variable models from selection_log
SINGLE_VAR_MODELS = []
if PKL_OK:
    sel = pd.DataFrame(PKL_RESULTS.get("selection_log",[]))
    if not sel.empty:
        single = sel[sel["n_features"]==1].sort_values("r2_loo", ascending=False)
        for _, row in single.iterrows():
            feat = row["features"][0]
            SINGLE_VAR_MODELS.append({
                "feature":  feat,
                "r2_loo":   round(row["r2_loo"], 4),
                "rmse_loo": round(row["rmse_loo"], 2),
            })

# ─────────────────────────────────────────────────────────────
# DESIGN TOKENS  (clean light theme)
# ─────────────────────────────────────────────────────────────
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

def card(extra=None):
    base = {
        "backgroundColor": C_WHITE,
        "borderRadius": "12px",
        "padding": "28px 32px",
        "marginBottom": "20px",
        "boxShadow": SHADOW_MD,
        "border": f"1px solid {C_BORDER}",
    }
    if extra: base.update(extra)
    return base

TAB_STYLE = {
    "fontFamily": FONT_SANS, "fontSize": "13px", "fontWeight": "500",
    "color": C_MUTED, "backgroundColor": C_WHITE, "border": "none",
    "borderBottom": f"2px solid transparent", "padding": "12px 20px",
    "letterSpacing": "0.02em",
}
TAB_SEL = {
    **TAB_STYLE, "color": C_ACCENT,
    "borderBottom": f"2px solid {C_ACCENT}",
    "fontWeight": "700",
}

PLT = dict(
    paper_bgcolor=C_WHITE, plot_bgcolor=C_BG,
    font=dict(family=FONT_SANS, size=12, color=C_TEXT),
    xaxis=dict(showgrid=True, gridcolor=C_BORDER, zeroline=False,
               linecolor=C_BORDER2, tickfont=dict(color=C_MUTED)),
    yaxis=dict(showgrid=True, gridcolor=C_BORDER, zeroline=False,
               linecolor=C_BORDER2, tickfont=dict(color=C_MUTED)),
    legend=dict(orientation="h", y=-0.22, bgcolor="rgba(0,0,0,0)",
                font=dict(color=C_MUTED, size=11)),
    margin=dict(l=56, r=20, t=32, b=70),
)

def fig_empty(msg="Sin datos"):
    f = go.Figure()
    f.update_layout(**PLT, height=340,
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=14, color=C_MUTED))])
    return f

# ─────────────────────────────────────────────────────────────
# COMPONENTES UI
# ─────────────────────────────────────────────────────────────
def section_title(text, sub=None):
    els = [html.H3(text, style={"fontFamily": FONT_SERIF, "fontSize": "19px",
                                 "color": C_TEXT, "fontWeight": "700",
                                 "marginBottom": "4px", "marginTop": "0"})]
    if sub:
        els.append(html.P(sub, style={"fontSize": "13px", "color": C_MUTED,
                                       "marginTop": "0", "marginBottom": "18px",
                                       "lineHeight": "1.5"}))
    return html.Div(els)

def kpi(label, value, unit="", color=C_ACCENT):
    return html.Div([
        html.P(label, style={"fontSize": "11px", "color": C_MUTED, "margin": "0 0 6px",
                              "textTransform": "uppercase", "letterSpacing": "0.08em",
                              "fontWeight": "600"}),
        html.Div([
            html.Span(value, style={"fontFamily": FONT_MONO, "fontSize": "26px",
                                     "fontWeight": "700", "color": color}),
            html.Span(f" {unit}" if unit else "", style={"fontSize": "13px", "color": C_MUTED}),
        ]),
    ], style={**card(), "padding": "20px 24px", "marginBottom": "0", "textAlign": "center"})

def badge(text, color=C_ACCENT):
    return html.Span(text, style={
        "fontSize": "11px", "fontWeight": "600", "padding": "3px 10px",
        "borderRadius": "20px", "backgroundColor": f"{color}18", "color": color,
        "border": f"1px solid {color}33", "letterSpacing": "0.03em",
    })

def metric_chip(label, value, color=C_ACCENT, sub=None):
    return html.Div([
        html.Div(label, style={"fontSize": "10px", "color": C_MUTED,
                               "textTransform": "uppercase", "letterSpacing": "0.08em",
                               "marginBottom": "6px", "fontWeight": "600"}),
        html.Div(value, style={"fontFamily": FONT_MONO, "fontSize": "22px",
                               "fontWeight": "700", "color": color}),
        html.Div(sub or "", style={"fontSize": "10px", "color": C_MUTED, "marginTop": "3px"}),
    ], style={
        "backgroundColor": C_WHITE, "borderRadius": "10px",
        "padding": "14px 18px", "textAlign": "center", "minWidth": "110px",
        "border": f"1px solid {C_BORDER}",
        "boxShadow": f"0 0 0 1px {color}22, {SHADOW_SM}",
    })

dd_style = {
    "backgroundColor": C_WHITE,
    "color": C_TEXT,
    "border": f"1px solid {C_BORDER2}",
    "borderRadius": "8px",
    "fontSize": "13px",
}

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Playfair+Display:wght@700&family=JetBrains+Mono:wght@400;500;600&display=swap",
        "https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="CSS Magdalena — Tesis EDA",
    meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}],
)
server = app.server   # ← Gunicorn / Railway / Render

app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
/* Reset & globals */
*{box-sizing:border-box}
body{margin:0;background:#F8F9FB;font-family:'Sora','DM Sans',sans-serif}
/* Scrollbar */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:#F1F3F7}
::-webkit-scrollbar-thumb{background:#C5CADB;border-radius:3px}
/* Dropdown overrides */
.Select-control{background:#fff!important;border-color:#D0D7E3!important;color:#1A2033!important;border-radius:8px!important}
.Select-value-label,.Select-placeholder,.Select--single>.Select-control .Select-value{color:#1A2033!important}
.Select-input input{color:#1A2033!important}
.Select-menu-outer{background:#fff!important;border-color:#D0D7E3!important;z-index:9999!important;border-radius:8px!important;box-shadow:0 8px 24px rgba(0,0,0,0.12)!important}
.Select-option{background:#fff!important;color:#1A2033!important;font-size:13px}
.Select-option:hover,.Select-option.is-focused{background:#EFF6FF!important;color:#2563EB!important}
.Select-option.is-selected{background:#DBEAFE!important;color:#1D4ED8!important;font-weight:600}
.Select--multi .Select-value{background:#EFF6FF!important;border-color:#BFDBFE!important;color:#2563EB!important}
/* Tab container */
.custom-tabs .tab-content{padding-top:0}
/* Checklist labels */
.dash-checklist label{font-size:13px!important;color:#3D4F7C!important;margin-right:8px!important}
/* Slider */
.rc-slider-tooltip-inner{background:#1A2033!important;color:#fff!important;font-size:11px!important}
/* Video responsive */
.video-wrapper{position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:10px}
.video-wrapper iframe,.video-wrapper video{position:absolute;top:0;left:0;width:100%;height:100%}
</style>
</head>
<body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>'''

# ─────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────────────────────
app.layout = html.Div(style={"backgroundColor": C_BG, "minHeight": "100vh"}, children=[

    dcc.Store(id="store-df"),

    # ── NAVBAR ──
    html.Div(style={
        "backgroundColor": C_WHITE,
        "borderBottom": f"1px solid {C_BORDER}",
        "padding": "0 48px",
        "display": "flex", "alignItems": "center", "gap": "16px", "height": "60px",
        "boxShadow": SHADOW_SM, "position": "sticky", "top": "0", "zIndex": "200",
    }, children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"10px"}, children=[
            html.Div(style={
                "width":"34px","height":"34px","borderRadius":"8px",
                "background":f"linear-gradient(135deg,{C_ACCENT} 0%,{C_PURPLE} 100%)",
                "display":"flex","alignItems":"center","justifyContent":"center",
            }, children=html.Span("〰", style={"color":"white","fontSize":"16px"})),
            html.Div([
                html.Span("CSS Magdalena", style={"fontFamily": FONT_SERIF, "fontSize": "15px",
                                                   "color": C_TEXT, "fontWeight": "700"}),
                html.Span("  Análisis Exploratorio · Universidad del Norte",
                          style={"fontSize": "12px", "color": C_MUTED}),
            ]),
        ]),
        html.Div(style={"marginLeft": "auto", "display":"flex","gap":"8px","alignItems":"center"}, children=[
            badge("Sentinel-2", C_ACCENT),
            badge("LISST", C_GREEN),
            badge("2025–2026", C_MUTED),
        ]),
    ]),

    # ── TABS ──
    html.Div(style={"padding": "0 48px", "maxWidth": "1400px", "margin": "0 auto"}, children=[
        dcc.Tabs(id="tabs", value="intro",
                 style={"borderBottom": f"1px solid {C_BORDER}", "marginTop": "8px"},
                 children=[
            dcc.Tab(label="Introducción",  value="intro",        style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Contexto",      value="contexto",     style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Problema",      value="problema",     style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Objetivo",      value="objetivo",     style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Marco Teórico", value="marco",        style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="EDA",           value="eda",          style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Modelo",        value="modelo",       style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Conclusiones",  value="conclusiones", style=TAB_STYLE, selected_style=TAB_SEL),
        ]),
        html.Div(id="tab-content", style={"paddingTop": "28px", "paddingBottom": "60px"}),
    ]),
])

# ═══════════════════════════════════════════════════════════════
# PESTAÑAS
# ═══════════════════════════════════════════════════════════════

def tab_intro():
    n_obs  = len(df) if not df.empty else "–"
    ssc_lo = int(df["SSC"].min()) if not df.empty and "SSC" in df else "–"
    ssc_hi = int(df["SSC"].max()) if not df.empty and "SSC" in df else "–"
    n_km   = df["km"].nunique()   if not df.empty else "–"

    return html.Div([

        # ── Hero ──
        html.Div(style={**card(), "borderLeft": f"4px solid {C_ACCENT}",
                        "background": f"linear-gradient(105deg,{C_ACCENT_L} 0%,{C_WHITE} 60%)"}, children=[
            html.Div(style={"display":"flex","gap":"8px","marginBottom":"14px","flexWrap":"wrap"},
                     children=[badge("Tesis de Pregrado",C_ACCENT),
                                badge("Geología",C_GREEN),
                                badge("Universidad del Norte",C_PURPLE)]),
            html.H1("Estimación de Concentración de Sedimentos en Suspensión "
                    "en el Río Magdalena mediante Imágenes Satelitales Sentinel-2",
                    style={"fontFamily": FONT_SERIF, "fontSize": "24px", "color": C_TEXT,
                           "lineHeight": "1.4", "marginTop": "0", "marginBottom": "18px"}),
            html.P("Este dashboard presenta el análisis exploratorio de datos de la tesis de pregrado "
                   "de Geología en la Universidad del Norte, enfocada en el desarrollo de un modelo empírico "
                   "para estimar la concentración de sedimentos en suspensión (SSC) en el tramo final del río "
                   "Magdalena a partir de reflectancia superficial obtenida con imágenes Sentinel-2 (ESA/Copernicus). "
                   "Los datos de campo fueron tomados con un perfilador LISST cada dos semanas entre Junio 2025 y "
                   "Marzo 2026, uniendo mediciones in situ con imágenes satelitales contemporáneas.",
                   style={"fontSize": "14.5px", "color": C_TEXT, "lineHeight": "1.8",
                          "maxWidth": "820px", "marginBottom": "24px"}),
            html.Div(style={"display":"flex","gap":"14px","flexWrap":"wrap"}, children=[
                kpi("Observaciones", str(n_obs), "puntos"),
                kpi("Período", "Jun 2025 – Mar 2026", "", C_GREEN),
                kpi("Rango SSC", f"{ssc_lo}–{ssc_hi}", "mg/L", C_AMBER),
                kpi("Estaciones", str(n_km), "km", C_PURPLE),
            ]),
        ]),

        # ── Video ──
        html.Div(style={**card()}, children=[
            section_title("Video explicativo del dashboard",
                          "Añade el enlace de tu video o sube el archivo directamente"),
            html.Div([
                # ── INSTRUCCIÓN: reemplaza el src por tu URL de YouTube/Vimeo,
                # o cambia la etiqueta <iframe> por <video controls> con tu archivo
                html.P("📹  Para agregar tu video:",
                       style={"fontWeight":"600","color":C_LABEL,"marginBottom":"8px","fontSize":"13px"}),
                html.Ul([
                    html.Li("YouTube: pega la URL de embed en el atributo src del iframe de abajo.", style={"fontSize":"13px","color":C_MUTED,"marginBottom":"4px"}),
                    html.Li("Video local: sube el archivo a la carpeta assets/ y usa <video> en lugar del iframe.", style={"fontSize":"13px","color":C_MUTED}),
                ], style={"paddingLeft":"18px","marginBottom":"16px"}),
                # Placeholder iframe — reemplaza src
                html.Div(className="video-wrapper", children=[
                    html.Iframe(
                        # ← REEMPLAZA ESTE src CON TU URL DE YOUTUBE EMBED
                        # Ejemplo: src="https://www.youtube.com/embed/TU_VIDEO_ID"
                        src="https://www.youtube.com/embed/dQw4w9WgXcQ",
                        style={"border":"none","borderRadius":"10px"},
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",
                    ),
                ]),
            ]),
        ]),

        # ── Autor ──
        html.Div(style={**card()}, children=[
            section_title("Autor"),
            html.Div(style={"display":"flex","alignItems":"center","gap":"20px","flexWrap":"wrap"}, children=[
                html.Div("FM", style={
                    "width":"52px","height":"52px","borderRadius":"50%",
                    "background":f"linear-gradient(135deg,{C_ACCENT} 0%,{C_PURPLE} 100%)",
                    "color":"white","display":"flex","alignItems":"center",
                    "justifyContent":"center","fontSize":"16px","fontWeight":"700",
                }),
                html.Div([
                    html.P("Francisco Javier Morales Carroll",
                           style={"fontSize":"15px","fontWeight":"700","color":C_TEXT,"margin":"0 0 2px"}),
                    html.P("Estudiante de Geología · Universidad del Norte",
                           style={"fontSize":"13px","color":C_MUTED,"margin":"0"}),
                ]),
                html.Div(style={"marginLeft":"auto","display":"flex","gap":"8px","flexWrap":"wrap"}, children=[
                    html.A("GitHub",    href="https://github.com/Franco1303", target="_blank", style={"fontSize":"12px","padding":"6px 14px","borderRadius":"6px","border":f"1px solid {C_BORDER2}","color":C_LABEL,"textDecoration":"none"}),
                    html.A("LinkedIn",  href="https://www.linkedin.com/in/francisco-morales-4715092a0", target="_blank", style={"fontSize":"12px","padding":"6px 14px","borderRadius":"6px","border":f"1px solid {C_BORDER2}","color":C_LABEL,"textDecoration":"none"}),
                    html.A("Correo",    href="mailto:fcarroll@uninorte.edu.co", target="_blank", style={"fontSize":"12px","padding":"6px 14px","borderRadius":"6px","border":f"1px solid {C_BORDER2}","color":C_LABEL,"textDecoration":"none"}),
                    html.A("Uninorte",  href="https://www.uninorte.edu.co", target="_blank", style={"fontSize":"12px","padding":"6px 14px","borderRadius":"6px","border":f"1px solid {C_ACCENT}","color":C_ACCENT,"textDecoration":"none","backgroundColor":C_ACCENT_L}),
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
        mode="markers+lines",
        marker=dict(size=11, color=C_ACCENT),
        text=["Km 0+250","Km 0+500","Km 1","Km 1+900","Km 3+500","Km 5+500","Km 7+900",
              "Km 11+200","Km 14+800","Km 17+600","Km 18+200","Km 19+800","Km 19+940"],
        hoverinfo="text",
        line=dict(color=C_ACCENT, width=2),
    ))
    fig.update_layout(
        mapbox=dict(style="carto-positron", center=dict(lat=11.05, lon=-74.82), zoom=11),
        margin=dict(l=0,r=0,t=0,b=0), height=480, paper_bgcolor=C_WHITE,
    )
    return html.Div([
        html.Div(style={**card()}, children=[
            section_title("Área de estudio", "Tramo estuarino del río Magdalena — Barranquilla, Colombia"),
            html.P("El río Magdalena es el ecosistema fluvial más importante del país, cubriendo un área "
                   "de 257 438 km² (24 % del territorio nacional). Con una descarga de sedimentos de "
                   "144 × 10⁶ t yr⁻¹ es el principal contribuyente de material terrígeno al Caribe colombiano "
                   "y el agente dominante en la dinámica costera del norte del país (Restrepo et al., 2006).",
                   style={"fontSize":"14px","color":C_TEXT,"lineHeight":"1.8","maxWidth":"820px","marginBottom":"20px"}),
            dcc.Graph(figure=fig, config={"displayModeBar":False}),
        ]),
        html.Div(style={**card()}, children=[
            section_title("Estaciones de muestreo"),
            html.P("Las estaciones Km 5 y Km 7 fueron descartadas del análisis final por estar afectadas "
                   "por actividades de dragado. Las estaciones Km 0, 1 y 3 también presentan este efecto en menor medida.",
                   style={"fontSize":"14px","color":C_MUTED,"marginBottom":"16px"}),
            html.Div(style={"display":"flex","gap":"10px","flexWrap":"wrap"}, children=[
                html.Div([
                    html.Div(f"Km {km}", style={"fontWeight":"700","color":KM_COLORS.get(km,C_ACCENT),"fontSize":"15px"}),
                    html.Div(f"{len(df[df['km']==km])} obs." if not df.empty else "–",
                             style={"fontSize":"12px","color":C_MUTED}),
                ], style={**card({"marginBottom":"0","padding":"14px 20px",
                                  "borderTop":f"3px solid {KM_COLORS.get(km,C_ACCENT)}"})}
                ) for km in sorted(df["km"].unique()) if not df.empty
            ]),
        ]),
    ])


def tab_problema():
    return html.Div([
        html.Div(style={**card(), "borderLeft":f"4px solid {C_AMBER}"}, children=[
            section_title("Planteamiento del problema"),
            html.P("El monitoreo de la SSC en ríos de gran caudal como el Magdalena representa un desafío "
                   "logístico y económico considerable. Los métodos tradicionales requieren campañas intensivas "
                   "con el perfilador LISST. La teledetección satelital con Sentinel-2 ofrece una alternativa "
                   "de bajo costo con cobertura sistemática cada 5 días, aunque en entornos estuarinos la "
                   "estimación de SSC es compleja por otros constituyentes ópticos y efectos de marea.",
                   style={"fontSize":"14.5px","color":C_TEXT,"lineHeight":"1.8","maxWidth":"820px"}),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"18px"}, children=[
            html.Div(style={**card()}, children=[
                html.H4("⚠ Limitaciones del monitoreo tradicional",
                        style={"fontSize":"14px","color":C_TEXT,"marginBottom":"12px","fontWeight":"700"}),
                html.Ul([html.Li(t, style={"fontSize":"13.5px","color":C_TEXT,"marginBottom":"8px","lineHeight":"1.6"})
                         for t in ["Alta demanda de recursos para campañas de campo",
                                   "Cobertura temporal limitada a fechas de muestreo",
                                   "Variabilidad espacial difícil de capturar puntualmente",
                                   "Influencia de dragados en el canal navegable"]],
                        style={"paddingLeft":"16px"}),
            ]),
            html.Div(style={**card()}, children=[
                html.H4("✓ Potencial de la teledetección",
                        style={"fontSize":"14px","color":C_TEXT,"marginBottom":"12px","fontWeight":"700"}),
                html.Ul([html.Li(t, style={"fontSize":"13.5px","color":C_TEXT,"marginBottom":"8px","lineHeight":"1.6"})
                         for t in ["Revisita cada 5 días con Sentinel-2",
                                   "Cobertura espacial continua del tramo fluvial",
                                   "Datos gratuitos via Google Earth Engine",
                                   "Reconstrucción histórica de series de SSC"]],
                        style={"paddingLeft":"16px"}),
            ]),
        ]),
    ])


def tab_objetivo():
    return html.Div([
        html.Div(style={**card(), "borderLeft":f"4px solid {C_ACCENT}"}, children=[
            section_title("Objetivo general"),
            html.P("Estimar la concentración superficial de sedimento en suspensión (SSC) en el sector fluvial "
                   "entre Calamar y Bocas de Ceniza mediante un modelo empírico derivado de variables espectrales "
                   "satelitales e información hidrosedimentológica in situ, orientado a caracterizar su "
                   "variabilidad espaciotemporal.",
                   style={"fontSize":"16px","color":C_TEXT,"lineHeight":"1.9","maxWidth":"820px"}),
        ]),
        html.Div(style={**card()}, children=[
            section_title("Objetivos específicos"),
            html.Div(style={"display":"flex","flexDirection":"column","gap":"14px"}, children=[
                html.Div(style={"display":"flex","gap":"14px","alignItems":"flex-start"}, children=[
                    html.Div(str(i+1), style={
                        "minWidth":"30px","height":"30px","borderRadius":"50%",
                        "backgroundColor":C_ACCENT,"color":"white","display":"flex",
                        "alignItems":"center","justifyContent":"center","fontWeight":"700","fontSize":"13px",
                    }),
                    html.P(t, style={"fontSize":"14px","color":C_TEXT,"lineHeight":"1.7","margin":"0"}),
                ])
                for i, t in enumerate([
                    "Caracterizar la respuesta espectral del agua asociada a diferentes concentraciones de SSC, "
                    "utilizando bandas del visible, NIR y SWIR e índices espectrales derivados, en el bajo río Magdalena.",
                    "Calibrar y validar un modelo empírico de estimación de SSC a partir de variables espectrales "
                    "satelitales, empleando información hidrosedimentológica in situ.",
                    "Cuantificar la variabilidad espaciotemporal de la SSC superficial a partir de la serie "
                    "satelital estimada, incluyendo estacionalidad, eventos extremos y tendencias.",
                ])
            ]),
        ]),
    ])


def tab_marco():
    indices_info = [
        ("RANS", "(Red+NIR)/(Red+NIR+Blue+Green+SWIR1+SWIR2)", "Índice normalizado para sedimentos"),
        ("VNES", "(Red+RE1+NIR)/(Blue+Green+Red+NIR+SWIR1+SWIR2)", "Variante extendida con red edge"),
        ("NDTI", "(Red−Green)/(Red+Green)", "Índice de turbidez normalizado"),
        ("NIR/RED", "NIR / Red", "Relación simple entre NIR y rojo"),
    ]
    return html.Div([
        html.Div(style={**card()}, children=[
            section_title("Teledetección de sedimentos en suspensión","Fundamentos físicos y estado del arte"),
            html.P("La estimación de SSC mediante teledetección se basa en la relación entre la reflectancia "
                   "espectral del agua y la concentración de partículas en suspensión. Los sedimentos aumentan "
                   "la reflectancia en las bandas roja y NIR al incrementar la retrodispersión. La relación "
                   "entre reflectancia y SSC suele ser potencial (log-log lineal), con saturación a altas "
                   "concentraciones en bandas del visible.",
                   style={"fontSize":"14px","color":C_TEXT,"lineHeight":"1.8","maxWidth":"820px"}),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"18px"}, children=[
            html.Div(style={**card()}, children=[
                section_title("Sentinel-2 MSI — Bandas utilizadas"),
                html.Div([
                    html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center",
                                    "padding":"8px 0","borderBottom":f"1px solid {C_BORDER}"}, children=[
                        html.Span(bname, style={"fontSize":"13px","fontWeight":"600","color":C_TEXT}),
                        html.Span(wl, style={"fontFamily":FONT_MONO,"fontSize":"11px","color":C_ACCENT}),
                        html.Span(res, style={"fontSize":"11px","color":C_MUTED}),
                    ])
                    for bname, wl, res in [
                        ("Blue (B2)",    "493 nm", "10 m"),
                        ("Green (B3)",   "560 nm", "10 m"),
                        ("Red (B4)",     "665 nm", "10 m"),
                        ("Red Edge (B5)","704 nm", "20 m"),
                        ("Red Edge (B7)","783 nm", "20 m"),
                        ("NIR (B8)",     "833 nm", "10 m"),
                        ("SWIR1 (B11)", "1614 nm", "20 m"),
                        ("SWIR2 (B12)", "2202 nm", "20 m"),
                    ]
                ]),
            ]),
            html.Div(style={**card()}, children=[
                section_title("Índices espectrales evaluados"),
                html.Div([
                    html.Div([
                        html.Div(n, style={"fontWeight":"700","color":C_ACCENT,"fontSize":"14px","marginBottom":"4px"}),
                        html.Code(f, style={"fontFamily":FONT_MONO,"fontSize":"11px","color":C_MUTED,
                                            "backgroundColor":C_BG,"padding":"2px 6px","borderRadius":"4px"}),
                        html.P(d, style={"fontSize":"13px","color":C_TEXT,"marginTop":"6px","marginBottom":"0","lineHeight":"1.5"}),
                    ], style={"marginBottom":"16px","paddingBottom":"16px",
                              "borderBottom":f"1px solid {C_BORDER}"})
                    for n,f,d in indices_info
                ]),
            ]),
        ]),
    ])


def tab_eda():
    return html.Div([

        # ── Filtro ──
        html.Div(style={**card(), "borderLeft":f"4px solid {C_ACCENT}",
                        "position":"sticky","top":"60px","zIndex":"100"}, children=[
            section_title("Filtro por estación km",
                          "Km 5 y 7 recomendado excluir (dragados). Km 0–3 pueden introducir ruido."),
            html.Div(style={"display":"flex","alignItems":"center","gap":"12px","flexWrap":"wrap"}, children=[
                dcc.Checklist(
                    id="km-filter",
                    options=[{"label": html.Span(f" Km {k} ",
                               style={"color":KM_COLORS.get(k,C_ACCENT),"fontWeight":"600",
                                      "fontSize":"13px"}), "value":k} for k in KMS_ALL],
                    value=KMS_ALL, inline=True,
                    inputStyle={"marginRight":"3px"},
                ),
                html.Div(id="km-count", style={"fontSize":"12px","color":C_MUTED,"marginLeft":"auto"}),
            ]),
        ]),

        # ── Stats ──
        html.Div(style={**card()}, children=[
            section_title("Estadísticas descriptivas"),
            html.Div(style={"display":"flex","gap":"8px","marginBottom":"14px","flexWrap":"wrap"}, children=[
                html.Button("Bandas",  id="pill-bandas",  n_clicks=0,
                            style={"fontSize":"12px","padding":"4px 12px","borderRadius":"20px",
                                   "border":f"1px solid {C_BORDER2}","cursor":"pointer",
                                   "backgroundColor":C_BG,"color":C_LABEL}),
                html.Button("Índices", id="pill-indices", n_clicks=0,
                            style={"fontSize":"12px","padding":"4px 12px","borderRadius":"20px",
                                   "border":f"1px solid {C_BORDER2}","cursor":"pointer",
                                   "backgroundColor":C_BG,"color":C_LABEL}),
                html.Button("SSC",     id="pill-ssc",     n_clicks=0,
                            style={"fontSize":"12px","padding":"4px 12px","borderRadius":"20px",
                                   "border":f"1px solid {C_BORDER2}","cursor":"pointer",
                                   "backgroundColor":C_BG,"color":C_LABEL}),
                dcc.Dropdown(id="stats-var", options=[{"label":v,"value":v} for v in BANDAS],
                             value=BANDAS[0] if BANDAS else None,
                             clearable=False, style={**dd_style,"minWidth":"160px"}),
            ]),
            dcc.Store(id="stats-group", data="bandas"),
            html.Div(id="stats-table"),
        ]),

        # ── Distribución ──
        html.Div(style={**card()}, children=[
            section_title("Distribución de SSC por estación",
                          "Histograma y boxplot por kilómetro"),
            html.Div(style={"display":"flex","gap":"12px","marginBottom":"14px","alignItems":"center"}, children=[
                html.Label("Variable:", style={"fontSize":"13px","color":C_MUTED,"fontWeight":"600"}),
                dcc.Dropdown(id="dist-var",
                             options=[{"label":v,"value":v} for v in ["SSC"]+BANDAS+INDICES],
                             value="SSC", clearable=False, style={**dd_style,"minWidth":"160px"}),
            ]),
            dcc.Graph(id="dist-plot", config={"displayModeBar":False}),
        ]),

        # ── Series de tiempo ──
        html.Div(style={**card()}, children=[
            section_title("Series de tiempo","Evolución temporal de SSC y reflectancia"),
            html.Div(style={"display":"flex","gap":"12px","marginBottom":"14px","alignItems":"center"}, children=[
                html.Label("Variable:", style={"fontSize":"13px","color":C_MUTED,"fontWeight":"600"}),
                dcc.Dropdown(id="ts-var",
                             options=[{"label":v,"value":v} for v in ["SSC"]+BANDAS+INDICES],
                             value="SSC", clearable=False, style={**dd_style,"width":"180px"}),
            ]),
            dcc.Graph(id="ts-plot", config={"displayModeBar":False}),
        ]),

        # ── Scatter ──
        html.Div(style={**card()}, children=[
            section_title("Relación reflectancia / SSC","Scatter con ajuste de regresión"),
            html.Div(style={"display":"flex","gap":"12px","marginBottom":"14px",
                            "alignItems":"center","flexWrap":"wrap"}, children=[
                html.Label("X:", style={"fontSize":"13px","color":C_MUTED,"fontWeight":"600"}),
                dcc.Dropdown(id="sc-x", options=[{"label":v,"value":v} for v in BANDAS+INDICES],
                             value="NIR", clearable=False, style={**dd_style,"minWidth":"130px"}),
                html.Label("Y:", style={"fontSize":"13px","color":C_MUTED,"fontWeight":"600"}),
                dcc.Dropdown(id="sc-y", options=[{"label":v,"value":v} for v in SSCS],
                             value="SSC", clearable=False, style={**dd_style,"minWidth":"100px"}),
                dcc.RadioItems(id="sc-tr", options=[{"label":" SSC","value":"linear"},
                                                     {"label":" ln(SSC)","value":"log"}],
                               value="log", inline=True, style={"fontSize":"13px","color":C_LABEL}),
                dcc.RadioItems(id="sc-color", options=[{"label":" Km","value":"km"},
                                                        {"label":" Ninguno","value":"none"}],
                               value="km", inline=True, style={"fontSize":"13px","color":C_LABEL}),
                dcc.RadioItems(id="sc-fit", options=[{"label":" Lineal","value":"lineal"},
                                                      {"label":" Potencial","value":"potencial"}],
                               value="lineal", inline=True, style={"fontSize":"13px","color":C_LABEL}),
            ]),
            dcc.Graph(id="sc-plot", config={"displayModeBar":False}),
            html.Div(id="sc-stats", style={"marginTop":"6px"}),
        ]),

        # ── Ranking correlaciones ──
        html.Div(style={**card()}, children=[
            section_title("Ranking de correlaciones con SSC","Pearson por banda e índice"),
            html.Div(style={"marginBottom":"12px"}, children=[
                dcc.RadioItems(id="corr-tr", options=[{"label":" SSC","value":"linear"},
                                                       {"label":" ln(SSC)","value":"log"}],
                               value="log", inline=True, style={"fontSize":"13px","color":C_LABEL}),
            ]),
            dcc.Graph(id="corrbar", config={"displayModeBar":False}),
        ]),

        # ── Firmas espectrales ──
        html.Div(style={**card()}, children=[
            section_title("Firmas espectrales","Reflectancia por banda coloreada por SSC"),
            html.Div(style={"marginBottom":"12px"}, children=[
                html.Label("Estación:", style={"fontSize":"13px","color":C_MUTED,"fontWeight":"600","marginRight":"8px"}),
                dcc.Dropdown(id="spec-km", value="all", clearable=False,
                             style={**dd_style,"display":"inline-block","minWidth":"130px"}),
            ]),
            dcc.Graph(id="spec-plot", config={"displayModeBar":False}, style={"height":"500px"}),
        ]),

        # ── Heatmap ──
        html.Div(style={**card()}, children=[
            section_title("Mapa de calor espacio-temporal",
                          "SSC promedio por estación y fecha de imagen"),
            html.Div(style={"marginBottom":"12px"}, children=[
                dcc.Dropdown(id="heat-var",
                             options=[{"label":v,"value":v} for v in ["SSC"]+BANDAS+INDICES],
                             value="SSC", clearable=False, style={**dd_style,"minWidth":"140px"}),
            ]),
            dcc.Graph(id="heat-plot", config={"displayModeBar":False}),
        ]),

        # ── Correlación ──
        html.Div(style={**card()}, children=[
            section_title("Matriz de correlación","Pearson entre bandas, índices y SSC"),
            dcc.RadioItems(id="corrmat-tr", options=[{"label":" SSC","value":"linear"},
                                                      {"label":" ln(SSC)","value":"log"}],
                           value="log", inline=True, style={"fontSize":"13px","color":C_LABEL,"marginBottom":"12px"}),
            dcc.Graph(id="corrmat", config={"displayModeBar":False}),
        ]),

        # ── Climatograma ──
        html.Div(style={**card()}, children=[
            section_title("Climatograma de SSC","Distribución mensual — dataset matcheado"),
            dcc.Graph(id="climo", config={"displayModeBar":False}),
        ]),
    ])


def tab_modelo():
    """Pestaña de modelo: solo regresiones lineales simples por variable."""
    return html.Div([

        # Header
        html.Div(style={**card(), "borderLeft":f"4px solid {C_ACCENT}",
                        "background":f"linear-gradient(105deg,{C_ACCENT_L} 0%,{C_WHITE} 60%)"}, children=[
            section_title("Modelos de regresión lineal por variable predictora",
                          "Comparación de R² LOOCV y RMSE para cada banda e índice individual"),
            html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","marginTop":"8px"}, children=[
                metric_chip("n calibración", str(PKL_RESULTS.get("n_train","–") if PKL_OK else "–"), C_ACCENT),
                metric_chip("Km incluidos",  ", ".join(str(k) for k in (PKL_RESULTS.get("kms_train",[]) if PKL_OK else [])), C_GREEN),
                metric_chip("Mejor variable",
                            PKL_RESULTS.get("best_features",["–"])[0] if PKL_OK else "–",
                            C_AMBER, "LOOCV"),
                metric_chip("Mejor R² LOOCV",
                            f"{PKL_RESULTS.get('best_r2_loo',0):.3f}" if PKL_OK else "–",
                            C_PURPLE),
            ]),
        ]),

        # Ranking visual
        html.Div(style={**card()}, children=[
            section_title("Ranking de variables — R² LOOCV",
                          "Regresión lineal simple SSC ~ cada variable. Validación Leave-One-Out."),
            dcc.Graph(id="model-ranking", config={"displayModeBar":False},
                      style={"height": str(max(300, len(SINGLE_VAR_MODELS)*44+80)) + "px"}),
        ]),

        # Selector + detalle
        html.Div(style={**card()}, children=[
            section_title("Detalle de modelo seleccionado"),
            html.Div(style={"display":"flex","gap":"16px","marginBottom":"18px",
                            "alignItems":"flex-end","flexWrap":"wrap"}, children=[
                html.Div([
                    html.Label("Variable predictora:", style={"fontSize":"13px","color":C_MUTED,
                                                               "fontWeight":"600","marginBottom":"6px","display":"block"}),
                    dcc.Dropdown(id="mdl-var",
                        options=[{"label":m["feature"],"value":m["feature"]} for m in SINGLE_VAR_MODELS],
                        value=SINGLE_VAR_MODELS[0]["feature"] if SINGLE_VAR_MODELS else "NIR",
                        clearable=False, style={**dd_style,"minWidth":"150px"}),
                ]),
            ]),
            html.Div(id="mdl-eq", style={
                "fontFamily":FONT_MONO,"fontSize":"14px",
                "background":C_ACCENT_L,"border":f"1px solid {C_ACCENT}33",
                "borderRadius":"8px","padding":"12px 20px","color":C_TEXT,"marginBottom":"18px",
            }),
            html.Div(id="mdl-metrics", style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"20px"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"18px"}, children=[
                html.Div(style={**card({"marginBottom":"0"})}, children=[
                    html.H4("SSC medido vs modelado (calibración)",
                            style={"fontSize":"13px","color":C_MUTED,"fontWeight":"600","marginTop":"0","marginBottom":"12px"}),
                    dcc.Graph(id="mdl-1to1", config={"displayModeBar":False}, style={"height":"360px"}),
                ]),
                html.Div(style={**card({"marginBottom":"0"})}, children=[
                    html.H4("Predicciones LOOCV vs medido",
                            style={"fontSize":"13px","color":C_MUTED,"fontWeight":"600","marginTop":"0","marginBottom":"12px"}),
                    dcc.Graph(id="mdl-loo", config={"displayModeBar":False}, style={"height":"360px"}),
                ]),
            ]),
        ]),

        # Tabla comparativa
        html.Div(style={**card()}, children=[
            section_title("Tabla comparativa de todas las variables",
                          "Ordenado por R² LOOCV — verde indica mejor desempeño"),
            html.Div(id="mdl-table"),
        ]),

        # Scatter residuos / ratio
        html.Div(style={**card()}, children=[
            section_title("Distribución de errores LOOCV",
                          "Histograma y ratio SSC_med / SSC_mod por estación"),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"18px"}, children=[
                dcc.Graph(id="mdl-errdist", config={"displayModeBar":False}, style={"height":"320px"}),
                dcc.Graph(id="mdl-ratio",   config={"displayModeBar":False}, style={"height":"320px"}),
            ]),
        ]),
    ])


def tab_conclusiones():
    return html.Div([
        html.Div(style={**card(), "borderLeft":f"4px solid {C_ACCENT}"}, children=[
            section_title("Conclusiones del análisis exploratorio"),
            html.Div(style={"display":"flex","flexDirection":"column","gap":"20px"}, children=[
                html.P("El proceso de control de calidad permitió consolidar entre 27 y 39 observaciones "
                       "de calibración distribuidas en los kilómetros 0, 1, 3, 14, 17, 18 y 19. "
                       "Las bandas Red, NIR y Rojo 3 presentaron las correlaciones más altas con SSC, "
                       "resultado consistente con la literatura para sistemas fluviales turbios. "
                       "La regresión lineal simple con NIR alcanzó R²_LOOCV = 0.796, mostrando el "
                       "mejor equilibrio entre simplicidad y desempeño generalizable.",
                       style={"fontSize":"14.5px","color":C_TEXT,"lineHeight":"1.9","margin":"0","textAlign":"justify"}),
                html.P("Con aproximadamente 29 observaciones, el dataset es adecuado para regresiones simples "
                       "validadas por LOOCV. Una ampliación futura permitiría implementar enfoques de machine "
                       "learning de manera más robusta. Los dragados en estaciones cercanas a la desembocadura "
                       "introducen ruido adicional en la señal espectral y deben considerarse en la interpretación.",
                       style={"fontSize":"14.5px","color":C_TEXT,"lineHeight":"1.9","margin":"0","textAlign":"justify"}),
            ]),
        ]),
        html.Div(style={**card()}, children=[
            section_title("Referencias"),
            html.Ul([
                html.Li(r, style={"fontSize":"13px","color":C_TEXT,"marginBottom":"8px","lineHeight":"1.7"})
                for r in [
                    "Qiu et al. (2024). Four-decades of sediment transport variations in the Yellow River using Landsat imagery. Remote Sensing of Environment, 306.",
                    "Qiu et al. (2024). Improving the observations of suspended sediment concentrations from Landsat to Sentinel-2. Int. J. Applied Earth Observation, 134.",
                    "Restrepo et al. (2006). Fluvial fluxes into the Caribbean Sea: The Magdalena River, Colombia. Global and Planetary Change, 50(1–2), 33–49.",
                    "Yepez et al. (2018). Retrieval of suspended sediment concentrations using Landsat-8 OLI in the Orinoco River. Comptes Rendus – Geoscience, 350(1–2), 20–30.",
                ]
            ], style={"paddingLeft":"18px"}),
        ]),
    ])


# ═══════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════

@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):
    return {"intro":tab_intro,"contexto":tab_contexto,"problema":tab_problema,
            "objetivo":tab_objetivo,"marco":tab_marco,"eda":tab_eda,
            "modelo":tab_modelo,"conclusiones":tab_conclusiones}.get(tab, tab_intro)()


@app.callback(Output("store-df","data"), Output("km-count","children"),
              Input("km-filter","value"))
def filter_store(kms):
    kms = kms or KMS_ALL
    filt = df[df["km"].isin(kms)] if not df.empty else df
    label = f"{len(filt)} obs · {len(kms)} estación(es)"
    return filt.to_json(date_format="iso", orient="split"), label


# ── Stats pills ──
@app.callback(Output("stats-group","data"),
              Input("pill-bandas","n_clicks"), Input("pill-indices","n_clicks"),
              Input("pill-ssc","n_clicks"), prevent_initial_call=True)
def set_group(b,i,s):
    return ctx.triggered_id.replace("pill-","")

@app.callback(Output("stats-var","options"), Output("stats-var","value"),
              Input("stats-group","data"))
def update_var_opts(g):
    m = {"bandas":BANDAS,"indices":INDICES,"ssc":SSCS}
    v = m.get(g, BANDAS)
    return [{"label":x,"value":x} for x in v], v[0]

@app.callback(Output("stats-table","children"),
              Input("stats-var","value"), Input("store-df","data"))
def stats_table(var, data):
    if not data or not var: return []
    dff = pd.read_json(io.StringIO(data), orient="split")
    if var not in dff.columns:
        return html.P("Variable no disponible.", style={"color":C_MUTED,"fontSize":"13px"})
    s = dff[var].describe()
    stats = {"Media":round(s["mean"],4),"Desv. Est.":round(s["std"],4),
             "Mín.":round(s["min"],4),"Mediana":round(dff[var].median(),4),
             "Máx.":round(s["max"],4)}
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Variable", style={"textAlign":"left","padding":"8px 14px","fontSize":"12px",
                                        "color":C_MUTED,"borderBottom":f"2px solid {C_BORDER}","fontWeight":"600"}),
            *[html.Th(k, style={"textAlign":"right","padding":"8px 14px","fontSize":"12px",
                                 "color":C_MUTED,"borderBottom":f"2px solid {C_BORDER}","fontWeight":"600"})
              for k in stats],
        ])),
        html.Tbody([html.Tr([
            html.Td(var, style={"padding":"8px 14px","fontSize":"13px","fontWeight":"700",
                                 "color":C_ACCENT,"borderBottom":f"1px solid {C_BORDER}"}),
            *[html.Td(str(v), style={"padding":"8px 14px","fontSize":"13px","textAlign":"right",
                                      "borderBottom":f"1px solid {C_BORDER}"}) for v in stats.values()]
        ])])
    ], style={"width":"100%","borderCollapse":"collapse"})


@app.callback(Output("dist-plot","figure"),
              Input("dist-var","value"), Input("store-df","data"))
def dist_plot(var, data):
    if not data: return fig_empty()
    dff = pd.read_json(io.StringIO(data), orient="split")
    if var not in dff.columns: return fig_empty()
    fig = make_subplots(rows=1,cols=2,subplot_titles=("Histograma","Boxplot"))
    for km in sorted(dff["km"].unique()):
        sub = dff[dff["km"]==km]; c = KM_COLORS.get(km,C_ACCENT)
        fig.add_trace(go.Histogram(x=sub[var],name=f"Km {km}",marker_color=c,opacity=0.7,nbinsx=12),row=1,col=1)
        fig.add_trace(go.Box(y=sub[var],name=f"Km {km}",marker_color=c,boxmean=True,showlegend=False),row=1,col=2)
    fig.update_layout(**PLT, height=360, barmode="overlay")
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor=C_BORDER)
    return fig


@app.callback(Output("ts-plot","figure"),
              Input("ts-var","value"), Input("store-df","data"))
def ts_plot(var, data):
    if not data: return fig_empty()
    dff = pd.read_json(io.StringIO(data), orient="split")
    dff["reflectance_date"] = pd.to_datetime(dff["reflectance_date"], errors="coerce")
    if var not in dff.columns: return fig_empty()
    fig = go.Figure()
    for km in sorted(dff["km"].unique()):
        sub = dff[dff["km"]==km].sort_values("reflectance_date"); c = KM_COLORS.get(km,C_ACCENT)
        fig.add_trace(go.Scatter(x=sub["reflectance_date"],y=sub[var],mode="lines+markers",
                                 name=f"Km {km}",line=dict(color=c,width=2),marker=dict(size=6,color=c)))
    fig.update_layout(**PLT, height=340, xaxis_title="Fecha", yaxis_title=var)
    return fig


@app.callback(Output("sc-plot","figure"), Output("sc-stats","children"),
              Input("sc-x","value"), Input("sc-tr","value"), Input("sc-color","value"),
              Input("store-df","data"), Input("sc-y","value"), Input("sc-fit","value"))
def scatter_plot(x_var, tr, color_by, data, y_var, ajuste):
    if not data: return fig_empty(), ""
    dff = pd.read_json(io.StringIO(data), orient="split")
    if x_var not in dff.columns or y_var not in dff.columns: return fig_empty(), ""
    x = dff[x_var]; y_raw = dff[y_var]
    y = np.log(y_raw) if tr=="log" else y_raw
    y_label = "ln(SSC)" if tr=="log" else "SSC (mg/L)"
    fig = go.Figure()
    if ajuste=="lineal":
        r, p = pearsonr(x,y); r2 = r**2
        m_c, b_c = np.polyfit(x,y,1)
        x_l = np.linspace(x.min(),x.max(),200); y_l = m_c*x_l+b_c
        eq = f"y = {m_c:.4f}x + {b_c:.4f}"
    else:
        mask=(x>0)&(y_raw>0); logx=np.log(x[mask]); logy=np.log(y_raw[mask])
        r,p=pearsonr(logx,logy); r2=r**2
        b_e,la=np.polyfit(logx,logy,1); a_e=np.exp(la)
        x_l=np.linspace(x[mask].min(),x[mask].max(),200); y_l=a_e*(x_l**b_e)
        if tr=="log": y_l=np.log(y_l)
        eq=f"y = {a_e:.4f}·x^{b_e:.4f}"
    if color_by=="km":
        for km in sorted(dff["km"].unique()):
            sub=dff[dff["km"]==km]; y_s=np.log(sub[y_var]) if tr=="log" else sub[y_var]
            fig.add_trace(go.Scatter(x=sub[x_var],y=y_s,mode="markers",name=f"Km {km}",
                                     marker=dict(size=9,color=KM_COLORS.get(km,C_ACCENT),
                                                 line=dict(width=1,color="white"))))
    else:
        fig.add_trace(go.Scatter(x=x,y=y,mode="markers",name="Datos",
                                 marker=dict(size=9,color=C_ACCENT,line=dict(width=1,color="white"))))
    fig.add_trace(go.Scatter(x=x_l,y=y_l,mode="lines",name="Regresión",
                             line=dict(color=C_RED,width=2,dash="dash")))
    fig.update_layout(**PLT,height=380,xaxis_title=x_var,yaxis_title=y_label)
    p_t = "< 0.0001" if p<0.0001 else f"{p:.4f}"
    chips = html.Div(style={"display":"flex","gap":"10px","flexWrap":"wrap"}, children=[
        html.Span(eq, style={"fontFamily":FONT_MONO,"fontSize":"12px","color":C_TEXT,
                              "background":C_ACCENT_L,"padding":"4px 10px","borderRadius":"4px"}),
        html.Span(f"R² = {r2:.3f}", style={"fontFamily":FONT_MONO,"fontSize":"12px","color":C_ACCENT,
                                             "fontWeight":"700","background":C_ACCENT_L,"padding":"4px 10px","borderRadius":"4px"}),
        html.Span(f"p = {p_t}", style={"fontFamily":FONT_MONO,"fontSize":"12px","color":C_MUTED,
                                        "background":C_BG,"padding":"4px 10px","borderRadius":"4px",
                                        "border":f"1px solid {C_BORDER}"}),
        html.Span(f"n = {len(dff)}", style={"fontFamily":FONT_MONO,"fontSize":"12px","color":C_MUTED,
                                              "background":C_BG,"padding":"4px 10px","borderRadius":"4px",
                                              "border":f"1px solid {C_BORDER}"}),
    ])
    return fig, chips


@app.callback(Output("spec-km","options"), Output("spec-km","value"),
              Input("store-df","data"), State("spec-km","value"))
def spec_km_opts(data, cur):
    if not data: return [{"label":"Todas","value":"all"}], "all"
    dff = pd.read_json(io.StringIO(data), orient="split")
    kms = sorted(dff["km"].unique())
    opts = [{"label":"Todas","value":"all"}]+[{"label":f"Km {k}","value":k} for k in kms]
    return opts, (cur if (cur=="all" or cur in kms) else "all")


@app.callback(Output("spec-plot","figure"), Input("spec-km","value"), Input("store-df","data"))
def spec_plot(km_sel, data):
    if not data: return fig_empty()
    dff = pd.read_json(io.StringIO(data), orient="split")
    BAND_NAMES=["aerosol","blue","green","red","rojo 1","rojo 2","rojo 3","NIR","rojo 4","SWIR1","SWIR2"]
    WL=[443.9,496.6,560,664.5,703.9,740.2,782.5,835.1,864.8,1613.7,2202.4]
    sub = dff if km_sel=="all" else dff[dff["km"]==km_sel]
    sub = sub.sort_values("SSC").reset_index(drop=True)
    if sub.empty or not all(b in sub.columns for b in BAND_NAMES): return fig_empty()
    smin,smax=sub["SSC"].min(),sub["SSC"].max()
    def ssc_col(s): t=(s-smin)/(smax-smin+1e-9); return f"rgb(255,{int(165*(1-t))},0)"
    WL_VIS=WL[:9]; BN_VIS=BAND_NAMES[:9]
    fig=make_subplots(rows=1,cols=2,column_widths=[0.73,0.27],shared_yaxes=True,
                      horizontal_spacing=0.04,subplot_titles=["Visible/NIR (400–950 nm)","SWIR"])
    for _,row in sub.iterrows():
        c=ssc_col(row["SSC"]); d=str(row["reflectance_date"])[:10]
        h=f"SSC:{row['SSC']:.1f} mg/L<br>Fecha:{d}"
        fig.add_trace(go.Scatter(x=WL_VIS,y=[row[b] for b in BN_VIS],mode="lines+markers",
                                 line=dict(color=c,width=1.8),marker=dict(size=6,color=c,line=dict(width=0.5,color="white")),
                                 hovertemplate=h+"<extra></extra>",showlegend=False),row=1,col=1)
        for si,wl_fict in [(9,10),(10,150)]:
            fig.add_trace(go.Scatter(x=[wl_fict],y=[row[BAND_NAMES[si]]],mode="markers",
                                     marker=dict(size=8,color=c,line=dict(width=0.5,color="white")),
                                     hovertemplate=f"{BAND_NAMES[si]}<br>"+h+"<extra></extra>",showlegend=False),row=1,col=2)
    fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",showlegend=False,hoverinfo="skip",
                             marker=dict(colorscale=[[0,"rgb(255,165,0)"],[1,"rgb(255,0,0)"]],
                                         cmin=smin,cmax=smax,color=[smin],showscale=True,
                                         colorbar=dict(title=dict(text="SSC (mg/L)",side="right"),
                                                       thickness=14,len=0.7))))
    fig.update_layout(**PLT,height=500,margin=dict(l=60,r=80,t=40,b=50))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor=C_BORDER)
    return fig


@app.callback(Output("corrbar","figure"), Input("corr-tr","value"), Input("store-df","data"))
def corrbar(tr, data):
    if not data: return fig_empty()
    dff = pd.read_json(io.StringIO(data), orient="split")
    cols=[c for c in BANDAS+INDICES if c in dff.columns]
    if "SSC" not in dff.columns: return fig_empty()
    yc = np.log(dff["SSC"]) if tr=="log" else dff["SSC"]
    res=[]
    for c in cols:
        try:
            r,p=pearsonr(dff[c].dropna(), yc[dff[c].notna()])
            res.append({"var":c,"r":r,"r_abs":abs(r),"p":p})
        except: pass
    df_r=pd.DataFrame(res).sort_values("r_abs",ascending=True)
    colors=[C_GREEN if r>=0 else C_RED for r in df_r["r"]]
    fig=go.Figure(go.Bar(x=df_r["r"],y=df_r["var"],orientation="h",
                          marker=dict(color=colors,line=dict(width=0)),
                          text=[f"r={r:.3f}" for r in df_r["r"]],
                          textposition="outside"))
    fig.add_vline(x=0,line=dict(color=C_MUTED,width=1,dash="dash"))
    fig.add_vline(x=0.7,line=dict(color=C_GREEN,width=1,dash="dot"),opacity=0.5)
    fig.add_vline(x=-0.7,line=dict(color=C_RED,width=1,dash="dot"),opacity=0.5)
    fig.update_layout(**PLT,height=max(300,len(df_r)*32+80),
                      xaxis=dict(title="Correlación de Pearson",range=[-1.1,1.1],
                                 showgrid=True,gridcolor=C_BORDER,zeroline=False),
                      margin=dict(l=80,r=100,t=30,b=50),showlegend=False)
    return fig


@app.callback(Output("heat-plot","figure"), Input("heat-var","value"), Input("store-df","data"))
def heat_plot(var, data):
    if not data: return fig_empty()
    dff = pd.read_json(io.StringIO(data), orient="split")
    dff["reflectance_date"]=pd.to_datetime(dff["reflectance_date"],errors="coerce")
    if var not in dff.columns: return fig_empty()
    pivot=(dff.groupby(["km",dff["reflectance_date"].dt.strftime("%Y-%m-%d")])[var]
             .mean().reset_index().pivot(index="km",columns="reflectance_date",values=var))
    pivot=pivot.sort_index(ascending=False)
    fig=go.Figure(go.Heatmap(z=pivot.values,x=pivot.columns.tolist(),
                              y=[f"Km {k}" for k in pivot.index],
                              colorscale="YlOrRd",xgap=2,ygap=2,
                              colorbar=dict(title=dict(text=var,side="right"),thickness=14),
                              hovertemplate="Fecha:%{x}<br>%{y}<br>"+var+":%{z:.1f}<extra></extra>"))
    fig.update_layout(**PLT,height=max(260,len(pivot)*55+100),
                      paper_bgcolor=C_WHITE,plot_bgcolor=C_WHITE,
                      xaxis=dict(title="Fecha",tickangle=-45,showgrid=False),
                      yaxis=dict(showgrid=False),margin=dict(l=80,r=60,t=20,b=80))
    return fig


@app.callback(Output("corrmat","figure"), Input("corrmat-tr","value"), Input("store-df","data"))
def corrmat(tr, data):
    if not data: return fig_empty()
    dff = pd.read_json(io.StringIO(data), orient="split")
    cols=[c for c in BANDAS+INDICES+["SSC"] if c in dff.columns]
    dc=dff[cols].copy()
    if tr=="log": dc["SSC"]=np.log(dc["SSC"])
    labels=[c if c!="SSC" else("ln(SSC)" if tr=="log" else "SSC") for c in cols]
    corr=dc.corr()
    fig=go.Figure(go.Heatmap(z=corr.values,x=labels,y=labels,colorscale="RdBu",zmid=0,zmin=-1,zmax=1,
                              text=np.round(corr.values,2),texttemplate="%{text}",
                              textfont={"size":9},hoverongaps=False))
    fig.update_layout(**PLT,height=480,paper_bgcolor=C_WHITE,plot_bgcolor=C_WHITE,
                      margin=dict(l=80,r=20,t=20,b=80),xaxis=dict(tickangle=-45,showgrid=False),
                      yaxis=dict(showgrid=False))
    return fig


@app.callback(Output("climo","figure"), Input("store-df","data"))
def climo(data):
    if not data: return fig_empty()
    dff = pd.read_json(io.StringIO(data), orient="split")
    dff["reflectance_date"]=pd.to_datetime(dff["reflectance_date"],errors="coerce")
    if "SSC" not in dff.columns: return fig_empty()
    dff["mes"]=dff["reflectance_date"].dt.month
    lbl=["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    fig=go.Figure()
    for mes in range(1,13):
        sub=dff[dff["mes"]==mes]
        if sub.empty: continue
        fig.add_trace(go.Box(y=sub["SSC"],x=[lbl[mes-1]]*len(sub),name=lbl[mes-1],
                             marker=dict(color=C_ACCENT,opacity=0.4,size=4),
                             line=dict(color=C_ACCENT),boxmean=True,showlegend=False))
    for km in sorted(dff["km"].unique()):
        sub_k=dff[dff["km"]==km]
        fig.add_trace(go.Scatter(x=[lbl[m-1] for m in sub_k["mes"]],y=sub_k["SSC"],mode="markers",
                                 name=f"Km {km}",marker=dict(size=8,color=KM_COLORS.get(km,C_ACCENT),
                                 line=dict(width=1,color="white"),opacity=0.85)))
    mm=dff.groupby("mes")["SSC"].mean().reindex(range(1,13))
    fig.add_trace(go.Scatter(x=[lbl[m-1] for m in mm.index if not pd.isna(mm[m])],
                              y=[v for v in mm.values if not pd.isna(v)],
                              mode="lines+markers",name="Media mensual",
                              line=dict(color=C_TEXT,width=2,dash="dash"),marker=dict(size=7,color=C_TEXT)))
    fig.update_layout(**PLT,height=400,xaxis=dict(title="Mes",categoryorder="array",
                       categoryarray=lbl,showgrid=False),
                      yaxis=dict(title="SSC (mg/L)",gridcolor=C_BORDER),boxmode="overlay")
    return fig


# ═══════════════════════════════════════════════════════════════
# CALLBACKS — MODELO (solo regresiones lineales simples)
# ═══════════════════════════════════════════════════════════════

def _fit_single(feature):
    """Fit linear regression SSC ~ feature on training data from pkl."""
    if not PKL_OK: return None, None, None, None, None
    y_true = np.array(PKL_RESULTS["y_true"])
    kms    = np.array(PKL_RESULTS["kms"])
    # Reconstruct X from the full dataframe used in training
    # We use df filtered by kms_train
    kms_train = PKL_RESULTS.get("kms_train",[14,17,18,19])
    dff = df[df["km"].isin(kms_train)].copy() if not df.empty else pd.DataFrame()
    if dff.empty or feature not in dff.columns: return None, None, None, None, None
    dff = dff.dropna(subset=[feature,"SSC"]).reset_index(drop=True)
    X = dff[[feature]].values
    y = dff["SSC"].values
    kms_arr = dff["km"].values
    # Calibration
    mdl = LinearRegression().fit(X, y)
    yhat_cal = mdl.predict(X)
    # LOOCV
    loo = LeaveOneOut()
    yhat_loo = np.zeros(len(y))
    for tr_i, te_i in loo.split(X):
        m = LinearRegression().fit(X[tr_i], y[tr_i])
        yhat_loo[te_i] = m.predict(X[te_i])
    return y, yhat_cal, yhat_loo, kms_arr, mdl


@app.callback(Output("model-ranking","figure"), Input("tabs","value"))
def model_ranking(_):
    if not SINGLE_VAR_MODELS: return fig_empty("No hay datos de selección de variables")
    df_s = pd.DataFrame(SINGLE_VAR_MODELS).sort_values("r2_loo", ascending=True)
    colors = [C_ACCENT if r2 == df_s["r2_loo"].max() else
              (C_GREEN if r2 >= 0.7 else (C_AMBER if r2 >= 0.5 else C_RED))
              for r2 in df_s["r2_loo"]]
    fig = go.Figure(go.Bar(
        x=df_s["r2_loo"], y=df_s["feature"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"R²={r:.3f}  RMSE={rm:.0f}" for r,rm in zip(df_s["r2_loo"],df_s["rmse_loo"])],
        textposition="outside",
        hovertemplate="%{y}<br>R²_LOOCV: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0.7, line=dict(color=C_GREEN,dash="dot",width=1.5), opacity=0.7,
                  annotation_text="R²=0.7", annotation_font=dict(size=10,color=C_GREEN))
    fig.update_layout(**PLT, height=max(300, len(df_s)*44+80),
                      xaxis=dict(title="R² LOOCV", range=[min(0, df_s["r2_loo"].min()-0.05), 1.0],
                                 showgrid=True, gridcolor=C_BORDER, zeroline=False),
                      yaxis=dict(showgrid=False),
                      margin=dict(l=80,r=140,t=30,b=50), showlegend=False)
    return fig


@app.callback(
    Output("mdl-eq","children"),
    Output("mdl-metrics","children"),
    Output("mdl-1to1","figure"),
    Output("mdl-loo","figure"),
    Output("mdl-errdist","figure"),
    Output("mdl-ratio","figure"),
    Input("mdl-var","value"),
)
def model_detail(feature):
    if not feature or not PKL_OK:
        return "Sin datos", [], fig_empty(), fig_empty(), fig_empty(), fig_empty()

    y, yhat_cal, yhat_loo, kms_arr, mdl = _fit_single(feature)
    if y is None:
        return "Variable no disponible en dataset de calibración", [], fig_empty(), fig_empty(), fig_empty(), fig_empty()

    coef = mdl.coef_[0]; inter = mdl.intercept_
    eq = f"SSC = {coef:.4f} × {feature} + {inter:.2f}"

    r2_cal  = r2_score(y, yhat_cal)
    rmse_cal= np.sqrt(mean_squared_error(y, yhat_cal))
    r2_loo  = r2_score(y, yhat_loo)
    rmse_loo= np.sqrt(mean_squared_error(y, yhat_loo))
    mape_val= 100*np.mean(np.abs((y-yhat_loo)/y))
    bias_val= float(np.mean(yhat_loo-y))

    chips = [
        metric_chip("R² Cal",    f"{r2_cal:.3f}",   C_ACCENT),
        metric_chip("RMSE Cal",  f"{rmse_cal:.1f}",  C_ACCENT, "mg/L"),
        metric_chip("R² LOOCV",  f"{r2_loo:.3f}",    C_GREEN),
        metric_chip("RMSE LOOCV",f"{rmse_loo:.1f}",  C_GREEN, "mg/L"),
        metric_chip("MAPE LOOCV",f"{mape_val:.1f}",  C_AMBER, "%"),
        metric_chip("Bias LOOCV",f"{bias_val:+.1f}", C_PURPLE, "mg/L"),
    ]

    lims = [min(y.min(),yhat_cal.min(),yhat_loo.min())*0.92,
            max(y.max(),yhat_cal.max(),yhat_loo.max())*1.08]

    # 1:1 cal
    fig_11 = go.Figure()
    fig_11.add_trace(go.Scatter(x=lims,y=lims,mode="lines",
                                line=dict(color=C_MUTED,dash="dash",width=1.5),name="1:1"))
    for km in sorted(set(kms_arr)):
        mask=kms_arr==km; kc=KM_COLORS.get(int(km),C_ACCENT)
        fig_11.add_trace(go.Scatter(x=y[mask],y=yhat_cal[mask],mode="markers",name=f"Km {int(km)}",
                                    marker=dict(size=9,color=kc,line=dict(width=1.5,color="white"))))
    fig_11.update_layout(**PLT,height=360,xaxis_title="SSC medido (mg/L)",yaxis_title="SSC modelado (mg/L)")

    # LOOCV 1:1
    fig_loo = go.Figure()
    fig_loo.add_trace(go.Scatter(x=lims,y=lims,mode="lines",
                                 line=dict(color=C_MUTED,dash="dash",width=1.5),name="1:1"))
    for km in sorted(set(kms_arr)):
        mask=kms_arr==km; kc=KM_COLORS.get(int(km),C_ACCENT)
        fig_loo.add_trace(go.Scatter(x=y[mask],y=yhat_loo[mask],mode="markers",name=f"Km {int(km)}",
                                     marker=dict(size=9,color=kc,line=dict(width=1.5,color="white"))))
    fig_loo.update_layout(**PLT,height=360,xaxis_title="SSC medido (mg/L)",yaxis_title="SSC LOOCV (mg/L)")

    # Error distribution
    errors = y - yhat_loo
    fig_err = go.Figure()
    fig_err.add_trace(go.Histogram(x=errors.tolist(),nbinsx=12,
                                    marker=dict(color=C_GREEN,opacity=0.75,
                                                line=dict(color=C_BORDER,width=1))))
    fig_err.add_vline(x=0,line=dict(color=C_MUTED,dash="dash",width=1.5))
    fig_err.add_vline(x=float(np.mean(errors)),line=dict(color=C_GREEN,dash="dot",width=1.5),
                      annotation_text=f"μ={np.mean(errors):.1f}",
                      annotation_position="top right",annotation_font=dict(color=C_GREEN,size=11))
    fig_err.update_layout(**PLT,height=320,xaxis_title="Error SSC_med − SSC_loo (mg/L)",yaxis_title="Frecuencia")

    # Ratio by km
    ratio = y / np.where(yhat_loo==0,np.nan,yhat_loo)
    fig_ratio = go.Figure()
    fig_ratio.add_hrect(y0=0.8,y1=1.2,fillcolor=f"{C_GREEN}15",line_width=0)
    fig_ratio.add_hrect(y0=0.7,y1=1.3,fillcolor=f"{C_AMBER}10",line_width=0)
    fig_ratio.add_hline(y=1.0,line=dict(color=C_MUTED,dash="dash",width=1.5))
    for km in sorted(set(kms_arr)):
        mask=kms_arr==km; rv=ratio[mask]; kc=KM_COLORS.get(int(km),C_ACCENT); n=mask.sum()
        med=np.nanmedian(rv); p25,p75=np.nanpercentile(rv,[25,75])
        km_lbl=f"Km {int(km)}"
        fig_ratio.add_trace(go.Scatter(x=[km_lbl]*n,y=rv.tolist(),mode="markers",
                                       marker=dict(size=8,color=kc,opacity=0.5),showlegend=False))
        fig_ratio.add_trace(go.Scatter(x=[km_lbl,km_lbl],y=[p25,p75],mode="lines",
                                       line=dict(color=kc,width=6),showlegend=False))
        fig_ratio.add_trace(go.Scatter(x=[km_lbl],y=[med],mode="markers",name=f"Km {int(km)}",
                                       marker=dict(size=14,color=kc,line=dict(width=2,color="white")),
                                       hovertemplate=f"Km {int(km)}<br>n={n}<br>Mediana:{med:.2f}<extra></extra>"))
    fig_ratio.update_layout(**PLT,height=320,xaxis_title="Estación",yaxis_title="SSC_med / SSC_loo")

    return eq, chips, fig_11, fig_loo, fig_err, fig_ratio


@app.callback(Output("mdl-table","children"), Input("tabs","value"))
def mdl_table(_):
    if not SINGLE_VAR_MODELS:
        return html.P("Sin datos.", style={"color":C_MUTED})
    best_r2 = max(m["r2_loo"] for m in SINGLE_VAR_MODELS)
    rows = []
    for m in sorted(SINGLE_VAR_MODELS, key=lambda x: -x["r2_loo"]):
        is_best = m["r2_loo"] == best_r2
        bg = C_ACCENT_L if is_best else C_WHITE
        rows.append(html.Tr([
            html.Td(html.Div([m["feature"],
                              html.Span(" ★", style={"color":C_ACCENT}) if is_best else ""],
                             style={"fontWeight":"700" if is_best else "400"}),
                   style={"padding":"9px 14px","fontSize":"13px","borderBottom":f"1px solid {C_BORDER}","backgroundColor":bg}),
            html.Td(f"{m['r2_loo']:.4f}",
                    style={"padding":"9px 14px","fontSize":"13px","textAlign":"right",
                           "fontFamily":FONT_MONO,"borderBottom":f"1px solid {C_BORDER}","backgroundColor":bg,
                           "color":C_GREEN if m["r2_loo"]>=0.7 else (C_AMBER if m["r2_loo"]>=0.5 else C_RED),
                           "fontWeight":"700" if is_best else "400"}),
            html.Td(f"{m['rmse_loo']:.1f} mg/L",
                    style={"padding":"9px 14px","fontSize":"13px","textAlign":"right",
                           "fontFamily":FONT_MONO,"borderBottom":f"1px solid {C_BORDER}","backgroundColor":bg}),
        ]))
    hdr_style = {"textAlign":"right","padding":"9px 14px","fontSize":"12px","color":C_MUTED,
                 "fontWeight":"600","borderBottom":f"2px solid {C_BORDER2}","backgroundColor":C_BG}
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Variable", style={**hdr_style,"textAlign":"left"}),
            html.Th("R² LOOCV", style=hdr_style),
            html.Th("RMSE LOOCV", style=hdr_style),
        ])),
        html.Tbody(rows),
    ], style={"width":"100%","borderCollapse":"collapse"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
