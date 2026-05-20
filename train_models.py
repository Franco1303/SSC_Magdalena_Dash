"""
train_models.py
───────────────────────────────────────────────────────────────
Entrenamiento, selección de variables y serialización de modelos
para el dashboard CSS Magdalena.

Ejecutar UNA vez antes de levantar el app:
    python train_models.py

Genera:
    models/results.pkl   ← todos los resultados, métricas y predicciones
    models/models.pkl    ← objetos sklearn entrenados listos para inferencia
───────────────────────────────────────────────────────────────
"""

import os, pickle, itertools, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
from tqdm import tqdm
from tqdm.auto import trange
os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
DATA_FILE    = "puntos_finales_calamar.csv"
TARGET       = "SSC"
KMS_TRAIN    = [14, 17, 18, 19]       # estaciones para calibración
MAX_FEATURES = 3                       # máximo predictores por combinación

CANDIDATES = [
    "blue", "green", "red", "rojo 1", "rojo 3",
    "NIR", "RANS", "VNES", "NIR/RED", "NDTI"
]

# Hiperparámetros — ajustados para n~29
RF_PARAMS  = dict(n_estimators=500, max_depth=4, min_samples_leaf=3,
                  max_features="sqrt", random_state=42, n_jobs=-1)
GBM_PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=3,
                  min_samples_leaf=3, subsample=0.8, random_state=42)
SVR_PARAMS = dict(kernel="rbf", C=400, epsilon=15, gamma="scale")

MODEL_DEFS = {
    "lineal": ("Regresión Lineal",       lambda: LinearRegression()),
    "rf":     ("Random Forest",          lambda: RandomForestRegressor(**RF_PARAMS)),
    "gbm":    ("Gradient Boosting",      lambda: GradientBoostingRegressor(**GBM_PARAMS)),
    "svr":    ("Support Vector (RBF)",   lambda: SVR(**SVR_PARAMS)),
}

# ─────────────────────────────────────────
# CARGA Y FILTRADO
# ─────────────────────────────────────────
print("Cargando datos...")
df_raw = pd.read_csv(DATA_FILE)
df_raw["km_num"] = pd.to_numeric(df_raw["km"], errors="coerce")
df_train = df_raw[df_raw["km_num"].isin(KMS_TRAIN)].copy()
df_train = df_train.dropna(subset=[TARGET] + CANDIDATES)
df_train = df_train.reset_index(drop=True)

print(f"  Puntos de calibración: {len(df_train)}")
print(f"  Km: {sorted(df_train['km_num'].unique().tolist())}")
print(f"  SSC rango: {df_train[TARGET].min():.1f} – {df_train[TARGET].max():.1f} mg/L")

y  = df_train[TARGET].values
kms= df_train["km_num"].values

# ─────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────
def mape(y_true, y_pred):
    mask = np.array(y_true) != 0
    return 100 * np.mean(np.abs(
        (np.array(y_true)[mask] - np.array(y_pred)[mask]) / np.array(y_true)[mask]
    ))

def bias(y_true, y_pred):
    return float(np.mean(np.array(y_pred) - np.array(y_true)))

def loocv_predict(model_fn, X, y, needs_scale=False):
    """LOOCV — devuelve predicciones out-of-sample para cada punto."""
    loo  = LeaveOneOut()
    yhat = np.zeros(len(y), dtype=float)
    for tr_idx, te_idx in loo.split(X):
        mdl = model_fn()
        Xtr, Xte = X[tr_idx], X[te_idx]
        if needs_scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        mdl.fit(Xtr, y[tr_idx])
        yhat[te_idx] = mdl.predict(Xte)
    return yhat

def metrics_dict(y_true, y_pred, prefix=""):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mp   = mape(y_true, y_pred)
    bi   = bias(y_true, y_pred)
    return {f"{prefix}r2":r2, f"{prefix}rmse":rmse,
            f"{prefix}mape":mp, f"{prefix}bias":bi}

# ─────────────────────────────────────────
# 1. SELECCIÓN DE VARIABLES POR R² LOOCV
# ─────────────────────────────────────────
print("\nSelección de variables (R² LOOCV Regresión Lineal)...")

avail = [c for c in CANDIDATES if c in df_train.columns]
best_combo = None
best_r2_loo = -np.inf
selection_log = []

for k in range(1, MAX_FEATURES + 1):
    for combo in itertools.combinations(avail, k):
        X_c = df_train[list(combo)].values
        yhat_loo = loocv_predict(LinearRegression, X_c, y)
        r2_loo   = r2_score(y, yhat_loo)
        rmse_loo = np.sqrt(mean_squared_error(y, yhat_loo))
        selection_log.append({
            "features": list(combo),
            "n_features": k,
            "r2_loo": r2_loo,
            "rmse_loo": rmse_loo,
        })
        if r2_loo > best_r2_loo:
            best_r2_loo = r2_loo
            best_combo  = list(combo)

sel_df = pd.DataFrame(selection_log).sort_values("r2_loo", ascending=False)
print(f"  Mejor combinación: {best_combo}  →  R²_LOOCV = {best_r2_loo:.3f}")
print(f"  Top 5:")
print(sel_df.head(5)[["features","n_features","r2_loo","rmse_loo"]].to_string(index=False))

# ─────────────────────────────────────────
# 2. ENTRENAR TODOS LOS MODELOS
#    con la mejor combinación de variables
# ─────────────────────────────────────────
print(f"\n{'═'*58}")
print("ETAPA 2/3 — Entrenamiento y LOOCV de modelos finales")
print("─"*58)
print(f"  Features: {best_combo}")
print(f"  n = {len(y)} puntos\n")

X_best  = df_train[best_combo].values
results = {}
trained = {}

for key, (label, fn) in MODEL_DEFS.items():
    print(f"  [{key.upper()}] {label}")
    needs_sc = key == "svr"

    # ── Calibración completa ──
    mdl = fn()
    Xtr = StandardScaler().fit_transform(X_best) if needs_sc else X_best
    sc_final = None
    if needs_sc:
        sc_final = StandardScaler()
        Xtr = sc_final.fit_transform(X_best)
        mdl.fit(Xtr, y)
    else:
        mdl.fit(X_best, y)
        Xtr = X_best

    yhat_cal = mdl.predict(Xtr)

    # ── LOOCV ──
    print(f"    LOOCV ({len(y)} iteraciones)...", end=" ", flush=True)
    yhat_loo = loocv_predict(fn, X_best, y, needs_scale=needs_sc)
    print("✓")

    # ── Métricas ──
    m_cal = metrics_dict(y, yhat_cal, "cal_")
    m_loo = metrics_dict(y, yhat_loo, "loo_")

    results[key] = {
        "label":    label,
        "features": best_combo,
        "y_true":   y.tolist(),
        "yhat_cal": yhat_cal.tolist(),
        "yhat_loo": yhat_loo.tolist(),
        "kms":      kms.tolist(),
        **m_cal, **m_loo,
    }
    trained[key] = {"model": mdl, "scaler": sc_final, "features": best_combo}

    # Equation string for linear
    if key == "lineal":
        coefs = mdl.coef_; inter = mdl.intercept_
        parts = [f"{c:.4f}·{f}" for c, f in zip(coefs, best_combo)]
        eq = "SSC = " + " + ".join(parts) + f" + {inter:.2f}"
        results[key]["equation"] = eq
        print(f"    Ecuación  : {eq}")
    else:
        results[key]["equation"] = f"{label}  |  features: {best_combo}"
    print(f"    R²_cal={m_cal['cal_r2']:.3f}  RMSE_cal={m_cal['cal_rmse']:.1f}")
    print(f"    R²_loo={m_loo['loo_r2']:.3f}  RMSE_loo={m_loo['loo_rmse']:.1f}  MAPE={m_loo['loo_mape']:.1f}%")
    print()

# ─────────────────────────────────────────
# 3. TABLA COMPARATIVA DE TODAS LAS COMBINACIONES
#    (top 10 por modelo para visualización)
# ─────────────────────────────────────────
print(f"{'═'*58}")
print("ETAPA 3/3 — Tabla comparativa (top 10 × 4 modelos)")
print("─"*58)
full_comparison = []
comp_combos = [(key, label, fn, row) 
               for key,(label,fn) in MODEL_DEFS.items() 
               for _,row in sel_df.head(10).iterrows()]

for key, label, fn, row in tqdm(comp_combos,
        desc="  Comparación extendida",
        bar_format="{l_bar}{bar:30}{r_bar}",
        colour="yellow", ncols=80):
    needs_sc = key == "svr"
    feats = row["features"]
    Xc    = df_train[feats].values
    try:
        yh_loo = loocv_predict(fn, Xc, y, needs_scale=needs_sc)
        r2l    = r2_score(y, yh_loo)
        rmsel  = np.sqrt(mean_squared_error(y, yh_loo))
        full_comparison.append({
            "model":    label,
            "features": " + ".join(feats),
            "n":        len(feats),
            "r2_loo":   round(r2l, 4),
            "rmse_loo": round(rmsel, 2),
        })
    except Exception:
        pass
comp_df = pd.DataFrame(full_comparison).sort_values(["model","r2_loo"], ascending=[True,False])

# ─────────────────────────────────────────
# 4. GUARDAR
# ─────────────────────────────────────────
payload = {
    "results":       results,
    "selection_log": sel_df.to_dict("records"),
    "comparison":    comp_df.to_dict("records"),
    "best_features": best_combo,
    "best_r2_loo":   best_r2_loo,
    "n_train":       len(df_train),
    "kms_train":     KMS_TRAIN,
    "y_true":        y.tolist(),
    "kms":           kms.tolist(),
}

with open("models/results.pkl", "wb") as f:
    pickle.dump(payload, f)
print(f"\n{'═'*58}")
print("GUARDADO")
print("─"*58)
print("  ✅ models/results.pkl")

with open("models/models.pkl", "wb") as f:
    pickle.dump(trained, f)
print("  ✅ models/models.pkl")

# ─────────────────────────────────────────
# 5. RESUMEN FINAL
# ─────────────────────────────────────────
print("\n" + "═"*58)
print("RESUMEN FINAL")
print("═"*58)
print(f"  Variables seleccionadas : {best_combo}")
print(f"  n calibración           : {len(df_train)}")
print(f"  Km incluidos            : {KMS_TRAIN}")
print()
print(f"  {'Modelo':<28} {'R²_cal':>7} {'R²_loo':>7} {'RMSE_loo':>9}")
print(f"  {'-'*55}")
for key, res in results.items():
    print(f"  {res['label']:<28} "
          f"{res['cal_r2']:>7.3f} "
          f"{res['loo_r2']:>7.3f} "
          f"{res['loo_rmse']:>9.1f} mg/L")
print("═"*58)
print("\nListo. Ahora puedes correr: python app.py")