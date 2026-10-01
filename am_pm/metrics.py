"""
Métricas de riesgo y retorno sobre series de valor de unidad de FICs.

Todas las funciones asumen series indexadas por fecha (DatetimeIndex) y
devuelven magnitudes anualizadas salvo indicación contraria.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DIAS_CALENDARIO_ANIO, DIAS_HABILES_ANIO

_FRECUENCIAS_CONOCIDAS = (12.0, 52.0, DIAS_HABILES_ANIO, DIAS_CALENDARIO_ANIO)


def periodos_por_anio(index: pd.Index) -> float:
    """
    Infiere la frecuencia efectiva de la serie.

    Los FICs colombianos publican valor de unidad en calendario corrido (365),
    mientras que ETFs y series bursátiles siguen el calendario bursátil (252).
    Se detecta automáticamente en vez de asumir una constante.
    """
    idx = pd.DatetimeIndex(index)
    if len(idx) < 3:
        return DIAS_HABILES_ANIO
    dias = (idx[-1] - idx[0]).days
    if dias <= 0:
        return DIAS_HABILES_ANIO
    obs_anio = (len(idx) - 1) * DIAS_CALENDARIO_ANIO / dias
    return min(_FRECUENCIAS_CONOCIDAS, key=lambda f: abs(f - obs_anio))


def observaciones_en(dias: float, ppa: float) -> int:
    """Número de observaciones que cubren `dias` calendario a la frecuencia `ppa`."""
    return max(1, int(round(dias * ppa / DIAS_CALENDARIO_ANIO)))


def retornos_simples(precios: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Retornos aritméticos período a período, sin relleno de faltantes."""
    return precios.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def retorno_anualizado(precios: pd.Series) -> float:
    """CAGR calculado sobre días calendario efectivos (robusto a huecos)."""
    serie = precios.dropna()
    if len(serie) < 2:
        return np.nan
    dias = (serie.index[-1] - serie.index[0]).days
    if dias <= 0 or serie.iloc[0] <= 0:
        return np.nan
    total = serie.iloc[-1] / serie.iloc[0]
    if total <= 0:
        return np.nan
    return float(total ** (DIAS_CALENDARIO_ANIO / dias) - 1.0)


def volatilidad_anualizada(retornos: pd.Series, ppa: float | None = None) -> float:
    """Desviación estándar anualizada de los retornos."""
    r = retornos.dropna()
    if len(r) < 5:
        return np.nan
    ppa = ppa or periodos_por_anio(r.index)
    return float(r.std(ddof=1) * np.sqrt(ppa))


def ratio_sharpe(retorno_anual: float, vol_anual: float, rf: float) -> float:
    """Exceso de retorno por unidad de volatilidad total."""
    if not np.isfinite(vol_anual) or vol_anual <= 1e-9 or not np.isfinite(retorno_anual):
        return np.nan
    return float((retorno_anual - rf) / vol_anual)


def desviacion_downside(retornos: pd.Series, rf: float, ppa: float | None = None) -> float:
    """
    Semidesviación anualizada respecto al MAR (r_f convertido a la frecuencia
    de la serie). Solo penaliza los retornos por debajo del umbral.
    """
    r = retornos.dropna()
    if len(r) < 5:
        return np.nan
    ppa = ppa or periodos_por_anio(r.index)
    mar = (1.0 + rf) ** (1.0 / ppa) - 1.0
    exceso = np.minimum(r.to_numpy(dtype=float) - mar, 0.0)
    return float(np.sqrt(np.mean(exceso ** 2)) * np.sqrt(ppa))


def ratio_sortino(
    retorno_anual: float, retornos: pd.Series, rf: float, ppa: float | None = None
) -> float:
    """Exceso de retorno por unidad de riesgo bajista."""
    dd = desviacion_downside(retornos, rf, ppa)
    if not np.isfinite(dd) or dd <= 1e-9 or not np.isfinite(retorno_anual):
        return np.nan
    return float((retorno_anual - rf) / dd)


def max_drawdown(precios: pd.Series) -> float:
    """Caída máxima pico-valle (valor negativo o cero)."""
    serie = precios.dropna()
    if len(serie) < 2:
        return np.nan
    maximo = serie.cummax()
    return float((serie / maximo - 1.0).min())


def ratio_calmar(retorno_anual: float, mdd: float) -> float:
    """Retorno anualizado sobre la caída máxima."""
    if not np.isfinite(mdd) or abs(mdd) < 1e-9 or not np.isfinite(retorno_anual):
        return np.nan
    return float(retorno_anual / abs(mdd))


def var_historico(retornos: pd.Series, alpha: float = 0.05) -> float:
    """VaR histórico de un período al nivel `alpha` (valor negativo)."""
    r = retornos.dropna()
    if len(r) < 20:
        return np.nan
    return float(np.quantile(r.to_numpy(dtype=float), alpha))


def cvar_historico(retornos: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall histórico al nivel `alpha`."""
    r = retornos.dropna()
    if len(r) < 20:
        return np.nan
    var = np.quantile(r.to_numpy(dtype=float), alpha)
    cola = r[r <= var]
    return float(cola.mean()) if len(cola) else float(var)


def beta_contra(retornos: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """
    Beta y R² de una regresión OLS del fondo contra un benchmark.

    Se usa para inferir la clase de activo de fondos cuyo nombre comercial no
    revela su política de inversión.
    """
    df = pd.concat([retornos, benchmark], axis=1, join="inner").dropna()
    if len(df) < 60:
        return np.nan, np.nan
    y = df.iloc[:, 0].to_numpy(dtype=float)
    x = df.iloc[:, 1].to_numpy(dtype=float)
    if np.std(x) < 1e-12:
        return np.nan, np.nan
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuo = y - X @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuo ** 2)) / ss_tot if ss_tot > 1e-18 else np.nan
    return float(coef[1]), r2


def resumen_metricas(precios: pd.Series, rf: float) -> dict[str, float]:
    """Ficha completa de desempeño de un fondo o de una curva de equity."""
    serie = precios.dropna()
    rets = retornos_simples(serie)
    ppa = periodos_por_anio(serie.index)
    r_anual = retorno_anualizado(serie)
    vol = volatilidad_anualizada(rets, ppa)
    mdd = max_drawdown(serie)
    return {
        "retorno_anual": r_anual,
        "vol_anual": vol,
        "sharpe": ratio_sharpe(r_anual, vol, rf),
        "sortino": ratio_sortino(r_anual, rets, rf, ppa),
        "max_drawdown": mdd,
        "calmar": ratio_calmar(r_anual, mdd),
        "var_95_diario": var_historico(rets),
        "cvar_95_diario": cvar_historico(rets),
        "pct_periodos_positivos": float((rets > 0).mean()) if len(rets) else np.nan,
        "n_obs": int(len(serie)),
        "periodos_anio": ppa,
    }


def matriz_covarianza(
    retornos: pd.DataFrame, ppa: float | None = None, shrinkage: float | None = None
) -> pd.DataFrame:
    """
    Covarianza anualizada con shrinkage hacia el modelo de correlación
    constante. Si `shrinkage` es None se estima la intensidad óptima con la
    aproximación de Ledoit-Wolf; el resultado se fuerza a ser definido positivo.
    """
    r = retornos.dropna(how="any")
    ppa = ppa or periodos_por_anio(r.index)
    muestra = r.cov().to_numpy(dtype=float) * ppa
    n = muestra.shape[0]
    if n == 0:
        return pd.DataFrame(muestra, index=r.columns, columns=r.columns)

    std = np.sqrt(np.clip(np.diag(muestra), 1e-18, None))
    corr = muestra / np.outer(std, std)
    fuera = corr[~np.eye(n, dtype=bool)]
    corr_media = float(np.mean(fuera)) if fuera.size else 0.0
    objetivo = np.full((n, n), corr_media) * np.outer(std, std)
    np.fill_diagonal(objetivo, np.diag(muestra))

    if shrinkage is None:
        # Aproximación de la intensidad óptima: crece con el ratio nº activos/nº obs.
        t = max(len(r), 2)
        shrinkage = float(np.clip(n / t, 0.05, 0.85))
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))

    cov = (1.0 - shrinkage) * muestra + shrinkage * objetivo
    cov = 0.5 * (cov + cov.T)
    autovalores = np.linalg.eigvalsh(cov)
    if autovalores.min() <= 0:
        cov += np.eye(n) * (abs(autovalores.min()) + 1e-10)
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)


def retornos_esperados(
    precios: pd.DataFrame, shrinkage_transversal: float = 0.60,
    grupos: pd.Series | None = None,
) -> pd.Series:
    """
    Vector de retornos esperados: CAGR histórico contraído hacia la media de
    su clase de activo (estimador tipo James-Stein). Reduce el sesgo de la
    optimización hacia los fondos con mejor desempeño reciente.

    El objetivo es la media de la clase y no la del universo: contraer un fondo
    de liquidez hacia una media que incluye renta variable le atribuye un
    exceso sobre r_f que no tiene, y con volatilidad de 0,3% ese exceso ficticio
    se convierte en un Sharpe que domina la optimización. Sin `grupos` se usa
    la media transversal del universo.
    """
    mu = precios.apply(retorno_anualizado)
    mu = mu.replace([np.inf, -np.inf], np.nan).dropna()
    if mu.empty:
        return mu
    gran_media = float(mu.mean())
    if grupos is None:
        objetivo = pd.Series(gran_media, index=mu.index)
    else:
        clase = grupos.reindex(mu.index)
        objetivo = mu.groupby(clase).transform("mean").fillna(gran_media)
    peso = float(np.clip(shrinkage_transversal, 0.0, 1.0))
    return peso * mu + (1.0 - peso) * objetivo
