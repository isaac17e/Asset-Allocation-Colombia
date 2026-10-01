"""
Rebalanceo táctico por bandas de tolerancia.

Un portafolio modelo se desvía de sus pesos objetivo por el simple movimiento
de los mercados. En lugar de rebalancear en calendario fijo (costoso) o nunca
(deriva de riesgo), se monitorean bandas: cuando una clase de activo se aparta
más de ±banda del objetivo, se dispara la alerta y se genera el plan de órdenes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORIAS_RV
from .utils import formato_cop, get_logger

log = get_logger("am_pm.rebalancing")


def derivar_pesos(
    pesos_objetivo: pd.Series, precios: pd.DataFrame, desde: pd.Timestamp | None = None
) -> pd.Series:
    """
    Pesos vigentes tras dejar correr el portafolio (buy & hold) desde `desde`.

    Es la forma correcta de medir la deriva: los pesos no se mueven por
    decisión del gestor sino por el desempeño relativo de cada fondo.
    """
    activos = [a for a in pesos_objetivo.index if a in precios.columns]
    panel = precios[activos].dropna(how="all")
    if desde is not None:
        panel = panel.loc[panel.index >= desde]
    if panel.empty or len(panel) < 2:
        return pesos_objetivo

    crecimiento = panel.iloc[-1] / panel.iloc[0]
    valor = pesos_objetivo.reindex(activos).fillna(0.0) * crecimiento
    total = float(valor.sum())
    if total <= 0:
        return pesos_objetivo
    return (valor / total).reindex(pesos_objetivo.index).fillna(0.0)


def pesos_por_categoria(pesos: pd.Series, categorias: pd.Series) -> pd.Series:
    """Agrega pesos de fondos a pesos por clase de activo."""
    agrupado = pesos.groupby(categorias.reindex(pesos.index), observed=True).sum()
    agrupado.index.name = "categoria"
    return agrupado


def evaluar_bandas(
    pesos_actuales: pd.Series, pesos_objetivo: pd.Series, categorias: pd.Series,
    banda: float = 0.05,
) -> pd.DataFrame:
    """
    Compara la composición vigente contra la objetivo por clase de activo.

    Devuelve la tabla de control con la desviación absoluta en puntos
    porcentuales, el estado (OK / ALERTA) y la acción sugerida.
    """
    actual = pesos_por_categoria(pesos_actuales, categorias)
    objetivo = pesos_por_categoria(pesos_objetivo, categorias)
    idx = objetivo.index.union(actual.index)
    actual, objetivo = actual.reindex(idx).fillna(0.0), objetivo.reindex(idx).fillna(0.0)

    desviacion = actual - objetivo
    tabla = pd.DataFrame(
        {
            "peso_objetivo": objetivo,
            "peso_actual": actual,
            "desviacion_pp": desviacion,
            "banda_inferior": objetivo - banda,
            "banda_superior": objetivo + banda,
        }
    )
    tabla["estado"] = np.where(tabla["desviacion_pp"].abs() > banda, "ALERTA", "OK")
    tabla["accion"] = np.select(
        [tabla["desviacion_pp"] > banda, tabla["desviacion_pp"] < -banda],
        ["VENDER / SOBREPONDERADO", "COMPRAR / SUBPONDERADO"],
        default="MANTENER",
    )
    tabla.index.name = "categoria"

    # Control adicional del techo/piso agregado de renta variable.
    rv_actual = float(actual.reindex(list(CATEGORIAS_RV)).fillna(0.0).sum())
    rv_objetivo = float(objetivo.reindex(list(CATEGORIAS_RV)).fillna(0.0).sum())
    fila_rv = pd.DataFrame(
        {
            "peso_objetivo": [rv_objetivo],
            "peso_actual": [rv_actual],
            "desviacion_pp": [rv_actual - rv_objetivo],
            "banda_inferior": [rv_objetivo - banda],
            "banda_superior": [rv_objetivo + banda],
            "estado": ["ALERTA" if abs(rv_actual - rv_objetivo) > banda else "OK"],
            "accion": [
                "REDUCIR RV" if rv_actual - rv_objetivo > banda
                else ("AUMENTAR RV" if rv_actual - rv_objetivo < -banda else "MANTENER")
            ],
        },
        index=pd.Index(["RENTA_VARIABLE_TOTAL"], name="categoria"),
    )
    return pd.concat([tabla, fila_rv])


def evaluar_bandas_fondo(
    pesos_actuales: pd.Series, pesos_objetivo: pd.Series, banda: float = 0.03
) -> pd.DataFrame:
    """Control de deriva a nivel de fondo individual."""
    idx = pesos_objetivo.index.union(pesos_actuales.index)
    objetivo = pesos_objetivo.reindex(idx).fillna(0.0)
    actual = pesos_actuales.reindex(idx).fillna(0.0)
    tabla = pd.DataFrame({"peso_objetivo": objetivo, "peso_actual": actual})
    tabla["desviacion_pp"] = tabla["peso_actual"] - tabla["peso_objetivo"]
    tabla["estado"] = np.where(tabla["desviacion_pp"].abs() > banda, "ALERTA", "OK")
    tabla.index.name = "fondo_id"
    return tabla[(tabla["peso_objetivo"] > 0) | (tabla["peso_actual"] > 0)]


def plan_ordenes(
    pesos_actuales: pd.Series, pesos_objetivo: pd.Series, patrimonio: float,
    costo_bps: float = 25.0, umbral_minimo: float = 0.005,
) -> pd.DataFrame:
    """
    Traduce la brecha de pesos en órdenes de suscripción y redención en COP.

    Las órdenes por debajo de `umbral_minimo` se omiten: su beneficio de
    tracking no compensa el costo operativo ni el impacto fiscal.
    """
    idx = pesos_objetivo.index.union(pesos_actuales.index)
    objetivo = pesos_objetivo.reindex(idx).fillna(0.0)
    actual = pesos_actuales.reindex(idx).fillna(0.0)
    delta = objetivo - actual
    delta[delta.abs() < umbral_minimo] = 0.0

    ordenes = pd.DataFrame(
        {
            "peso_actual": actual,
            "peso_objetivo": objetivo,
            "delta_peso": delta,
            "monto_cop": delta * patrimonio,
        }
    )
    ordenes["operacion"] = np.select(
        [ordenes["delta_peso"] > 0, ordenes["delta_peso"] < 0],
        ["SUSCRIBIR", "REDIMIR"], default="SIN OPERACION",
    )
    ordenes["costo_estimado_cop"] = ordenes["monto_cop"].abs() * (costo_bps / 10_000.0)
    ordenes.index.name = "fondo_id"
    return ordenes[ordenes["delta_peso"] != 0.0].sort_values("monto_cop", ascending=False)


def resumen_rebalanceo(ordenes: pd.DataFrame, patrimonio: float) -> dict[str, float]:
    """Métricas agregadas del rebalanceo: rotación y costo."""
    if ordenes.empty:
        return {"turnover": 0.0, "costo_cop": 0.0, "costo_pct": 0.0, "n_ordenes": 0}
    turnover = float(ordenes["delta_peso"].abs().sum() / 2.0)
    costo = float(ordenes["costo_estimado_cop"].sum())
    return {
        "turnover": turnover,
        "costo_cop": costo,
        "costo_pct": costo / patrimonio if patrimonio > 0 else np.nan,
        "n_ordenes": int(len(ordenes)),
    }


def informe_tactico(
    pesos_objetivo: pd.Series, precios: pd.DataFrame, categorias: pd.Series,
    patrimonio: float, banda_categoria: float = 0.05, banda_fondo: float = 0.03,
    costo_bps: float = 25.0, dias_deriva: int = 91,
) -> dict[str, pd.DataFrame | dict]:
    """
    Informe completo de rebalanceo táctico.

    Simula la deriva del portafolio modelo durante los últimos `dias_deriva`
    días calendario y evalúa las bandas contra los pesos objetivo vigentes.
    """
    inicio = precios.index[-1] - pd.Timedelta(days=dias_deriva)
    desde = precios.index[precios.index >= inicio][0]
    actuales = derivar_pesos(pesos_objetivo, precios, desde)
    bandas_cat = evaluar_bandas(actuales, pesos_objetivo, categorias, banda_categoria)
    bandas_fondo = evaluar_bandas_fondo(actuales, pesos_objetivo, banda_fondo)
    ordenes = plan_ordenes(actuales, pesos_objetivo, patrimonio, costo_bps)
    resumen = resumen_rebalanceo(ordenes, patrimonio)

    alertas = int((bandas_cat["estado"] == "ALERTA").sum())
    log.info(
        "Deriva desde %s: %d alertas de categoría | turnover %.2f%% | costo %s",
        desde.date(), alertas, resumen["turnover"] * 100, formato_cop(resumen["costo_cop"]),
    )
    return {
        "fecha_referencia": desde,
        "pesos_actuales": actuales,
        "bandas_categoria": bandas_cat,
        "bandas_fondo": bandas_fondo,
        "ordenes": ordenes,
        "resumen": resumen,
    }
