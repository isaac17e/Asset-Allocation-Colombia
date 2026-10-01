"""
Backtest walk-forward de los portafolios modelo.

Protocolo (sin mirada al futuro):

  1. En cada fecha de rebalanceo se estiman r_f, μ y Σ **sólo** con la ventana
     de datos previa.
  2. Se reconstruye el universo elegible de esa fecha (fondos con historia
     completa en la ventana) y las restricciones del perfil sobre ese universo.
  3. Se optimiza y se mantienen las posiciones hasta el siguiente rebalanceo,
     dejando derivar los pesos con el mercado (buy & hold intra-período).
  4. La rotación entre los pesos derivados y los nuevos objetivos se cobra como
     costo de transacción.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ConfigBacktest, ConfigOptimizacion
from .metrics import (
    matriz_covarianza,
    observaciones_en,
    periodos_por_anio,
    resumen_metricas,
    retornos_esperados,
    retornos_simples,
)
from .optimizers import limpiar_pesos, resolver
from .profiles import PERFILES, construir_restricciones
from .universe import UniversoCurado, calcular_rf_dinamica, clasificar_fondos
from .allocation import aplicar_cardinalidad
from .utils import get_logger

log = get_logger("am_pm.backtest")

SEPARADOR = "::"


@dataclass
class ResultadoBacktest:
    """Curvas de equity, métricas realizadas y bitácora de rebalanceos."""

    equity: pd.DataFrame
    metricas: pd.DataFrame
    rebalanceos: pd.DataFrame
    #: r_f realizada en el mismo período del backtest (referencia de Sharpe y Sortino).
    rf_realizada: float = np.nan
    pesos_historicos: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def estrategias(self) -> list[str]:
        return list(self.equity.columns)


def _clave(perfil: str, metodo: str) -> str:
    return f"{perfil}{SEPARADOR}{metodo}"


def _crecimiento_tramo(precios_tramo: pd.DataFrame, pesos: pd.Series) -> pd.Series:
    """
    Valor del portafolio dentro de un tramo bajo buy & hold.

    Cada peso se capitaliza con el crecimiento de su fondo, de modo que la
    deriva de las ponderaciones queda incorporada de forma exacta.
    """
    activos = [a for a in pesos.index if a in precios_tramo.columns]
    p = precios_tramo[activos]
    w = pesos.reindex(activos).fillna(0.0).to_numpy(dtype=float)
    crecimiento = p.div(p.iloc[0], axis=1).to_numpy(dtype=float)
    return pd.Series(crecimiento @ w, index=p.index)


def _pesos_derivados(precios_tramo: pd.DataFrame, pesos: pd.Series) -> pd.Series:
    """Pesos al cierre del tramo tras la deriva de mercado."""
    activos = [a for a in pesos.index if a in precios_tramo.columns]
    p = precios_tramo[activos]
    valor = pesos.reindex(activos).fillna(0.0) * (p.iloc[-1] / p.iloc[0])
    total = float(valor.sum())
    return valor / total if total > 0 else pesos


def _universo_ventana(precios: pd.DataFrame, ventana: pd.DataFrame, min_vol: float = 1e-6) -> list[str]:
    """Fondos con historia completa y variación efectiva dentro de la ventana."""
    completos = ventana.columns[ventana.notna().all()]
    variables = [c for c in completos if float(ventana[c].pct_change().std()) > min_vol]
    return [c for c in variables if c in precios.columns]


def ejecutar_backtest(
    universo: UniversoCurado, cfg_bt: ConfigBacktest, cfg_opt: ConfigOptimizacion,
    perfiles: tuple[str, ...] | None = None,
) -> ResultadoBacktest:
    """Corre el walk-forward para todos los perfiles y métodos configurados."""
    precios = universo.precios.sort_index()
    categorias = universo.mapa_categoria()
    gestores = universo.mapa_gestor()
    nombres_perfil = perfiles or tuple(PERFILES)

    n_obs = len(precios)
    ppa_panel = periodos_por_anio(precios.index)
    min_obs = observaciones_en(cfg_bt.min_dias, ppa_panel)
    if n_obs < min_obs:
        raise ValueError(
            f"Historia insuficiente para el backtest: {n_obs} obs (< {min_obs})."
        )

    ventana_obs = observaciones_en(cfg_bt.ventana_estimacion_dias, ppa_panel)
    paso_obs = observaciones_en(cfg_bt.paso_rebalanceo_dias, ppa_panel)
    ventana_rf = (
        None if cfg_opt.ventana_rf_dias is None
        else observaciones_en(cfg_opt.ventana_rf_dias, ppa_panel)
    )
    puntos = list(range(ventana_obs, n_obs - 1, paso_obs))
    log.info("Backtest: %d rebalanceos | ventana=%d días (%d obs) | paso=%d días (%d obs) | costo=%.0f bps",
             len(puntos), cfg_bt.ventana_estimacion_dias, ventana_obs,
             cfg_bt.paso_rebalanceo_dias, paso_obs, cfg_bt.costo_bps)

    claves = [_clave(p, m) for p in nombres_perfil for m in cfg_opt.metodos]
    valores: dict[str, list[pd.Series]] = {k: [] for k in claves}
    nivel: dict[str, float] = {k: 1.0 for k in claves}
    pesos_previos: dict[str, pd.Series] = {}
    historial_pesos: dict[str, list[pd.Series]] = {k: [] for k in claves}
    bitacora: list[dict] = []

    t0 = time.time()
    for i, t in enumerate(puntos):
        fin = puntos[i + 1] if i + 1 < len(puntos) else n_obs - 1
        ventana = precios.iloc[t - ventana_obs : t]
        tramo = precios.iloc[t : fin + 1]
        fecha = precios.index[t]

        activos = _universo_ventana(precios, ventana)
        if len(activos) < 5:
            log.warning("Rebalanceo %s omitido: sólo %d fondos elegibles", fecha.date(), len(activos))
            continue

        ventana_act = ventana[activos]
        retornos_act = retornos_simples(ventana_act)
        ppa = periodos_por_anio(ventana_act.index)
        # La clase de activo se reestima con la ventana: la que asigna el panel
        # completo usa volatilidades y betas que en esa fecha no se conocían.
        categorias_t = clasificar_fondos(retornos_act, universo.fondos.loc[activos])["categoria"]
        n_reclasificados = int((categorias_t != categorias.reindex(activos)).sum())
        mu = retornos_esperados(ventana_act, cfg_opt.shrinkage_mu, categorias_t)
        cov = matriz_covarianza(retornos_act, ppa, cfg_opt.shrinkage_cov)
        rf_t = calcular_rf_dinamica(
            ventana_act, categorias_t, None, ventana_rf, cfg_opt.rf_fallback
        )

        for nombre in nombres_perfil:
            perfil = PERFILES[nombre]
            restr, _ = construir_restricciones(perfil, categorias_t, gestores.reindex(activos))
            for metodo in cfg_opt.metodos:
                clave = _clave(nombre, metodo)
                w = resolver(metodo, mu, cov, rf_t, restr, cfg_opt.n_arranques, cfg_opt.max_iter)
                w = limpiar_pesos(w, restr, cfg_opt.peso_minimo_operativo)
                w = aplicar_cardinalidad(w, perfil.max_fondos, restr, universo.fondos["aum_cop"])

                anterior = pesos_previos.get(clave)
                turnover = 1.0 if anterior is None else _turnover(anterior, w)
                costo = turnover * cfg_bt.costo_bps / 10_000.0
                nivel[clave] *= 1.0 - costo

                curva = _crecimiento_tramo(tramo, w) * nivel[clave]
                valores[clave].append(curva.iloc[1:] if i > 0 else curva)
                nivel[clave] = float(curva.iloc[-1])

                pesos_previos[clave] = _pesos_derivados(tramo, w)
                historial_pesos[clave].append(w.rename(fecha))
                bitacora.append(
                    {
                        "fecha": fecha, "perfil": nombre, "metodo": metodo,
                        "turnover": turnover, "costo_pct": costo,
                        "n_posiciones": int((w > 1e-6).sum()), "rf_ventana": rf_t,
                        "n_reclasificados": n_reclasificados,
                    }
                )
        log.info("Rebalanceo %d/%d — %s (%d fondos elegibles, %d con otra clase que en el panel)",
                 i + 1, len(puntos), fecha.date(), len(activos), n_reclasificados)

    equity = pd.DataFrame({k: pd.concat(v) for k, v in valores.items() if v}).sort_index()
    equity = equity[~equity.index.duplicated(keep="first")]
    equity = pd.concat([_benchmarks(precios, categorias, equity.index), equity], axis=1)

    # El Sharpe realizado se mide contra la caja de ese mismo período, no contra
    # la r_f del panel completo: en un ciclo de tasas, la r_f de otro período
    # domina el exceso de retorno de los portafolios de baja volatilidad.
    rf_realizada = calcular_rf_dinamica(
        precios.reindex(equity.index), categorias, universo.fondos["aum_cop"],
        None, universo.rf,
    )
    metricas = _metricas_equity(equity, rf_realizada)
    rebalanceos = pd.DataFrame(bitacora)
    pesos_hist = {k: pd.DataFrame(v).fillna(0.0) for k, v in historial_pesos.items() if v}
    log.info("Backtest completado en %.1fs | %d estrategias | %d fechas",
             time.time() - t0, equity.shape[1], equity.shape[0])
    return ResultadoBacktest(
        equity, metricas, rebalanceos, rf_realizada=rf_realizada, pesos_historicos=pesos_hist
    )


def _turnover(anterior: pd.Series, nuevo: pd.Series) -> float:
    """Rotación de una cara: media suma de diferencias absolutas de pesos."""
    idx = anterior.index.union(nuevo.index)
    a = anterior.reindex(idx).fillna(0.0)
    b = nuevo.reindex(idx).fillna(0.0)
    return float((b - a).abs().sum() / 2.0)


def _benchmarks(
    precios: pd.DataFrame, categorias: pd.Series, indice: pd.Index
) -> pd.DataFrame:
    """
    Referencias pasivas: la caja (RF_CORTO equiponderada), que es el costo de
    oportunidad real del cliente, y el universo completo equiponderado.
    """
    cols = {}
    ids_rf = [c for c in precios.columns if categorias.get(c) == "RF_CORTO"]
    if ids_rf:
        cols["BENCH::CAJA_RF_CORTO"] = _indice_equiponderado(precios[ids_rf], indice)
    cols["BENCH::UNIVERSO_1_N"] = _indice_equiponderado(precios, indice)
    return pd.DataFrame(cols, index=indice)


def _indice_equiponderado(precios: pd.DataFrame, indice: pd.Index) -> pd.Series:
    """Índice buy & hold equiponderado, normalizado a 1 en la fecha inicial."""
    panel = precios.reindex(indice).ffill().dropna(axis=1, how="any")
    if panel.empty:
        return pd.Series(1.0, index=indice)
    crecimiento = panel.div(panel.iloc[0], axis=1)
    return crecimiento.mean(axis=1)


def _metricas_equity(equity: pd.DataFrame, rf: float) -> pd.DataFrame:
    """Ficha de desempeño realizado de cada curva de equity."""
    filas = {col: resumen_metricas(equity[col].dropna(), rf) for col in equity.columns}
    tabla = pd.DataFrame(filas).T
    partes = [c.split(SEPARADOR) for c in tabla.index]
    tabla.insert(0, "perfil", [p[0] for p in partes])
    tabla.insert(1, "metodo", [p[1] if len(p) > 1 else "" for p in partes])
    tabla.index.name = "estrategia"
    return tabla.sort_values(["perfil", "sharpe"], ascending=[True, False])
