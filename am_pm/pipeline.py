"""
Orquestación end-to-end del Asset Allocation Manager.

Encadena las cuatro etapas del proceso de inversión:
extracción → universo curado → asignación por perfil → validación y artefactos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .allocation import ResultadoAsignacion, construir_portafolios
from .backtest import ResultadoBacktest, ejecutar_backtest
from .config import CATEGORIAS, ConfigAM
from .ingestion import cargar_datos
from .profiles import PERFILES
from .rebalancing import informe_tactico
from .reporting import (
    generar_informe,
    mostrar_backtest,
    mostrar_composicion,
    mostrar_pesos,
    mostrar_politica,
    mostrar_rebalanceo,
    mostrar_tabla_comparativa,
    mostrar_universo,
)
from .universe import UniversoCurado, construir_universo
from .utils import formato_cop, get_logger

log = get_logger("am_pm.pipeline")


@dataclass
class ResultadoAMPM:
    """Salida completa de una corrida del AM-PM."""

    universo: UniversoCurado
    asignacion: ResultadoAsignacion
    backtest: ResultadoBacktest | None
    rebalanceo: dict[str, dict]
    artefactos: dict[str, Path] = field(default_factory=dict)
    duracion_seg: float = 0.0


def ejecutar(
    cfg: ConfigAM, perfiles: tuple[str, ...] | None = None, con_backtest: bool = True
) -> ResultadoAMPM:
    """Ejecuta el pipeline completo y deja los artefactos en `cfg.outdir`."""
    t0 = time.time()
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nombres = perfiles or tuple(PERFILES)

    log.info("=" * 78)
    log.info("AM-PM · Asset Allocation Manager | perfiles: %s", ", ".join(nombres))
    log.info("=" * 78)

    # 1 — Extracción -------------------------------------------------------
    datos = cargar_datos(cfg.datos, offline=cfg.modo_offline, semilla=cfg.semilla)

    # 2 — Universo curado --------------------------------------------------
    universo = construir_universo(datos, cfg.datos, cfg.optimizacion)

    # 3 — Asignación por perfil -------------------------------------------
    asignacion = construir_portafolios(universo, cfg.optimizacion, nombres)

    # 4 — Rebalanceo táctico ----------------------------------------------
    rebalanceo: dict[str, dict] = {}
    for nombre in nombres:
        res = asignacion.perfiles[nombre]
        # El portafolio de referencia operativo es el de máximo Sharpe.
        metodo_ref = "MARKOWITZ_SHARPE" if "MARKOWITZ_SHARPE" in res.pesos.columns else res.pesos.columns[0]
        rebalanceo[nombre] = informe_tactico(
            pesos_objetivo=res.pesos[metodo_ref],
            precios=universo.precios,
            categorias=universo.mapa_categoria(),
            patrimonio=cfg.patrimonio_cop,
            banda_categoria=res.perfil.banda_rebalanceo,
            banda_fondo=cfg.rebalanceo.banda_fondo,
            costo_bps=cfg.rebalanceo.costo_bps,
            dias_deriva=cfg.rebalanceo.dias_deriva,
        )

    # 5 — Backtest walk-forward -------------------------------------------
    backtest: ResultadoBacktest | None = None
    if con_backtest:
        try:
            backtest = ejecutar_backtest(universo, cfg.backtest, cfg.optimizacion, nombres)
        except ValueError as exc:
            log.error("Backtest omitido: %s", exc)

    # 6 — Presentación de resultados --------------------------------------
    artefactos = _publicar_resultados(
        universo, asignacion, backtest, rebalanceo, cfg, outdir
    )

    duracion = time.time() - t0
    resultado = ResultadoAMPM(universo, asignacion, backtest, rebalanceo, artefactos, duracion)
    imprimir_resumen(resultado, cfg)
    return resultado


def _publicar_resultados(
    universo: UniversoCurado, asignacion: ResultadoAsignacion,
    backtest: ResultadoBacktest | None, rebalanceo: dict[str, dict],
    cfg: ConfigAM, outdir: Path,
) -> dict[str, Path]:
    """
    Vuelca todo el detalle tabular a la terminal y consolida los gráficos en un
    único informe HTML. No se escriben CSV ni imágenes sueltas: la mesa lee las
    cifras en consola y revisa los gráficos en un solo documento con scroll.
    """
    mostrar_universo(universo, max_filas=cfg.max_filas_tabla)
    mostrar_politica(PERFILES)
    mostrar_composicion(asignacion)
    mostrar_pesos(asignacion, universo.fondos, max_filas=cfg.max_filas_tabla)
    mostrar_rebalanceo(rebalanceo, cfg.patrimonio_cop)
    mostrar_backtest(backtest)
    mostrar_tabla_comparativa(asignacion, backtest)

    artefactos: dict[str, Path] = {}
    if cfg.generar_html:
        artefactos["informe_html"] = generar_informe(
            universo, asignacion, backtest, outdir / cfg.nombre_informe
        )
    return artefactos


# --------------------------------------------------------------------------- #
# Resumen ejecutivo
# --------------------------------------------------------------------------- #
def imprimir_resumen(resultado: ResultadoAMPM, cfg: ConfigAM) -> None:
    """Informe de comité: qué se construyó, con qué datos y qué resultó."""
    u = resultado.universo
    lineas: list[str] = []
    add = lineas.append

    add("")
    add("=" * 78)
    add("  AM-PM · RESUMEN EJECUTIVO DE ASIGNACIÓN")
    add("=" * 78)
    add(f"  Fecha de corte     : {u.fecha_corte.date()}")
    add(f"  Origen de datos    : {u.origen}"
        + ("   *** DATOS SIMULADOS: NO USAR PARA DECISIONES ***" if u.origen == "SINTETICO" else ""))
    add(f"  Universo curado    : {len(u.fondos)} fondos | {u.precios.shape[0]} observaciones")
    add(f"  Ventana histórica  : {u.precios.index[0].date()} a {u.precios.index[-1].date()}")
    add(f"  Tasa libre riesgo  : {u.rf:.2%} anual (proxy compuesto RF_CORTO)")
    add(f"  Patrimonio mandato : {formato_cop(cfg.patrimonio_cop)}")
    add("")

    add("  UNIVERSO POR CLASE DE ACTIVO")
    add("  " + "-" * 74)
    resumen_cat = u.fondos.groupby("categoria", observed=True).agg(
        fondos=("aum_cop", "size"), aum_bn=("aum_cop", lambda s: s.sum() / 1e12),
        ret=("retorno_anual", "median"), vol=("vol_anual", "median"), sharpe=("sharpe", "median"),
    ).reindex(CATEGORIAS).dropna(how="all")
    add(f"  {'Categoría':<20}{'Fondos':>8}{'AUM (bn COP)':>15}{'Ret med':>10}{'Vol med':>10}{'Sharpe':>9}")
    for cat, fila in resumen_cat.iterrows():
        add(f"  {cat:<20}{int(fila['fondos']):>8}{fila['aum_bn']:>15.1f}"
            f"{fila['ret']:>9.1%}{fila['vol']:>10.1%}{fila['sharpe']:>9.2f}")
    add("")

    for nombre, res in resultado.asignacion.perfiles.items():
        add(f"  PERFIL {nombre} — {res.perfil.descripcion}")
        add("  " + "-" * 74)
        add(f"  {'Método':<16}{'Ret esp':>9}{'Vol esp':>9}{'Sharpe':>9}{'RV':>8}{'Pos':>6}{'DivRatio':>10}")
        for metodo, fila in res.estadisticas.iterrows():
            add(f"  {metodo:<16}{fila['retorno_esperado']:>8.1%}{fila['vol_esperada']:>9.1%}"
                f"{fila['sharpe_ex_ante']:>9.2f}{fila['peso_RENTA_VARIABLE']:>8.0%}"
                f"{int(fila['n_posiciones']):>6}{fila['ratio_diversificacion']:>10.2f}")
        alertas = resultado.rebalanceo[nombre]["bandas_categoria"]
        n_alertas = int((alertas["estado"] == "ALERTA").sum())
        turnover = resultado.rebalanceo[nombre]["resumen"]["turnover"]
        add(f"  Bandas ±{res.perfil.banda_rebalanceo:.0%}: {n_alertas} alerta(s) | "
            f"turnover requerido {turnover:.1%} | "
            f"costo {formato_cop(resultado.rebalanceo[nombre]['resumen']['costo_cop'])}")
        if res.notas:
            add(f"  Relajaciones de mandato: {len(res.notas)}")
            for nota in res.notas:
                add(f"    · {nota}")
        add("")

    if resultado.backtest is not None:
        add("  BACKTEST WALK-FORWARD (neto de costos de transacción)")
        add(f"  r_f realizada en el período: {resultado.backtest.rf_realizada:.2%} "
            "(referencia del Sharpe realizado)")
        add("  " + "-" * 74)
        add(f"  {'Estrategia':<34}{'CAGR':>9}{'Vol':>9}{'Sharpe':>9}{'MaxDD':>9}")
        for estrategia, fila in resultado.backtest.metricas.iterrows():
            add(f"  {estrategia:<34}{fila['retorno_anual']:>8.1%}{fila['vol_anual']:>9.1%}"
                f"{fila['sharpe']:>9.2f}{fila['max_drawdown']:>9.1%}")
        add("")

    if resultado.artefactos:
        add("  INFORME GRÁFICO")
        add("  " + "-" * 74)
        for ruta in resultado.artefactos.values():
            add(f"  · {ruta}")
            add(f"    Abrir con:  open '{ruta}'")
        add("")
    add(f"  Corrida completada en {resultado.duracion_seg:.1f}s")
    add("=" * 78)

    print("\n".join(lineas))
