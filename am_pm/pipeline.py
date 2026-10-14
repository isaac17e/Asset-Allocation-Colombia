"""
Orquestación end-to-end del Asset Allocation Manager.

Encadena las cuatro etapas del proceso de inversión:
extracción → universo curado → asignación por perfil → validación y artefactos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .allocation import ResultadoAsignacion, construir_portafolios
from .backtest import ResultadoBacktest, ejecutar_backtest
from .config import CATEGORIAS, ConfigAM
from .ingestion import cargar_datos
from .profiles import PERFILES
from .rebalancing import informe_tactico
from .reporting import (
    generar_informe,
    imprimir_tabla,
    imprimir_titulo,
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
    imprimir_titulo("AM-PM · Resumen ejecutivo de asignación")
    print(f"  Fecha de corte     : {u.fecha_corte.date()}")
    print(f"  Origen de datos    : {u.origen}"
          + ("   *** DATOS SIMULADOS: NO USAR PARA DECISIONES ***" if u.origen == "SINTETICO" else ""))
    print(f"  Universo curado    : {len(u.fondos)} fondos | {u.precios.shape[0]} observaciones")
    print(f"  Ventana histórica  : {u.precios.index[0].date()} a {u.precios.index[-1].date()}")
    print(f"  Tasa libre riesgo  : {u.rf:.2%} anual (proxy compuesto RF_CORTO)")
    print(f"  Patrimonio mandato : {formato_cop(cfg.patrimonio_cop)}")

    resumen_cat = u.fondos.groupby("categoria", observed=True).agg(
        fondos=("aum_cop", "size"), aum_bn=("aum_cop", lambda s: s.sum() / 1e12),
        ret=("retorno_anual", "median"), vol=("vol_anual", "median"), sharpe=("sharpe", "median"),
    ).reindex(CATEGORIAS).dropna(how="all")
    imprimir_tabla(
        resumen_cat.reset_index(),
        titulo="Universo por clase de activo",
        formatos={"fondos": "int", "aum_bn": "num", "ret": "pct", "vol": "pct", "sharpe": "num"},
        encabezados={"categoria": "Categoría", "fondos": "Fondos", "aum_bn": "AUM (bn COP)",
                     "ret": "Ret med", "vol": "Vol med", "sharpe": "Sharpe"},
    )

    for nombre, res in resultado.asignacion.perfiles.items():
        rebalanceo = resultado.rebalanceo[nombre]
        n_alertas = int((rebalanceo["bandas_categoria"]["estado"] == "ALERTA").sum())
        imprimir_tabla(
            res.estadisticas.rename_axis("metodo").reset_index()[
                ["metodo", "retorno_esperado", "vol_esperada", "sharpe_ex_ante",
                 "peso_RENTA_VARIABLE", "n_posiciones", "ratio_diversificacion"]
            ],
            titulo=f"PERFIL {nombre} — {res.perfil.descripcion}",
            formatos={"retorno_esperado": "pct", "vol_esperada": "pct", "sharpe_ex_ante": "num",
                      "peso_RENTA_VARIABLE": "pct", "n_posiciones": "int",
                      "ratio_diversificacion": "num"},
            encabezados={"metodo": "Método", "retorno_esperado": "Ret esp", "vol_esperada": "Vol esp",
                         "sharpe_ex_ante": "Sharpe", "peso_RENTA_VARIABLE": "RV",
                         "n_posiciones": "Pos", "ratio_diversificacion": "DivRatio"},
            nota=(f"Bandas ±{res.perfil.banda_rebalanceo:.0%}: {n_alertas} alerta(s) | "
                  f"turnover requerido {rebalanceo['resumen']['turnover']:.1%} | "
                  f"costo {formato_cop(rebalanceo['resumen']['costo_cop'])}"),
        )
        if res.notas:
            print(f"  Relajaciones de mandato: {len(res.notas)}")
            for nota in res.notas:
                print(f"    · {nota}")

    if resultado.backtest is not None:
        imprimir_tabla(
            resultado.backtest.metricas[["retorno_anual", "vol_anual", "sharpe", "max_drawdown"]]
            .reset_index(),
            titulo="BACKTEST WALK-FORWARD (neto de costos de transacción)",
            formatos={"retorno_anual": "pct", "vol_anual": "pct", "sharpe": "num", "max_drawdown": "pct"},
            encabezados={"estrategia": "Estrategia", "retorno_anual": "CAGR", "vol_anual": "Vol",
                         "sharpe": "Sharpe", "max_drawdown": "MaxDD"},
            nota=(f"r_f realizada en el período: {resultado.backtest.rf_realizada:.2%} "
                  "(referencia del Sharpe realizado)"),
        )

    for ruta in resultado.artefactos.values():
        imprimir_titulo("Informe gráfico", nivel=2)
        print(f"  · {ruta}")
        print(f"    Abrir con:  open '{ruta}'")
    print(f"\n  Corrida completada en {resultado.duracion_seg:.1f}s")
