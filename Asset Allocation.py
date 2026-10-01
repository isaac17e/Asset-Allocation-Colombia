from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from dataclasses import replace
from pathlib import Path

from am_pm.config import METODOS, PERFILES as NOMBRES_PERFIL, ConfigAM
from am_pm.pipeline import ejecutar
from am_pm.utils import configurar_logging


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Asset Allocation.py",
        description="Asset Allocation Manager para FICs, ETFs e índices colombianos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g_datos = p.add_argument_group("datos")
    g_datos.add_argument("--dataset", default=None, help="ID del dataset SODA (por defecto qhpu-8ixx)")
    g_datos.add_argument("--app-token", default=None, help="App token de datos.gov.co (evita el throttling)")
    g_datos.add_argument("--fecha-corte", default=None, help="Fecha de corte AAAA-MM-DD (por defecto, la última publicada)")
    g_datos.add_argument("--lookback", type=float, default=4.0, help="Años de historia a descargar (def. 4)")
    g_datos.add_argument("--top-n", type=int, default=70, help="Máximo de fondos en el universo curado (def. 70)")
    g_datos.add_argument("--aum-minimo", type=float, default=30e9, help="AUM mínimo por fondo en COP (def. 30e9)")
    g_datos.add_argument("--sin-cache", action="store_true", help="Ignora la cache local y vuelve a descargar")
    g_datos.add_argument("--offline", action="store_true", help="Usa datos sintéticos reproducibles")
    g_datos.add_argument("--respaldo-sintetico", action="store_true",
                         help="Si la API falla, continúa con datos sintéticos en vez de detenerse")

    g_opt = p.add_argument_group("optimización")
    g_opt.add_argument("--perfil", action="append", choices=list(NOMBRES_PERFIL),
                       help="Perfil a calcular (repetible). Por defecto, los tres.")
    g_opt.add_argument("--metodo", action="append", choices=list(METODOS),
                       help="Método de optimización (repetible). Por defecto, los cuatro.")
    g_opt.add_argument("--shrinkage-mu", type=float, default=0.60,
                       help="Peso de la media histórica frente a la transversal (def. 0.60)")
    g_opt.add_argument("--patrimonio", type=float, default=1e9, help="Patrimonio del mandato en COP (def. 1e9)")

    g_reb = p.add_argument_group("rebalanceo y backtest")
    g_reb.add_argument("--banda", type=float, default=None, help="Banda de rebalanceo por categoría (def. 0.05)")
    g_reb.add_argument("--costo-bps", type=float, default=25.0, help="Costo de transacción en bps (def. 25)")
    g_reb.add_argument("--ventana-estimacion", type=int, default=365,
                       help="Días calendario por ventana de estimación (def. 365)")
    g_reb.add_argument("--paso-rebalanceo", type=int, default=91,
                       help="Días calendario entre rebalanceos (def. 91)")
    g_reb.add_argument("--sin-backtest", action="store_true", help="Omite el backtest walk-forward")

    g_out = p.add_argument_group("presentación de resultados")
    g_out.add_argument("--outdir", default="outputs",
                       help="Directorio donde se deja el informe HTML (def. outputs)")
    g_out.add_argument("--html", default=None,
                       help="Ruta explícita del informe HTML (anula --outdir)")
    g_out.add_argument("--sin-html", action="store_true",
                       help="Sólo tablas en consola: no genera el informe gráfico")
    g_out.add_argument("--abrir", action="store_true",
                       help="Abre el informe HTML en el navegador al terminar")
    g_out.add_argument("--max-filas", type=int, default=None,
                       help="Tope de filas por tabla en consola (def. sin recorte)")

    p.add_argument("--verbose", "-v", action="store_true", help="Logging en nivel DEBUG")
    return p


def construir_config(args: argparse.Namespace) -> ConfigAM:
    """Traduce los argumentos de consola a la configuración del pipeline."""
    cfg = ConfigAM()
    datos = replace(
        cfg.datos,
        dataset_id=args.dataset or cfg.datos.dataset_id,
        app_token=args.app_token,
        fecha_corte=args.fecha_corte,
        lookback_anios=args.lookback,
        top_n_fondos=args.top_n,
        aum_minimo_cop=args.aum_minimo,
        usar_cache=not args.sin_cache,
        respaldo_sintetico=args.respaldo_sintetico,
    )
    metodos = tuple(args.metodo) if args.metodo else cfg.optimizacion.metodos
    optimizacion = replace(cfg.optimizacion, metodos=metodos, shrinkage_mu=args.shrinkage_mu)
    rebalanceo = replace(cfg.rebalanceo, costo_bps=args.costo_bps)
    if args.banda is not None:
        rebalanceo = replace(rebalanceo, banda_categoria=args.banda)
    backtest = replace(
        cfg.backtest,
        ventana_estimacion_dias=args.ventana_estimacion,
        paso_rebalanceo_dias=args.paso_rebalanceo,
        costo_bps=args.costo_bps,
    )
    if args.html:
        destino = Path(args.html)
        outdir, nombre_informe = destino.parent, destino.name
    else:
        outdir, nombre_informe = Path(args.outdir), cfg.nombre_informe

    return replace(
        cfg, datos=datos, optimizacion=optimizacion, rebalanceo=rebalanceo, backtest=backtest,
        outdir=outdir, nombre_informe=nombre_informe, patrimonio_cop=args.patrimonio,
        modo_offline=args.offline, generar_html=not args.sin_html,
        max_filas_tabla=args.max_filas,
    )


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    configurar_logging(logging.DEBUG if args.verbose else logging.INFO)

    cfg = construir_config(args)
    perfiles = tuple(args.perfil) if args.perfil else None

    # Las bandas de rebalanceo por perfil se sobrescriben sólo si el usuario las fija.
    if args.banda is not None:
        from dataclasses import replace as _replace
        from am_pm import profiles

        for clave, perfil in profiles.PERFILES.items():
            profiles.PERFILES[clave] = _replace(perfil, banda_rebalanceo=args.banda)

    try:
        resultado = ejecutar(cfg, perfiles=perfiles, con_backtest=not args.sin_backtest)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("am_pm").exception("La corrida falló: %s", exc)
        return 1

    informe = resultado.artefactos.get("informe_html")
    if args.abrir and informe is not None:
        webbrowser.open(informe.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
