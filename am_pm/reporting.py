"""
Presentación de resultados: tablas en terminal e informe gráfico HTML.

El módulo no escribe CSV ni PNG. Toda la información tabular se imprime en la
consola con formato de mesa de inversión, y los gráficos —construidos en
[charts.py](charts.py) como SVG interactivo— se consolidan en un único
documento HTML autocontenido que se recorre con scroll.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .charts import (
    CSS_INTERACCION,
    ETIQUETA_CATEGORIA,
    ETIQUETA_METODO,
    JS_INTERACCION,
    REJILLA,
    SUPERFICIE as SUPERFICIE_TARJETA,
    TINTA_PRIMARIA,
    TINTA_SECUNDARIA,
    TINTA_TENUE,
    Grafico,
    grafico_composicion,
    grafico_equity,
    grafico_universo,
)
from .config import CATEGORIAS, CATEGORIAS_RV
from .config import PERFILES as ORDEN_PERFILES
from .utils import get_logger

log = get_logger("am_pm.reporting")

# =========================================================================== #
# 1. TABLAS EN TERMINAL
# =========================================================================== #
_FORMATOS = {
    "pct": lambda v: f"{v:.1%}",
    "pct2": lambda v: f"{v:.2%}",
    "pp": lambda v: f"{v:+.1%}",
    "num": lambda v: f"{v:,.2f}",
    "num3": lambda v: f"{v:,.3f}",
    "int": lambda v: f"{v:,.0f}",
    "cop": lambda v: f"${v:,.0f}",
    "bn": lambda v: f"{v / 1e12:,.2f}",
    "mm": lambda v: f"{v / 1e9:,.1f}",
}


def _formatear(valor: object, formato: str | None, ancho_texto: int) -> str:
    """Convierte un valor a su representación de celda."""
    if valor is None or (isinstance(valor, float) and not np.isfinite(valor)):
        return "—"
    if pd.isna(valor):
        return "—"
    if formato and formato in _FORMATOS:
        try:
            return _FORMATOS[formato](float(valor))
        except (TypeError, ValueError):
            return str(valor)
    if isinstance(valor, (int, np.integer)):
        return f"{int(valor):,}"
    if isinstance(valor, (float, np.floating)):
        return f"{float(valor):,.4f}"
    texto = str(valor)
    return texto if len(texto) <= ancho_texto else texto[: ancho_texto - 1] + "…"


def imprimir_titulo(texto: str, nivel: int = 1) -> None:
    """Encabezado de sección de la consola."""
    if nivel == 1:
        print()
        print("═" * 118)
        print(f"  {texto.upper()}")
        print("═" * 118)
    else:
        print()
        print(f"  {texto}")
        print("  " + "─" * 114)


def imprimir_tabla(
    df: pd.DataFrame,
    titulo: str | None = None,
    formatos: dict[str, str] | None = None,
    encabezados: dict[str, str] | None = None,
    nota: str | None = None,
    max_filas: int | None = None,
    ancho_texto: int = 40,
    sangria: str = "  ",
) -> None:
    """
    Imprime un DataFrame como tabla alineada.

    Los números se alinean a la derecha y los textos a la izquierda; el ancho de
    cada columna se ajusta a su contenido, de modo que la tabla sea legible sin
    depender de un visor externo.
    """
    if titulo:
        imprimir_titulo(titulo, nivel=2)
    if df is None or df.empty:
        print(f"{sangria}(sin registros)")
        return

    datos = df
    omitidas = 0
    if max_filas is not None and len(datos) > max_filas:
        omitidas = len(datos) - max_filas
        datos = datos.head(max_filas)

    formatos = formatos or {}
    encabezados = encabezados or {}
    columnas = list(datos.columns)

    celdas: dict[str, list[str]] = {}
    alinear_derecha: dict[str, bool] = {}
    for col in columnas:
        celdas[col] = [_formatear(v, formatos.get(col), ancho_texto) for v in datos[col]]
        alinear_derecha[col] = col in formatos or pd.api.types.is_numeric_dtype(datos[col])

    titulos = {col: encabezados.get(col, col) for col in columnas}
    anchos = {
        col: max(len(titulos[col]), *(len(c) for c in celdas[col])) for col in columnas
    }

    encabezado = sangria + "  ".join(
        titulos[col].rjust(anchos[col]) if alinear_derecha[col] else titulos[col].ljust(anchos[col])
        for col in columnas
    )
    print(encabezado.rstrip())
    print(sangria + "─" * (len(encabezado.rstrip()) - len(sangria)))
    for i in range(len(datos)):
        fila = sangria + "  ".join(
            celdas[col][i].rjust(anchos[col]) if alinear_derecha[col] else celdas[col][i].ljust(anchos[col])
            for col in columnas
        )
        print(fila.rstrip())
    if omitidas:
        print(f"{sangria}… {omitidas} fila(s) adicionales omitidas (usar --max-filas para ampliar)")
    if nota:
        print(f"{sangria}{nota}")


# --------------------------------------------------------------------------- #
# Vistas de negocio
# --------------------------------------------------------------------------- #
def mostrar_universo(universo, max_filas: int | None = None) -> None:
    """Ficha del universo curado: clasificación y métricas por fondo."""
    imprimir_titulo("Universo curado de fondos invertibles")
    tabla = (
        universo.fondos.reset_index()
        .sort_values(["categoria", "aum_cop"], ascending=[True, False])
        [[
            "fondo_id", "nombre_patrimonio", "nombre_entidad", "categoria",
            "metodo_clasificacion", "aum_cop", "retorno_anual", "vol_anual",
            "sharpe", "sortino", "max_drawdown",
        ]]
    )
    imprimir_tabla(
        tabla,
        formatos={
            "aum_cop": "mm", "retorno_anual": "pct", "vol_anual": "pct",
            "sharpe": "num", "sortino": "num", "max_drawdown": "pct",
        },
        encabezados={
            "fondo_id": "ID", "nombre_patrimonio": "Fondo", "nombre_entidad": "Gestora",
            "categoria": "Clase de activo", "metodo_clasificacion": "Clasificado por",
            "aum_cop": "AUM (kMM)", "retorno_anual": "Retorno", "vol_anual": "Vol",
            "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "Max DD",
        },
        max_filas=max_filas,
        ancho_texto=44,
        nota=f"r_f dinámica = {universo.rf:.2%} anual · corte {universo.fecha_corte.date()} · fuente {universo.origen}",
    )

    reclasificados = universo.fondos[
        (universo.fondos["metodo_clasificacion"] == "MODELO_RIESGO")
        & (universo.fondos["categoria_regla"] != "")
    ]
    if not reclasificados.empty:
        imprimir_tabla(
            reclasificados.reset_index()[
                ["nombre_patrimonio", "categoria_regla", "categoria", "vol_anual",
                 "beta_rv_local", "beta_internacional", "observacion"]
            ],
            titulo="Discrepancias nombre vs. riesgo observado (revisión del comité)",
            formatos={"vol_anual": "pct2", "beta_rv_local": "num", "beta_internacional": "num"},
            encabezados={
                "nombre_patrimonio": "Fondo", "categoria_regla": "Dice el nombre",
                "categoria": "Dice el riesgo", "vol_anual": "Vol",
                "beta_rv_local": "β RV local", "beta_internacional": "β interncl.",
                "observacion": "Observación",
            },
            ancho_texto=46,
        )


def mostrar_politica(perfiles: dict) -> None:
    """Bandas de asignación y límites de concentración de cada mandato."""
    imprimir_titulo("Política de inversión por perfil (P.Perfil)")
    filas = []
    for perfil in perfiles.values():
        fila = {"perfil": perfil.nombre}
        for cat in CATEGORIAS:
            lo, hi = perfil.limite(cat)
            fila[cat] = f"{lo:.0%}–{hi:.0%}"
        fila["RV_TOTAL"] = f"{perfil.rv_min:.0%}–{perfil.rv_max:.0%}"
        fila["max_fondo"] = perfil.max_peso_fondo
        fila["max_gestor"] = perfil.max_peso_gestor
        fila["max_pos"] = perfil.max_fondos
        fila["banda"] = perfil.banda_rebalanceo
        filas.append(fila)
    imprimir_tabla(
        pd.DataFrame(filas),
        formatos={"max_fondo": "pct", "max_gestor": "pct", "banda": "pct", "max_pos": "int"},
        encabezados={
            "perfil": "Perfil", "RV_TOTAL": "RV total", "max_fondo": "Tope fondo",
            "max_gestor": "Tope gestora", "max_pos": "Máx pos.", "banda": "Banda",
            **{c: ETIQUETA_CATEGORIA[c] for c in CATEGORIAS},
        },
    )


def mostrar_pesos(asignacion, fondos: pd.DataFrame, max_filas: int | None = None) -> None:
    """Portafolios modelo: posiciones de cada perfil y método."""
    imprimir_titulo("Portafolios modelo · pesos por fondo")
    for perfil in _orden_perfiles(asignacion.perfiles):
        resultado = asignacion.perfiles[perfil]
        for metodo in resultado.pesos.columns:
            serie = resultado.pesos[metodo]
            posiciones = serie[serie > 1e-6].sort_values(ascending=False)
            if posiciones.empty:
                continue
            tabla = pd.DataFrame(
                {
                    "fondo_id": posiciones.index,
                    "nombre_patrimonio": fondos.loc[posiciones.index, "nombre_patrimonio"].to_numpy(),
                    "nombre_entidad": fondos.loc[posiciones.index, "nombre_entidad"].to_numpy(),
                    "categoria": fondos.loc[posiciones.index, "categoria"].to_numpy(),
                    "peso": posiciones.to_numpy(),
                    "retorno_anual": fondos.loc[posiciones.index, "retorno_anual"].to_numpy(),
                    "vol_anual": fondos.loc[posiciones.index, "vol_anual"].to_numpy(),
                    "sharpe": fondos.loc[posiciones.index, "sharpe"].to_numpy(),
                }
            )
            imprimir_tabla(
                tabla,
                titulo=f"{perfil} · {ETIQUETA_METODO.get(metodo, metodo)} "
                       f"({len(posiciones)} posiciones)",
                formatos={"peso": "pct", "retorno_anual": "pct", "vol_anual": "pct", "sharpe": "num"},
                encabezados={
                    "fondo_id": "ID", "nombre_patrimonio": "Fondo", "nombre_entidad": "Gestora",
                    "categoria": "Clase", "peso": "Peso", "retorno_anual": "Retorno",
                    "vol_anual": "Vol", "sharpe": "Sharpe",
                },
                max_filas=max_filas,
                ancho_texto=42,
            )


def mostrar_composicion(asignacion) -> None:
    """Asignación agregada por clase de activo."""
    imprimir_titulo("Composición por clase de activo")
    for perfil in _orden_perfiles(asignacion.perfiles):
        comp = asignacion.perfiles[perfil].composicion.reindex(CATEGORIAS).fillna(0.0)
        tabla = comp.reset_index()
        tabla["categoria"] = tabla["categoria"].map(ETIQUETA_CATEGORIA)
        total_rv = comp.reindex(list(CATEGORIAS_RV)).sum()
        fila_rv = pd.DataFrame([{"categoria": "→ Renta variable total", **total_rv.to_dict()}])
        imprimir_tabla(
            pd.concat([tabla, fila_rv], ignore_index=True),
            titulo=f"{perfil}",
            formatos={m: "pct" for m in comp.columns},
            encabezados={"categoria": "Clase de activo",
                         **{m: ETIQUETA_METODO.get(m, m) for m in comp.columns}},
        )


def mostrar_tabla_comparativa(asignacion, backtest, separador: str = "::") -> pd.DataFrame:
    """Cruce de métricas esperadas (ex-ante) y realizadas (backtest)."""
    imprimir_titulo("Tabla comparativa · esperado vs. realizado")
    filas = []
    for perfil in _orden_perfiles(asignacion.perfiles):
        resultado = asignacion.perfiles[perfil]
        for metodo, stats in resultado.estadisticas.iterrows():
            filas.append(
                {
                    "perfil": perfil,
                    "metodo": ETIQUETA_METODO.get(metodo, metodo),
                    "_clave": f"{perfil}{separador}{metodo}",
                    "ex_ret": stats["retorno_esperado"],
                    "ex_vol": stats["vol_esperada"],
                    "ex_sharpe": stats["sharpe_ex_ante"],
                    "rv": stats["peso_RENTA_VARIABLE"],
                    "riesgo_rv": stats["riesgo_RENTA_VARIABLE"],
                    "n_pos": stats["n_posiciones"],
                    "div": stats["ratio_diversificacion"],
                }
            )
    tabla = pd.DataFrame(filas)

    if backtest is not None and not backtest.metricas.empty:
        met = backtest.metricas
        tabla["bt_ret"] = tabla["_clave"].map(met["retorno_anual"])
        tabla["bt_vol"] = tabla["_clave"].map(met["vol_anual"])
        tabla["bt_sharpe"] = tabla["_clave"].map(met["sharpe"])
        tabla["bt_sortino"] = tabla["_clave"].map(met["sortino"])
        tabla["bt_mdd"] = tabla["_clave"].map(met["max_drawdown"])
        rot = backtest.rebalanceos.groupby(["perfil", "metodo"])["turnover"].mean()
        costo = backtest.rebalanceos.groupby(["perfil", "metodo"])["costo_pct"].sum()
        claves = list(zip(tabla["perfil"], [m for m in _metodos_originales(asignacion)]))
        tabla["bt_turnover"] = [rot.get(k, np.nan) for k in claves]
        tabla["bt_costo"] = [costo.get(k, np.nan) for k in claves]

    imprimir_tabla(
        tabla.drop(columns=["_clave"]),
        formatos={
            "ex_ret": "pct", "ex_vol": "pct", "ex_sharpe": "num", "rv": "pct",
            "riesgo_rv": "pct", "n_pos": "int", "div": "num", "bt_ret": "pct",
            "bt_vol": "pct", "bt_sharpe": "num", "bt_sortino": "num", "bt_mdd": "pct",
            "bt_turnover": "pct", "bt_costo": "pct2",
        },
        encabezados={
            "perfil": "Perfil", "metodo": "Método", "ex_ret": "Ret esp",
            "ex_vol": "Vol esp", "ex_sharpe": "Sharpe esp", "rv": "Peso RV",
            "riesgo_rv": "Riesgo RV", "n_pos": "Pos", "div": "Div ratio",
            "bt_ret": "CAGR bt", "bt_vol": "Vol bt", "bt_sharpe": "Sharpe bt",
            "bt_sortino": "Sortino bt", "bt_mdd": "MaxDD bt",
            "bt_turnover": "Turnover", "bt_costo": "Costo acum",
        },
        nota=("Ex-ante: esperado por el optimizador, contra r_f del panel. "
              "Bt: realizado walk-forward, neto de costos, contra r_f del período del backtest."),
    )
    return tabla


def _metodos_originales(asignacion) -> list[str]:
    """Claves internas de método, en el mismo orden que la tabla comparativa."""
    salida: list[str] = []
    for perfil in _orden_perfiles(asignacion.perfiles):
        salida.extend(list(asignacion.perfiles[perfil].estadisticas.index))
    return salida


def mostrar_rebalanceo(informes: dict[str, dict], patrimonio: float) -> None:
    """Bandas tácticas y plan de órdenes."""
    imprimir_titulo("Rebalanceo táctico · bandas de tolerancia y órdenes")
    for perfil, informe in informes.items():
        bandas = informe["bandas_categoria"].reset_index()
        bandas["categoria"] = bandas["categoria"].map(
            lambda c: ETIQUETA_CATEGORIA.get(c, c.replace("_", " ").title())
        )
        imprimir_tabla(
            bandas,
            titulo=f"{perfil} · deriva desde {informe['fecha_referencia'].date()}",
            formatos={
                "peso_objetivo": "pct", "peso_actual": "pct", "desviacion_pp": "pp",
                "banda_inferior": "pct", "banda_superior": "pct",
            },
            encabezados={
                "categoria": "Clase de activo", "peso_objetivo": "Objetivo",
                "peso_actual": "Vigente", "desviacion_pp": "Desvío",
                "banda_inferior": "Banda inf", "banda_superior": "Banda sup",
                "estado": "Estado", "accion": "Acción",
            },
        )
        ordenes = informe["ordenes"]
        if not ordenes.empty:
            imprimir_tabla(
                ordenes.reset_index(),
                titulo=f"{perfil} · órdenes sobre patrimonio de ${patrimonio:,.0f}",
                formatos={
                    "peso_actual": "pct", "peso_objetivo": "pct", "delta_peso": "pp",
                    "monto_cop": "cop", "costo_estimado_cop": "cop",
                },
                encabezados={
                    "fondo_id": "ID", "peso_actual": "Vigente", "peso_objetivo": "Objetivo",
                    "delta_peso": "Δ Peso", "monto_cop": "Monto", "operacion": "Operación",
                    "costo_estimado_cop": "Costo est.",
                },
            )
        resumen = informe["resumen"]
        n = int(resumen["n_ordenes"])
        print(f"    Turnover {resumen['turnover']:.2%} · {n} orden{'' if n == 1 else 'es'} · "
              f"costo ${resumen['costo_cop']:,.0f} ({resumen['costo_pct']:.3%} del patrimonio)")


def mostrar_backtest(backtest, separador: str = "::") -> None:
    """Desempeño realizado del walk-forward."""
    if backtest is None:
        return
    imprimir_titulo("Backtest walk-forward · desempeño realizado")
    met = backtest.metricas.reset_index()
    met["estrategia"] = met["estrategia"].str.replace(separador, " · ", regex=False)
    imprimir_tabla(
        met[["estrategia", "retorno_anual", "vol_anual", "sharpe", "sortino",
             "max_drawdown", "calmar", "var_95_diario", "pct_periodos_positivos", "n_obs"]],
        formatos={
            "retorno_anual": "pct", "vol_anual": "pct", "sharpe": "num", "sortino": "num",
            "max_drawdown": "pct", "calmar": "num", "var_95_diario": "pct2",
            "pct_periodos_positivos": "pct", "n_obs": "int",
        },
        encabezados={
            "estrategia": "Estrategia", "retorno_anual": "CAGR", "vol_anual": "Vol",
            "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "Max DD",
            "calmar": "Calmar", "var_95_diario": "VaR 95%", "n_obs": "Obs",
            "pct_periodos_positivos": "% días +",
        },
        ancho_texto=38,
        nota=(f"Sharpe y Sortino contra r_f realizada en el período del backtest = "
              f"{backtest.rf_realizada:.2%} anual (RF_CORTO ponderada por AUM)."),
    )
    bitacora = backtest.rebalanceos.groupby(["perfil", "metodo"]).agg(
        rebalanceos=("turnover", "size"),
        turnover_medio=("turnover", "mean"),
        costo_acumulado=("costo_pct", "sum"),
        pos_medias=("n_posiciones", "mean"),
    ).reset_index()
    bitacora["metodo"] = bitacora["metodo"].map(lambda m: ETIQUETA_METODO.get(m, m))
    imprimir_tabla(
        bitacora,
        titulo="Bitácora de rebalanceos",
        formatos={
            "rebalanceos": "int", "turnover_medio": "pct", "costo_acumulado": "pct2",
            "pos_medias": "num",
        },
        encabezados={
            "perfil": "Perfil", "metodo": "Método", "rebalanceos": "N° reb.",
            "turnover_medio": "Turnover medio", "costo_acumulado": "Costo acum.",
            "pos_medias": "Posiciones",
        },
    )


def _orden_perfiles(perfiles: dict) -> list[str]:
    """Perfiles en la escalera de riesgo del mandato, no en orden alfabético."""
    ordenados = [p for p in ORDEN_PERFILES if p in perfiles]
    return ordenados + [p for p in perfiles if p not in ordenados]



# =========================================================================== #
# 2. INFORME HTML INTERACTIVO
# =========================================================================== #
_PLANTILLA_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titulo}</title>
<style>
  :root {{
    color-scheme: light;
    --superficie: #fcfcfb;
    --tarjeta: {tarjeta};
    --tinta: {tinta};
    --tinta-2: {tinta2};
    --tinta-3: {tinta3};
    --borde: {borde};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--superficie);
    color: var(--tinta);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .envoltura {{ max-width: 1520px; margin: 0 auto; padding: 0 28px 72px; }}
  header.principal {{
    padding: 44px 0 26px; border-bottom: 1px solid var(--borde); margin-bottom: 30px;
  }}
  h1 {{ font-size: 27px; margin: 0 0 6px; letter-spacing: -0.015em; font-weight: 650; }}
  .subtitulo {{ color: var(--tinta-2); font-size: 14px; margin: 0; }}
  .aviso {{
    margin-top: 16px; padding: 11px 15px; border-radius: 8px;
    background: #fff6e5; border: 1px solid #f0d9a8; color: #6b4c00; font-size: 13.5px;
  }}
  .tarjetas {{
    display: grid; gap: 12px; margin-top: 24px;
    grid-template-columns: repeat(auto-fit, minmax(178px, 1fr));
  }}
  .tarjeta-dato {{
    background: var(--tarjeta); border: 1px solid var(--borde);
    border-radius: 10px; padding: 14px 16px;
  }}
  .tarjeta-dato .rotulo {{
    font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.055em;
    color: var(--tinta-3); margin-bottom: 5px;
  }}
  .tarjeta-dato .valor {{ font-size: 21px; font-weight: 620; letter-spacing: -0.01em; }}
  nav.indice {{
    position: sticky; top: 0; z-index: 10;
    background: #fcfcfbf2; backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--borde);
    display: flex; gap: 22px; padding: 13px 0; margin-bottom: 8px;
    font-size: 13.5px; overflow-x: auto;
  }}
  nav.indice a {{ color: var(--tinta-2); text-decoration: none; white-space: nowrap; }}
  nav.indice a:hover {{ color: var(--tinta); text-decoration: underline; }}
  section.grafica {{ margin: 40px 0 0; scroll-margin-top: 62px; }}
  section.grafica h2 {{
    font-size: 18.5px; margin: 0 0 4px; font-weight: 620; letter-spacing: -0.01em;
  }}
  section.grafica p.nota {{
    color: var(--tinta-2); font-size: 13.5px; margin: 0 0 14px; max-width: 92ch;
  }}
  .lienzo {{
    background: var(--tarjeta); border: 1px solid var(--borde);
    border-radius: 12px; padding: 18px; overflow-x: auto;
  }}
  svg.grafico {{ width: 100%; height: auto; display: block; min-width: 860px; }}
  footer {{
    margin-top: 52px; padding-top: 18px; border-top: 1px solid var(--borde);
    color: var(--tinta-3); font-size: 12.5px;
  }}
{css_interaccion}
  @media print {{
    nav.indice {{ display: none; }}
    .lienzo {{ break-inside: avoid; }}
    .tooltip {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="envoltura">
  <header class="principal">
    <h1>{titulo}</h1>
    <p class="subtitulo">{subtitulo}</p>
    {aviso}
    <div class="tarjetas">{tarjetas}</div>
  </header>
  <nav class="indice">{indice}</nav>
  {secciones}
  <footer>{pie}</footer>
</div>
<script>{js_interaccion}</script>
</body>
</html>
"""


def construir_informe_html(
    secciones: list[tuple[str, str, str, Grafico]],
    tarjetas: list[tuple[str, str]],
    ruta: Path,
    titulo: str = "AM-PM · Asset Allocation Manager",
    subtitulo: str = "",
    aviso: str | None = None,
    pie: str = "",
) -> Path:
    """
    Consolida todos los gráficos en un único documento HTML autocontenido.

    `secciones` es una lista de (ancla, título, descripción, gráfico). Cada
    gráfico aporta su SVG y una carga útil JSON que la capa de interacción lee
    para construir los tooltips. El resultado no depende de red ni de librerías
    externas: se abre con doble clic.
    """
    bloques_tarjetas = "".join(
        f'<div class="tarjeta-dato"><div class="rotulo">{r}</div>'
        f'<div class="valor">{v}</div></div>'
        for r, v in tarjetas
    )
    indice = "".join(f'<a href="#{a}">{t}</a>' for a, t, _, _ in secciones)
    bloques = "".join(
        f'<section class="grafica" id="{ancla}">'
        f"<h2>{titulo_seccion}</h2>"
        f'<p class="nota">{nota}</p>'
        f'<div class="lienzo">{grafico.svg}</div>'
        f'<script type="application/json" id="datos-{grafico.id}">{grafico.json_datos()}</script>'
        f"</section>"
        for ancla, titulo_seccion, nota, grafico in secciones
    )
    html = _PLANTILLA_HTML.format(
        titulo=titulo,
        subtitulo=subtitulo,
        aviso=f'<div class="aviso">{aviso}</div>' if aviso else "",
        tarjetas=bloques_tarjetas,
        indice=indice,
        secciones=bloques,
        pie=pie,
        tarjeta=SUPERFICIE_TARJETA,
        tinta=TINTA_PRIMARIA,
        tinta2=TINTA_SECUNDARIA,
        tinta3=TINTA_TENUE,
        borde=REJILLA,
        css_interaccion=CSS_INTERACCION,
        js_interaccion=JS_INTERACCION,
    )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(html, encoding="utf-8")
    log.info("Informe gráfico interactivo: %s", ruta)
    return ruta


def generar_informe(
    universo, asignacion, backtest, ruta: Path, fecha_generacion: datetime | None = None
) -> Path:
    """Arma el informe HTML completo a partir de los resultados de la corrida."""
    fecha_generacion = fecha_generacion or datetime.now()
    secciones: list[tuple[str, str, str, Grafico]] = [
        (
            "universo",
            "Universo curado: riesgo y retorno por clase de activo",
            "Cada panel resalta una clase de activo sobre el telón del universo completo. "
            "La línea punteada marca la tasa libre de riesgo dinámica: los fondos por debajo "
            "no compensan el costo de oportunidad de la liquidez. "
            "<strong>Acerca el cursor a un punto</strong> para identificar el fondo, su gestora "
            "y cómo fue clasificado.",
            grafico_universo(universo.fondos, universo.rf),
        ),
        (
            "composicion",
            "Composición estratégica por clase de activo",
            "Asignación resultante de cada método de optimización, dentro de las bandas del "
            "mandato de cada perfil. La escalera de riesgo debe leerse de izquierda a derecha. "
            "<strong>Pasa el cursor por un segmento</strong> para ver su peso exacto.",
            grafico_composicion({p: r.composicion for p, r in asignacion.perfiles.items()}),
        ),
    ]
    if backtest is not None and not backtest.equity.empty:
        secciones.append(
            (
                "equity",
                "Backtest walk-forward: capital acumulado neto de costos",
                "Reestimación completa de r_f, μ y Σ en cada rebalanceo, sin mirada al futuro. "
                "Las referencias pasivas (gris) son la caja en RF corto y el universo "
                "equiponderado. <strong>Recorre el gráfico con el cursor</strong> para comparar "
                "las cuatro estrategias en cualquier fecha.",
                grafico_equity(backtest.equity),
            )
        )

    tarjetas = [
        ("Fecha de corte", str(universo.fecha_corte.date())),
        ("Fondos en universo", f"{len(universo.fondos)}"),
        ("Observaciones", f"{universo.precios.shape[0]:,}"),
        ("Tasa libre de riesgo", f"{universo.rf:.2%}"),
        ("Ventana", f"{universo.precios.index[0].date()} → {universo.precios.index[-1].date()}"),
        ("Origen de datos", universo.origen),
    ]
    aviso = (
        "Datos simulados: esta corrida se ejecutó en modo offline y los resultados "
        "<strong>no representan al mercado</strong>. No usar para decisiones de inversión."
        if universo.origen == "SINTETICO" else None
    )
    return construir_informe_html(
        secciones,
        tarjetas,
        ruta,
        subtitulo=(
            "Asignación estratégica y táctica sobre Fondos de Inversión Colectiva, ETFs e "
            "índices del mercado colombiano · Superintendencia Financiera vía datos.gov.co"
        ),
        aviso=aviso,
        pie=(
            f"Generado el {fecha_generacion:%Y-%m-%d %H:%M} · "
            "Las tablas de detalle se imprimen en la terminal durante la corrida. "
            "Documento informativo: no constituye recomendación de inversión."
        ),
    )
