"""
Generación de gráficos SVG interactivos, sin dependencias de terceros.

Cada gráfico se emite como SVG nativo más una carga útil JSON que consume la
capa de interacción del informe. No se usa ninguna librería de visualización:
el documento se abre sin red y pesa una fracción de una imagen rasterizada.

Reglas de interacción (ver `JS_INTERACCION`):
  * Líneas: retícula vertical que engancha la fecha más cercana y un único
    tooltip que lista **todas** las series en esa fecha.
  * Barras: la marca es el objetivo del cursor y se realza al pasar por encima.
  * Puntos: capa de punto más cercano — en los conglomerados de renta fija los
    fondos se solapan, así que basta con estar cerca, no encima.
  * Los nombres de fondos provienen de una API pública: se insertan en el DOM
    con `textContent`, nunca concatenando HTML.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CATEGORIAS, CATEGORIAS_RV
from .config import PERFILES as ORDEN_PERFILES

# --------------------------------------------------------------------------- #
# Sistema visual (paleta categórica validada para daltonismo)
# --------------------------------------------------------------------------- #
SUPERFICIE = "#ffffff"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#8a8880"
REJILLA = "#e6e5e0"

PALETA = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
COLOR_CATEGORIA = dict(zip(CATEGORIAS, PALETA))

ETIQUETA_CATEGORIA = {
    "RF_CORTO": "RF Corto plazo",
    "RF_MEDIANO_LARGO": "RF Mediano/Largo",
    "MIXTO": "Mixtos",
    "RV_LOCAL": "RV Local",
    "RV_INTERNACIONAL": "RV Internacional",
}
ETIQUETA_METODO = {
    "MARKOWITZ_SHARPE": "Markowitz",
    "RISK_PARITY": "Risk Parity",
    "HRP": "HRP",
    "EQUIPONDERADO": "1/N",
}
ESTILO_BENCHMARK = {
    "CAJA_RF_CORTO": ("2 3", "Caja (RF corto)"),
    "UNIVERSO_1_N": ("7 4", "Universo 1/N"),
}


@dataclass
class Grafico:
    """Un gráfico listo para incrustar: marcado SVG y datos para el tooltip."""

    id: str
    svg: str
    datos: dict = field(default_factory=dict)

    def json_datos(self) -> str:
        """JSON seguro para incrustar dentro de una etiqueta <script>."""
        return json.dumps(self.datos, ensure_ascii=False).replace("<", "\\u003c")


# --------------------------------------------------------------------------- #
# Utilidades de dibujo
# --------------------------------------------------------------------------- #
def esc(texto: object) -> str:
    """Escapa texto para insertarlo en atributos o nodos SVG."""
    return (
        str(texto)
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&#39;")
    )


def _n(valor: float) -> str:
    """Número compacto para atributos SVG (evita ruido de coma flotante)."""
    return f"{valor:.2f}".rstrip("0").rstrip(".") or "0"


@dataclass(frozen=True)
class Escala:
    """Transformación lineal dominio → rango de coordenadas del lienzo."""

    d0: float
    d1: float
    r0: float
    r1: float

    def __call__(self, valor: float) -> float:
        if self.d1 == self.d0:
            return (self.r0 + self.r1) / 2
        return self.r0 + (valor - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


@dataclass(frozen=True)
class Marco:
    """Área de trazado de un panel dentro del lienzo."""

    x: float
    y: float
    ancho: float
    alto: float

    @property
    def x1(self) -> float:
        return self.x + self.ancho

    @property
    def y1(self) -> float:
        return self.y + self.alto

    @property
    def centro_x(self) -> float:
        return self.x + self.ancho / 2

    def escala_x(self, d0: float, d1: float) -> Escala:
        return Escala(d0, d1, self.x, self.x1)

    def escala_y(self, d0: float, d1: float) -> Escala:
        # El eje Y del SVG crece hacia abajo: el rango se invierte.
        return Escala(d0, d1, self.y1, self.y)


def ticks_lindos(vmin: float, vmax: float, objetivo: int = 5) -> list[float]:
    """Marcas de eje en valores redondos (algoritmo de números agradables)."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return [vmin if np.isfinite(vmin) else 0.0]
    bruto = (vmax - vmin) / max(objetivo, 1)
    magnitud = 10 ** math.floor(math.log10(bruto)) if bruto > 0 else 1.0
    paso = magnitud * 10
    for mult in (1, 2, 2.5, 5, 10):
        if bruto <= magnitud * mult:
            paso = magnitud * mult
            break
    valor = math.ceil(vmin / paso) * paso
    marcas: list[float] = []
    while valor <= vmax + paso * 1e-9:
        marcas.append(round(valor, 10))
        valor += paso
    return marcas or [vmin, vmax]


def _texto(
    x: float, y: float, contenido: str, *, tam: float = 11, color: str = TINTA_SECUNDARIA,
    ancla: str = "middle", peso: str = "normal", base: str = "middle", rotar: bool = False,
) -> str:
    """Nodo de texto. Los textos nunca capturan el cursor (ver CSS)."""
    atributos = (
        f'x="{_n(x)}" y="{_n(y)}" font-size="{_n(tam)}" fill="{color}" '
        f'text-anchor="{ancla}" dominant-baseline="{base}"'
    )
    if peso != "normal":
        atributos += f' font-weight="{peso}"'
    if rotar:
        atributos += f' transform="rotate(-90 {_n(x)} {_n(y)})"'
    return f"<text {atributos}>{esc(contenido)}</text>"


def _rejilla_y(marco: Marco, ticks: list[float], escala: Escala, formato) -> str:
    """Rejilla horizontal recesiva con sus rótulos."""
    piezas: list[str] = []
    for valor in ticks:
        y = escala(valor)
        if not (marco.y - 1 <= y <= marco.y1 + 1):
            continue
        piezas.append(
            f'<line x1="{_n(marco.x)}" y1="{_n(y)}" x2="{_n(marco.x1)}" y2="{_n(y)}" '
            f'stroke="{REJILLA}" stroke-width="1"/>'
        )
        piezas.append(_texto(marco.x - 8, y, formato(valor), tam=10, ancla="end"))
    return "".join(piezas)


def _linea_base(marco: Marco) -> str:
    return (
        f'<line x1="{_n(marco.x)}" y1="{_n(marco.y1)}" x2="{_n(marco.x1)}" '
        f'y2="{_n(marco.y1)}" stroke="{REJILLA}" stroke-width="1"/>'
    )


def _titulo_panel(marco: Marco, texto: str) -> str:
    return _texto(marco.centro_x, marco.y - 22, texto, tam=13, color=TINTA_PRIMARIA, peso="600")


def _leyenda(y: float, entradas: list[tuple[str, str, str]], ancho_total: float) -> str:
    """
    Leyenda centrada. `entradas` es (etiqueta, color, forma): 'rect' para
    rellenos y 'linea'/'guion' para series de línea — la clave refleja la marca.
    """
    if not entradas:
        return ""
    anchos = [len(e[0]) * 6.4 + 32 for e in entradas]
    cursor = (ancho_total - sum(anchos)) / 2
    piezas: list[str] = []
    for (etiqueta, color, forma), ancho in zip(entradas, anchos):
        if forma == "rect":
            piezas.append(
                f'<rect x="{_n(cursor)}" y="{_n(y - 5.5)}" width="11" height="11" '
                f'rx="2.5" fill="{color}"/>'
            )
        else:
            guion = ' stroke-dasharray="5 3"' if forma == "guion" else ""
            piezas.append(
                f'<line x1="{_n(cursor)}" y1="{_n(y)}" x2="{_n(cursor + 15)}" y2="{_n(y)}" '
                f'stroke="{color}" stroke-width="2.4" stroke-linecap="round"{guion}/>'
            )
        piezas.append(_texto(cursor + 21, y, etiqueta, tam=11, ancla="start"))
        cursor += ancho
    return "".join(piezas)


def _envolver_svg(ancho: float, alto: float, gid: str, cuerpo: str, titulo: str) -> str:
    return (
        f'<svg viewBox="0 0 {_n(ancho)} {_n(alto)}" xmlns="http://www.w3.org/2000/svg" '
        f'class="grafico" data-gid="{gid}" preserveAspectRatio="xMidYMid meet" '
        f'role="img" aria-label="{esc(titulo)}">{cuerpo}</svg>'
    )


def _fmt(valor: float, patron: str = "{:.2f}") -> str:
    return patron.format(valor) if np.isfinite(valor) else "—"


def _monto_corto(valor: float) -> str:
    """Monto en COP con la unidad propia de su magnitud."""
    if not np.isfinite(valor):
        return "—"
    if abs(valor) >= 1e12:
        return f"${valor / 1e12:,.2f} billones"
    if abs(valor) >= 1e9:
        return f"${valor / 1e9:,.1f} mil millones"
    if abs(valor) >= 1e6:
        return f"${valor / 1e6:,.1f} millones"
    return f"${valor:,.0f}"


# =========================================================================== #
# Gráfico 1 — Universo: riesgo/retorno en múltiplos pequeños
# =========================================================================== #
def grafico_universo(fondos: pd.DataFrame, rf: float) -> Grafico:
    """
    Un panel por clase de activo, resaltada sobre el telón del universo completo.
    Evita depender de la discriminación de cinco colores simultáneos.
    """
    presentes = [c for c in CATEGORIAS if (fondos["categoria"] == c).any()]
    n_paneles = max(len(presentes), 1)

    ancho, alto = 1240.0, 330.0
    izq, der, sup, inf = 56.0, 18.0, 46.0, 50.0
    hueco = 28.0
    ancho_panel = (ancho - izq - der - hueco * (n_paneles - 1)) / n_paneles

    vol = (fondos["vol_anual"] * 100).to_numpy(dtype=float)
    ret = (fondos["retorno_anual"] * 100).to_numpy(dtype=float)
    x_max = float(np.nanmax(vol)) * 1.08 if len(vol) else 1.0
    y_min = min(float(np.nanmin(ret)), rf * 100) if len(ret) else 0.0
    y_max = max(float(np.nanmax(ret)), rf * 100) if len(ret) else 1.0
    colchon = (y_max - y_min) * 0.12 or 1.0
    y_min, y_max = y_min - colchon, y_max + colchon

    piezas: list[str] = []
    paneles_json: list[dict] = []

    for i, categoria in enumerate(presentes):
        marco = Marco(izq + i * (ancho_panel + hueco), sup, ancho_panel, alto - sup - inf)
        ex = marco.escala_x(0.0, x_max)
        ey = marco.escala_y(y_min, y_max)

        piezas.append(_rejilla_y(marco, ticks_lindos(y_min, y_max, 5), ey, lambda v: f"{v:.0f}%"))
        piezas.append(_linea_base(marco))
        for valor in ticks_lindos(0, x_max, 4):
            piezas.append(_texto(ex(valor), marco.y1 + 15, f"{valor:.0f}", tam=10))

        # Telón de fondo: el universo completo, recesivo.
        for v, r in zip(vol, ret):
            if np.isfinite(v) and np.isfinite(r):
                piezas.append(f'<circle cx="{_n(ex(v))}" cy="{_n(ey(r))}" r="3" fill="{REJILLA}"/>')

        y_rf = ey(rf * 100)
        piezas.append(
            f'<line x1="{_n(marco.x)}" y1="{_n(y_rf)}" x2="{_n(marco.x1)}" y2="{_n(y_rf)}" '
            f'stroke="{TINTA_TENUE}" stroke-width="1.2" stroke-dasharray="2 3"/>'
        )
        if i == 0:
            piezas.append(_texto(marco.x1 - 2, y_rf - 9, f"r_f = {rf:.1%}", tam=10, ancla="end"))

        subconjunto = fondos[fondos["categoria"] == categoria]
        color = COLOR_CATEGORIA[categoria]
        puntos_json: list[dict] = []
        for k, (_, fila) in enumerate(subconjunto.iterrows()):
            v, r = float(fila["vol_anual"] * 100), float(fila["retorno_anual"] * 100)
            if not (np.isfinite(v) and np.isfinite(r)):
                continue
            cx, cy = ex(v), ey(r)
            piezas.append(
                f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="5" fill="{color}" stroke="{SUPERFICIE}" '
                f'stroke-width="2" class="punto" data-panel="{i}" data-i="{k}" tabindex="0" '
                f'role="button" aria-label="{esc(fila["nombre_patrimonio"])}"/>'
            )
            puntos_json.append(
                {
                    "cx": round(cx, 2), "cy": round(cy, 2),
                    "titulo": str(fila["nombre_patrimonio"]),
                    "color": color,
                    "filas": [
                        ["Retorno anual", f"{fila['retorno_anual']:.2%}"],
                        ["Volatilidad", f"{fila['vol_anual']:.2%}"],
                        ["Sharpe", _fmt(float(fila["sharpe"]))],
                        ["Sortino", _fmt(float(fila["sortino"]))],
                        ["Max drawdown", f"{fila['max_drawdown']:.1%}"],
                        ["AUM", _monto_corto(float(fila["aum_cop"]))],
                        ["Clase", ETIQUETA_CATEGORIA.get(categoria, categoria)],
                        ["Gestora", str(fila["nombre_entidad"])],
                        ["Clasificado por", str(fila["metodo_clasificacion"]).replace("_", " ").title()],
                    ],
                }
            )

        # Capa de punto más cercano: en los conglomerados los fondos se solapan.
        piezas.append(
            f'<rect class="zona-puntos" x="{_n(marco.x)}" y="{_n(marco.y)}" '
            f'width="{_n(marco.ancho)}" height="{_n(marco.alto)}" fill="transparent" '
            f'data-panel="{i}"/>'
        )
        piezas.append(_titulo_panel(marco, f"{ETIQUETA_CATEGORIA[categoria]} · {len(subconjunto)} fondos"))
        piezas.append(_texto(marco.centro_x, marco.y1 + 34, "Volatilidad anual (%)",
                             tam=10, color=TINTA_TENUE))
        paneles_json.append({"puntos": puntos_json})

    piezas.append(_texto(15, sup + (alto - sup - inf) / 2, "Retorno anual (%)",
                         tam=10, color=TINTA_TENUE, rotar=True))
    return Grafico(
        id="universo",
        svg=_envolver_svg(ancho, alto, "universo", "".join(piezas),
                          "Riesgo y retorno del universo curado por clase de activo"),
        datos={"tipo": "puntos", "paneles": paneles_json},
    )


# =========================================================================== #
# Gráfico 2 — Composición: barras apiladas
# =========================================================================== #
def grafico_composicion(composiciones: dict[str, pd.DataFrame]) -> Grafico:
    """Asignación por clase de activo: un panel por perfil, una barra por método."""
    perfiles = [p for p in ORDEN_PERFILES if p in composiciones]
    perfiles += [p for p in composiciones if p not in perfiles]
    n_paneles = max(len(perfiles), 1)

    ancho, alto = 1240.0, 480.0
    izq, der, sup, inf = 56.0, 18.0, 48.0, 92.0
    hueco = 48.0
    ancho_panel = (ancho - izq - der - hueco * (n_paneles - 1)) / n_paneles

    piezas: list[str] = []
    marcas_json: list[dict] = []
    indice = 0

    for i, perfil in enumerate(perfiles):
        comp = composiciones[perfil].reindex(CATEGORIAS).fillna(0.0)
        metodos = list(comp.columns)
        marco = Marco(izq + i * (ancho_panel + hueco), sup, ancho_panel, alto - sup - inf)
        ey = marco.escala_y(0.0, 1.0)
        piezas.append(_rejilla_y(marco, [0, 0.2, 0.4, 0.6, 0.8, 1.0], ey, lambda v: f"{v:.0%}"))

        paso = marco.ancho / max(len(metodos), 1)
        ancho_barra = min(paso * 0.62, 78.0)
        for j, metodo in enumerate(metodos):
            centro = marco.x + paso * (j + 0.5)
            x0 = centro - ancho_barra / 2
            acumulado = 0.0
            for categoria in CATEGORIAS:
                valor = float(comp.loc[categoria, metodo])
                if valor <= 1e-9:
                    continue
                y_sup, y_inf = ey(acumulado + valor), ey(acumulado)
                # Separador de 2 unidades entre segmentos apilados.
                altura = max(y_inf - y_sup - 2, 0.8)
                piezas.append(
                    f'<rect x="{_n(x0)}" y="{_n(y_sup)}" width="{_n(ancho_barra)}" '
                    f'height="{_n(altura)}" fill="{COLOR_CATEGORIA[categoria]}" rx="1.5" '
                    f'class="marca" data-i="{indice}" tabindex="0" role="button" '
                    f'aria-label="{esc(ETIQUETA_CATEGORIA[categoria])}: {valor:.0%}"/>'
                )
                if valor >= 0.06:
                    piezas.append(_texto(centro, (y_sup + y_inf) / 2, f"{valor:.0%}",
                                         tam=11, color=TINTA_PRIMARIA))
                marcas_json.append(
                    {
                        "titulo": ETIQUETA_CATEGORIA[categoria],
                        "color": COLOR_CATEGORIA[categoria],
                        "filas": [
                            ["Peso", f"{valor:.1%}"],
                            ["Perfil", perfil.capitalize()],
                            ["Método", ETIQUETA_METODO.get(metodo, metodo)],
                        ],
                    }
                )
                indice += 1
                acumulado += valor
            piezas.append(_texto(centro, marco.y1 + 17, ETIQUETA_METODO.get(metodo, metodo), tam=11))
            total_rv = float(comp.reindex(list(CATEGORIAS_RV))[metodo].sum())
            piezas.append(_texto(centro, marco.y1 + 33, f"RV {total_rv:.0%}", tam=10, color=TINTA_TENUE))

        piezas.append(_linea_base(marco))
        piezas.append(_titulo_panel(marco, perfil.capitalize()))

    piezas.append(_leyenda(alto - 22,
                           [(ETIQUETA_CATEGORIA[c], COLOR_CATEGORIA[c], "rect") for c in CATEGORIAS],
                           ancho))
    piezas.append(_texto(15, sup + (alto - sup - inf) / 2, "Peso del portafolio",
                         tam=10, color=TINTA_TENUE, rotar=True))
    return Grafico(
        id="composicion",
        svg=_envolver_svg(ancho, alto, "composicion", "".join(piezas),
                          "Composición estratégica por clase de activo y perfil"),
        datos={"tipo": "marcas", "marcas": marcas_json},
    )


# =========================================================================== #
# Gráfico 3 — Curvas de equity con retícula
# =========================================================================== #
def grafico_equity(equity: pd.DataFrame, separador: str = "::") -> Grafico:
    """
    Capital acumulado por perfil. Cada panel lleva una retícula vertical que
    engancha la fecha más próxima al cursor y un tooltip con todas las series.
    """
    columnas = [c for c in equity.columns if not c.startswith("BENCH")]
    presentes = {c.split(separador)[0] for c in columnas}
    perfiles = [p for p in ORDEN_PERFILES if p in presentes]
    perfiles += sorted(presentes - set(perfiles))
    benchmarks = [c for c in equity.columns if c.startswith("BENCH")]
    n_paneles = max(len(perfiles), 1)

    ancho, alto = 1240.0, 450.0
    izq, der, sup, inf = 58.0, 62.0, 48.0, 80.0
    hueco = 78.0
    ancho_panel = (ancho - izq - der - hueco * (n_paneles - 1)) / n_paneles

    fechas = pd.DatetimeIndex(equity.index)
    n_obs = len(fechas)
    piezas: list[str] = []
    paneles_json: list[dict] = []

    for i, perfil in enumerate(perfiles):
        series = [c for c in columnas if c.startswith(f"{perfil}{separador}")]
        todas = benchmarks + series
        marco = Marco(izq + i * (ancho_panel + hueco), sup, ancho_panel, alto - sup - inf)

        valores = equity[todas].to_numpy(dtype=float)
        y_min, y_max = float(np.nanmin(valores)), float(np.nanmax(valores))
        colchon = (y_max - y_min) * 0.06 or 0.01
        y_min, y_max = y_min - colchon, y_max + colchon

        ex = marco.escala_x(0, max(n_obs - 1, 1))
        ey = marco.escala_y(y_min, y_max)
        piezas.append(_rejilla_y(marco, ticks_lindos(y_min, y_max, 5), ey, lambda v: f"{v:.2f}"))
        piezas.append(_linea_base(marco))
        for pos in np.unique(np.linspace(0, n_obs - 1, 5).astype(int)):
            piezas.append(_texto(ex(pos), marco.y1 + 18, f"{fechas[pos]:%Y-%m}", tam=10))

        etiquetas_finales: list[tuple[float, str]] = []
        series_json: list[dict] = []
        for col in todas:
            serie = equity[col].to_numpy(dtype=float)
            es_bench = col.startswith("BENCH")
            clave = col.split(separador)[1]
            if es_bench:
                guiones, nombre = ESTILO_BENCHMARK.get(clave, ("5 3", clave.title()))
                color, grosor, dash = TINTA_TENUE, 1.4, f' stroke-dasharray="{guiones}"'
            else:
                nombre = ETIQUETA_METODO.get(clave, clave)
                color = PALETA[series.index(col) % len(PALETA)]
                grosor, dash = 2.0, ""
            puntos = " ".join(
                f"{_n(ex(k))},{_n(ey(v))}" for k, v in enumerate(serie) if np.isfinite(v)
            )
            piezas.append(
                f'<polyline points="{puntos}" fill="none" stroke="{color}" '
                f'stroke-width="{grosor}"{dash} stroke-linejoin="round" stroke-linecap="round"/>'
            )
            finitos = serie[np.isfinite(serie)]
            if len(finitos):
                etiquetas_finales.append((float(finitos[-1]), f"{finitos[-1]:.2f}x"))
            series_json.append(
                {
                    "nombre": nombre, "color": color,
                    "valores": [None if not np.isfinite(v) else round(float(v), 5) for v in serie],
                }
            )

        # Etiquetas directas al final, separadas para que no se solapen.
        separacion = (y_max - y_min) * 0.045
        ordenadas = sorted(etiquetas_finales, key=lambda e: -e[0])
        posiciones: list[float] = []
        for valor, _ in ordenadas:
            posiciones.append(valor if not posiciones else min(valor, posiciones[-1] - separacion))
        for (_, texto), y in zip(ordenadas, posiciones):
            piezas.append(_texto(marco.x1 + 7, ey(y), texto, tam=10, ancla="start"))

        # Capa de interacción: retícula, resaltes y zona sensible (encima de todo).
        piezas.append(
            f'<line class="reticula" x1="0" y1="{_n(marco.y)}" x2="0" y2="{_n(marco.y1)}" '
            f'stroke="{TINTA_TENUE}" stroke-width="1" opacity="0" data-panel="{i}"/>'
        )
        for k, serie in enumerate(series_json):
            piezas.append(
                f'<circle class="foco" r="4.5" fill="{serie["color"]}" stroke="{SUPERFICIE}" '
                f'stroke-width="2" opacity="0" data-panel="{i}" data-serie="{k}"/>'
            )
        piezas.append(
            f'<rect class="zona" x="{_n(marco.x)}" y="{_n(marco.y)}" width="{_n(marco.ancho)}" '
            f'height="{_n(marco.alto)}" fill="transparent" data-panel="{i}" tabindex="0" '
            f'role="application" aria-label="Serie temporal {esc(perfil)}. '
            f'Flechas para recorrer las fechas."/>'
        )
        piezas.append(_titulo_panel(marco, perfil.capitalize()))

        paneles_json.append(
            {
                "titulo": perfil.capitalize(),
                "x": round(marco.x, 2), "y": round(marco.y, 2),
                "ancho": round(marco.ancho, 2), "alto": round(marco.alto, 2),
                "limites": [round(y_min, 6), round(y_max, 6)],
                "n": n_obs, "series": series_json,
            }
        )

    entradas = [
        (ESTILO_BENCHMARK[b.split(separador)[1]][1], TINTA_TENUE, "guion")
        for b in benchmarks if b.split(separador)[1] in ESTILO_BENCHMARK
    ]
    if perfiles:
        primeros = [c for c in columnas if c.startswith(f"{perfiles[0]}{separador}")]
        entradas += [
            (ETIQUETA_METODO.get(c.split(separador)[1], c.split(separador)[1]),
             PALETA[k % len(PALETA)], "linea")
            for k, c in enumerate(primeros)
        ]
    piezas.append(_leyenda(alto - 20, entradas, ancho))
    piezas.append(_texto(15, sup + (alto - sup - inf) / 2, "Capital acumulado (base 1.0)",
                         tam=10, color=TINTA_TENUE, rotar=True))
    return Grafico(
        id="equity",
        svg=_envolver_svg(ancho, alto, "equity", "".join(piezas),
                          "Backtest walk-forward: capital acumulado por perfil"),
        datos={"tipo": "lineas", "fechas": [f"{f:%Y-%m-%d}" for f in fechas],
               "paneles": paneles_json},
    )


# =========================================================================== #
# Capa de interacción (CSS + JS, embebidos una sola vez en el informe)
# =========================================================================== #
CSS_INTERACCION = """
  .tooltip {
    position: fixed; z-index: 50; pointer-events: none;
    background: #ffffff; color: var(--tinta);
    border: 1px solid var(--borde); border-radius: 9px;
    box-shadow: 0 6px 22px rgba(0,0,0,.14);
    padding: 10px 12px; font-size: 12.5px; line-height: 1.5;
    max-width: 340px; opacity: 0; transition: opacity .09s ease;
  }
  .tooltip[data-visible="1"] { opacity: 1; }
  .tooltip .tt-titulo {
    font-weight: 620; margin-bottom: 6px; padding-bottom: 6px;
    border-bottom: 1px solid var(--borde);
    display: flex; gap: 7px; align-items: center;
  }
  .tooltip .tt-clave { width: 14px; height: 3px; border-radius: 2px; flex: none; }
  .tooltip .tt-fila {
    display: flex; justify-content: space-between; gap: 20px; align-items: baseline;
  }
  .tooltip .tt-fila .et { color: var(--tinta-3); font-size: 11.5px; }
  .tooltip .tt-fila .vl { font-weight: 620; font-variant-numeric: tabular-nums; }
  .tooltip .tt-fila .lk {
    display: inline-block; width: 12px; height: 3px; border-radius: 2px;
    margin-right: 6px; vertical-align: middle;
  }
  /* El texto nunca intercepta el cursor: taparía la marca que rotula. */
  svg.grafico text { pointer-events: none; user-select: none; }
  svg.grafico .reticula, svg.grafico .foco { pointer-events: none; }
  svg.grafico .marca, svg.grafico .punto { transition: opacity .1s ease; }
  svg.grafico .marca { cursor: pointer; }
  svg.grafico .zona, svg.grafico .zona-puntos { cursor: crosshair; }
  /* Atenuar lo demás enfoca la marca sin volver ilegibles sus rótulos. */
  svg.grafico .atenuado { opacity: .72; }
  svg.grafico .resaltado { stroke: #0b0b0b; stroke-width: 2.2; }
  svg.grafico .punto.resaltado { r: 7; }
  svg.grafico [tabindex]:focus { outline: none; }
  svg.grafico [tabindex]:focus-visible { outline: 2px solid #2a78d6; outline-offset: 2px; }
  .pista-interaccion {
    color: var(--tinta-3); font-size: 12px; margin: 0 0 10px;
    display: flex; align-items: center; gap: 7px;
  }
"""

JS_INTERACCION = r"""
(function () {
  "use strict";

  var tip = document.createElement("div");
  tip.className = "tooltip";
  tip.setAttribute("role", "tooltip");
  document.body.appendChild(tip);

  /* Los nombres de fondos vienen de una API pública: se insertan como texto,
     nunca como HTML concatenado. */
  function pintar(titulo, color, filas) {
    tip.textContent = "";
    var cab = document.createElement("div");
    cab.className = "tt-titulo";
    if (color) {
      var clave = document.createElement("span");
      clave.className = "tt-clave";
      clave.style.background = color;
      cab.appendChild(clave);
    }
    cab.appendChild(document.createTextNode(titulo));
    tip.appendChild(cab);
    filas.forEach(function (fila) {
      var linea = document.createElement("div");
      linea.className = "tt-fila";
      var et = document.createElement("span");
      et.className = "et";
      if (fila[2]) {
        var lk = document.createElement("span");
        lk.className = "lk";
        lk.style.background = fila[2];
        et.appendChild(lk);
      }
      et.appendChild(document.createTextNode(fila[0]));
      var vl = document.createElement("span");
      vl.className = "vl";
      vl.textContent = fila[1];
      linea.appendChild(et);
      linea.appendChild(vl);
      tip.appendChild(linea);
    });
  }

  function ubicar(x, y) {
    var caja = tip.getBoundingClientRect();
    var izq = x + 16, arr = y + 16;
    if (izq + caja.width > window.innerWidth - 12) izq = x - caja.width - 16;
    if (arr + caja.height > window.innerHeight - 12) arr = y - caja.height - 16;
    tip.style.left = Math.max(8, izq) + "px";
    tip.style.top = Math.max(8, arr) + "px";
    tip.dataset.visible = "1";
  }

  function ocultar() { tip.dataset.visible = "0"; }

  function centro(el) {
    var c = el.getBoundingClientRect();
    return { x: c.left + c.width / 2, y: c.top + c.height / 2 };
  }

  function aCoordSvg(svg, ev) {
    var ctm = svg.getScreenCTM();
    if (!ctm) return null;
    var p = svg.createSVGPoint();
    p.x = ev.clientX; p.y = ev.clientY;
    return p.matrixTransform(ctm.inverse());
  }

  /* --- Barras apiladas: la marca es el objetivo ----------------------- */
  function activarMarcas(svg, datos) {
    var marcas = datos.marcas || [];
    var todas = svg.querySelectorAll(".marca");

    function entrar(el, x, y) {
      var d = marcas[+el.dataset.i];
      if (!d) return;
      todas.forEach(function (m) { if (m !== el) m.classList.add("atenuado"); });
      el.classList.add("resaltado");
      pintar(d.titulo, d.color, d.filas);
      ubicar(x, y);
    }
    function salir() {
      todas.forEach(function (m) {
        m.classList.remove("atenuado");
        m.classList.remove("resaltado");
      });
      ocultar();
    }
    todas.forEach(function (el) {
      el.addEventListener("pointerenter", function (ev) { entrar(el, ev.clientX, ev.clientY); });
      el.addEventListener("pointermove", function (ev) { ubicar(ev.clientX, ev.clientY); });
      el.addEventListener("pointerleave", salir);
      el.addEventListener("focus", function () { var c = centro(el); entrar(el, c.x, c.y); });
      el.addEventListener("blur", salir);
    });
  }

  /* --- Dispersión: capa de punto más cercano -------------------------- */
  function activarPuntos(svg, datos) {
    var pintados = svg.querySelectorAll(".punto");

    function resaltar(ip, k, x, y) {
      var d = datos.paneles[ip] && datos.paneles[ip].puntos[k];
      if (!d) return;
      pintados.forEach(function (p) { p.classList.remove("resaltado"); });
      var el = svg.querySelector('.punto[data-panel="' + ip + '"][data-i="' + k + '"]');
      if (el) el.classList.add("resaltado");
      pintar(d.titulo, d.color, d.filas);
      ubicar(x, y);
    }
    function limpiar() {
      pintados.forEach(function (p) { p.classList.remove("resaltado"); });
      ocultar();
    }

    svg.querySelectorAll(".zona-puntos").forEach(function (zona) {
      var ip = +zona.dataset.panel;
      var puntos = (datos.paneles[ip] || {}).puntos || [];
      zona.addEventListener("pointermove", function (ev) {
        var p = aCoordSvg(svg, ev);
        if (!p) return;
        var mejor = -1, mejorDist = Infinity;
        for (var k = 0; k < puntos.length; k++) {
          var dx = puntos[k].cx - p.x, dy = puntos[k].cy - p.y;
          var dist = dx * dx + dy * dy;
          if (dist < mejorDist) { mejorDist = dist; mejor = k; }
        }
        if (mejor >= 0) resaltar(ip, mejor, ev.clientX, ev.clientY);
      });
      zona.addEventListener("pointerleave", limpiar);
    });

    pintados.forEach(function (el) {
      el.addEventListener("focus", function () {
        var c = centro(el);
        resaltar(+el.dataset.panel, +el.dataset.i, c.x, c.y);
      });
      el.addEventListener("blur", limpiar);
    });
  }

  /* --- Líneas: retícula que engancha la fecha ------------------------- */
  function activarLineas(svg, datos) {
    var fechas = datos.fechas || [];

    datos.paneles.forEach(function (panel, ip) {
      var zona = svg.querySelector('.zona[data-panel="' + ip + '"]');
      var linea = svg.querySelector('.reticula[data-panel="' + ip + '"]');
      var focos = svg.querySelectorAll('.foco[data-panel="' + ip + '"]');
      if (!zona) return;
      var indice = -1;

      function posY(v) {
        var lim = panel.limites;
        return panel.y + panel.alto - ((v - lim[0]) / (lim[1] - lim[0])) * panel.alto;
      }

      function dibujar(i, cx, cy) {
        if (i < 0 || i >= panel.n) return;
        indice = i;
        var x = panel.n > 1 ? panel.x + (i / (panel.n - 1)) * panel.ancho : panel.x;
        linea.setAttribute("x1", x);
        linea.setAttribute("x2", x);
        linea.setAttribute("opacity", "1");

        var filas = [];
        panel.series.forEach(function (serie, k) {
          var v = serie.valores[i];
          var foco = focos[k];
          if (v === null || v === undefined) {
            if (foco) foco.setAttribute("opacity", "0");
            return;
          }
          if (foco) {
            foco.setAttribute("cx", x);
            foco.setAttribute("cy", posY(v));
            foco.setAttribute("opacity", "1");
          }
          filas.push([serie.nombre, v.toFixed(3) + "x", serie.color, v]);
        });
        filas.sort(function (a, b) { return b[3] - a[3]; });
        pintar(panel.titulo + " · " + (fechas[i] || ""), null, filas);
        ubicar(cx, cy);
      }

      function limpiar() {
        linea.setAttribute("opacity", "0");
        focos.forEach(function (f) { f.setAttribute("opacity", "0"); });
        ocultar();
      }

      zona.addEventListener("pointermove", function (ev) {
        var p = aCoordSvg(svg, ev);
        if (!p) return;
        var t = (p.x - panel.x) / panel.ancho;
        var i = Math.round(t * (panel.n - 1));
        dibujar(Math.max(0, Math.min(panel.n - 1, i)), ev.clientX, ev.clientY);
      });
      zona.addEventListener("pointerleave", limpiar);
      zona.addEventListener("blur", limpiar);
      zona.addEventListener("focus", function () {
        var c = centro(zona);
        dibujar(indice < 0 ? panel.n - 1 : indice, c.x, c.y);
      });
      zona.addEventListener("keydown", function (ev) {
        var salto = ev.shiftKey ? 20 : 1, nuevo = indice;
        if (ev.key === "ArrowRight") nuevo = indice + salto;
        else if (ev.key === "ArrowLeft") nuevo = indice - salto;
        else if (ev.key === "Home") nuevo = 0;
        else if (ev.key === "End") nuevo = panel.n - 1;
        else if (ev.key === "Escape") { limpiar(); return; }
        else return;
        ev.preventDefault();
        var c = centro(zona);
        dibujar(Math.max(0, Math.min(panel.n - 1, nuevo)), c.x, c.y);
      });
    });
  }

  document.querySelectorAll("svg.grafico").forEach(function (svg) {
    var nodo = document.getElementById("datos-" + svg.dataset.gid);
    if (!nodo) return;
    var datos = JSON.parse(nodo.textContent);
    if (datos.tipo === "lineas") activarLineas(svg, datos);
    else if (datos.tipo === "puntos") activarPuntos(svg, datos);
    else activarMarcas(svg, datos);
  });

  window.addEventListener("scroll", ocultar, { passive: true });
})();
"""
