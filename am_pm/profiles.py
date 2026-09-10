"""
Perfiles de riesgo y política de inversión (P.Perfil).

Cada perfil define un mandato: bandas por clase de activo, piso/techo de renta
variable y límites de concentración por fondo y por gestora (riesgo de emisor y
riesgo fiduciario). El mandato se traduce a un conjunto de restricciones
lineales que consume el motor de optimización.

Cuando el universo disponible no permite cumplir el mandato (p. ej. no hay
fondos de renta variable internacional con historia suficiente), las
restricciones se relajan de forma ordenada y trazable en lugar de fallar: el
comité debe enterarse de la desviación, no recibir un error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .config import CATEGORIAS, CATEGORIAS_RV
from .optimizers import GrupoRestriccion, RestriccionesPortafolio
from .utils import get_logger

log = get_logger("am_pm.profiles")


@dataclass(frozen=True)
class PerfilRiesgo:
    """Mandato de inversión de un perfil."""

    nombre: str
    descripcion: str
    horizonte_meses: int
    #: Banda (mínimo, máximo) de peso por clase de activo.
    limites_categoria: Mapping[str, tuple[float, float]]
    #: Piso y techo de la exposición agregada a renta variable.
    rv_min: float
    rv_max: float
    #: Concentración máxima admitida en un solo fondo.
    max_peso_fondo: float
    #: Concentración máxima admitida en una sola gestora/fiduciaria.
    max_peso_gestor: float
    #: Número máximo de posiciones (control operativo y de costos).
    max_fondos: int
    #: Banda de tolerancia antes de disparar rebalanceo táctico.
    banda_rebalanceo: float = 0.05

    def limite(self, categoria: str) -> tuple[float, float]:
        return self.limites_categoria.get(categoria, (0.0, 0.0))


#: Política estándar de la casa. Los topes de renta variable siguen la
#: segmentación regulatoria y comercial habitual del mercado colombiano.
PERFILES: dict[str, PerfilRiesgo] = {
    "CONSERVADOR": PerfilRiesgo(
        nombre="CONSERVADOR",
        descripcion=(
            "Preservación de capital y liquidez. Núcleo en renta fija de corto "
            "plazo, con renta variable estrictamente accesoria (<=10%)."
        ),
        horizonte_meses=12,
        limites_categoria={
            "RF_CORTO": (0.35, 0.80),
            "RF_MEDIANO_LARGO": (0.15, 0.55),
            "MIXTO": (0.00, 0.15),
            "RV_LOCAL": (0.00, 0.06),
            "RV_INTERNACIONAL": (0.00, 0.08),
        },
        rv_min=0.00,
        rv_max=0.10,
        max_peso_fondo=0.20,
        max_peso_gestor=0.35,
        max_fondos=8,
        banda_rebalanceo=0.05,
    ),
    "MODERADO": PerfilRiesgo(
        nombre="MODERADO",
        descripcion=(
            "Balance entre renta fija, fondos mixtos y renta variable. "
            "Exposición accionaria acotada al 40% del portafolio."
        ),
        horizonte_meses=36,
        limites_categoria={
            "RF_CORTO": (0.10, 0.40),
            "RF_MEDIANO_LARGO": (0.15, 0.45),
            "MIXTO": (0.00, 0.30),
            "RV_LOCAL": (0.05, 0.25),
            "RV_INTERNACIONAL": (0.05, 0.30),
        },
        rv_min=0.20,
        rv_max=0.40,
        max_peso_fondo=0.15,
        max_peso_gestor=0.30,
        max_fondos=10,
        banda_rebalanceo=0.05,
    ),
    "AGRESIVO": PerfilRiesgo(
        nombre="AGRESIVO",
        descripcion=(
            "Maximización de crecimiento de largo plazo. Mínimo 50% en renta "
            "variable local e internacional; renta fija como amortiguador."
        ),
        horizonte_meses=60,
        limites_categoria={
            "RF_CORTO": (0.02, 0.15),
            "RF_MEDIANO_LARGO": (0.00, 0.20),
            "MIXTO": (0.00, 0.25),
            "RV_LOCAL": (0.15, 0.45),
            "RV_INTERNACIONAL": (0.15, 0.55),
        },
        rv_min=0.50,
        rv_max=0.85,
        max_peso_fondo=0.12,
        max_peso_gestor=0.25,
        max_fondos=12,
        banda_rebalanceo=0.05,
    ),
}


def obtener_perfil(nombre: str) -> PerfilRiesgo:
    clave = nombre.strip().upper()
    if clave not in PERFILES:
        raise KeyError(f"Perfil desconocido: {nombre}. Opciones: {list(PERFILES)}")
    return PERFILES[clave]


# --------------------------------------------------------------------------- #
# Traducción del mandato a restricciones
# --------------------------------------------------------------------------- #
def construir_restricciones(
    perfil: PerfilRiesgo, categorias: pd.Series, gestores: pd.Series
) -> tuple[RestriccionesPortafolio, list[str]]:
    """
    Traduce el mandato del perfil al conjunto factible sobre el universo dado.

    Devuelve las restricciones y la bitácora de relajaciones aplicadas.
    """
    activos = tuple(categorias.index)
    n = len(activos)
    notas: list[str] = []

    cota_inf = np.zeros(n)
    cota_sup = np.full(n, perfil.max_peso_fondo)

    idx_por_categoria = {
        cat: tuple(i for i, a in enumerate(activos) if categorias[a] == cat)
        for cat in CATEGORIAS
    }

    grupos: list[GrupoRestriccion] = []
    for cat, indices in idx_por_categoria.items():
        minimo, maximo = perfil.limite(cat)
        if not indices:
            if minimo > 0:
                notas.append(
                    f"Sin fondos en {cat}: se anula el mínimo de {minimo:.0%} del mandato."
                )
            continue
        # Capacidad real: nº de fondos disponibles x tope individual.
        capacidad = len(indices) * perfil.max_peso_fondo
        if capacidad < minimo:
            notas.append(
                f"{cat}: mínimo {minimo:.0%} recortado a {capacidad:.0%} "
                f"(sólo {len(indices)} fondos elegibles)."
            )
            minimo = capacidad
        grupos.append(GrupoRestriccion(cat, indices, minimo, min(maximo, capacidad)))

    indices_rv = tuple(
        i for i, a in enumerate(activos) if categorias[a] in CATEGORIAS_RV
    )
    if indices_rv:
        capacidad_rv = len(indices_rv) * perfil.max_peso_fondo
        rv_min = min(perfil.rv_min, capacidad_rv)
        if rv_min < perfil.rv_min:
            notas.append(
                f"Renta variable: piso {perfil.rv_min:.0%} recortado a {rv_min:.0%} por capacidad."
            )
        grupos.append(GrupoRestriccion("RENTA_VARIABLE", indices_rv, rv_min, perfil.rv_max))
    elif perfil.rv_min > 0:
        notas.append("Sin fondos de renta variable: se anula el piso de RV del mandato.")

    for gestor, indices in _indices_por_grupo(gestores, activos).items():
        grupos.append(
            GrupoRestriccion(f"GESTOR::{gestor}", indices, 0.0, perfil.max_peso_gestor)
        )

    restr = RestriccionesPortafolio(
        activos=activos,
        cota_inferior=cota_inf,
        cota_superior=cota_sup,
        grupos=tuple(grupos),
        etiqueta=perfil.nombre,
    )
    return _asegurar_factibilidad(restr, perfil, notas)


def _indices_por_grupo(serie: pd.Series, activos: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    """Agrupa índices posicionales de activos por valor de la serie (gestora)."""
    mapa: dict[str, list[int]] = {}
    for i, a in enumerate(activos):
        clave = str(serie.get(a, "NA"))
        mapa.setdefault(clave, []).append(i)
    return {k: tuple(v) for k, v in mapa.items()}


def _relajar_gestores(restr: RestriccionesPortafolio, tope: float) -> RestriccionesPortafolio:
    grupos = tuple(
        GrupoRestriccion(g.nombre, g.indices, g.minimo, max(g.maximo, tope))
        if g.nombre.startswith("GESTOR::") else g
        for g in restr.grupos
    )
    return RestriccionesPortafolio(
        restr.activos, restr.cota_inferior, restr.cota_superior, grupos, restr.etiqueta
    )


def _relajar_maximos_categoria(restr: RestriccionesPortafolio, factor: float) -> RestriccionesPortafolio:
    grupos = tuple(
        GrupoRestriccion(g.nombre, g.indices, g.minimo, min(1.0, g.maximo * factor))
        if g.nombre in CATEGORIAS else g
        for g in restr.grupos
    )
    return RestriccionesPortafolio(
        restr.activos, restr.cota_inferior, restr.cota_superior, grupos, restr.etiqueta
    )


def _sin_minimos(restr: RestriccionesPortafolio) -> RestriccionesPortafolio:
    grupos = tuple(GrupoRestriccion(g.nombre, g.indices, 0.0, g.maximo) for g in restr.grupos)
    return RestriccionesPortafolio(
        restr.activos, restr.cota_inferior, restr.cota_superior, grupos, restr.etiqueta
    )


def _asegurar_factibilidad(
    restr: RestriccionesPortafolio, perfil: PerfilRiesgo, notas: list[str]
) -> tuple[RestriccionesPortafolio, list[str]]:
    """
    Verifica factibilidad con un LP y, si no la hay, relaja en cascada
    **acumulativa**: cada etapa conserva las relajaciones de las anteriores.

    El orden refleja la jerarquía de la política de inversión: primero se ceden
    los límites de concentración (operativos), después los mínimos estratégicos
    por clase de activo y sólo al final los máximos, que son el corazón del
    mandato de riesgo del perfil.
    """
    if restr.punto_factible() is not None:
        for nota in notas:
            log.warning("[%s] %s", perfil.nombre, nota)
        return restr, notas

    # Etapa 1 — tope por gestora.
    for factor in (1.25, 1.5, 2.0, 4.0):
        tope = min(1.0, perfil.max_peso_gestor * factor)
        candidato = _relajar_gestores(restr, tope)
        if candidato.punto_factible() is not None:
            notas.append(
                f"Tope por gestora relajado de {perfil.max_peso_gestor:.0%} a {tope:.0%}."
            )
            return _finalizar(candidato, perfil, notas)
        restr = candidato  # se acumula la relajación para la siguiente etapa
    notas.append(f"Tope por gestora elevado a {min(1.0, perfil.max_peso_gestor * 4.0):.0%}.")

    # Etapa 2 — tope por fondo.
    for factor in (1.25, 1.5, 2.0, 3.0):
        tope = min(1.0, perfil.max_peso_fondo * factor)
        candidato = RestriccionesPortafolio(
            restr.activos, restr.cota_inferior, np.full(restr.n, tope),
            restr.grupos, restr.etiqueta,
        )
        if candidato.punto_factible() is not None:
            notas.append(f"Tope por fondo relajado de {perfil.max_peso_fondo:.0%} a {tope:.0%}.")
            return _finalizar(candidato, perfil, notas)
        restr = candidato
    notas.append(f"Tope por fondo elevado a {min(1.0, perfil.max_peso_fondo * 3.0):.0%}.")

    # Etapa 3 — mínimos estratégicos por categoría.
    candidato = _sin_minimos(restr)
    if candidato.punto_factible() is not None:
        notas.append("Mínimos por categoría anulados: el universo no admite el mandato completo.")
        return _finalizar(candidato, perfil, notas)
    restr = candidato

    # Etapa 4 — máximos por categoría (último recurso: se desvirtúa el perfil).
    for factor in (1.5, 2.0, 3.0, 100.0):
        candidato = _relajar_maximos_categoria(restr, factor)
        if candidato.punto_factible() is not None:
            notas.append(
                f"ALERTA: máximos por categoría ampliados x{factor:g}. El portafolio "
                "resultante NO refleja el mandato del perfil; revisar el universo."
            )
            return _finalizar(candidato, perfil, notas)

    notas.append(
        "CRÍTICO: no existe portafolio factible ni con todas las restricciones relajadas."
    )
    return _finalizar(restr, perfil, notas)


def _finalizar(
    restr: RestriccionesPortafolio, perfil: PerfilRiesgo, notas: list[str]
) -> tuple[RestriccionesPortafolio, list[str]]:
    for nota in notas:
        log.warning("[%s] %s", perfil.nombre, nota)
    return restr, notas


def resumen_politica() -> pd.DataFrame:
    """Tabla legible de la política de inversión de los tres perfiles."""
    filas = []
    for perfil in PERFILES.values():
        fila = {
            "perfil": perfil.nombre,
            "horizonte_meses": perfil.horizonte_meses,
            "rv_min": perfil.rv_min,
            "rv_max": perfil.rv_max,
            "max_peso_fondo": perfil.max_peso_fondo,
            "max_peso_gestor": perfil.max_peso_gestor,
            "max_fondos": perfil.max_fondos,
            "banda_rebalanceo": perfil.banda_rebalanceo,
        }
        for cat in CATEGORIAS:
            lo, hi = perfil.limite(cat)
            fila[f"{cat}_min"] = lo
            fila[f"{cat}_max"] = hi
        filas.append(fila)
    return pd.DataFrame(filas).set_index("perfil")
