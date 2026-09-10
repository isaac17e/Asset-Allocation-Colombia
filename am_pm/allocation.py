"""
Asignación estratégica por perfil: del universo curado a los portafolios modelo.

Para cada perfil se construyen las restricciones del mandato y se resuelven los
cuatro métodos de optimización sobre el mismo conjunto factible, de modo que
las diferencias observadas sean atribuibles al criterio de asignación y no a
distintos grados de libertad.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CATEGORIAS, CATEGORIAS_RV, ConfigOptimizacion
from .metrics import matriz_covarianza, periodos_por_anio, retornos_esperados
from .optimizers import (
    RestriccionesPortafolio,
    contribuciones_riesgo,
    limpiar_pesos,
    resolver,
)
from .profiles import PERFILES, PerfilRiesgo, construir_restricciones
from .universe import UniversoCurado
from .utils import get_logger

log = get_logger("am_pm.allocation")


@dataclass
class ResultadoPerfil:
    """Portafolios modelo y diagnóstico de un perfil de riesgo."""

    perfil: PerfilRiesgo
    pesos: pd.DataFrame                 # index: fondo_id, columns: método
    composicion: pd.DataFrame           # index: categoría, columns: método
    estadisticas: pd.DataFrame          # index: método, métricas ex-ante
    restricciones: RestriccionesPortafolio
    notas: list[str] = field(default_factory=list)


@dataclass
class ResultadoAsignacion:
    """Resultado consolidado de los tres perfiles."""

    perfiles: dict[str, ResultadoPerfil]
    mu: pd.Series
    cov: pd.DataFrame
    rf: float

    def pesos_largos(self) -> pd.DataFrame:
        """Formato largo (tidy) de todos los pesos, listo para exportar."""
        filas = []
        for nombre, res in self.perfiles.items():
            for metodo in res.pesos.columns:
                serie = res.pesos[metodo]
                for fondo, peso in serie[serie > 0].items():
                    filas.append({"perfil": nombre, "metodo": metodo, "fondo_id": fondo, "peso": peso})
        return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Estadísticas ex-ante
# --------------------------------------------------------------------------- #
def estadisticas_ex_ante(
    w: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float, categorias: pd.Series
) -> dict[str, float]:
    """Perfil de riesgo/retorno esperado del portafolio y su diversificación."""
    activos = list(w.index)
    w_v = w.to_numpy(dtype=float)
    cov_m = cov.reindex(index=activos, columns=activos).to_numpy(dtype=float)
    mu_v = mu.reindex(activos).to_numpy(dtype=float)

    vol = float(np.sqrt(max(w_v @ cov_m @ w_v, 1e-18)))
    ret = float(w_v @ mu_v)
    vols_individuales = np.sqrt(np.clip(np.diag(cov_m), 1e-18, None))
    hhi = float(np.sum(w_v ** 2))

    stats = {
        "retorno_esperado": ret,
        "vol_esperada": vol,
        "sharpe_ex_ante": (ret - rf) / vol if vol > 1e-9 else np.nan,
        "ratio_diversificacion": float(w_v @ vols_individuales) / vol if vol > 1e-9 else np.nan,
        "n_efectivo": 1.0 / hhi if hhi > 0 else np.nan,
        "n_posiciones": int((w_v > 1e-6).sum()),
        "peso_maximo": float(w_v.max()),
    }

    rc = contribuciones_riesgo(w_v, cov_m)
    total_rc = float(np.sum(rc))
    for cat in CATEGORIAS:
        mascara = np.array([categorias.get(a) == cat for a in activos])
        stats[f"peso_{cat}"] = float(w_v[mascara].sum()) if mascara.any() else 0.0
        stats[f"riesgo_{cat}"] = (
            float(rc[mascara].sum() / total_rc) if mascara.any() and abs(total_rc) > 1e-12 else 0.0
        )
    stats["peso_RENTA_VARIABLE"] = sum(stats[f"peso_{c}"] for c in CATEGORIAS_RV)
    stats["riesgo_RENTA_VARIABLE"] = sum(stats[f"riesgo_{c}"] for c in CATEGORIAS_RV)
    return stats


def _restringir_a(
    restr: RestriccionesPortafolio, conservar: set[str]
) -> RestriccionesPortafolio:
    """Copia las restricciones anulando la cota superior de los activos excluidos."""
    cota_sup = np.array(
        [restr.cota_superior[i] if a in conservar else 0.0 for i, a in enumerate(restr.activos)]
    )
    return RestriccionesPortafolio(
        restr.activos, restr.cota_inferior, cota_sup, restr.grupos, restr.etiqueta
    )


def _seleccion_cardinalidad(
    pesos: pd.Series, restr: RestriccionesPortafolio, k: int
) -> set[str]:
    """
    Elige los `k` fondos a conservar.

    No basta con tomar los de mayor peso: si todos pertenecen a la misma clase
    de activo, el portafolio reducido no puede cumplir los mínimos del mandato.
    Por eso primero se reserva cupo para cada grupo con mínimo exigido y sólo
    después se completa por tamaño de posición.
    """
    orden = list(pesos.sort_values(ascending=False).index)
    seleccion: list[str] = []

    for g in restr.grupos:
        if g.minimo <= 1e-9 or not g.indices:
            continue
        activos_grupo = {restr.activos[i] for i in g.indices}
        tope_individual = float(np.max(restr.cota_superior[list(g.indices)]))
        necesarios = int(np.ceil(g.minimo / max(tope_individual, 1e-9)))
        candidatos = [a for a in orden if a in activos_grupo][:necesarios]
        seleccion.extend(candidatos)

    for activo in orden:
        if len(set(seleccion)) >= k:
            break
        seleccion.append(activo)
    return set(seleccion)


def aplicar_cardinalidad(
    pesos: pd.Series, max_fondos: int, restr: RestriccionesPortafolio
) -> pd.Series:
    """
    Limita el número de posiciones del portafolio.

    En vez de truncar y renormalizar (lo que rompería las bandas del mandato),
    se anula la cota superior de los fondos excluidos y se reproyecta, de modo
    que el resultado siga siendo factible dentro del subconjunto elegido. Si el
    límite es incompatible con los mínimos por clase de activo, se amplía la
    cardinalidad al mínimo número de posiciones que sí respeta el mandato: el
    límite de posiciones es un control operativo, no una restricción de riesgo.
    """
    activas = int((pesos > 1e-6).sum())
    if activas <= max_fondos:
        return pesos

    for k in range(max_fondos, min(activas, restr.n) + 1):
        conservar = _seleccion_cardinalidad(pesos, restr, k)
        restr_reducida = _restringir_a(restr, conservar)
        if restr_reducida.punto_factible() is None:
            continue
        w = restr_reducida.proyectar(pesos.to_numpy(dtype=float))
        if k > max_fondos:
            log.info(
                "[%s] Cardinalidad ampliada de %d a %d posiciones por los mínimos del mandato.",
                restr.etiqueta, max_fondos, k,
            )
        return pd.Series(w, index=pesos.index, name=pesos.name)

    log.warning(
        "[%s] No hay subconjunto factible con %d posiciones; se conservan %d.",
        restr.etiqueta, max_fondos, activas,
    )
    return pesos


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def optimizar_perfil(
    universo: UniversoCurado, perfil: PerfilRiesgo, mu: pd.Series, cov: pd.DataFrame,
    cfg: ConfigOptimizacion,
) -> ResultadoPerfil:
    """Resuelve los cuatro métodos de asignación para un perfil."""
    categorias = universo.mapa_categoria()
    gestores = universo.mapa_gestor()
    restr, notas = construir_restricciones(perfil, categorias, gestores)

    pesos: dict[str, pd.Series] = {}
    stats: dict[str, dict[str, float]] = {}
    for metodo in cfg.metodos:
        w = resolver(metodo, mu, cov, universo.rf, restr, cfg.n_arranques, cfg.max_iter)
        w = limpiar_pesos(w, restr, cfg.peso_minimo_operativo)
        w = aplicar_cardinalidad(w, perfil.max_fondos, restr)
        violaciones = restr.violaciones(w.to_numpy(dtype=float), tol=1e-4)
        if violaciones:
            log.warning("[%s/%s] Restricciones con holgura numérica: %s",
                        perfil.nombre, metodo, {k: round(v, 4) for k, v in violaciones.items()})
        pesos[metodo] = w
        stats[metodo] = estadisticas_ex_ante(w, mu, cov, universo.rf, categorias)
        log.info(
            "[%s/%s] ret=%.2f%% vol=%.2f%% sharpe=%.2f RV=%.1f%% n=%d",
            perfil.nombre, metodo, stats[metodo]["retorno_esperado"] * 100,
            stats[metodo]["vol_esperada"] * 100, stats[metodo]["sharpe_ex_ante"],
            stats[metodo]["peso_RENTA_VARIABLE"] * 100, stats[metodo]["n_posiciones"],
        )

    df_pesos = pd.DataFrame(pesos)
    composicion = (
        df_pesos.groupby(categorias.reindex(df_pesos.index), observed=True).sum()
        .reindex(CATEGORIAS).fillna(0.0)
    )
    composicion.index.name = "categoria"
    return ResultadoPerfil(
        perfil=perfil,
        pesos=df_pesos,
        composicion=composicion,
        estadisticas=pd.DataFrame(stats).T,
        restricciones=restr,
        notas=notas,
    )


def construir_portafolios(
    universo: UniversoCurado, cfg: ConfigOptimizacion, perfiles: tuple[str, ...] | None = None
) -> ResultadoAsignacion:
    """Genera los portafolios modelo de todos los perfiles solicitados."""
    ppa = periodos_por_anio(universo.precios.index)
    mu = retornos_esperados(universo.precios, cfg.shrinkage_mu)
    cov = matriz_covarianza(universo.retornos, ppa, cfg.shrinkage_cov)
    activos = [a for a in universo.precios.columns if a in mu.index and a in cov.index]
    mu, cov = mu.reindex(activos), cov.reindex(index=activos, columns=activos)
    log.info("Insumos de optimización: %d activos | r_f=%.2f%% | mu medio=%.2f%%",
             len(activos), universo.rf * 100, mu.mean() * 100)

    nombres = perfiles or tuple(PERFILES)
    resultados = {
        nombre: optimizar_perfil(universo, PERFILES[nombre], mu, cov, cfg) for nombre in nombres
    }
    return ResultadoAsignacion(perfiles=resultados, mu=mu, cov=cov, rf=universo.rf)
