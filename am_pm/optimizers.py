"""
Motor de optimización de portafolios con restricciones institucionales.

Métodos implementados (todos sujetos al mismo conjunto de restricciones, de
modo que la comparación entre ellos sea limpia):

  * MARKOWITZ_SHARPE — máximo ratio de Sharpe con r_f dinámica.
  * RISK_PARITY      — igualación de contribuciones marginales al riesgo.
  * HRP              — Hierarchical Risk Parity (López de Prado), proyectado
                       al conjunto factible.
  * EQUIPONDERADO    — 1/N, proyectado al conjunto factible.

Las restricciones (bandas por clase de activo, tope por fondo, tope por
gestora, piso/techo de renta variable) son lineales, por lo que la factibilidad
se verifica con programación lineal antes de lanzar el optimizador no lineal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

from .utils import get_logger

log = get_logger("am_pm.optimizers")


@dataclass(frozen=True)
class GrupoRestriccion:
    """Restricción lineal de suma de pesos sobre un subconjunto de activos."""

    nombre: str
    indices: tuple[int, ...]
    minimo: float = 0.0
    maximo: float = 1.0


@dataclass
class RestriccionesPortafolio:
    """
    Conjunto factible: presupuesto pleno, cotas por activo y grupos lineales.

    Todo el motor trabaja sobre esta abstracción, lo que permite añadir nuevos
    límites (por emisor, por moneda, por liquidez) sin tocar los optimizadores.
    """

    activos: tuple[str, ...]
    cota_inferior: np.ndarray
    cota_superior: np.ndarray
    grupos: tuple[GrupoRestriccion, ...] = field(default_factory=tuple)
    etiqueta: str = ""
    #: Resultado memorizado del LP de viabilidad. Las restricciones no se
    #: modifican tras construirse (relajar crea una instancia nueva), y el LP se
    #: pide decenas de veces por optimización: semillas, proyecciones, limpieza.
    _factible: tuple[np.ndarray | None] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def n(self) -> int:
        return len(self.activos)

    # ------------------------------------------------------------------ #
    # Representación matricial
    # ------------------------------------------------------------------ #
    def matrices_desigualdad(self) -> tuple[np.ndarray, np.ndarray]:
        """Construye A_ub, b_ub tal que A_ub @ w <= b_ub."""
        filas: list[np.ndarray] = []
        cotas: list[float] = []
        for g in self.grupos:
            indicador = np.zeros(self.n)
            indicador[list(g.indices)] = 1.0
            if g.maximo < 1.0 - 1e-12:
                filas.append(indicador)
                cotas.append(g.maximo)
            if g.minimo > 1e-12:
                filas.append(-indicador)
                cotas.append(-g.minimo)
        if not filas:
            return np.zeros((0, self.n)), np.zeros(0)
        return np.vstack(filas), np.asarray(cotas, dtype=float)

    def bounds(self) -> list[tuple[float, float]]:
        return list(zip(self.cota_inferior.tolist(), self.cota_superior.tolist()))

    def restricciones_scipy(self) -> list[dict]:
        """Restricciones en el formato de `scipy.optimize.minimize` (SLSQP)."""
        A_ub, b_ub = self.matrices_desigualdad()
        cons: list[dict] = [
            {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": np.ones_like}
        ]
        if A_ub.shape[0]:
            cons.append(
                {"type": "ineq", "fun": lambda w, A=A_ub, b=b_ub: b - A @ w,
                 "jac": lambda w, A=A_ub: -A}
            )
        return cons

    # ------------------------------------------------------------------ #
    # Factibilidad y proyección
    # ------------------------------------------------------------------ #
    def punto_factible(self) -> np.ndarray | None:
        """Halla un punto factible resolviendo un LP de viabilidad (memorizado)."""
        if self._factible is None:
            self._factible = (self._resolver_lp_factible(),)
        punto = self._factible[0]
        return None if punto is None else punto.copy()

    def _resolver_lp_factible(self) -> np.ndarray | None:
        A_ub, b_ub = self.matrices_desigualdad()
        res = linprog(
            c=np.zeros(self.n),
            A_ub=A_ub if A_ub.shape[0] else None,
            b_ub=b_ub if A_ub.shape[0] else None,
            A_eq=np.ones((1, self.n)),
            b_eq=np.array([1.0]),
            bounds=self.bounds(),
            method="highs",
        )
        return np.asarray(res.x, dtype=float) if res.success else None

    def es_factible(self, w: np.ndarray, tol: float = 1e-6) -> bool:
        return not self.violaciones(w, tol)

    def violaciones(self, w: np.ndarray, tol: float = 1e-6) -> dict[str, float]:
        """Diagnóstico de restricciones incumplidas (para auditoría)."""
        fallos: dict[str, float] = {}
        if abs(float(np.sum(w)) - 1.0) > tol:
            fallos["presupuesto"] = float(np.sum(w)) - 1.0
        exceso_sup = float(np.max(w - self.cota_superior)) if self.n else 0.0
        if exceso_sup > tol:
            fallos["cota_superior_activo"] = exceso_sup
        exceso_inf = float(np.max(self.cota_inferior - w)) if self.n else 0.0
        if exceso_inf > tol:
            fallos["cota_inferior_activo"] = exceso_inf
        for g in self.grupos:
            peso = float(np.sum(w[list(g.indices)])) if g.indices else 0.0
            if peso > g.maximo + tol:
                fallos[f"max::{g.nombre}"] = peso - g.maximo
            if peso < g.minimo - tol:
                fallos[f"min::{g.nombre}"] = g.minimo - peso
        return fallos

    def proyectar(self, w_objetivo: np.ndarray) -> np.ndarray:
        """
        Proyección euclídea sobre el conjunto factible: el portafolio factible
        más cercano al vector propuesto. Es lo que permite llevar soluciones
        no restringidas (HRP, 1/N) al mandato del perfil sin desnaturalizarlas.
        """
        inicio = self.punto_factible()
        if inicio is None:
            log.error("Conjunto factible vacío para %s; se devuelve el objetivo normalizado.",
                      self.etiqueta)
            w = np.clip(w_objetivo, self.cota_inferior, self.cota_superior)
            return w / w.sum() if w.sum() > 0 else np.full(self.n, 1.0 / self.n)

        res = minimize(
            lambda w: float(np.sum((w - w_objetivo) ** 2)),
            x0=inicio,
            jac=lambda w: 2.0 * (w - w_objetivo),
            method="SLSQP",
            bounds=self.bounds(),
            constraints=self.restricciones_scipy(),
            options={"maxiter": 300, "ftol": 1e-12},
        )
        w = np.asarray(res.x, dtype=float) if res.success else inicio
        return _sanear(w, self)


# --------------------------------------------------------------------------- #
# Utilidades numéricas
# --------------------------------------------------------------------------- #
def _sanear(w: np.ndarray, restr: RestriccionesPortafolio) -> np.ndarray:
    """Corrige ruido numérico: recorta a las cotas y renormaliza a 1."""
    w = np.clip(np.nan_to_num(w, nan=0.0), restr.cota_inferior, restr.cota_superior)
    total = float(np.sum(w))
    if total <= 0:
        return np.full(restr.n, 1.0 / restr.n)
    return w / total


def _puntos_iniciales(restr: RestriccionesPortafolio, n_arranques: int, semilla: int) -> list[np.ndarray]:
    """Arranques múltiples: LP factible, 1/N proyectado y aleatorios Dirichlet."""
    rng = np.random.default_rng(semilla)
    inicios: list[np.ndarray] = []
    base = restr.punto_factible()
    if base is not None:
        inicios.append(base)
    inicios.append(restr.proyectar(np.full(restr.n, 1.0 / restr.n)))
    for _ in range(max(0, n_arranques - len(inicios))):
        inicios.append(restr.proyectar(rng.dirichlet(np.ones(restr.n))))
    return inicios


def contribuciones_riesgo(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Contribución de cada activo al riesgo total del portafolio."""
    vol = float(np.sqrt(max(w @ cov @ w, 1e-18)))
    return w * (cov @ w) / vol


# --------------------------------------------------------------------------- #
# Optimizadores
# --------------------------------------------------------------------------- #
def optimizar_min_varianza(
    cov: np.ndarray, restr: RestriccionesPortafolio, max_iter: int = 500
) -> np.ndarray:
    """Portafolio de mínima varianza sujeto a las restricciones del mandato."""
    inicio = restr.punto_factible()
    if inicio is None:
        return restr.proyectar(np.full(restr.n, 1.0 / restr.n))
    res = minimize(
        lambda w: float(w @ cov @ w), x0=inicio, jac=lambda w: 2.0 * (cov @ w),
        method="SLSQP", bounds=restr.bounds(), constraints=restr.restricciones_scipy(),
        options={"maxiter": max_iter, "ftol": 1e-12},
    )
    return _sanear(np.asarray(res.x, dtype=float) if res.success else inicio, restr)


def _max_sharpe_convexo(
    exceso: np.ndarray, cov: np.ndarray, restr: RestriccionesPortafolio, max_iter: int
) -> np.ndarray | None:
    """
    Máximo Sharpe por la transformación de Schaible.

    Maximizar (μ-r_f)'w / √(w'Σw) es un problema fraccional que SLSQP resuelve
    mal cuando la volatilidad objetivo es muy baja (caso típico del perfil
    conservador: vol ~0.5%, donde el cociente explota numéricamente).

    Con el cambio de variable y = κw, κ > 0, el problema equivale a

        min  y'Σy   s.a.  (μ-r_f)'y = 1,  1'y = κ,  A y <= b κ,  lκ <= y <= uκ

    que es un programa cuadrático convexo — bien condicionado y con óptimo
    global garantizado. La solución original se recupera como w = y/κ.
    """
    n = restr.n
    A_ub, b_ub = restr.matrices_desigualdad()
    cota_inf, cota_sup = restr.cota_inferior, restr.cota_superior

    semillas = _semillas_convexas(exceso, restr)
    if not semillas:
        return None

    def objetivo(z: np.ndarray) -> float:
        y = z[:n]
        return float(y @ cov @ y)

    def gradiente(z: np.ndarray) -> np.ndarray:
        y = z[:n]
        return np.concatenate([2.0 * (cov @ y), [0.0]])

    restricciones = [
        {"type": "eq", "fun": lambda z: float(exceso @ z[:n] - 1.0),
         "jac": lambda z: np.concatenate([exceso, [0.0]])},
        {"type": "eq", "fun": lambda z: float(np.sum(z[:n]) - z[n]),
         "jac": lambda z: np.concatenate([np.ones(n), [-1.0]])},
        {"type": "ineq", "fun": lambda z: cota_sup * z[n] - z[:n],
         "jac": lambda z: np.column_stack([-np.eye(n), cota_sup])},
        {"type": "ineq", "fun": lambda z: z[:n] - cota_inf * z[n],
         "jac": lambda z: np.column_stack([np.eye(n), -cota_inf])},
    ]
    if A_ub.shape[0]:
        restricciones.append(
            {"type": "ineq", "fun": lambda z: b_ub * z[n] - A_ub @ z[:n],
             "jac": lambda z: np.column_stack([-A_ub, b_ub])}
        )

    for x0 in semillas:
        res = minimize(
            objetivo, x0=x0, jac=gradiente, method="SLSQP",
            bounds=[(0.0, None)] * n + [(1e-9, None)],
            constraints=restricciones, options={"maxiter": max_iter, "ftol": 1e-12},
        )
        if not res.success:
            continue
        y, kappa = np.asarray(res.x[:n], dtype=float), float(res.x[n])
        if kappa <= 1e-9:
            continue
        w = _sanear(y / kappa, restr)
        if restr.es_factible(w, tol=1e-4):
            return w
    return None


def _semillas_convexas(
    exceso: np.ndarray, restr: RestriccionesPortafolio
) -> list[np.ndarray]:
    """
    Arranques del problema transformado: portafolios factibles con exceso de
    retorno positivo, reescalados para cumplir (μ-r_f)'y = 1.

    Se ofrecen varios porque SLSQP es sensible al punto inicial cuando las
    restricciones de grupo están activas en el óptimo.
    """
    candidatos: list[np.ndarray] = []
    base = restr.punto_factible()
    if base is not None:
        candidatos.append(base)
    candidatos.append(restr.proyectar(np.full(restr.n, 1.0 / restr.n)))
    # Sesgado hacia el activo de mayor exceso y hacia los tres mejores.
    for k in (1, 3):
        objetivo = np.zeros(restr.n)
        mejores = np.argsort(exceso)[-k:]
        objetivo[mejores] = 1.0 / k
        candidatos.append(restr.proyectar(objetivo))

    semillas: list[np.ndarray] = []
    for w in candidatos:
        excedente = float(exceso @ w)
        if excedente > 1e-9:
            kappa = 1.0 / excedente
            semillas.append(np.concatenate([w * kappa, [kappa]]))
    return semillas


def optimizar_max_sharpe(
    mu: np.ndarray, cov: np.ndarray, rf: float, restr: RestriccionesPortafolio,
    n_arranques: int = 6, max_iter: int = 500, semilla: int = 42,
) -> np.ndarray:
    """
    Markowitz en su forma de máximo Sharpe (cartera tangente con restricciones).

    Se resuelve por la vía convexa y sólo se recurre al método no lineal
    directo si aquélla no converge.
    """
    exceso = mu - rf
    if float(np.max(exceso)) <= 1e-9:
        log.warning(
            "[%s] Ningún fondo supera r_f=%.2f%%: la cartera tangente no existe; "
            "se resuelve mínima varianza.", restr.etiqueta, rf * 100,
        )
        return optimizar_min_varianza(cov, restr, max_iter)

    w = _max_sharpe_convexo(exceso, cov, restr, max_iter)
    if w is not None and restr.es_factible(w, tol=1e-4):
        return w

    log.warning("[%s] Transformación convexa no convergió; se usa SLSQP directo.",
                restr.etiqueta)

    def objetivo(w_: np.ndarray) -> float:
        vol = float(np.sqrt(max(w_ @ cov @ w_, 1e-18)))
        return -float((w_ @ exceso) / vol)

    def gradiente(w_: np.ndarray) -> np.ndarray:
        cov_w = cov @ w_
        vol = float(np.sqrt(max(w_ @ cov_w, 1e-18)))
        return -(exceso / vol - float(w_ @ exceso) * cov_w / vol ** 3)

    return _resolver_no_lineal(
        objetivo, restr, n_arranques, max_iter, semilla, "max_sharpe", gradiente
    )


def optimizar_risk_parity(
    cov: np.ndarray, restr: RestriccionesPortafolio,
    n_arranques: int = 6, max_iter: int = 500, semilla: int = 42,
) -> np.ndarray:
    """
    Paridad de riesgo: minimiza la dispersión de las contribuciones al riesgo.

    Bajo restricciones activas la paridad exacta puede ser inalcanzable; el
    objetivo cuadrático entrega la solución más cercana admisible.
    """

    def objetivo(w: np.ndarray) -> float:
        rc = contribuciones_riesgo(w, cov)
        return float(np.sum((rc - rc.mean()) ** 2)) * 1e4

    def gradiente(w: np.ndarray) -> np.ndarray:
        # Con rc_i = w_i (Σw)_i / σ y c = rc - media(rc), como Σc = 0 el
        # término de la media se anula y ∇f = 2·10⁴ · J'c, con
        # J'c = [(Σw)∘c + Σ(w∘c)] / σ − (Σw) · ((w∘Σw)·c) / σ³.
        cov_w = cov @ w
        vol = float(np.sqrt(max(w @ cov_w, 1e-18)))
        c = w * cov_w / vol
        c = c - c.mean()
        jt_c = (cov_w * c + cov @ (w * c)) / vol - cov_w * float((w * cov_w) @ c) / vol ** 3
        return 2e4 * jt_c

    return _resolver_no_lineal(
        objetivo, restr, n_arranques, max_iter, semilla, "risk_parity", gradiente
    )


def _resolver_no_lineal(
    objetivo, restr: RestriccionesPortafolio, n_arranques: int, max_iter: int,
    semilla: int, etiqueta: str, gradiente=None,
) -> np.ndarray:
    """SLSQP con multi-arranque; se queda con el mejor óptimo factible."""
    mejor_w: np.ndarray | None = None
    mejor_valor = np.inf
    for x0 in _puntos_iniciales(restr, n_arranques, semilla):
        try:
            res = minimize(
                objetivo, x0=x0, jac=gradiente, method="SLSQP", bounds=restr.bounds(),
                constraints=restr.restricciones_scipy(),
                options={"maxiter": max_iter, "ftol": 1e-10},
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("Arranque fallido en %s: %s", etiqueta, exc)
            continue
        w = _sanear(np.asarray(res.x, dtype=float), restr)
        valor = float(objetivo(w))
        if res.success and valor < mejor_valor and restr.es_factible(w, tol=1e-4):
            mejor_valor, mejor_w = valor, w
    if mejor_w is None:
        log.warning("%s sin solución factible en %s; se usa proyección de 1/N.",
                    etiqueta, restr.etiqueta)
        return restr.proyectar(np.full(restr.n, 1.0 / restr.n))
    return mejor_w


def optimizar_equiponderado(restr: RestriccionesPortafolio) -> np.ndarray:
    """1/N llevado al conjunto factible (benchmark ingenuo de referencia)."""
    return restr.proyectar(np.full(restr.n, 1.0 / restr.n))


# --------------------------------------------------------------------------- #
# HRP — Hierarchical Risk Parity
# --------------------------------------------------------------------------- #
def _quasi_diagonalizar(enlace: np.ndarray) -> list[int]:
    """Reordena los activos siguiendo el dendrograma (López de Prado, 2016)."""
    enlace = enlace.astype(int)
    orden = pd.Series([enlace[-1, 0], enlace[-1, 1]])
    n_items = enlace[-1, 3]
    while orden.max() >= n_items:
        orden.index = range(0, orden.shape[0] * 2, 2)
        clusters = orden[orden >= n_items]
        i, j = clusters.index, clusters.to_numpy() - n_items
        orden[i] = enlace[j, 0]
        orden = pd.concat([orden, pd.Series(enlace[j, 1], index=i + 1)]).sort_index()
        orden.index = range(orden.shape[0])
    return orden.tolist()


def _varianza_cluster(cov: np.ndarray, indices: list[int]) -> float:
    """Varianza de un subcluster ponderado por el inverso de la varianza."""
    sub = cov[np.ix_(indices, indices)]
    inv_var = 1.0 / np.clip(np.diag(sub), 1e-18, None)
    w = inv_var / inv_var.sum()
    return float(w @ sub @ w)


def _bisecar(cov: np.ndarray, orden: list[int]) -> np.ndarray:
    """Asignación recursiva de pesos por bisección del árbol jerárquico."""
    w = np.ones(len(orden))
    pesos = pd.Series(w, index=orden)
    clusters = [orden]
    while clusters:
        clusters = [
            c[k:m]
            for c in clusters
            for k, m in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            izq, der = clusters[i], clusters[i + 1]
            v_izq, v_der = _varianza_cluster(cov, izq), _varianza_cluster(cov, der)
            alfa = 1.0 - v_izq / (v_izq + v_der) if (v_izq + v_der) > 0 else 0.5
            pesos[izq] *= alfa
            pesos[der] *= 1.0 - alfa
    return pesos.sort_index().to_numpy()


def optimizar_hrp(cov: np.ndarray, restr: RestriccionesPortafolio) -> np.ndarray:
    """
    HRP: clustering jerárquico sobre la distancia de correlación, bisección
    recursiva y proyección final al conjunto factible del perfil.

    No requiere invertir la matriz de covarianza, lo que lo hace estable con
    universos amplios y series cortas — el caso típico de los FICs locales.
    """
    n = cov.shape[0]
    if n < 2:
        return np.ones(n)
    std = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = np.clip(cov / np.outer(std, std), -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    enlace = linkage(squareform(dist, checks=False), method="single")
    orden = _quasi_diagonalizar(enlace)
    w_hrp = _bisecar(cov, orden)
    w_hrp = w_hrp / w_hrp.sum()
    return restr.proyectar(w_hrp)


# --------------------------------------------------------------------------- #
# Despachador y post-proceso
# --------------------------------------------------------------------------- #
def resolver(
    metodo: str, mu: pd.Series, cov: pd.DataFrame, rf: float,
    restr: RestriccionesPortafolio, n_arranques: int = 6, max_iter: int = 500,
    semilla: int = 42,
) -> pd.Series:
    """Ejecuta el método solicitado y devuelve los pesos indexados por fondo."""
    mu_v = mu.reindex(list(restr.activos)).to_numpy(dtype=float)
    cov_m = cov.reindex(index=list(restr.activos), columns=list(restr.activos)).to_numpy(dtype=float)

    if metodo == "MARKOWITZ_SHARPE":
        w = optimizar_max_sharpe(mu_v, cov_m, rf, restr, n_arranques, max_iter, semilla)
    elif metodo == "RISK_PARITY":
        w = optimizar_risk_parity(cov_m, restr, n_arranques, max_iter, semilla)
    elif metodo == "HRP":
        w = optimizar_hrp(cov_m, restr)
    elif metodo == "EQUIPONDERADO":
        w = optimizar_equiponderado(restr)
    else:
        raise ValueError(f"Método de optimización no soportado: {metodo}")
    return pd.Series(w, index=list(restr.activos), name=metodo)


def limpiar_pesos(
    pesos: pd.Series, restr: RestriccionesPortafolio, piso: float = 0.01
) -> pd.Series:
    """
    Higiene operativa: elimina posiciones testimoniales (por debajo del piso
    de suscripción razonable) y reproyecta para no romper las restricciones.
    """
    if piso <= 0:
        return pesos
    w = pesos.to_numpy(dtype=float).copy()
    w[w < piso] = 0.0
    if w.sum() <= 0:
        return pesos
    w = w / w.sum()
    if not restr.es_factible(w, tol=1e-4):
        w = restr.proyectar(w)
    return pd.Series(w, index=pesos.index, name=pesos.name)
