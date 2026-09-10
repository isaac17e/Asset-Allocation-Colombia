"""
Construcción del universo curado de fondos invertibles.

El reto práctico del mercado colombiano es que el nombre comercial de la
mayoría de FICs no revela su política de inversión ("Fiducuenta", "Sumar",
"Fidugob" concentran una porción mayoritaria del AUM). Por eso la
clasificación en clases de activo es **híbrida**:

  * Capa 1 — reglas: taxonomía por palabras clave y subtipo de patrimonio,
    con precedencia explícita (renta fija antes que bursátil, Colombia antes
    que global) para evitar falsos positivos.
  * Capa 2 — inferencia cuantitativa: para los fondos sin regla aplicable se
    estima la clase a partir del riesgo realizado (volatilidad anualizada) y
    de las betas contra un proxy de renta variable local y uno de exposición
    internacional/FX.
  * Capa 3 — conciliación: si la regla contradice al riesgo observado (p. ej.
    un fondo "efectivo" en dólares con 12% de volatilidad), manda el dato y la
    discrepancia queda registrada para revisión del comité de inversiones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CATEGORIAS, ConfigDatos, ConfigOptimizacion
from .ingestion import DatosFIC, id_fondo
from .metrics import (
    beta_contra,
    max_drawdown,
    periodos_por_anio,
    ratio_calmar,
    ratio_sharpe,
    ratio_sortino,
    retorno_anualizado,
    retornos_simples,
    volatilidad_anualizada,
)
from .utils import alinear_precios, get_logger, normalizar_texto, serie_estancada_max

log = get_logger("am_pm.universe")

# --------------------------------------------------------------------------- #
# Patrones de la capa de reglas (sobre nombre normalizado, sin tildes)
# --------------------------------------------------------------------------- #
P_RF_EXPLICITA = re.compile(
    r"\b(TES|GBI|RENTA FIJA|BONO|DEUDA|CREDITO PRIVADO|FACTORING|LIBRANZ|"
    r"TITULARIZ|HIPOTECAR|CARTERA DE CREDITO)\b"
)
P_ACCIONES = re.compile(
    r"\b(ACCION|ACCIONES|RENTA VARIABLE|EQUITY|COLCAP|BURSATIL|INDICE|"
    r"MSCI|S&P|NASDAQ|SELECT)\b"
)
P_LOCAL = re.compile(r"\b(COLOMBIA|COLCAP|LOCAL|NACIONAL|PAIS)\b")
P_INTERNACIONAL = re.compile(
    r"\b(GLOBAL\w*|INTERNACION\w*|MUNDIAL\w*|WORLD|ACWI|EEUU|USA|ESTADOS UNIDOS|"
    r"EUROPA|ASIA|EMERGENT\w*|LATAM|NASDAQ|DOLAR\w*|USD)\b"
)
P_MIXTO = re.compile(
    r"\b(BALANCEAD\w*|MIXTO|MULTIACTIVO|MULTIESTRATEGIA|DIVERSIFICAD\w*|"
    r"ASIGNACION|ESTRATEGIC\w*|PERFIL|ALTERNATIVO)\b"
)
P_MERCADO_MONETARIO = re.compile(
    r"\b(ALTA LIQUIDEZ|LIQUIDEZ|EFECTIVO|VISTA|MERCADO MONETARIO|MONEY MARKET|"
    r"CASH|TESORERIA|DISPONIBLE|AHORRO|CORTO PLAZO|1525)\b"
)
P_PLAZO_LARGO = re.compile(r"\b(PACTO DE PERMANENCIA|MEDIANO PLAZO|LARGO PLAZO|\d{2,3} DIAS)\b")

#: Umbrales de la capa cuantitativa (volatilidad anualizada en COP).
UMBRAL_VOL_RF_CORTO = 0.015
UMBRAL_VOL_RF_LARGO = 0.050
UMBRAL_VOL_MIXTO = 0.100
UMBRAL_BETA_RV = 0.55
UMBRAL_BETA_INTL = 0.45
UMBRAL_R2 = 0.30


@dataclass
class UniversoCurado:
    """Universo invertible: fichas de fondos + panel de precios alineado."""

    fondos: pd.DataFrame           # una fila por fondo, indexada por fondo_id
    precios: pd.DataFrame          # panel de valor de unidad alineado
    retornos: pd.DataFrame         # retornos simples del panel
    rf: float                      # tasa libre de riesgo dinámica anualizada
    fecha_corte: pd.Timestamp
    origen: str

    @property
    def ids(self) -> list[str]:
        return list(self.fondos.index)

    def por_categoria(self) -> dict[str, list[str]]:
        return {
            cat: list(grupo.index)
            for cat, grupo in self.fondos.groupby("categoria", observed=True)
        }

    def mapa_categoria(self) -> pd.Series:
        return self.fondos["categoria"]

    def mapa_gestor(self) -> pd.Series:
        return self.fondos["codigo_entidad"]


# --------------------------------------------------------------------------- #
# Panel de precios
# --------------------------------------------------------------------------- #
def construir_panel(historia: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivotea la historia larga a paneles ancho de valor de unidad y de AUM."""
    if historia.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = historia.dropna(subset=["fecha_corte", "valor_unidad_operaciones"]).copy()
    df["fondo_id"] = [
        id_fondo(c, t) for c, t in zip(df["codigo_negocio"], df["tipo_participacion"])
    ]
    df = df.sort_values("fecha_corte").drop_duplicates(["fondo_id", "fecha_corte"], keep="last")
    precios = df.pivot(index="fecha_corte", columns="fondo_id", values="valor_unidad_operaciones")
    aum = df.pivot(index="fecha_corte", columns="fondo_id", values="valor_fondo_cierre_dia_t")
    return precios.sort_index(), aum.sort_index()


def filtrar_series(precios: pd.DataFrame, cfg: ConfigDatos) -> pd.DataFrame:
    """
    Descarta series no aptas para optimización: historia corta, precios no
    positivos o valores de unidad congelados durante períodos prolongados
    (típico de vehículos con valoración no diaria o en liquidación).
    """
    if precios.empty:
        return precios
    validas: list[str] = []
    descartes: dict[str, int] = {"historia": 0, "precio": 0, "estancada": 0}
    for col in precios.columns:
        serie = precios[col].dropna()
        if len(serie) < cfg.min_obs_historia:
            descartes["historia"] += 1
            continue
        if (serie <= 0).any():
            descartes["precio"] += 1
            continue
        if serie_estancada_max(serie) > cfg.max_dias_estancados:
            descartes["estancada"] += 1
            continue
        validas.append(col)
    log.info(
        "Filtro de series: %d válidas | descartes -> historia=%d, precio=%d, estancada=%d",
        len(validas), descartes["historia"], descartes["precio"], descartes["estancada"],
    )
    return precios[validas]


# --------------------------------------------------------------------------- #
# Capa 1: reglas
# --------------------------------------------------------------------------- #
def clasificar_por_reglas(nombre: str, subtipo: str) -> str | None:
    """
    Asigna clase de activo por nomenclatura. Devuelve None si el nombre no es
    concluyente, delegando la decisión a la capa cuantitativa.

    La precedencia importa: "Fondo Bursátil Global X TES Colombia COP GBI" es
    renta fija pese a contener 'bursátil' y 'select'.
    """
    n = normalizar_texto(nombre)
    s = normalizar_texto(subtipo)

    if P_RF_EXPLICITA.search(n):
        return "RF_CORTO" if P_MERCADO_MONETARIO.search(n) else "RF_MEDIANO_LARGO"

    if P_ACCIONES.search(n):
        if P_LOCAL.search(n):
            return "RV_LOCAL"
        if P_INTERNACIONAL.search(n):
            return "RV_INTERNACIONAL"
        return "RV_LOCAL"  # por defecto, la renta variable local domina la oferta

    if P_MIXTO.search(n):
        return "MIXTO"

    if P_MERCADO_MONETARIO.search(n) and not P_INTERNACIONAL.search(n):
        return "RF_CORTO"

    if P_PLAZO_LARGO.search(n):
        return "RF_MEDIANO_LARGO"

    if "MERCADO MONETARIO" in s:
        return "RF_CORTO"

    return None


# --------------------------------------------------------------------------- #
# Capa 2: inferencia cuantitativa
# --------------------------------------------------------------------------- #
def seleccionar_proxies(
    retornos: pd.DataFrame, metadatos: pd.DataFrame
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Elige los factores de referencia del clasificador cuantitativo:
    un proxy de renta variable local (idealmente el ETF de COLCAP) y uno de
    exposición internacional/FX (fondo en dólares o de acciones globales).

    Si no hay candidatos por nombre, se toma el fondo de mayor volatilidad
    dentro de cada bloque como aproximación.
    """
    nombres = metadatos["nombre_patrimonio"].map(normalizar_texto)
    vol = retornos.std() * np.sqrt(periodos_por_anio(retornos.index))

    def _elegir(patron: re.Pattern[str], excluir: re.Pattern[str] | None = None) -> pd.Series | None:
        candidatos = [
            fid for fid, nom in nombres.items()
            if fid in retornos.columns
            and patron.search(nom)
            and not (excluir and excluir.search(nom))
        ]
        if not candidatos:
            return None
        elegido = vol[candidatos].idxmax()
        log.info("Proxy seleccionado: %s -> %s", patron.pattern[:28], nombres.get(elegido))
        return retornos[elegido]

    proxy_local = _elegir(re.compile(r"\b(COLCAP|ACCIONES COLOMBIA|RENTA VARIABLE COLOMBIA)\b"))
    if proxy_local is None:
        proxy_local = _elegir(P_ACCIONES, excluir=P_INTERNACIONAL)
    proxy_intl = _elegir(re.compile(r"\b(DOLAR\w*|USD|ACCIONES GLOBAL\w*|GLOBAL\w*)\b"), excluir=P_LOCAL)

    if proxy_local is None:
        log.warning("Sin proxy de renta variable local: la capa cuantitativa usará solo volatilidad.")
    if proxy_intl is None:
        log.warning("Sin proxy internacional/FX: no se podrá separar RV_INTERNACIONAL por betas.")
    return proxy_local, proxy_intl


def clasificar_por_riesgo(
    vol_anual: float, beta_local: float, r2_local: float, beta_intl: float, r2_intl: float
) -> str:
    """Árbol de decisión sobre riesgo realizado y sensibilidad a factores."""
    beta_local = beta_local if np.isfinite(beta_local) else 0.0
    beta_intl = beta_intl if np.isfinite(beta_intl) else 0.0
    r2_local = r2_local if np.isfinite(r2_local) else 0.0
    r2_intl = r2_intl if np.isfinite(r2_intl) else 0.0

    if beta_local >= UMBRAL_BETA_RV and r2_local >= UMBRAL_R2:
        return "RV_LOCAL"
    if beta_intl >= UMBRAL_BETA_INTL and r2_intl >= UMBRAL_R2:
        return "RV_INTERNACIONAL"
    if not np.isfinite(vol_anual):
        return "MIXTO"
    if vol_anual <= UMBRAL_VOL_RF_CORTO:
        return "RF_CORTO"
    if vol_anual <= UMBRAL_VOL_RF_LARGO:
        return "RF_MEDIANO_LARGO"
    if vol_anual <= UMBRAL_VOL_MIXTO:
        return "MIXTO"
    return "RV_INTERNACIONAL" if beta_intl > beta_local else "RV_LOCAL"


def _conciliar(regla: str | None, cuantitativa: str, vol_anual: float) -> tuple[str, str, str]:
    """
    Resuelve el conflicto entre nomenclatura y riesgo observado.

    Devuelve (categoría final, método, observación). La regla se respeta salvo
    cuando implica un riesgo incompatible con el dato: un fondo etiquetado como
    liquidez no puede exhibir volatilidad de renta variable.
    """
    if regla is None:
        return cuantitativa, "MODELO_RIESGO", "Nombre no concluyente"
    if regla == cuantitativa:
        return regla, "REGLA", ""
    if regla == "RF_CORTO" and vol_anual > UMBRAL_VOL_RF_LARGO:
        return cuantitativa, "MODELO_RIESGO", f"Regla decía RF_CORTO con vol {vol_anual:.1%}"
    if regla in ("RF_CORTO", "RF_MEDIANO_LARGO") and vol_anual > UMBRAL_VOL_MIXTO:
        return cuantitativa, "MODELO_RIESGO", f"Regla decía {regla} con vol {vol_anual:.1%}"
    if regla in ("RV_LOCAL", "RV_INTERNACIONAL") and vol_anual < UMBRAL_VOL_RF_LARGO:
        return cuantitativa, "MODELO_RIESGO", f"Regla decía {regla} con vol {vol_anual:.1%}"
    if regla == "MIXTO" and vol_anual < UMBRAL_VOL_RF_CORTO:
        # Un balanceado sin riesgo de mercado es, para efectos de presupuesto
        # de riesgo, un fondo de liquidez con otro nombre comercial.
        return cuantitativa, "MODELO_RIESGO", f"Regla decía MIXTO con vol {vol_anual:.2%}"
    return regla, "REGLA", f"Modelo sugería {cuantitativa}"


# --------------------------------------------------------------------------- #
# Tasa libre de riesgo dinámica
# --------------------------------------------------------------------------- #
def calcular_rf_dinamica(
    precios: pd.DataFrame,
    categorias: pd.Series,
    aum: pd.Series | None = None,
    ventana: int = 252,
    fallback: float = 0.085,
) -> float:
    """
    Proxy de la tasa libre de riesgo: rendimiento anualizado compuesto de la
    categoría RF_CORTO, ponderado por AUM y estimado sobre la ventana reciente.

    Es la referencia natural para un inversionista colombiano: el costo de
    oportunidad real es el fondo de liquidez, no un bono teórico.
    """
    ids = [c for c in precios.columns if categorias.get(c) == "RF_CORTO"]
    if not ids:
        log.warning("Sin fondos RF_CORTO; r_f = fallback %.2f%%", fallback * 100)
        return fallback

    ventana_efectiva = min(int(ventana), len(precios))
    tramo = precios[ids].tail(ventana_efectiva)
    cagr = tramo.apply(retorno_anualizado).dropna()
    cagr = cagr[(cagr > -0.5) & (cagr < 1.0)]
    if cagr.empty:
        return fallback

    if aum is not None:
        pesos = aum.reindex(cagr.index).fillna(0.0).clip(lower=0.0)
        if pesos.sum() > 0:
            rf = float((cagr * pesos).sum() / pesos.sum())
            log.info("r_f dinámica (RF_CORTO ponderada por AUM, %d fondos): %.2f%%",
                     len(cagr), rf * 100)
            return rf
    rf = float(cagr.median())
    log.info("r_f dinámica (mediana RF_CORTO, %d fondos): %.2f%%", len(cagr), rf * 100)
    return rf


# --------------------------------------------------------------------------- #
# Orquestador
# --------------------------------------------------------------------------- #
def construir_universo(
    datos: DatosFIC, cfg_datos: ConfigDatos, cfg_opt: ConfigOptimizacion
) -> UniversoCurado:
    """
    Genera el `universo_curado`: fichas de fondos con clase de activo, métricas
    de riesgo/retorno y panel de precios listo para optimizar.
    """
    precios_crudos, aum_panel = construir_panel(datos.historia)
    if precios_crudos.empty:
        raise ValueError("La historia descargada está vacía: no hay universo que construir.")

    precios = filtrar_series(precios_crudos, cfg_datos)
    precios = alinear_precios(precios, min_obs=cfg_datos.min_obs_panel)
    if precios.shape[1] < 5:
        raise ValueError(f"Universo insuficiente tras filtros: {precios.shape[1]} fondos.")
    log.info("Panel alineado: %d fondos x %d fechas (%s a %s)", precios.shape[1],
             precios.shape[0], precios.index[0].date(), precios.index[-1].date())

    metadatos = _metadatos_fondos(datos.snapshot, precios.columns)
    retornos = retornos_simples(precios)
    ppa = periodos_por_anio(precios.index)

    # --- estadísticos base y factores -------------------------------------
    vol = retornos.apply(lambda s: volatilidad_anualizada(s, ppa))
    ret_anual = precios.apply(retorno_anualizado)
    proxy_local, proxy_intl = seleccionar_proxies(retornos, metadatos)

    filas: list[dict] = []
    for fid in precios.columns:
        meta = metadatos.loc[fid]
        b_loc, r2_loc = beta_contra(retornos[fid], proxy_local) if proxy_local is not None else (np.nan, np.nan)
        b_int, r2_int = beta_contra(retornos[fid], proxy_intl) if proxy_intl is not None else (np.nan, np.nan)
        regla = clasificar_por_reglas(meta["nombre_patrimonio"], meta["nombre_subtipo_patrimonio"])
        modelo = clasificar_por_riesgo(vol[fid], b_loc, r2_loc, b_int, r2_int)
        categoria, metodo, nota = _conciliar(regla, modelo, vol[fid])
        filas.append(
            {
                "fondo_id": fid,
                "nombre_patrimonio": meta["nombre_patrimonio"],
                "nombre_entidad": meta["nombre_entidad"],
                "codigo_entidad": meta["codigo_entidad"],
                "nombre_subtipo_patrimonio": meta["nombre_subtipo_patrimonio"],
                "categoria": categoria,
                "categoria_regla": regla or "",
                "categoria_modelo": modelo,
                "metodo_clasificacion": metodo,
                "observacion": nota,
                "aum_cop": meta["valor_fondo_cierre_dia_t"],
                "beta_rv_local": b_loc,
                "r2_rv_local": r2_loc,
                "beta_internacional": b_int,
                "r2_internacional": r2_int,
                "retorno_anual": ret_anual[fid],
                "vol_anual": vol[fid],
                "max_drawdown": max_drawdown(precios[fid]),
                "n_obs": int(precios[fid].notna().sum()),
            }
        )

    fondos = pd.DataFrame(filas).set_index("fondo_id")

    # --- r_f dinámica y métricas ajustadas por riesgo ----------------------
    aum_ultimo = aum_panel.reindex(columns=precios.columns).ffill().iloc[-1] if not aum_panel.empty else None
    rf = calcular_rf_dinamica(
        precios, fondos["categoria"], aum_ultimo, cfg_opt.ventana_rf, cfg_opt.rf_fallback
    )
    fondos["sharpe"] = [
        ratio_sharpe(fondos.loc[f, "retorno_anual"], fondos.loc[f, "vol_anual"], rf)
        for f in fondos.index
    ]
    fondos["sortino"] = [
        ratio_sortino(fondos.loc[f, "retorno_anual"], retornos[f], rf, ppa) for f in fondos.index
    ]
    fondos["calmar"] = [
        ratio_calmar(fondos.loc[f, "retorno_anual"], fondos.loc[f, "max_drawdown"])
        for f in fondos.index
    ]

    fondos = _recortar_top_n(fondos, cfg_datos.top_n_fondos)
    precios = precios[fondos.index]
    retornos = retornos[fondos.index]

    _reportar_composicion(fondos)
    return UniversoCurado(
        fondos=fondos,
        precios=precios,
        retornos=retornos,
        rf=rf,
        fecha_corte=datos.fecha_corte,
        origen=datos.origen,
    )


def _metadatos_fondos(snapshot: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    """Indexa el snapshot por fondo_id y lo alinea con el panel de precios."""
    snap = snapshot.copy()
    snap["fondo_id"] = [
        id_fondo(c, t) for c, t in zip(snap["codigo_negocio"], snap["tipo_participacion"])
    ]
    snap = snap.drop_duplicates("fondo_id").set_index("fondo_id")
    columnas = [
        "nombre_patrimonio", "nombre_entidad", "codigo_entidad",
        "nombre_subtipo_patrimonio", "valor_fondo_cierre_dia_t",
    ]
    for col in columnas:
        if col not in snap.columns:
            snap[col] = np.nan
    meta = snap.reindex(ids)[columnas]
    meta["nombre_patrimonio"] = meta["nombre_patrimonio"].fillna("SIN NOMBRE")
    meta["nombre_entidad"] = meta["nombre_entidad"].fillna("SIN GESTOR")
    meta["codigo_entidad"] = meta["codigo_entidad"].fillna("NA").astype(str)
    meta["nombre_subtipo_patrimonio"] = meta["nombre_subtipo_patrimonio"].fillna("")
    meta["valor_fondo_cierre_dia_t"] = meta["valor_fondo_cierre_dia_t"].fillna(0.0)
    return meta


def _recortar_top_n(fondos: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Recorta el universo preservando representación de todas las categorías:
    primero se garantiza un mínimo por clase de activo (para que las
    restricciones por perfil sean factibles) y luego se completa por AUM.
    """
    if len(fondos) <= top_n:
        return fondos
    minimo_por_categoria = 3
    elegidos: list[str] = []
    for _cat, grupo in fondos.groupby("categoria", observed=True):
        elegidos.extend(grupo.sort_values("aum_cop", ascending=False).head(minimo_por_categoria).index)
    restantes = fondos.drop(index=elegidos).sort_values("aum_cop", ascending=False)
    faltan = max(0, top_n - len(elegidos))
    elegidos.extend(restantes.head(faltan).index)
    return fondos.loc[fondos.index.isin(elegidos)]


def _reportar_composicion(fondos: pd.DataFrame) -> None:
    """Traza la composición del universo curado por categoría."""
    resumen = fondos.groupby("categoria", observed=True).agg(
        fondos=("aum_cop", "size"),
        aum=("aum_cop", "sum"),
        vol_media=("vol_anual", "mean"),
    )
    for cat in CATEGORIAS:
        if cat not in resumen.index:
            log.warning("Categoría sin representantes en el universo: %s", cat)
    partes = [
        f"{cat}: {int(f.fondos)} fondos / {f.aum / 1e12:.1f} bn / vol {f.vol_media:.1%}"
        for cat, f in resumen.iterrows()
    ]
    log.info("Universo curado — %s", " | ".join(partes))
    por_modelo = int((fondos["metodo_clasificacion"] == "MODELO_RIESGO").sum())
    log.info("Clasificación: %d por regla, %d por modelo de riesgo",
             len(fondos) - por_modelo, por_modelo)
