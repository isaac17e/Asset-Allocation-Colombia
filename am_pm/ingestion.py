"""
Extracción de datos de FICs desde la API SODA del Portal de Datos Abiertos.

Estrategia en dos etapas para no descargar el dataset completo (≈3M de filas):

  1. *Screening*: se trae un único corte diario con todo el mercado y se
     selecciona el universo elegible por AUM, liquidez y tipo de vehículo.
  2. *Historia*: se descargan las series diarias sólo de los pares
     (código de negocio, tipo de participación) seleccionados, paginando.

Si la API no está disponible, `cargar_datos` detiene la corrida; sólo degrada
al generador sintético reproducible si `respaldo_sintetico` está activo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:  # sodapy es la vía oficial; urllib queda como respaldo del respaldo
    from sodapy import Socrata

    _SODAPY_DISPONIBLE = True
except ImportError:  # pragma: no cover
    Socrata = None  # type: ignore[assignment]
    _SODAPY_DISPONIBLE = False

from .config import COLUMNAS_HISTORIA, COLUMNAS_SNAPSHOT, DATASETS_FIC_ALTERNOS, ConfigDatos
from .utils import EXT_CACHE, a_numerico, get_logger, guardar_cache, hash_clave, leer_cache

log = get_logger("am_pm.ingestion")

_CAMPOS_NUMERICOS = (
    "valor_unidad_operaciones",
    "valor_fondo_cierre_dia_t",
    "numero_inversionistas",
    "rentabilidad_anual",
    "principal_compartimento",
)


@dataclass
class DatosFIC:
    """Contenedor de la extracción: metadatos del corte + panel histórico."""

    snapshot: pd.DataFrame
    historia: pd.DataFrame
    fecha_corte: pd.Timestamp
    origen: str  # "SODA" | "CACHE" | "SINTETICO"

    @property
    def es_sintetico(self) -> bool:
        return self.origen == "SINTETICO"


class ClienteFIC:
    """Envoltorio delgado sobre `sodapy.Socrata` con paginación y reintentos."""

    def __init__(self, cfg: ConfigDatos) -> None:
        if not _SODAPY_DISPONIBLE:
            raise RuntimeError("sodapy no está instalado: `pip install sodapy`")
        self.cfg = cfg
        self.dataset_id = cfg.dataset_id
        self._cliente = Socrata(cfg.dominio, cfg.app_token, timeout=cfg.timeout)

    # ------------------------------------------------------------------ #
    # Infraestructura de consulta
    # ------------------------------------------------------------------ #
    def _get(self, *, reintentos: int = 3, **kwargs) -> list[dict]:
        """Ejecuta una consulta SODA con backoff exponencial ante fallos."""
        ultimo_error: Exception | None = None
        for intento in range(reintentos):
            try:
                return self._cliente.get(self.dataset_id, **kwargs)
            except Exception as exc:  # noqa: BLE001 - se reintenta cualquier fallo de red/API
                ultimo_error = exc
                espera = 2.0 * (intento + 1)
                log.warning("Fallo consulta SODA (%s). Reintento en %.0fs", exc, espera)
                time.sleep(espera)
        raise ConnectionError(f"No fue posible consultar {self.dataset_id}: {ultimo_error}")

    def _get_paginado(self, *, where: str, select: str, etiqueta: str) -> pd.DataFrame:
        """Descarga todas las páginas de una consulta usando `$offset`."""
        filas: list[dict] = []
        offset = 0
        page = self.cfg.page_size
        while True:
            lote = self._get(
                select=select, where=where, limit=page, offset=offset, order=":id"
            )
            filas.extend(lote)
            log.debug("%s: +%d filas (acumulado %d)", etiqueta, len(lote), len(filas))
            if len(lote) < page:
                break
            offset += page
            time.sleep(0.2)  # cortesía con el rate-limit sin app_token
        return pd.DataFrame.from_records(filas)

    def verificar_dataset(self) -> str:
        """Valida el dataset configurado y cae a los alternos si responde 404."""
        candidatos = [self.dataset_id, *DATASETS_FIC_ALTERNOS]
        for ds in candidatos:
            try:
                self._cliente.get(ds, limit=1)
                if ds != self.dataset_id:
                    log.warning("Dataset %s no disponible; se usa %s", self.dataset_id, ds)
                self.dataset_id = ds
                return ds
            except Exception as exc:  # noqa: BLE001
                log.warning("Dataset %s descartado (%s)", ds, exc)
        raise ConnectionError("Ningún dataset de FICs respondió en datos.gov.co")

    # ------------------------------------------------------------------ #
    # Consultas de negocio
    # ------------------------------------------------------------------ #
    def ultima_fecha_corte(self) -> pd.Timestamp:
        respuesta = self._get(select="max(fecha_corte) as mx")
        return pd.to_datetime(respuesta[0]["mx"])

    def snapshot(self, fecha: pd.Timestamp) -> pd.DataFrame:
        """Corte transversal del mercado en una fecha dada."""
        subtipos = ",".join(f"'{s}'" for s in self.cfg.subtipos)
        where = (
            f"fecha_corte='{fecha:%Y-%m-%dT00:00:00.000}' "
            f"AND nombre_subtipo_patrimonio in ({subtipos})"
        )
        df = self._get_paginado(
            where=where, select=",".join(COLUMNAS_SNAPSHOT), etiqueta="snapshot"
        )
        return _tipificar(df)

    def historia(
        self, pares: list[tuple[str, str]], inicio: pd.Timestamp, fin: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Series diarias para los pares (código de negocio, tipo de participación)
        seleccionados, descargadas por lotes.
        """
        marcos: list[pd.DataFrame] = []
        chunk = max(1, self.cfg.chunk_codigos)
        lotes = [pares[i : i + chunk] for i in range(0, len(pares), chunk)]
        for i, lote in enumerate(lotes, start=1):
            filtro_pares = " OR ".join(
                f"(codigo_negocio='{cod}' AND tipo_participacion='{part}')"
                for cod, part in lote
            )
            where = (
                f"fecha_corte >= '{inicio:%Y-%m-%dT00:00:00.000}' "
                f"AND fecha_corte <= '{fin:%Y-%m-%dT00:00:00.000}' AND ({filtro_pares})"
            )
            df = self._get_paginado(
                where=where, select=",".join(COLUMNAS_HISTORIA), etiqueta=f"historia {i}/{len(lotes)}"
            )
            marcos.append(df)
            log.info("Historia lote %d/%d: %d filas", i, len(lotes), len(df))
        if not marcos:
            return pd.DataFrame(columns=list(COLUMNAS_HISTORIA))
        return _tipificar(pd.concat(marcos, ignore_index=True))


def _tipificar(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos: fechas a datetime, numéricos a float."""
    if df.empty:
        return df
    out = df.copy()
    if "fecha_corte" in out.columns:
        out["fecha_corte"] = pd.to_datetime(out["fecha_corte"], errors="coerce")
    for col in _CAMPOS_NUMERICOS:
        if col in out.columns:
            out[col] = a_numerico(out[col])
    for col in ("codigo_negocio", "tipo_participacion", "codigo_entidad"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    return out


def id_fondo(codigo_negocio: str, tipo_participacion: str) -> str:
    """Identificador único de la clase de participación invertible."""
    return f"{codigo_negocio}-{tipo_participacion}"


# --------------------------------------------------------------------------- #
# Generador sintético (modo offline / pruebas deterministas)
# --------------------------------------------------------------------------- #
_PLANTILLA_SINTETICA: tuple[tuple[str, str, float, float, float, int], ...] = (
    # (prefijo, categoría implícita, retorno, vol, beta_mercado, nº de fondos)
    ("FIC LIQUIDEZ", "RF_CORTO", 0.092, 0.004, 0.00, 8),
    ("FIC RENTA FIJA LP", "RF_MEDIANO_LARGO", 0.105, 0.045, 0.10, 7),
    ("FIC BALANCEADO", "MIXTO", 0.115, 0.085, 0.45, 5),
    ("FIC ACCIONES COLOMBIA", "RV_LOCAL", 0.145, 0.180, 1.00, 5),
    ("FIC ACCIONES GLOBALES", "RV_INTERNACIONAL", 0.130, 0.150, 0.35, 5),
)
_GESTORAS_SINTETICAS = (
    "Fiduciaria Alfa", "Fiduciaria Beta", "Comisionista Gamma",
    "Fiduciaria Delta", "Asset Management Epsilon",
)


def generar_datos_sinteticos(cfg: ConfigDatos, semilla: int = 42) -> DatosFIC:
    """
    Panel sintético con estructura factorial (factor de mercado local + factor
    global) y las mismas columnas del dataset real. Permite ejecutar y validar
    el pipeline completo sin conexión.
    """
    rng = np.random.default_rng(semilla)
    fin = pd.Timestamp(cfg.fecha_corte) if cfg.fecha_corte else pd.Timestamp.today().normalize()
    inicio = fin - pd.Timedelta(days=int(cfg.lookback_anios * 365))
    fechas = pd.date_range(inicio, fin, freq="D")
    n = len(fechas)
    ppa = 365.0

    f_local = rng.normal(0, 0.18 / np.sqrt(ppa), n)
    f_global = rng.normal(0, 0.14 / np.sqrt(ppa), n)

    filas_hist: list[pd.DataFrame] = []
    filas_snap: list[dict] = []
    contador = 0
    for prefijo, _cat, mu, vol, beta, cuantos in _PLANTILLA_SINTETICA:
        for k in range(cuantos):
            contador += 1
            codigo = f"9{contador:04d}"
            participacion = "800"
            gestora = _GESTORAS_SINTETICAS[contador % len(_GESTORAS_SINTETICAS)]
            mu_i = mu + rng.normal(0, 0.012)
            vol_idio = max(vol * 0.45, 1e-4)
            carga_global = beta if "GLOBAL" in prefijo else beta * 0.25
            ret = (
                (1 + mu_i) ** (1 / ppa) - 1
                + beta * f_local * (0.0 if "GLOBAL" in prefijo else 1.0)
                + carga_global * f_global
                + rng.normal(0, vol_idio / np.sqrt(ppa), n)
            )
            precio = 1000.0 * np.cumprod(1 + ret)
            aum = float(rng.uniform(cfg.aum_minimo_cop * 1.5, cfg.aum_minimo_cop * 60))
            filas_hist.append(
                pd.DataFrame(
                    {
                        "fecha_corte": fechas,
                        "codigo_negocio": codigo,
                        "tipo_participacion": participacion,
                        "valor_unidad_operaciones": precio,
                        "valor_fondo_cierre_dia_t": aum * (precio / precio[0]),
                    }
                )
            )
            filas_snap.append(
                {
                    "fecha_corte": fin,
                    "codigo_entidad": str(100 + contador % len(_GESTORAS_SINTETICAS)),
                    "nombre_entidad": gestora,
                    "nombre_tipo_entidad": "SOCIEDADES FIDUCIARIAS",
                    "codigo_negocio": codigo,
                    "nombre_patrimonio": f"{prefijo} {k + 1}",
                    "nombre_subtipo_patrimonio": "FIC DE TIPO GENERAL",
                    "tipo_participacion": participacion,
                    "principal_compartimento": 1.0,
                    "valor_unidad_operaciones": float(precio[-1]),
                    "valor_fondo_cierre_dia_t": float(aum * precio[-1] / precio[0]),
                    "numero_inversionistas": int(rng.integers(200, 20000)),
                    "rentabilidad_anual": mu_i * 100,
                }
            )
    return DatosFIC(
        snapshot=pd.DataFrame(filas_snap),
        historia=pd.concat(filas_hist, ignore_index=True),
        fecha_corte=fin,
        origen="SINTETICO",
    )


# --------------------------------------------------------------------------- #
# Orquestador de extracción
# --------------------------------------------------------------------------- #
def cargar_datos(cfg: ConfigDatos, *, offline: bool = False, semilla: int = 42) -> DatosFIC:
    """
    Punto de entrada de la capa de datos.

    Devuelve el corte transversal y el panel histórico de los fondos elegibles,
    resolviendo cache en disco. Si la API falla, la corrida se detiene salvo
    que `cfg.respaldo_sintetico` autorice continuar con datos sintéticos.
    """
    if offline:
        log.warning("Modo offline: se generan datos sintéticos reproducibles.")
        return generar_datos_sinteticos(cfg, semilla)

    try:
        cliente = ClienteFIC(cfg)
        cliente.verificar_dataset()
        fecha = pd.to_datetime(cfg.fecha_corte) if cfg.fecha_corte else cliente.ultima_fecha_corte()
        log.info("Fecha de corte del análisis: %s", fecha.date())

        clave = hash_clave(cliente.dataset_id, fecha.date(), cfg.subtipos)
        ruta_snap = Path(cfg.cache_dir) / f"snapshot_{clave}{EXT_CACHE}"
        snap = leer_cache(ruta_snap) if cfg.usar_cache else None
        if snap is None:
            snap = cliente.snapshot(fecha)
            if cfg.usar_cache:
                guardar_cache(snap, ruta_snap)
        snap = _tipificar(snap)
        log.info("Snapshot: %d clases de participación en %d fondos",
                 len(snap), snap["codigo_negocio"].nunique())

        pares = _preseleccionar_pares(snap, cfg)
        log.info("Preselección para descarga histórica: %d clases", len(pares))

        inicio = fecha - pd.Timedelta(days=int(cfg.lookback_anios * 365))
        clave_h = hash_clave(cliente.dataset_id, fecha.date(), inicio.date(), tuple(pares))
        ruta_hist = Path(cfg.cache_dir) / f"historia_{clave_h}{EXT_CACHE}"
        hist = leer_cache(ruta_hist) if cfg.usar_cache else None
        origen = "CACHE" if hist is not None else "SODA"
        if hist is None:
            hist = cliente.historia(pares, inicio, fecha)
            if cfg.usar_cache:
                guardar_cache(hist, ruta_hist)
        hist = _tipificar(hist)
        log.info("Historia: %d filas desde %s (%s)", len(hist), inicio.date(), origen)
        return DatosFIC(snapshot=snap, historia=hist, fecha_corte=fecha, origen=origen)

    except Exception as exc:  # noqa: BLE001
        log.error("Extracción desde datos.gov.co fallida (%s).", exc)
        if not cfg.respaldo_sintetico:
            raise ConnectionError(
                "No se pudieron obtener datos de mercado. Reintente más tarde, use "
                "--app-token, o --respaldo-sintetico para continuar con datos simulados."
            ) from exc
        log.warning("Se continúa con datos sintéticos: los resultados NO son de mercado.")
        return generar_datos_sinteticos(cfg, semilla)


def _preseleccionar_pares(snapshot: pd.DataFrame, cfg: ConfigDatos) -> list[tuple[str, str]]:
    """
    Elige la clase de participación representativa de cada fondo (la de mayor
    AUM) y recorta a los `top_n` fondos por tamaño, ampliando el margen para
    absorber bajas por filtros de calidad posteriores.
    """
    df = snapshot.dropna(subset=["valor_unidad_operaciones", "valor_fondo_cierre_dia_t"]).copy()
    df = df[df["valor_unidad_operaciones"] > 0]
    if "numero_inversionistas" in df.columns:
        df = df[df["numero_inversionistas"].fillna(0) >= cfg.min_inversionistas]

    aum_fondo = df.groupby("codigo_negocio")["valor_fondo_cierre_dia_t"].sum()
    elegibles = aum_fondo[aum_fondo >= cfg.aum_minimo_cop]
    # Margen de 60% sobre top_n: algunos fondos se caerán por historia insuficiente.
    limite = int(cfg.top_n_fondos * 1.6)
    top = elegibles.sort_values(ascending=False).head(limite).index

    df = df[df["codigo_negocio"].isin(top)]
    idx = df.groupby("codigo_negocio")["valor_fondo_cierre_dia_t"].idxmax()
    seleccion = df.loc[idx, ["codigo_negocio", "tipo_participacion"]]
    return list(seleccion.itertuples(index=False, name=None))
