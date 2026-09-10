"""Utilidades transversales: logging, cache en disco y helpers de series."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

_LOG_FORMATO = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"


def configurar_logging(nivel: int = logging.INFO) -> None:
    """Configura el logging raíz una sola vez, en formato legible para consola."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=nivel, format=_LOG_FORMATO, datefmt="%H:%M:%S")
    root.setLevel(nivel)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sodapy").setLevel(logging.ERROR)


def get_logger(nombre: str) -> logging.Logger:
    return logging.getLogger(nombre)


def hash_clave(*partes: object) -> str:
    """Hash corto y estable para nombrar archivos de cache."""
    crudo = "|".join(str(p) for p in partes)
    return hashlib.sha1(crudo.encode("utf-8")).hexdigest()[:12]


#: La cache de descarga se guarda en binario comprimido, no en CSV: conserva los
#: tipos (fechas y flotantes) sin reinterpretarlos al releer, pesa una fracción
#: y no ensucia el proyecto con archivos que se puedan confundir con resultados.
EXT_CACHE = ".pkl.gz"


def guardar_cache(df: pd.DataFrame, ruta: Path) -> None:
    """Persiste un DataFrame en la cache local de descargas."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(ruta, compression="gzip")


def leer_cache(ruta: Path) -> pd.DataFrame | None:
    """Recupera un DataFrame de la cache; devuelve None si falta o está dañado."""
    if not ruta.exists():
        return None
    try:
        return pd.read_pickle(ruta, compression="gzip")
    except Exception:  # cache corrupta o de otra versión: se vuelve a descargar
        return None


def normalizar_texto(texto: object) -> str:
    """Mayúsculas sin tildes ni dobles espacios, para reglas de clasificación."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return ""
    s = str(texto).upper().strip()
    reemplazos = {
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "Ä": "A", "Ë": "E", "Ï": "I", "Ö": "O", "Ü": "U", "Ñ": "N",
    }
    for viejo, nuevo in reemplazos.items():
        s = s.replace(viejo, nuevo)
    return " ".join(s.split())


def a_numerico(serie: pd.Series) -> pd.Series:
    """Convierte a float tolerando strings, nulos y notación con comas."""
    if serie.dtype.kind in "if":
        return serie.astype(float)
    limpia = serie.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(limpia, errors="coerce")


def serie_estancada_max(precios: pd.Series) -> int:
    """Máxima racha de observaciones consecutivas sin variación de precio."""
    sin_cambio = precios.diff().fillna(0).eq(0)
    if not sin_cambio.any():
        return 0
    grupos = (~sin_cambio).cumsum()
    return int(sin_cambio.groupby(grupos).sum().max())


def alinear_precios(
    precios: pd.DataFrame, min_cobertura: float = 0.90, min_obs: int = 252
) -> pd.DataFrame:
    """
    Recorta el panel al rectángulo (fondos x fechas) más informativo.

    Cada fondo adicional obliga a acortar la historia común al arranque del más
    joven, y cada fecha adicional obliga a excluir fondos. En vez de fijar el
    compromiso a dedo, se elige la fecha de inicio que maximiza el producto
    fondos x fechas sujeto a un piso de observaciones: la estimación de la
    matriz de covarianza degrada mucho más rápido por falta de historia que por
    falta de activos.

    Dentro de la ventana elegida se descartan las fechas con cobertura
    insuficiente (festivos parciales), se interpolan huecos internos cortos y se
    eliminan los fondos que sigan incompletos.
    """
    if precios.empty:
        return precios

    panel = precios.sort_index()
    inicios = panel.apply(lambda s: s.first_valid_index()).dropna()
    if inicios.empty:
        return panel.dropna(axis=1, how="any")

    mejor: tuple[int, pd.Timestamp, list[str]] | None = None
    for fecha in sorted(set(inicios)):
        columnas = list(inicios[inicios <= fecha].index)
        n_fechas = int((panel.index >= fecha).sum())
        if n_fechas < min_obs or len(columnas) < 5:
            continue
        puntaje = len(columnas) * n_fechas
        if mejor is None or puntaje > mejor[0]:
            mejor = (puntaje, fecha, columnas)

    if mejor is None:  # ningún corte cumple el piso: se usa el panel completo
        recorte = panel
    else:
        _, fecha_inicio, columnas = mejor
        recorte = panel.loc[panel.index >= fecha_inicio, columnas]

    cobertura = recorte.notna().mean(axis=1)
    recorte = recorte.loc[cobertura >= min_cobertura]
    recorte = recorte.interpolate(method="time", limit=5, limit_area="inside")
    return recorte.dropna(axis=1, how="any")


def formato_cop(valor: float) -> str:
    """Formatea montos en COP con separadores de miles."""
    return f"${valor:,.0f}"
