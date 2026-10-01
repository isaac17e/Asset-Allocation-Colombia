"""Configuración centralizada del AM-PM (constantes, parámetros y rutas)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------- #
# Fuente de datos: Portal de Datos Abiertos de Colombia (SODA API)
# --------------------------------------------------------------------------- #
DOMINIO_SODA: str = "www.datos.gov.co"

#: Dataset vigente de la Superintendencia Financiera con la información diaria
#: de Fondos de Inversión Colectiva (valor de unidad, AUM, rentabilidades).
DATASET_FIC: str = "qhpu-8ixx"

#: Datasets alternos/históricos. Se prueban en orden si el principal falla.
DATASETS_FIC_ALTERNOS: tuple[str, ...] = ("m64f-9559", "kfgv-akjs")

#: Columnas del dataset que consume el pipeline (proyección server-side).
COLUMNAS_SNAPSHOT: tuple[str, ...] = (
    "fecha_corte",
    "codigo_entidad",
    "nombre_entidad",
    "nombre_tipo_entidad",
    "codigo_negocio",
    "nombre_patrimonio",
    "nombre_subtipo_patrimonio",
    "tipo_participacion",
    "principal_compartimento",
    "valor_unidad_operaciones",
    "valor_fondo_cierre_dia_t",
    "numero_inversionistas",
    "rentabilidad_anual",
)
COLUMNAS_HISTORIA: tuple[str, ...] = (
    "fecha_corte",
    "codigo_negocio",
    "tipo_participacion",
    "valor_unidad_operaciones",
    "valor_fondo_cierre_dia_t",
)

# --------------------------------------------------------------------------- #
# Taxonomía de clases de activo
# --------------------------------------------------------------------------- #
CATEGORIAS: tuple[str, ...] = (
    "RF_CORTO",
    "RF_MEDIANO_LARGO",
    "MIXTO",
    "RV_LOCAL",
    "RV_INTERNACIONAL",
)
CATEGORIAS_RV: tuple[str, ...] = ("RV_LOCAL", "RV_INTERNACIONAL")
CATEGORIAS_RF: tuple[str, ...] = ("RF_CORTO", "RF_MEDIANO_LARGO")

#: Subtipos de patrimonio con liquidez y valoración diaria aptos para el
#: universo invertible. Se excluyen FCP e inmobiliarias (iliquidez, valoración
#: no diaria, compromisos de capital).
SUBTIPOS_INVERTIBLES: tuple[str, ...] = (
    "FIC DE TIPO GENERAL",
    "FIC DE MERCADO MONETARIO",
    "FIC BURSATILES",
)

PERFILES: tuple[str, ...] = ("CONSERVADOR", "MODERADO", "AGRESIVO")
METODOS: tuple[str, ...] = ("MARKOWITZ_SHARPE", "RISK_PARITY", "HRP", "EQUIPONDERADO")

DIAS_CALENDARIO_ANIO: float = 365.0
DIAS_HABILES_ANIO: float = 252.0


# --------------------------------------------------------------------------- #
# Parámetros por bloque del pipeline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConfigDatos:
    """Parámetros de extracción y filtrado del universo."""

    dominio: str = DOMINIO_SODA
    dataset_id: str = DATASET_FIC
    app_token: str | None = None
    timeout: int = 120
    #: Años de historia diaria a descargar.
    lookback_anios: float = 4.0
    #: Fecha de corte del análisis (None = último corte publicado).
    fecha_corte: str | None = None
    #: Máximo de fondos en el universo curado (ordenados por AUM).
    top_n_fondos: int = 70
    #: AUM mínimo por fondo, en COP (filtro de capacidad/liquidez).
    aum_minimo_cop: float = 30_000_000_000.0
    #: Mínimo de observaciones de valor de unidad exigidas.
    min_obs_historia: int = 500
    #: Piso de observaciones del panel alineado (condiciona el ancho de la ventana).
    min_obs_panel: int = 500
    #: Máximo de días consecutivos con valor de unidad congelado (serie viciada).
    max_dias_estancados: int = 20
    #: Mínimo de inversionistas (descarta vehículos a la medida de un cliente).
    min_inversionistas: int = 25
    subtipos: tuple[str, ...] = SUBTIPOS_INVERTIBLES
    usar_cache: bool = True
    #: Si la API falla, continuar con datos sintéticos en vez de detener la
    #: corrida. Apagado por defecto: un portafolio sobre datos simulados que
    #: llega al comité por un timeout es peor que una corrida fallida.
    respaldo_sintetico: bool = False
    cache_dir: Path = Path("data/cache")
    #: Tamaño de lote de códigos de negocio por request y paginación SODA.
    chunk_codigos: int = 25
    page_size: int = 50_000


@dataclass(frozen=True)
class ConfigOptimizacion:
    """Parámetros del motor de optimización."""

    metodos: tuple[str, ...] = METODOS
    #: Intensidad de shrinkage de la matriz de covarianza (None = Ledoit-Wolf).
    shrinkage_cov: float | None = None
    #: Peso de la media histórica frente a la media de su clase de activo (James-Stein).
    shrinkage_mu: float = 0.60
    #: Ventana de la tasa libre de riesgo, en días calendario. None = la misma
    #: ventana con que se estima μ: el exceso de retorno sólo tiene sentido si
    #: ambos rendimientos cubren el mismo período (y el mismo ciclo de tasas).
    ventana_rf_dias: int | None = None
    #: Fallback de r_f si no hay fondos RF_CORTO utilizables.
    rf_fallback: float = 0.085
    #: Multi-arranque del optimizador no lineal.
    n_arranques: int = 6
    max_iter: int = 500
    #: Piso operativo: pesos por debajo se truncan a cero y se renormaliza.
    peso_minimo_operativo: float = 0.01


@dataclass(frozen=True)
class ConfigRebalanceo:
    """Bandas tácticas de rebalanceo."""

    banda_categoria: float = 0.05
    banda_fondo: float = 0.03
    costo_bps: float = 25.0
    #: Período de deriva simulada, en días calendario (un trimestre).
    dias_deriva: int = 91


@dataclass(frozen=True)
class ConfigBacktest:
    """
    Parámetros del backtest walk-forward.

    Las ventanas se expresan en días calendario y se traducen a observaciones
    con la frecuencia detectada del panel: los FICs publican 365 valores de
    unidad al año y los ETFs ~252, así que un número fijo de observaciones no
    representa el mismo lapso en ambos casos.
    """

    ventana_estimacion_dias: int = 365
    paso_rebalanceo_dias: int = 91
    costo_bps: float = 25.0
    #: Historia mínima requerida para ejecutar el backtest (año y medio).
    min_dias: int = 548


@dataclass(frozen=True)
class ConfigAM:
    """Configuración raíz del Asset Allocation Manager."""

    datos: ConfigDatos = field(default_factory=ConfigDatos)
    optimizacion: ConfigOptimizacion = field(default_factory=ConfigOptimizacion)
    rebalanceo: ConfigRebalanceo = field(default_factory=ConfigRebalanceo)
    backtest: ConfigBacktest = field(default_factory=ConfigBacktest)
    outdir: Path = Path("outputs")
    #: Nombre del informe gráfico HTML (único artefacto en disco).
    nombre_informe: str = "informe_am_pm.html"
    #: Genera el informe HTML con los gráficos consolidados.
    generar_html: bool = True
    #: Tope de filas por tabla en consola (None = sin recorte).
    max_filas_tabla: int | None = None
    #: Patrimonio del mandato, usado para traducir pesos a órdenes en COP.
    patrimonio_cop: float = 1_000_000_000.0
    semilla: int = 42
    modo_offline: bool = False

    def con_outdir(self, ruta: str | Path) -> "ConfigAM":
        return replace(self, outdir=Path(ruta))
