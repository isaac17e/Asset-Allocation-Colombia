"""
Pruebas del AM-PM.

Se ejecutan íntegramente sobre el generador sintético, de modo que no dependen
de la disponibilidad de datos.gov.co ni de la fecha de corte publicada.

    python -m pytest tests/ -v      (o bien:  python tests/test_am_pm.py)
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from am_pm.allocation import aplicar_cardinalidad, construir_portafolios, estadisticas_ex_ante
from am_pm.config import CATEGORIAS, ConfigAM, ConfigDatos, ConfigOptimizacion
from am_pm.ingestion import generar_datos_sinteticos
from am_pm.metrics import (
    matriz_covarianza,
    max_drawdown,
    periodos_por_anio,
    ratio_sharpe,
    ratio_sortino,
    retorno_anualizado,
    retornos_simples,
    volatilidad_anualizada,
)
from am_pm.optimizers import GrupoRestriccion, RestriccionesPortafolio, resolver
from am_pm.profiles import PERFILES, construir_restricciones
from am_pm.rebalancing import derivar_pesos, evaluar_bandas, plan_ordenes
from am_pm.universe import calcular_rf_dinamica, clasificar_por_reglas, construir_universo


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def test_retorno_anualizado_serie_conocida() -> None:
    """Una serie que duplica su valor en dos años debe dar ~41.4% anual."""
    fechas = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    precios = pd.Series(np.linspace(100, 200, len(fechas)), index=fechas)
    esperado = 2.0 ** (365.0 / (fechas[-1] - fechas[0]).days) - 1.0
    assert abs(retorno_anualizado(precios) - esperado) < 1e-9


def test_volatilidad_escala_con_raiz_del_tiempo() -> None:
    rng = np.random.default_rng(7)
    fechas = pd.date_range("2024-01-01", periods=1000, freq="D")
    rets = pd.Series(rng.normal(0, 0.01, 1000), index=fechas)
    vol = volatilidad_anualizada(rets)
    assert abs(vol - 0.01 * np.sqrt(365)) < 0.02


def test_periodos_por_anio_detecta_frecuencia() -> None:
    """La frecuencia se infiere del calendario, no se asume."""
    calendario = pd.date_range("2023-01-01", periods=800, freq="D")
    habil = pd.date_range("2023-01-01", periods=500, freq="B")
    assert periodos_por_anio(calendario) == 365.0
    assert periodos_por_anio(habil) == 252.0


def test_sortino_supera_sharpe_con_sesgo_positivo() -> None:
    """Sin caídas relevantes, el riesgo bajista es menor que la volatilidad total."""
    rng = np.random.default_rng(3)
    fechas = pd.date_range("2024-01-01", periods=800, freq="D")
    rets = pd.Series(np.abs(rng.normal(0.0006, 0.004, 800)), index=fechas)
    precios = 100 * (1 + rets).cumprod()
    r_anual = retorno_anualizado(precios)
    vol = volatilidad_anualizada(rets)
    assert ratio_sortino(r_anual, rets, 0.05) > ratio_sharpe(r_anual, vol, 0.05)


def test_max_drawdown_es_negativo_y_acotado() -> None:
    precios = pd.Series(
        [100, 120, 90, 130], index=pd.date_range("2024-01-01", periods=4, freq="D")
    )
    assert abs(max_drawdown(precios) - (90 / 120 - 1)) < 1e-12


def test_covarianza_es_definida_positiva_con_pocos_datos() -> None:
    """El shrinkage debe rescatar el caso nº activos > nº observaciones."""
    rng = np.random.default_rng(11)
    datos = pd.DataFrame(
        rng.normal(0, 0.01, (40, 60)),
        index=pd.date_range("2024-01-01", periods=40, freq="D"),
    )
    cov = matriz_covarianza(datos)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > 0


# --------------------------------------------------------------------------- #
# Clasificación del universo
# --------------------------------------------------------------------------- #
def test_reglas_priorizan_renta_fija_sobre_bursatil() -> None:
    """'Fondo Bursátil Global X TES Colombia' es renta fija, no acciones."""
    assert clasificar_por_reglas(
        "FONDO BURSÁTIL GLOBAL X TES COLOMBIA COP GBI EM ID ETF", "FIC BURSATILES"
    ) == "RF_MEDIANO_LARGO"


def test_reglas_priorizan_colombia_sobre_global() -> None:
    """'Global X' es el emisor del ETF; el subyacente es Colombia."""
    assert clasificar_por_reglas(
        "FONDO BURSÁTIL GLOBAL X COLOMBIA SELECT DE S&P", "FIC BURSATILES"
    ) == "RV_LOCAL"
    assert clasificar_por_reglas(
        "CREDICORP CAPITAL ACCIONES GLOBALES", "FIC DE TIPO GENERAL"
    ) == "RV_INTERNACIONAL"


def test_reglas_devuelven_none_si_el_nombre_no_informa() -> None:
    """Los nombres comerciales opacos deben delegarse al modelo de riesgo."""
    assert clasificar_por_reglas("FONDO ABIERTO ALIANZA", "FIC DE TIPO GENERAL") is None
    assert clasificar_por_reglas("FIC VALOR PLUS I", "FIC DE TIPO GENERAL") is None


def test_rf_dinamica_refleja_la_categoria_corto_plazo() -> None:
    fechas = pd.date_range("2024-01-01", periods=400, freq="D")
    tasa = 0.09
    serie = 1000 * (1 + tasa) ** (np.arange(len(fechas)) / 365.0)
    precios = pd.DataFrame({"A": serie, "B": serie}, index=fechas)
    categorias = pd.Series({"A": "RF_CORTO", "B": "RF_CORTO"})
    assert abs(calcular_rf_dinamica(precios, categorias, ventana=365) - tasa) < 0.002


def test_rf_dinamica_usa_fallback_sin_fondos_de_liquidez() -> None:
    fechas = pd.date_range("2024-01-01", periods=100, freq="D")
    precios = pd.DataFrame({"A": np.linspace(100, 110, 100)}, index=fechas)
    categorias = pd.Series({"A": "RV_LOCAL"})
    assert calcular_rf_dinamica(precios, categorias, fallback=0.077) == 0.077


# --------------------------------------------------------------------------- #
# Restricciones y optimizadores
# --------------------------------------------------------------------------- #
def _restricciones_prueba(n: int = 12) -> tuple[RestriccionesPortafolio, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(5)
    activos = tuple(f"F{i}" for i in range(n))
    A = rng.normal(size=(n, n))
    cov = pd.DataFrame(A @ A.T / 200 + np.eye(n) * 0.005, index=activos, columns=activos)
    mu = pd.Series(rng.uniform(0.07, 0.20, n), index=list(activos))
    restr = RestriccionesPortafolio(
        activos=activos,
        cota_inferior=np.zeros(n),
        cota_superior=np.full(n, 0.20),
        grupos=(
            GrupoRestriccion("RENTA_VARIABLE", tuple(range(8, n)), 0.20, 0.40),
            GrupoRestriccion("GESTOR::A", (0, 1, 2), 0.0, 0.30),
        ),
        etiqueta="PRUEBA",
    )
    return restr, mu, cov


def test_todos_los_metodos_producen_pesos_factibles() -> None:
    restr, mu, cov = _restricciones_prueba()
    for metodo in ("MARKOWITZ_SHARPE", "RISK_PARITY", "HRP", "EQUIPONDERADO"):
        w = resolver(metodo, mu, cov, 0.08, restr)
        assert abs(w.sum() - 1.0) < 1e-6, metodo
        assert (w >= -1e-9).all(), metodo
        assert not restr.violaciones(w.to_numpy(), tol=1e-4), (metodo, restr.violaciones(w.to_numpy()))


def test_markowitz_maximiza_el_sharpe_ex_ante() -> None:
    """La cartera tangente debe dominar a los demás métodos en Sharpe esperado."""
    restr, mu, cov = _restricciones_prueba()
    rf = 0.08
    sharpes = {}
    for metodo in ("MARKOWITZ_SHARPE", "RISK_PARITY", "HRP", "EQUIPONDERADO"):
        w = resolver(metodo, mu, cov, rf, restr).to_numpy()
        vol = float(np.sqrt(w @ cov.to_numpy() @ w))
        sharpes[metodo] = (w @ mu.to_numpy() - rf) / vol
    assert sharpes["MARKOWITZ_SHARPE"] >= max(
        v for k, v in sharpes.items() if k != "MARKOWITZ_SHARPE"
    ) - 1e-6


def test_risk_parity_iguala_contribuciones_al_riesgo() -> None:
    """Sin restricciones activas, las contribuciones deben ser homogéneas."""
    rng = np.random.default_rng(1)
    n = 8
    activos = tuple(f"F{i}" for i in range(n))
    A = rng.normal(size=(n, n))
    cov = pd.DataFrame(A @ A.T / 100 + np.eye(n) * 0.01, index=activos, columns=activos)
    restr = RestriccionesPortafolio(activos, np.zeros(n), np.ones(n), (), "LIBRE")
    w = resolver("RISK_PARITY", pd.Series(0.1, index=list(activos)), cov, 0.05, restr).to_numpy()
    cov_m = cov.to_numpy()
    rc = w * (cov_m @ w) / np.sqrt(w @ cov_m @ w)
    assert rc.std() / rc.mean() < 0.05


def test_proyeccion_devuelve_punto_factible() -> None:
    restr, _, _ = _restricciones_prueba()
    objetivo = np.zeros(restr.n)
    objetivo[0] = 1.0  # portafolio inviable: todo en un fondo
    w = restr.proyectar(objetivo)
    assert not restr.violaciones(w, tol=1e-6)


def test_conjunto_infactible_se_detecta() -> None:
    """Si las cotas no permiten sumar 1, no existe punto factible."""
    n = 4
    restr = RestriccionesPortafolio(
        tuple(f"F{i}" for i in range(n)), np.zeros(n), np.full(n, 0.10), (), "IMPOSIBLE"
    )
    assert restr.punto_factible() is None


# --------------------------------------------------------------------------- #
# Perfiles
# --------------------------------------------------------------------------- #
def test_perfiles_respetan_los_topes_de_renta_variable() -> None:
    assert PERFILES["CONSERVADOR"].rv_max <= 0.10
    assert PERFILES["MODERADO"].rv_max <= 0.40
    assert PERFILES["AGRESIVO"].rv_min >= 0.50


def test_mandato_infactible_se_relaja_en_vez_de_fallar() -> None:
    """Con un universo diminuto el mandato es imposible: debe relajarse y avisar."""
    categorias = pd.Series({"a": "RF_CORTO", "b": "RF_CORTO", "c": "RF_MEDIANO_LARGO",
                            "d": "MIXTO", "e": "RV_LOCAL", "f": "RV_INTERNACIONAL"})
    gestores = pd.Series({"a": "G1", "b": "G1", "c": "G2", "d": "G2", "e": "G3", "f": "G3"})
    for perfil in PERFILES.values():
        restr, notas = construir_restricciones(perfil, categorias, gestores)
        assert restr.punto_factible() is not None, perfil.nombre
        assert notas, "una relajación silenciosa sería inaceptable para el comité"


def test_cardinalidad_se_amplia_si_choca_con_los_minimos() -> None:
    """El límite de posiciones es operativo: cede ante los mínimos de riesgo."""
    n = 12
    activos = tuple(f"F{i}" for i in range(n))
    restr = RestriccionesPortafolio(
        activos, np.zeros(n), np.full(n, 0.20),
        (GrupoRestriccion("RV", (10, 11), 0.30, 0.40),), "PRUEBA",
    )
    pesos = pd.Series(np.full(n, 1.0 / n), index=list(activos))
    w = aplicar_cardinalidad(pesos, 3, restr)
    assert not restr.violaciones(w.to_numpy(), tol=1e-4)
    assert abs(w.sum() - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# Rebalanceo
# --------------------------------------------------------------------------- #
def test_deriva_aumenta_el_peso_del_activo_ganador() -> None:
    fechas = pd.date_range("2024-01-01", periods=100, freq="D")
    precios = pd.DataFrame(
        {"A": np.linspace(100, 200, 100), "B": np.full(100, 100.0)}, index=fechas
    )
    objetivo = pd.Series({"A": 0.5, "B": 0.5})
    derivados = derivar_pesos(objetivo, precios)
    assert derivados["A"] > 0.66 and abs(derivados.sum() - 1.0) < 1e-9


def test_bandas_disparan_alerta_al_superar_el_umbral() -> None:
    categorias = pd.Series({"A": "RV_LOCAL", "B": "RF_CORTO"})
    tabla = evaluar_bandas(
        pd.Series({"A": 0.70, "B": 0.30}), pd.Series({"A": 0.50, "B": 0.50}),
        categorias, banda=0.05,
    )
    assert tabla.loc["RV_LOCAL", "estado"] == "ALERTA"
    assert "VENDER" in tabla.loc["RV_LOCAL", "accion"]
    assert tabla.loc["RF_CORTO", "accion"].startswith("COMPRAR")


def test_ordenes_omiten_ajustes_irrelevantes() -> None:
    """Un desvío de 1 pb no justifica una orden."""
    ordenes = plan_ordenes(
        pd.Series({"A": 0.5001, "B": 0.4999}), pd.Series({"A": 0.50, "B": 0.50}),
        patrimonio=1e9, umbral_minimo=0.005,
    )
    assert ordenes.empty


# --------------------------------------------------------------------------- #
# Integración
# --------------------------------------------------------------------------- #
def test_pipeline_sintetico_end_to_end() -> None:
    """El universo se construye, se clasifica y se optimiza sin intervención."""
    cfg_datos = replace(ConfigDatos(), min_obs_historia=300, min_obs_panel=300, top_n_fondos=25)
    cfg_opt = ConfigOptimizacion()
    datos = generar_datos_sinteticos(cfg_datos, semilla=42)
    universo = construir_universo(datos, cfg_datos, cfg_opt)

    assert len(universo.fondos) >= 15
    assert set(universo.fondos["categoria"]).issubset(set(CATEGORIAS))
    assert 0.0 < universo.rf < 0.30
    assert universo.precios.notna().all().all()

    asignacion = construir_portafolios(universo, cfg_opt)
    for nombre, resultado in asignacion.perfiles.items():
        perfil = PERFILES[nombre]
        for metodo in resultado.pesos.columns:
            w = resultado.pesos[metodo]
            assert abs(w.sum() - 1.0) < 1e-4, (nombre, metodo)
            rv = resultado.composicion.loc[list(("RV_LOCAL", "RV_INTERNACIONAL")), metodo].sum()
            assert rv <= perfil.rv_max + 1e-3, (nombre, metodo, rv)


def test_estadisticas_ex_ante_descomponen_el_riesgo() -> None:
    """Las contribuciones al riesgo por categoría deben sumar 100%."""
    restr, mu, cov = _restricciones_prueba()
    categorias = pd.Series(
        {a: ("RV_LOCAL" if i >= 8 else "RF_CORTO") for i, a in enumerate(restr.activos)}
    )
    w = resolver("EQUIPONDERADO", mu, cov, 0.08, restr)
    stats = estadisticas_ex_ante(w, mu, cov, 0.08, categorias)
    total = sum(stats[f"riesgo_{c}"] for c in CATEGORIAS)
    assert abs(total - 1.0) < 1e-6
    assert abs(stats["peso_RENTA_VARIABLE"] - stats["peso_RV_LOCAL"]) < 1e-9


if __name__ == "__main__":
    fallos = 0
    pruebas = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    for nombre, prueba in pruebas:
        try:
            prueba()
            print(f"  PASS  {nombre}")
        except Exception as exc:  # noqa: BLE001
            fallos += 1
            print(f"  FAIL  {nombre}: {type(exc).__name__}: {exc}")
    print(f"\n{len(pruebas) - fallos}/{len(pruebas)} pruebas superadas")
    sys.exit(1 if fallos else 0)
