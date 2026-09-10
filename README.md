# AM-PM · Asset Allocation Manager

Motor de asignación estratégica y táctica de capital (Top-Down) sobre **Fondos de
Inversión Colectiva, ETFs e índices del mercado colombiano**, segmentado por
perfiles de riesgo Conservador, Moderado y Agresivo.

El sistema no selecciona acciones ni bonos individuales: distribuye capital
entre vehículos colectivos, con los datos oficiales diarios que publica la
Superintendencia Financiera de Colombia en el Portal de Datos Abiertos.

---

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requiere Python 3.10 o superior.

## Uso

```bash
# Corrida completa: universo, 3 perfiles x 4 métodos, rebalanceo y backtest
python "Asset Allocation.py"

# Un solo perfil, 5 años de historia, patrimonio de COP 5.000 millones
python "Asset Allocation.py" --perfil AGRESIVO --lookback 5 --patrimonio 5e9

# Corrida rápida sin backtest, con token de datos.gov.co (evita el throttling)
python "Asset Allocation.py" --sin-backtest --app-token $SODA_APP_TOKEN

# Modo offline reproducible (datos sintéticos) para pruebas
python "Asset Allocation.py" --offline

# Abrir el informe gráfico en el navegador al terminar
python "Asset Allocation.py" --abrir

# Sólo tablas en consola, sin informe gráfico
python "Asset Allocation.py" --sin-html

# Pruebas
python tests/test_am_pm.py        # o: python -m pytest tests/ -v
```

`python "Asset Allocation.py" --help` lista los parámetros disponibles (bandas,
costos, ventanas, topes de concentración, shrinkage, presentación).

---

## Arquitectura

| Módulo | Responsabilidad |
|---|---|
| [config.py](am_pm/config.py) | Parámetros, taxonomía y constantes en dataclasses inmutables |
| [ingestion.py](am_pm/ingestion.py) | Cliente SODA (`sodapy`), paginación, cache y respaldo sintético |
| [metrics.py](am_pm/metrics.py) | Retorno, volatilidad, Sharpe, Sortino, drawdown, VaR, covarianza |
| [universe.py](am_pm/universe.py) | Universo curado y clasificación híbrida en 5 clases de activo |
| [profiles.py](am_pm/profiles.py) | Mandatos por perfil y traducción a restricciones lineales |
| [optimizers.py](am_pm/optimizers.py) | Markowitz, Risk Parity, HRP y 1/N sobre el mismo conjunto factible |
| [allocation.py](am_pm/allocation.py) | Portafolios modelo, cardinalidad y estadísticas ex-ante |
| [rebalancing.py](am_pm/rebalancing.py) | Bandas tácticas, deriva de pesos y plan de órdenes |
| [backtest.py](am_pm/backtest.py) | Walk-forward con costos de transacción |
| [charts.py](am_pm/charts.py) | Gráficos SVG interactivos y su capa de tooltips, sin librerías externas |
| [reporting.py](am_pm/reporting.py) | Tablas en terminal e informe gráfico HTML autocontenido |
| [pipeline.py](am_pm/pipeline.py) | Orquestación end-to-end y resumen ejecutivo |

---

## Metodología

### 1. Extracción en dos etapas

El dataset `qhpu-8ixx` tiene ~3 millones de filas. Descargarlo completo es
inviable, así que el pipeline hace *screening* sobre un único corte diario
(todo el mercado, ~200 fondos), selecciona la clase de participación de mayor
AUM de cada fondo y sólo entonces descarga las series históricas de los
seleccionados, paginando con `$offset` y cacheando en disco.

Se excluyen fondos de capital privado e inmobiliarios: sin valoración diaria ni
liquidez, no son asignables en un mandato líquido.

### 2. Clasificación híbrida en clases de activo

El obstáculo real del mercado colombiano es que **el nombre comercial no revela
la política de inversión**: "Fiducuenta", "Sumar", "Fidugob" o "Valor Plus"
concentran la mayor parte del AUM y no dicen nada. Un clasificador por palabras
clave deja sin resolver ~60% del patrimonio. Por eso hay tres capas:

1. **Reglas** con precedencia explícita. "Fondo Bursátil Global X **TES**
   Colombia" es renta fija pese a decir *bursátil*; "Global X **Colombia**
   Select" es renta variable local pese a decir *global* (ahí "Global X" es el
   emisor del ETF, no el subyacente).
2. **Modelo de riesgo** para los nombres no concluyentes: volatilidad
   anualizada realizada más betas contra un proxy de renta variable local
   (ETF de COLCAP) y uno de exposición internacional/FX.
3. **Conciliación**: si la regla contradice al riesgo observado, manda el dato y
   la discrepancia queda registrada. Así el sistema detecta que "Acción
   Sociedad Fiduciaria" no es un fondo de acciones (volatilidad 0,3%) y que un
   "efectivo en dólares" tiene riesgo de renta variable para un inversionista en
   pesos (volatilidad 11% por el tipo de cambio).

Cada fondo queda con su categoría, el método que la determinó y la observación
correspondiente en la tabla del universo curado que se imprime en consola,
auditable por el comité. Las discrepancias entre nombre y riesgo observado se
listan aparte, en su propia tabla de revisión.

### 3. Tasa libre de riesgo dinámica

`r_f` no es un supuesto: es el rendimiento anualizado compuesto de la categoría
`RF_CORTO`, ponderado por AUM sobre la ventana reciente. Es el costo de
oportunidad real del cliente colombiano —el fondo de liquidez donde ya está su
plata—, y se reestima en cada ventana del backtest.

### 4. Restricciones del mandato

Bandas por clase de activo, piso/techo agregado de renta variable (≤10%
conservador, ≤40% moderado, ≥50% agresivo), tope por fondo y tope por gestora
(riesgo fiduciario). Todas son lineales, así que la factibilidad se **verifica
con programación lineal** antes de optimizar.

Si el universo no admite el mandato, las restricciones se relajan en cascada
acumulativa —concentración primero, mínimos estratégicos después, máximos de
riesgo al final— y cada relajación se reporta. Un mandato incumplido en
silencio sería peor que un error.

### 5. Optimización

Los cuatro métodos resuelven sobre **el mismo conjunto factible**, de modo que
las diferencias sean atribuibles al criterio y no a distintos grados de libertad.

- **Markowitz (máximo Sharpe)** vía la transformación de Schaible: el cociente
  `(μ−r_f)'w / √(w'Σw)` es un problema fraccional que SLSQP resuelve mal cuando
  la volatilidad objetivo es de 0,5% (el caso del perfil conservador). Con
  `y = κw` se convierte en un programa cuadrático convexo con óptimo global.
- **Risk Parity**: minimiza la dispersión de las contribuciones al riesgo.
- **HRP**: clustering jerárquico y bisección recursiva, sin invertir Σ.
- **1/N**: referencia ingenua.

HRP y 1/N se llevan al mandato por **proyección euclídea** sobre el conjunto
factible: el portafolio admisible más cercano al propuesto.

Σ se estima con shrinkage hacia correlación constante (intensidad tipo
Ledoit-Wolf) y μ se contrae hacia la media transversal, para no perseguir al
fondo que mejor lo hizo el trimestre pasado.

### 6. Rebalanceo táctico

Se simula la deriva buy-and-hold del portafolio modelo y se comparan los pesos
vigentes contra los objetivos. Una desviación mayor a ±5 puntos porcentuales en
una clase de activo dispara alerta, con la acción sugerida y el plan de órdenes
en pesos, incluyendo rotación y costo estimado.

### 7. Backtest walk-forward

En cada rebalanceo se reestiman `r_f`, μ y Σ **sólo** con la ventana previa, se
reconstruye el universo elegible de esa fecha y se re-optimiza. Entre
rebalanceos los pesos derivan con el mercado, y la rotación se cobra a la tarifa
configurada. Se comparan contra dos referencias pasivas: la caja (`RF_CORTO`
equiponderada) y el universo completo equiponderado.

---

## Salida de resultados

El sistema **no escribe CSV ni imágenes sueltas**. Los resultados se entregan
por dos canales:

### 1. Tablas en la terminal

Todo el detalle tabular se imprime durante la corrida, con formato de mesa de
inversión (columnas alineadas, porcentajes y montos en COP ya formateados):

| Tabla | Contenido |
|---|---|
| Universo curado | Ficha de cada fondo: clase de activo, método de clasificación, AUM, retorno, volatilidad, Sharpe, Sortino, drawdown |
| Discrepancias nombre vs. riesgo | Fondos donde el nombre comercial contradice al riesgo observado, con sus betas |
| Política de inversión | Bandas por clase de activo y topes de concentración de los tres mandatos |
| Composición por clase de activo | Pesos agregados por perfil y método, con el total de renta variable |
| Portafolios modelo | Posiciones de cada perfil × método: fondo, gestora, peso y métricas |
| Bandas de rebalanceo | Objetivo, peso vigente, desvío, estado y acción sugerida por categoría |
| Órdenes de rebalanceo | Suscripciones y redenciones en COP con costo estimado |
| Backtest | Desempeño realizado por estrategia y bitácora de rebalanceos |
| Tabla comparativa | Métricas ex-ante cruzadas con las realizadas en el walk-forward |
| Resumen ejecutivo | Cierre de comité: universo, r_f, portafolios, alertas y backtest |

Con `--max-filas N` se recorta el número de filas por tabla en corridas largas.

### 2. Informe gráfico HTML interactivo

Un **único documento autocontenido** (`outputs/informe_am_pm.html`) que se
recorre con scroll. Los gráficos se generan como SVG nativo desde
[charts.py](am_pm/charts.py): sin matplotlib, sin CDN, sin red. Incluye índice
fijo, tarjetas con las cifras de la corrida y tres secciones:

| Sección | Contenido | Al pasar el cursor |
|---|---|---|
| Universo curado | Mapa riesgo/retorno en múltiplos pequeños, un panel por clase de activo | Nombre del fondo, gestora, retorno, volatilidad, Sharpe, Sortino, drawdown, AUM y **cómo fue clasificado** |
| Composición estratégica | Barras apiladas por perfil y método | Peso exacto del segmento, con su perfil y método |
| Backtest walk-forward | Curvas de capital acumulado neto de costos | Retícula que engancha la fecha y tooltip con **las seis series** de ese día, ordenadas |

Detalles de la capa de interacción:

- **Dispersión con punto más cercano.** En el conglomerado de renta fija corta
  los fondos se solapan; el cursor sólo tiene que estar *cerca*, no encima.
- **Teclado.** Los puntos y segmentos son focalizables con Tab; en las curvas,
  las flechas recorren fechas (Shift para saltos de 20, Inicio/Fin para los
  extremos, Esc para salir).
- **Nombres tratados como dato no confiable.** Vienen de una API pública, así
  que se insertan con `textContent`, nunca concatenando HTML.
- **Paleta validada para daltonismo** (separación CVD ΔE ≥ 8 en pares
  adyacentes) con etiquetado directo: la identidad de una serie nunca depende
  sólo del color, y el tooltip nunca es la única vía a un valor.

Se abre con doble clic, o con `--abrir` al terminar la corrida. `--html RUTA`
fija la ubicación y `--sin-html` lo omite.

---

## Advertencias

- El backtest reoptimiza sobre el universo **vigente hoy**: los fondos
  liquidados o fusionados no están en el dataset actual, así que hay sesgo de
  supervivencia. Los resultados sirven para comparar métodos entre sí, no como
  proyección de rentabilidad.
- El costo de transacción es un parámetro plano (`--costo-bps`). No modela
  penalidades por retiro anticipado ni pactos de permanencia, que en varios FICs
  colombianos son significativos.
- La clasificación cuantitativa infiere la clase de activo del comportamiento
  del precio, no del prospecto. Es un insumo para el comité, no un sustituto de
  la revisión del reglamento del fondo.
- `--offline` genera datos sintéticos: sirve para probar el código, jamás para
  decidir.
