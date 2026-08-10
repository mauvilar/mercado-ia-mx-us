# Mercado de trabajo de IA: México vs Estados Unidos — Diseño

**Fecha:** 2026-08-07
**Autor:** Mau Vilar
**Estado:** aprobado, pendiente de plan de implementación

---

## 1. Motivación

El punto de partida era descargar un dataset de Kaggle con salarios de vacantes de AI Engineer
y desglosarlo por zona (CDMX, Querétaro, …) y por Estados Unidos. La verificación empírica de
los datasets candidatos mostró que ese camino no existe. Tres hallazgos, todos comprobados
descargando y abriendo los archivos, no leyendo sus descripciones:

**1.1 Kaggle no tiene datos salariales de México a nivel ciudad.**
El mejor dataset real y actualizado (`mannacharya/ai-job-listings-bi-weekly-updated`, 7,066
vacantes scrapeadas entre abril y junio de 2026) contiene **12 filas cuya ubicación menciona
México, y ninguna de ellas trae salario**. El dataset de Adzuna+USAJobs
(`atharvasoundankar/ai-job-market-global-2026`, 5,773 filas) cubre únicamente US, UK, CA, DE y AU.

**1.2 El dataset más popular de la categoría es sintético.**
`m0sm71/ai-jobs-dataset-2026` se anuncia como «50,000+ scrapeados de LinkedIn, Indeed, Glassdoor,
Wellfound y RemoteOK». El conteo por país lo desmiente:

```
Australia 4436 · Canada 4396 · United States 4372 · Singapore 4345 · Japan 4338
United Kingdom 4322 · Ireland 4319 · Switzerland 4315 · India 4305 · Netherlands 4269
Germany 4263 · France 4252
```

Doce países con ~4,300 filas cada uno y **100 % de las filas con salario**. Ningún agregador
real produce esa uniformidad; el dataset real de referencia tiene 19 % de salarios publicados y
está fuertemente sesgado a San Francisco. Es data generada. Queda descartado.

**1.3 En México el salario simplemente no se publica.**
Consulta en vivo al conector de Indeed (agosto 2026), 20 vacantes de perfil IA en CDMX y
Querétaro: **20 de 20 con `Compensation: N/A`**.

**1.4 El título «AI Engineer» todavía no es una categoría discreta en México.**
La búsqueda de «AI Engineer» en CDMX devuelve mayoritariamente SRE de Mastercard, SDE de Amazon
y un QA intern. El trabajo de IA existe, pero viene etiquetado como Software o Data Engineer.
Esto no es ruido: es uno de los hallazgos del proyecto.

La conclusión es que el dataset que la pregunta necesita **no existe y hay que construirlo**.
Eso convierte al proyecto en algo mejor que un análisis de CSV descargado.

---

## 2. Tesis y preguntas

> **¿Cuánto paga el trabajo de IA en México frente a Estados Unidos en 2026 — y por qué es tan
> difícil averiguarlo?**

La opacidad salarial mexicana no se trata como obstáculo sino como segundo objeto de estudio.

**Preguntas que el proyecto responde:**

1. ¿Cuánto paga una vacante de IA en CDMX, Querétaro, Monterrey y Guadalajara, y en los
   principales metros de EE. UU.?
2. ¿Cuánta de esa brecha sobrevive al ajuste por poder adquisitivo (PPP)?
3. ¿Qué skills cargan prima salarial, y son las mismas en ambos mercados?
4. ¿Qué tan opaco es el mercado mexicano y **qué tipo de empresa sí** publica sueldo?
5. ¿Cuántas vacantes en México son de IA sin llamarse así? (núcleo vs anillo)
6. ¿Existe arbitraje salarial en el remoto-para-EE. UU. desde México, y de qué tamaño?

---

## 3. Alcance

**Dentro:**
- Países: México y Estados Unidos.
- Zonas MX: CDMX/Valle de México, Querétaro, Monterrey, Guadalajara, más remoto-desde-México.
- Zonas US: SF Bay Area, NYC, Seattle, Austin, Boston, Los Ángeles, más remoto-US.
- Ventana: vacantes vivas recolectadas desde el arranque, acumulándose semanalmente. La serie
  histórica 2021→2026 a nivel país viene del dataset de ai-jobs.net.
- Roles: núcleo y anillo según la taxonomía de §5.

**Fuera:**
- Otros países (el dataset global de ai-jobs.net se usa sólo como contexto histórico).
- Compensación en acciones, bonos y prestaciones: casi nunca aparecen en la publicación y
  estimarlos sería inventar.
- Scraping de sitios que lo prohíben en sus términos. Toda ingesta va por API pública o
  dataset publicado con licencia abierta.

---

## 4. Fuentes de datos

| Fuente | Rol | Acceso | Aporta |
|---|---|---|---|
| **Adzuna API** (`mx`, `us`) | Primaria | API key gratuita | Mismo esquema para ambos países, nivel ciudad, refresco semanal. Endpoints `search`, `salary histogram`, `salary history` |
| **`mannacharya/ai-job-listings-bi-weekly-updated`** | Profundidad US | Kaggle API, CC0 | 7,066 vacantes reales, 19 % con salario, descripción completa para extraer skills |
| **ai-jobs.net** (`foorilla/ai-jobs-net-salaries`, CSV en GitHub) | Serie de tiempo | HTTP público, CC0 | 151,445 filas 2020→2025 a nivel país; provee el eje temporal que las vacantes vivas no tienen. **Verificado el 2026-08-10:** el ref de Kaggle que traía este spec estaba muerto (ese usuario no existe); la fuente viva es el repo del proyecto. Trae 113 filas de México, 102 de ellas 2024–2025 — autorreportadas y con sesgo de selección hacia empresas internacionales, así que sirven de ancla y techo, no de sustituto de la recolección por ciudad |
| **Conector Indeed (MCP)** | Validación | Ya autenticado | Muestreo puntual para verificar que la cobertura de Adzuna es real |
| **Factor PPP, Banco Mundial** (`PA.NUS.PPP`) | Normalización | API abierta | Convierte la comparación MX/US en algo defendible |
| **OCC / Computrabajo** | Refuerzo condicional | Sólo si el día 0 lo exige, y sólo si sus términos lo permiten | Cobertura extra de salarios MX publicados |

Sobre el refuerzo condicional: §3 prohíbe ingerir de sitios cuyos términos lo impidan, así que
antes de tocar OCC o Computrabajo hay que leer sus términos en el día 0. Si no lo permiten, las
alternativas por orden de preferencia son (a) su tabulador salarial público, que es contenido
publicado y no scraping de listados, (b) ampliar el abanico de queries y ciudades en Adzuna MX,
y (c) alargar la ventana de acumulación, que es gratis porque el pipeline ya corre solo.

**Fuentes descartadas y por qué** — esta tabla va también en el README público:

| Descartada | Razón |
|---|---|
| `m0sm71/ai-jobs-dataset-2026` | Sintética. Distribución uniforme por país (~4,300 × 12) y 100 % de salarios presentes |
| `waddahali/global-ai-job-market-and-agentic-surge-2025-2026` | Declara ser sintética en su propia descripción |
| `ruchi798/data-science-job-salaries` | Real pero congelada en 2022; el mercado de IA de 2026 no existía |
| Glassdoor scraping | Sus términos lo prohíben |

### 4.1 Trampa crítica de Adzuna

Adzuna expone el flag `salary_is_predicted`. La API **modela** salarios cuando la vacante no los
publica. Si no se filtra, el «salario observado» del análisis es en realidad el modelo interno de
otra empresa y toda conclusión queda contaminada. Regla: se filtra desde la ingesta, se conserva
en una columna aparte, y hay un test automatizado que impide que una fila predicha entre al
conjunto observado (§10).

---

## 5. Esquema canónico

Una fila = una observación de vacante en un snapshot.

| Campo | Tipo | Notas |
|---|---|---|
| `posting_id` | str | `sha1(source + id_nativo)`, estable entre corridas |
| `snapshot_date` | date | Fecha de la corrida de recolección |
| `source` | enum | `adzuna_mx`, `adzuna_us`, `kaggle_mannacharya`, `aijobs_net`, `indeed` |
| `title_raw` / `title_norm` | str | Crudo y normalizado |
| `company` | str | |
| `company_posting_count` | int | Vacantes de esa empresa en el corpus. Proxy calculable de tamaño/actividad, porque ninguna fuente confiable publica tamaño de empresa |
| `company_is_multinational` | bool | Contra lista curada en `config/`. Sostiene el análisis «multinacional vs local» |
| `category` | str | Categoría del agregador (Adzuna la provee). Proxy de sector |
| `country` | enum | `MX`, `US` |
| `state`, `city`, `metro` | str | Ciudad canonizada vía `config/cities.yml` |
| `is_remote` | bool | |
| `remote_scope` | enum | `local`, `nacional`, `us_desde_mx`, `global` |
| `posted_date` | date | |
| `salary_min_raw`, `salary_max_raw` | float | Tal como vienen |
| `salary_currency` | enum | `MXN`, `USD` |
| `salary_period` | enum | `hora`, `mes`, `año` |
| `salary_is_predicted` | bool | **Clave.** Ver §4.1 |
| `salary_observed` | bool | `True` sólo si viene publicado **y** no es predicho |
| `salary_annual_local` | float | Derivado |
| `salary_annual_usd` | float | Derivado, tipo de cambio del snapshot |
| `salary_annual_usd_ppp` | float | Derivado, factor PPP del país |
| `seniority` | enum | `intern`, `jr`, `mid`, `sr`, `lead`, `staff`, `principal`, `unknown` |
| `tier` | enum | `nucleo`, `anillo`, `fuera` |
| `skills` | list[str] | Del léxico de `config/skills.yml` |
| `description_text` | str | |
| `description_lang` | enum | `es`, `en` |
| `url` | str | |

### 5.1 Taxonomía núcleo / anillo

- **Núcleo** — el título lo declara. Regex sobre `title_norm` que exige un término de IA
  (`ai`, `artificial intelligence`, `inteligencia artificial`, `ml`, `machine learning`,
  `deep learning`, `llm`, `nlp`, `computer vision`, `mlops`, `genai`) junto a un término de rol
  (`engineer`, `ingenier[oa]`, `developer`, `desarrollador`, `scientist`, `architect`,
  `arquitect[oa]`).
- **Anillo** — no cumple núcleo, pero la descripción contiene **≥ 2 skills** del grupo «IA
  aplicada» (LLM, RAG, fine-tuning, PyTorch, TensorFlow, LangChain, vector database, embeddings,
  transformers, Hugging Face, prompt engineering, MLOps, feature store). El umbral de 2 evita
  los falsos positivos de vacantes que mencionan «AI» una vez como relleno de marketing.
- **Fuera** — el resto. Se conserva en crudo pero no entra a los agregados salariales.

Núcleo y anillo se reportan **siempre por separado**, nunca mezclados en un mismo promedio.

---

## 6. Métodos

**Normalización.** México publica mensual y en MXN; EE. UU. anual y en USD. Se reconcilia
periodicidad → moneda (tipo de cambio de la fecha del snapshot) → PPP (factor Banco Mundial).
Las tres versiones se conservan; los gráficos comparativos usan PPP y lo dicen en el eje.

**Deduplicación.** La misma vacante aparece en varios agregadores. Llave por
`hash(title_norm + company + city)` más fuzzy matching sobre el título para los casi-duplicados.

**Extracción.** Seniority y skills se extraen del **texto de la descripción**, no del título,
porque el título miente (ver §1.4). Léxico versionado en `config/skills.yml` con alias.

**Estadística.**
- Mediana con intervalo de confianza bootstrap. Nunca promedio simple: las distribuciones
  salariales tienen cola derecha larga y la media es engañosa.
- **Regla n ≥ 30 para graficar.** Los cortes por debajo del umbral se reportan explícitamente
  como muestra insuficiente en vez de omitirse en silencio.
- Pruebas de hipótesis formales: Mann-Whitney U para comparar medianas entre ciudades
  mexicanas; Welch sobre log-salario para MX vs US.
- Todo corte publicado va acompañado de su `n`.

---

## 7. Modelo de imputación

Estima rangos salariales para las vacantes mexicanas que no publican sueldo.

- **Estimador:** `HistGradientBoostingRegressor` de scikit-learn — soporta categóricas y
  ausentes de forma nativa y no agrega dependencias fuera del stack ya usado.
- **Target:** `log(salary_annual_usd_ppp)`.
- **Features:** `title_norm` agrupado, `seniority`, `skills` (multi-hot), `category`,
  `company_posting_count`, `company_is_multinational`, `is_remote`, `remote_scope`, `country`,
  `metro`, `tier`. No se usa «tamaño de empresa» porque ninguna fuente confiable lo publica; el
  conteo de vacantes por empresa dentro del corpus es el proxy honesto y sí es calculable.
- **Entrenamiento:** filas con `salary_observed = True`, **excluyendo México por completo**.
- **Hold-out primario:** *todas* las filas mexicanas con `salary_observed = True`. El modelo
  nunca las ve durante el entrenamiento.
- **Baseline obligatorio:** mediana por (país × seniority). El modelo debe superarlo.
- **Métrica:** MdAPE (mediana del error porcentual absoluto) sobre el hold-out mexicano.

**Criterio de publicación:** las imputaciones se publican **sólo si** MdAPE ≤ 35 % en el hold-out
mexicano **y** el modelo supera al baseline. Si no se cumple, las estimaciones no se publican y
el fallo se reporta como hallazgo — un modelo que no transfiere entre mercados es un resultado
legítimo e informativo. Las columnas imputadas viven separadas de las observadas y jamás se
mezclan en un mismo agregado.

---

## 8. Arquitectura

Ubicación: `~/Desktop/DA/Proyectos/mercado-ia-mx-us/`. Andamio heredado de CREDIT_RISK_V2
(`uv`, Python 3.12, `pyproject.toml` con rangos pinneados, Makefile con `help`, ruff/black/mypy/
pytest). Stack de análisis heredado de los notebooks de TripleTen: `pandas`, `numpy`,
`matplotlib`, `seaborn`, `scipy`, `scikit-learn`. Se añade `requests` (Adzuna), `pyarrow`
(parquet) y `pyyaml` (config). Nada más.

```
mercado-ia-mx-us/
├── README.md                      # tesis · hallazgos · reproducir · fuentes descartadas
├── pyproject.toml
├── Makefile                       # setup · collect · build · analyze · test · lint
├── .env.example                   # ADZUNA_APP_ID / ADZUNA_APP_KEY / KAGGLE_*
├── .github/workflows/collect.yml
├── config/
│   ├── cities.yml                 # canonización de ciudades y agrupación en metros
│   ├── taxonomy.yml               # patrones de título (núcleo) y gatillos (anillo)
│   └── skills.yml                 # léxico con alias
├── src/
│   ├── data/
│   │   ├── adzuna.py              # cliente: paginación, rate limit, backoff, reintentos
│   │   ├── kaggle_sources.py      # descarga y valida los dos datasets reales
│   │   ├── collect.py             # CLI de una corrida → parquet crudo + manifest
│   │   └── schema.py              # esquema canónico y validaciones
│   ├── features/
│   │   ├── normalize.py           # periodicidad, moneda, PPP
│   │   ├── taxonomy.py            # núcleo / anillo / fuera
│   │   ├── extract.py             # seniority y skills desde la descripción
│   │   └── dedupe.py
│   ├── analysis/
│   │   ├── stats.py               # mediana, bootstrap CI, regla n ≥ 30
│   │   └── hypothesis.py          # Mann-Whitney, Welch
│   ├── models/
│   │   ├── impute.py
│   │   └── evaluate.py            # hold-out MX, baseline, criterio de publicación
│   └── utils/                     # config.py, logging.py
├── data/
│   ├── raw/                       # snapshots por fecha, inmutables (parquet)
│   ├── interim/
│   └── processed/                 # lo que se publica en Kaggle
├── notebooks/
├── reports/figures/
└── tests/
```

---

## 9. Notebooks

Cuatro, en español, con la estructura narrativa de los proyectos de TripleTen: encabezado
(`# Título` → `## Sección` → `### Subsección`, `### DA: *Mau Vilar*`, enlace a GitHub) y un
bloque de conclusiones al cierre de cada sección. Todos corren de punta a punta desde
`data/processed/` sin volver a llamar a ninguna API.

1. **`01_recoleccion_y_calidad.ipynb`** — auditoría de fuentes. Incluye la autopsia del dataset
   sintético con la evidencia del conteo por país. Limpieza, deduplicación, cobertura.
2. **`02_brecha_salarial.ipynb`** — el análisis central. CDMX vs Querétaro vs Monterrey vs
   Guadalajara, metros de EE. UU., brecha nominal, brecha PPP, pruebas de hipótesis.
3. **`03_skills_y_transparencia.ipynb`** — primas salariales por skill en ambos mercados; quién
   publica sueldo y quién no, cortado por `category` (sector), `company_is_multinational` y
   `company_posting_count`.
4. **`04_modelo_imputacion.ipynb`** — entrenamiento, hold-out mexicano, baseline y veredicto
   sobre si las estimaciones se publican.

---

## 10. Automatización, errores y pruebas

**Automatización.** GitHub Action con cron semanal: recolecta, escribe un snapshot nuevo en
`data/raw/`, reconstruye `data/processed/` y sube una versión al dataset de Kaggle. El corpus se
acumula, así que la muestra mexicana crece con el tiempo — en la semana 4 hay ~4× la de la
semana 1.

**Errores.** Backoff exponencial ante rate limit de Adzuna. Un fallo parcial no tumba la corrida:
se guarda lo recibido y el hueco queda anotado en un `manifest.json` por snapshot (fecha,
queries, conteos, tasa de éxito), que además alimenta la serie de cobertura. `data/raw/` nunca se
sobrescribe. Los secretos van por `.env` y GitHub Secrets, nunca al repo.

**Pruebas.**
- **Candado de integridad:** el test falla si una fila con `salary_is_predicted = True` entra al
  conjunto `salary_observed`. Es el error que hundiría el análisis completo, así que se vuelve
  imposible por construcción.
- Normalización: 45,000 MXN mensuales → anual → USD → PPP, ida y vuelta.
- Taxonomía: un «Senior Software Engineer» cuya descripción pide RAG y PyTorch **debe** caer en
  `anillo`; uno que menciona «AI» una sola vez debe caer en `fuera`.
- Deduplicación: la misma vacante desde dos fuentes colapsa a una fila.
- Regla n ≥ 30: un corte con n = 29 se marca insuficiente.
- Cliente de Adzuna contra respuestas mockeadas. CI nunca pega a la API real.

---

## 11. Criterios de éxito

1. Al menos 3 de las 4 ciudades mexicanas alcanzan n ≥ 30 de salarios **observados** dentro de
   las primeras 6 semanas de recolección. Si no, el resultado se reporta explícitamente como
   insuficiente en vez de forzar un número.
2. Los cuatro notebooks corren de punta a punta desde `data/processed/`.
3. `make test` en verde; ruff y mypy limpios.
4. Dataset publicado en Kaggle con documentación de columnas y licencia.
5. README con la tabla de fuentes descartadas y su evidencia.
6. Entrada del proyecto publicada en el portafolio Next.js.

---

## 12. Riesgos

| Riesgo | Mitigación |
|---|---|
| Adzuna MX tiene cobertura salarial baja | La prueba de viabilidad del día 0 lo mide antes de invertir la semana. Si falla, se refuerza con OCC/Computrabajo |
| La muestra mexicana nunca llega a n ≥ 30 | Se reporta como hallazgo (la opacidad es parte de la tesis), no se rellena con imputaciones disfrazadas de observaciones |
| El modelo no transfiere de US a MX | Criterio de publicación explícito en §7. El resultado negativo se publica |
| Cambio de términos o de API | Los snapshots ya recolectados son inmutables y sobreviven |
| El alcance se infla | Notebooks fijos en 4; ciudades y roles congelados en `config/` |

---

## 13. Calendario

| Etapa | Trabajo |
|---|---|
| **Día 0** | API key de Adzuna, una corrida de prueba, conteo de salarios MX observados → **go / no-go** |
| **Días 1–2** | Ingesta, esquema, normalización, primeras pruebas |
| **Día 3** | Taxonomía, extracción de skills, deduplicación |
| **Días 4–5** | Notebooks 01 y 02 |
| **Día 6** | Notebook 03 |
| **Día 7** | Modelo de imputación y notebook 04 |
| **Día 8** | README, GitHub Action, publicación en Kaggle, entrada en el portafolio |

El día 0 es una compuerta real: si Adzuna MX no entrega salarios suficientes, se decide ahí el
refuerzo con OCC/Computrabajo antes de escribir el resto del pipeline.

---

## 14. Decisiones tomadas

| Decisión | Elección | Razón |
|---|---|---|
| Origen de los datos de México | Construir dataset propio vía API de Adzuna | Kaggle tiene 0 salarios de México; «construí el dataset» pesa más que «bajé un CSV» |
| Manejo del hueco salarial | Observados + opacidad como hallazgo + modelo de imputación | Añade componente de ML real y convierte la limitación en objeto de estudio |
| Definición de vacante de IA | Taxonomía de dos niveles (núcleo + anillo) | En México el título no declara el trabajo de IA; permite medir cuántas vacantes son de IA sin llamarse así |
| Entregable | Repo modular + pipeline semanal + dataset público en Kaggle | El dataset se mantiene vivo; cierra el círculo de la motivación del proyecto |
| Estilo | Repo tipo CREDIT_RISK_V2, narrativa tipo TripleTen | Consistencia con el trabajo previo; sin dependencias exóticas |
