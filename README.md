# Mercado de trabajo de IA: México vs Estados Unidos

**¿Cuánto paga el trabajo de IA en México frente a Estados Unidos — y por qué es tan difícil averiguarlo?**

Un pipeline propio que recolecta vacantes de IA de la API de Adzuna cada semana, las clasifica,
normaliza sus salarios a poder adquisitivo comparable y las analiza. El corpus se acumula solo.

---

## Los hallazgos

**1. En México el dato prácticamente no existe, y ahora está medido.**
De 3,781 vacantes de IA en el corpus, **750 estadounidenses publican salario y sólo 4 mexicanas** —
20.1 % contra 8.9 %. Las cuatro mexicanas están todas en CDMX y van de 60,000 a 600,000 MXN anuales:
un rango de 10x sobre cuatro observaciones. No hay mediana mexicana que reportar, y este proyecto se
niega a inventar una.

**2. Cinco metros estadounidenses sí se sostienen** (mediana anual en USD, IC bootstrap del 90 %):

| Metro | n | Mediana | IC 90 % |
|---|---:|---:|---|
| SF Bay Area | 319 | $225,000 | $216,300 – $226,800 |
| Seattle | 63 | $202,500 | $185,000 – $207,500 |
| NYC | 116 | $200,000 | $185,000 – $210,000 |
| Austin | 38 | $197,575 | $168,044 – $209,750 |
| Boston | 64 | $182,500 | $175,000 – $195,000 |

**3. El prompt engineering está castigado, no premiado.**
Contra una mediana nacional estadounidense de $200,000, `prompt_eng` paga $138,950 y `langchain`
$140,400 — unos 30 % **por debajo** del mercado. `agents` y `llm` encabezan la tabla pero sin prima:
`agents` cae exactamente en la mediana. El spread entre la skill mejor y peor pagada es de $61,050.
Y en la importancia por permutación del modelo las skills marcan **cero**: la prima es composicional
—quien pide `agents` contrata más senior, en metros más caros— no incremental.

**4. Las multinacionales publican sueldo menos que las empresas locales.** En los dos países, y va
contra la intuición: EE. UU. 8.2 % contra 28.1 %; México 18.2 % contra 31.9 %. No se afirma como
causal: la lista de multinacionales es curada y subcuenta, y la ley de transparencia salarial
estadounidense es estatal. Pero el patrón se repite en México, donde esa ley no existe.

**5. El modelo de imputación no se publica, y ese es el resultado.**
Un gradient boosting entrenado con 750 salarios estadounidenses, validado contra un hold-out con
**todos** los salarios mexicanos observados. MdAPE 958 % contra un umbral de 35 %: **no publicable**.
El argumento de fondo pesa más que el número — con n=4 no existe experimento capaz de validar la
transferencia en ninguna dirección. Un modelo cuya precisión no se puede medir no tiene derecho a
publicar estimaciones.

---

## Por qué existe este dataset

El proyecto empezó como «bajar un dataset de Kaggle y desglosarlo por zona». Esa vía no existe:

- El mejor dataset real y actualizado (`mannacharya/ai-job-listings`, 7,066 vacantes de 2026)
  contiene **12 filas de México y ninguna con salario**.
- El más popular de la categoría —50,000 filas, miles de descargas, anunciado como scrapeado de
  LinkedIn, Indeed y Glassdoor— **es data generada**. Se detecta por su forma: doce países con
  ~4,300 filas cada uno y 100 % de las filas con salario. Ningún agregador real produce esa
  uniformidad; el dataset real de referencia tiene 19 % de salarios y está dominado por San
  Francisco.

Así que el dataset se construyó. El detector que descartó al sintético vive en el código
(`verificar_no_sintetico`) y su evidencia se reproduce en el notebook 01.

### Fuentes descartadas y por qué

| Descartada | Razón |
|---|---|
| `m0sm71/ai-jobs-dataset-2026` | Sintética. CV de 0.013 entre países y 100 % de salarios presentes |
| `waddahali/global-ai-job-market-and-agentic-surge-2025-2026` | Declara ser sintética en su propia descripción |
| `ruchi798/data-science-job-salaries` | Real pero congelada en 2022 |
| `aijobs/global-salaries-in-ai-ml-data-science` | Ref muerto: ese usuario de Kaggle no existe. Reemplazado por el CSV público del propio proyecto ai-jobs.net |
| Scraping de Glassdoor | Sus términos lo prohíben |

---

## Advertencias metodológicas

Están aquí porque condicionan cómo leer todo lo anterior.

- **Regla n ≥ 30.** Ningún corte por debajo de 30 observaciones se grafica ni se resume en una cifra:
  se reporta como insuficiente, con su n a la vista. Por eso no hay medianas mexicanas.
- **Mediana, nunca media.** Los salarios tienen cola derecha larga. Todo va con intervalo de
  confianza bootstrap.
- **`salary_is_predicted` se filtra desde la ingesta.** Adzuna *modela* salarios cuando la vacante no
  los publica. Sin ese filtro, Nueva York habría aportado 151 «observaciones» de las cuales 150 son
  el modelo interno de otra empresa. Un test automatizado impide que una fila predicha entre al
  conjunto observado.
- **Núcleo y anillo se reportan por separado.** *Núcleo* = el título declara la vacante de IA.
  *Anillo* = el título no lo dice pero la descripción exige ≥ 2 skills de IA aplicada. El anillo es
  10.6 % en EE. UU. y 4.4 % en México.
- **El buscador de Adzuna México devuelve mucho ruido.** En la sonda del día 0, 38 de 42 vacantes
  mexicanas con salario resultaron ser abogados, community managers y un técnico HVAC. La taxonomía
  los filtra; las cifras de México salen de lo que sobrevivió ese filtro, no de una muestra
  aleatoria del mercado.
- **PPP.** La comparación entre países usa el factor `PA.NUS.PPP` del Banco Mundial (MX 10.33,
  US 1.0) aplicado sobre el monto en moneda local, no sobre el ya convertido a dólares.
- **Pérdida conocida.** 3,706 filas quedan fuera por país distinto de MX/US; parte son
  estadounidenses cuyo país no mapeó porque la fuente de Kaggle trae «Santa Clara» o «CA» en esa
  columna. Está medida en el log de cada corrida.

---

## Cómo reproducirlo

```bash
make setup                 # crea el venv (Python 3.12) e instala todo
cp .env.example .env       # y pon tu app_id/app_key de developer.adzuna.com

make probe                 # compuerta día 0: mide cuánta señal hay antes de invertir
make collect               # una corrida -> data/raw/<fecha>/adzuna.parquet + manifest
make build                 # data/raw/* -> data/processed/vacantes.parquet
make notebooks             # ejecuta los cuatro notebooks de punta a punta

make test                  # 85 pruebas
make lint                  # ruff + mypy
```

Las llaves de Adzuna son gratuitas. Kaggle se lee de `~/.kaggle/kaggle.json`.

---

## Estructura

```
src/
  data/      adzuna.py · mapping.py · collect.py · kaggle_sources.py · schema.py · probe.py
  features/  normalize.py · taxonomy.py · extract.py · dedupe.py · build.py
  analysis/  stats.py · hypothesis.py
  models/    impute.py · evaluate.py
notebooks/   01 calidad · 02 brecha salarial · 03 skills y transparencia · 04 modelo
config/      cities.yml · skills.yml · taxonomy.yml
docs/superpowers/  specs/ (diseño y veredicto del día 0) · plans/
```

`adzuna.py` habla HTTP y no conoce el esquema; `mapping.py` traduce al esquema canónico y no conoce
HTTP. Cuando Adzuna cambie un campo, sólo se toca `mapping.py`.

**Stack:** Python 3.12 · uv · pandas · numpy · scipy · scikit-learn · matplotlib · seaborn ·
requests · pyarrow · pytest · ruff · mypy

---

## Licencia

MIT. Los datos recolectados provienen de la API pública de Adzuna y de datasets de Kaggle publicados
bajo CC0; se redistribuyen bajo los términos de sus fuentes.

Mau Vilar — [github.com/mauvilar](https://github.com/mauvilar)
