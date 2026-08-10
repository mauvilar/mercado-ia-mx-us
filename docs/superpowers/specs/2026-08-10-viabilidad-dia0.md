# Compuerta del día 0 — veredicto

**Fecha:** 2026-08-10
**Ejecutado con:** app "NyxAI Studio's App", plan Trial Access de Adzuna
**Decisión:** seguir, pero con el alcance de México reformulado. Ver §5.

---

## 1. Lo que midió la sonda

```
=== COBERTURA POR ZONA ===
  mx|Ciudad de Mexico          total= 123  observado=  17 (14%)  predicho=   0
  mx|Guadalajara               total=  44  observado=  16 (36%)  predicho=   0
  mx|Monterrey                 total=  26  observado=   7 (27%)  predicho=   0
  mx|Queretaro                 total=  16  observado=   4 (25%)  predicho=   0
  us|Austin                    total= 153  observado=  26 (17%)  predicho= 127
  us|New York                  total= 151  observado=   1 (1%)   predicho= 150
  us|San Francisco             total= 150  observado=  59 (39%)  predicho=  91
```

Deduplicando por `id` de Adzuna: **42 vacantes mexicanas únicas** con salario real
(la sonda reportaba 44 filas; el sesgo por duplicados es despreciable) y **83 estadounidenses**.

## 2. La trampa de `salary_is_predicted`, confirmada en vivo

Nueva York devuelve **151 vacantes y sólo 1 con salario real**. Las otras 150 son
estimaciones del modelo interno de Adzuna. Sin el filtro de §4.1 del spec, NY habría
aportado 151 «observaciones» salariales al análisis, de las cuales 150 serían el modelo
de otra empresa. San Francisco: 59 reales de 150. Austin: 26 de 153.

México es el caso contrario: **Adzuna no predice ni un solo salario mexicano** (`predicho=0`
en las cuatro ciudades). Todo lo que llega de MX es publicado o no llega.

## 3. Periodicidad: la asunción se confirma

`mapping.py` documentaba `PERIODO_ADZUNA = "año"` como asunción pendiente de esta sonda.
**Se confirma.** Los 42 salarios mexicanos se agrupan en 144,000 · 168,000 · 180,000 ·
216,000 · 240,000 — todos divisibles entre 12 y equivalentes a 12,000–20,000 MXN
mensuales, que es el rango real del mercado. Adzuna anualiza.

Un caso desentona: «SR Software Engineer (Data focus) — (Remote, LATAM)» a 64,800–94,800,
que como anual en pesos es imposible para un senior y como anual en dólares es exacto.
Es un outlier de una vacante remota pagada en USD, no un patrón. Queda anotado; si
aparecen más, habrá que inferir moneda por fila.

## 4. El hallazgo que cambia el proyecto

De las **42** vacantes mexicanas con salario real, la taxonomía núcleo/anillo clasifica:

| tier | n |
|---|---|
| núcleo | 4 |
| anillo | 0 |
| **fuera** | **38** |

Las 38 descartadas incluyen: abogado corporativo, técnico HVAC, fotógrafo junior,
community manager (×3), maestra de matemáticas, ejecutivo de bodas y convenciones,
gerente de RRHH, analista fiscal, diseñador publicitario. **La búsqueda de Adzuna México
empareja por palabras sueltas y devuelve casi puro ruido.**

Las 4 reales, todas en CDMX:

```
nucleo      60,000  Ingeniero de Preventa Google Cloud e Inteligencia Artificial
nucleo      81,000  RPA Engineer AI Development Experience
nucleo     240,000  Ingeniero en Inteligencia Artificial
nucleo     600,000  Machine Learning Engineer
```

**Señal útil: 4 de 42 por corrida (10%). Querétaro, Monterrey y Guadalajara: cero.**

La taxonomía hizo su trabajo: sin ella, la «mediana salarial de vacantes de IA en
Guadalajara» se habría calculado sobre community managers y un técnico HVAC.

## 5. Veredicto y consecuencias

La regla de decisión del plan era: `0 < mx_obs < 10` → seguir, ampliar consultas, y el
modelo de imputación carga más peso. **Estamos en 4.** Se sigue, con tres correcciones:

1. **El desglose por ciudad mexicana no se sostiene con Adzuna sola.** CDMX podría llegar
   a n≥30 en ~8 semanas *si* cada corrida trajera vacantes nuevas, pero con
   `max_days_old=60` las corridas semanales se solapan casi por completo. Querétaro,
   Monterrey y Guadalajara probablemente nunca lleguen. El criterio de éxito §11.1 del
   spec («3 de 4 ciudades con n≥30 en 6 semanas») hoy no es alcanzable por esta vía.
2. **Mitigaciones por orden de costo:** (a) `collect.py` ya usa 9 consultas contra las 4
   de la sonda, y más ciudades — medir cuánto sube el rendimiento antes de nada más;
   (b) bajar `max_days_old` para que cada corrida aporte vacantes genuinamente nuevas;
   (c) revisar términos de OCC/Computrabajo; (d) usar las 113 filas de México de
   ai-jobs.net como ancla a nivel país, asumidas autorreportadas y sesgadas.
3. **El lado estadounidense sí se sostiene:** 83 vacantes con salario real en una corrida,
   con SF y Austin holgadamente por encima del umbral.

Lo que este veredicto no toca: la tesis del proyecto sale **fortalecida**. La pregunta era
«¿cuánto paga el trabajo de IA en México y por qué es tan difícil averiguarlo?», y ahora
hay números duros para la segunda mitad: 10 % de señal útil, cero predicción de salarios
en MX, y un mercado donde el título «AI Engineer» es tan raro que el buscador del propio
agregador no distingue una vacante de IA de un técnico de refrigeración.
