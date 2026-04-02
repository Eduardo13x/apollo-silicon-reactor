# ANE/Accelerate Analysis — Apollo Optimizer Learning Engine

**Date:** 2026-04-02
**Agent:** Agente 4 — ANE/Accelerate
**Verdict:** Vectorization not viable at current array sizes. No implementation.

---

## 1. Auditoría Cuantitativa

### 1a. Tamaños reales de los arrays (medidos desde `/var/lib/apollo/learned_state.json`)

| Subsistema | Estructura | Entries actuales | Cap máximo |
|---|---|---|---|
| OutcomeTracker.weights | HashMap<String, PatternWeight> | **228** | ~500 (GC a 5+ throttles) |
| OutcomeTracker.co_occurrence | Vec<(String,String,u32)> | **102** | 100-150 (GC pruning) |
| OutcomeTracker.experience_memory | VecDeque<ExperienceRecord> | **0** (vacío) | 300 |
| OutcomeTracker.hop_groups | HashMap<WorkloadHop, HopGroupWeight> | **5** | 6 (WorkloadHop variants) |
| SkillRegistry.skills | HashMap<String, OptimizationSkill> | **307** | no cap fijo |
| SpecialistAccuracy.accuracy | Vec<f64> | **4** | fijo |

**Observación crítica:** `experience_memory` está persistida en el JSON como `experience_records: []` — se vacía en cada restart. Los 112,049 throttles observados generan experience records en memoria durante la sesión, pero no sobreviven reinicio. Esto descalifica ANE/CoreML de entrada.

### 1b. Naturaleza de las operaciones de aprendizaje

**Las operaciones NO son full-array passes por ciclo.** Son actualizaciones event-driven:

- `OutcomeTracker.tick()`: drena outcomes pendientes con >30s de antigüedad → actualiza 0-N entries de `weights` (N = throttles que maduran en este ciclo). El único full-scan es el filtro `low_value_vs_baseline` al final, que itera los 228 entries.
- `SkillRegistry.record_result()`: actualiza 1 skill por llamada.
- `SkillRegistry.matching_skills()`: itera 307 skills para filtrar matching. Se llama 1x/ciclo en la rama de decisión.
- `CausalGraph.evaluate()`: procesa pending actions (queue, no array), actualiza 0-N edges.
- `HopGroupWeight.record()`: 1 entry por outcome (6 grupos max).

### 1c. Timing medido (Python ~10x más lento que Rust → dividir por ~10)

| Operación | Python µs/iter | Rust estimado µs |
|---|---|---|
| low_value scan (N=228) | 27.3 µs | ~2.7 µs |
| skills matching scan (N=307) | 21.3 µs | ~2.1 µs |
| co-occurrence sort (N=102) | 0.66 µs | ~0.07 µs |
| specialist accuracy EMA (N=4) | 0.30 µs | ~0.03 µs |
| **Total learning ops por ciclo** | | **~5-10 µs** |

**Ciclo del daemon:** mediana 62ms, P95 86ms.
**Porcentaje de ciclo en aprendizaje:** ~5-10 µs / 62,000 µs = **0.008% - 0.016%**

---

## 2. Threshold de Break-Even para vDSP

Para que `vDSP_vsma` (vectorized scalar-multiply-add) sea beneficioso sobre Rust escalar:

```
vDSP_time(N) ≈ setup_overhead + N × throughput_neon
scalar_time(N) ≈ N × scalar_ns

setup_overhead ≈ 1–3 µs (función call + alignment check + NEON dispatch)
throughput_neon ≈ 0.3 ns/element (f32, vectorized)
scalar_ns ≈ 2 ns/element (f64, branch, memory access)

Break-even: 2µs = N × (2ns - 0.3ns) = N × 1.7ns
→ N_break_even ≈ 1,176 elements
```

**Todos los arrays del daemon están por debajo de este umbral:**

```
228 entries  < 1,176  (weights)
307 entries  < 1,176  (skills)
102 entries  < 1,176  (co_occurrence)
0   entries  ≪ 1,176  (experience_memory — vacío)
4   entries  ≪ 1,176  (specialist_accuracy)
5   entries  ≪ 1,176  (hop_groups)
```

Aplicar vDSP en cualquiera de estos arrays **añadiría latencia**, no la reduciría.

---

## 3. Evaluación SoA (Struct of Arrays)

La misión propone refactorizar `HashMap<String, PatternWeight>` a arrays paralelos. Análisis:

**Costo de migración:**
- `PatternWeight` tiene 2 campos u32. SoA daría `Vec<u32>` × 2 + `Vec<String>` para índices.
- El acceso es hoy por nombre (`weights.get_mut(&process_name)`). SoA requeriría una lookup de índice (HashMap o binary search) + acceso al array. Sin ganancia de localidad porque el proceso de búsqueda ya domina.
- Las actualizaciones son individuales (1 proceso a la vez), no bulk.

**Conclusión SoA:** No aplica. SoA es beneficioso cuando se hacen operaciones bulk sobre todos los elementos. Aquí el patrón es `get_mut(name)` → update 1 campo → done. La HashMap es el acceso correcto para este patrón.

---

## 4. Evaluación ANE/CoreML

**Requisito mínimo:** ≥100 ejemplos por clase para un modelo de 3 inputs → 1 output.

**Estado actual:**
- `experience_memory`: **0 records** en disco (no persistida entre reinicios)
- Datos históricos: 112,049 throttles registrados en `weights`, pero sin features temporales (pressure_at_action, pressure_drop) — solo conteos agregados.
- Las features necesarias (process_category, pressure, workload_mode) → expected_pressure_drop sí están en `ExperienceRecord`, pero esos records no se acumulan entre reinicios.

**Prerequisito para ANE:** primero hay que **persistir experience_memory** correctamente (el campo existe en `OutcomeTrackerPersisted.experience_records` y el código de serialización está implementado, pero los records se vacían antes de persisistir — o la sesión actual es demasiado nueva). Con un daemon corriendo >30min debería acumular records. Verificar que `persist_improved()` se llame mientras el daemon tiene records en vuelo.

Incluso si se acumulan records, con 300 records máximos y 6+ categorías de proceso, el dataset es marginal para CoreML. Un modelo de 3 inputs con 100 ejemplos por clase necesita 600 registros mínimo — el doble del cap actual.

**Conclusión ANE:** No viable. Insuficiente training data. El prerequisito real es aumentar el cap de experience_memory o persistir journales de outcome para training offline.

---

## 5. Dónde SÍ valdría vectorización en el futuro

Si alguna de estas condiciones se cumple, revisar:

| Condición | Threshold | Acción |
|---|---|---|
| `weights` crece > 1,200 entries | N > 1,176 | SoA + vDSP para low_value scan |
| `experience_memory` crece > 1,200 records | N > 1,176 | vDSP para similarity query inner loop |
| Se añade un subsistema con array global de scores | N > 1,176 | vDSP directo |
| Se necesita batch scoring de skills | N > 1,176 | vDSP_vsma sobre success_rate array |

Para llegar a N=1,176 en `weights`, el daemon necesitaría throttlear ~1,176 procesos distintos sin GC. Con el GC actual (retiene entries con ≥5 throttles), en la práctica el cap efectivo está ~300-400. No llegará a 1,176.

---

## 6. Recomendación

**No implementar vectorización.** Los arrays son 5-8x menores que el break-even threshold de vDSP (~1,176 elementos). Las operaciones de aprendizaje consumen <0.016% del ciclo de 62ms. El overhead de setup de vDSP (~2µs) supera el costo total de las operaciones escalares actuales (~5-10µs total para todos los subsistemas).

**Oportunidad real (no vectorización, sino fix):** `experience_memory` debería estar acumulándose entre reinicios para que el sistema aprenda de sesión en sesión. Si no hay records en el JSON tras múltiples horas de daemon activo, hay un bug silencioso en el flujo de persist. Esto vale más que cualquier optimización SIMD.

---

## Apéndice: Datos de runtime

- Daemon activo (v0.7.0): `/usr/local/libexec/apollo-optimizerd`
- Ciclos completados: 1,213 (sesión actual)
- Ciclo mediano: 62ms, P95: 86ms
- Throttles aplicados (sesión): 636
- Throttles históricos totales (weights): 112,049
- Presión actual: 72.7%
