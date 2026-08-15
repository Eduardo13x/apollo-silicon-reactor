# Recuperar el pragmatismo de Apollo

## Objetivo

Recuperar una ruta inmediata y pragmatica para aceleraciones reversibles sin
debilitar la seguridad, el rollback ni la deliberacion profunda de Apollo.

## Contrato global

- La ruta refleja solo puede producir QoS interactivo, ajuste reversible de
  nice, liberacion de restricciones de I/O, boost temporal y prewarm Markov.
- Freeze, throttle, purge, sysctl y memorystatus permanecen en la ruta
  deliberativa existente.
- Reutilizar validacion de seguridad, identidad de PID, effect ledger, TTL y
  rollback existentes. No introducir actuadores privilegiados nuevos.
- Reglas deterministas pueden proponer desde el arranque. World Model, NARS,
  Markov, MPC y causal solo ajustan o vetan con evidencia local confiable y
  decisiva; ausencia o incertidumbre son neutrales.
- El razonamiento profundo usa snapshots inmutables, resultados versionados y
  colas latest-wins acotadas. Resultados con mas de dos ciclos o identidad
  distinta se descartan y nunca bloquean el ciclo principal.
- M1 conserva perfil secuencial; M4 usa adaptive-multicore. El deploy oficial
  rechaza artefactos incompatibles con el hardware detectado.
- La telemetria distingue soporte de modelos de propuestas/admisiones/acciones
  reales y conserva compatibilidad de serializacion.
- La ruta inicia en sombra y solo se activa automaticamente tras 500 ciclos si
  no hay violaciones de seguridad o rollback, p95 <= 75 ms, regresion p95 <=
  10%, churn <= 10% y el perfil compilado coincide con el esperado.
- Si un criterio falla, permanece en sombra y publica el bloqueo exacto. Debe
  existir un interruptor de configuracion para deshabilitarla sin reinstalar.

## Tareas

1. Congelar matriz de escenarios y contratos publicos a partir de auditorias.
2. Implementar ReflexBroker e intents tipados con TDD.
3. Implementar snapshot/worker latest-wins y arbitraje unico con TDD.
4. Corregir artefactos y despliegue por perfil de hardware con TDD.
5. Integrar metricas, dashboard y activacion sombra con TDD.
6. Ejecutar revision adversarial consolidada y un solo pase de correccion.
7. Ejecutar verificacion integrada release; preparar despliegue por separado.

## Compatibilidad y limites

- Todos los campos persistidos nuevos usan defaults Serde.
- No resetear ni descartar cambios locales previos.
- Una sola lane de Cargo; ningun agente puede ejecutar Cargo en paralelo.
- Ninguna afirmacion de cierre sin evidencia mecanica fresca.
