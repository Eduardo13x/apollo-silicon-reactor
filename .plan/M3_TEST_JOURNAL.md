# PR #6 — M3b: Tests journal.rs + lock_ext.rs + capabilities.rs

**Rama**: `test/m3-io-safety`
**Base**: `main` post-PR #5 (o independiente, no hay dependencia real)
**Deuda resuelta**: DEBT-008 (parcial — 3 módulos más), DEBT-009 (parcial)
**Riesgo daemon**: NINGUNO — solo se agregan tests y un dev-dependency
**Archivos tocados**: `Cargo.toml`, `src/engine/journal.rs`, `src/engine/lock_ext.rs`, `src/engine/capabilities.rs`

---

## Contexto

### ¿Por qué estos tres módulos juntos?

- `journal.rs`: seguridad de I/O — tiene lógica de protección contra symlinks que merece un test dedicado de seguridad
- `lock_ext.rs`: resilencia del daemon — el trait `LockRecover` es lo que previene que un panic en un thread derribe el daemon; necesita un test de envenenamiento
- `capabilities.rs`: detección en runtime — función pura que debería ser trivial de testear

### Código real de journal.rs (75 líneas — completo, del audit)

```rust
// Funciones públicas:
pub fn append_journal(path: &Path, entry: &JournalEntry) -> anyhow::Result<()>
pub fn read_journal(path: &Path) -> anyhow::Result<Vec<JournalEntry>>

// Constante:
const MAX_JOURNAL_BYTES: u64 = 10 * 1024 * 1024;  // 10 MB

// Comportamiento:
// 1. append_journal: rechaza symlinks (path Y parent), rota si > 10MB, escribe JSONL
// 2. read_journal: lee JSONL, ignora líneas malformadas, devuelve Vec<JournalEntry>
```

### Código real de lock_ext.rs (42 líneas — completo, del audit)

```rust
pub trait LockRecover<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T>;
}
impl<T> LockRecover<T> for Mutex<T> { /* unwrap_or_else(|e| e.into_inner()) */ }

pub trait RwLockRecover<T> {
    fn read_recover(&self) -> RwLockReadGuard<'_, T>;
    fn write_recover(&self) -> RwLockWriteGuard<'_, T>;
}
impl<T> RwLockRecover<T> for RwLock<T> { /* misma estrategia */ }
```

### Código real de capabilities.rs (41 líneas — completo, del audit)

```rust
pub fn detect_capabilities() -> CapabilityReport {
    // can_taskpolicy: cfg!(target_os = "macos")
    // can_sysctl: sysctl_direct::exists("kern.ostype")
    // can_mdutil: Path::new("/usr/bin/mdutil").exists()
    // can_tmutil: Path::new("/usr/bin/tmutil").exists()
    // is_root: libc::geteuid() == 0
}
```

---

## Commit A — Dev-dependency + Tests journal.rs

**Commit message**: `test(journal): symlink protection, rotation, roundtrip + add tempfile dev-dep`

### Parte 1: Agregar tempfile a Cargo.toml

`journal.rs` necesita crear archivos temporales en tests. `tempfile` crea y limpia automáticamente.

**En `Cargo.toml`**, agregar sección `[dev-dependencies]`:

```toml
[dev-dependencies]
tempfile = "3"
```

> **Verificar primero** que no existe ya:
> ```bash
> grep -n "dev-dependencies\|tempfile" Cargo.toml
> ```

### Parte 2: Agregar tests a `src/engine/journal.rs`

Al final del archivo (después de `read_journal`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::types::JournalEntry;
    use std::fs;
    use tempfile::TempDir;

    fn make_entry(action: &str) -> JournalEntry {
        // Construir un JournalEntry válido para tests.
        // Verificar los campos de JournalEntry con:
        // grep -A20 "struct JournalEntry" src/engine/types.rs
        //
        // Patrón probable:
        JournalEntry {
            timestamp: chrono::Utc::now(),
            action: action.to_string(),
            // ... otros campos con valores por defecto
            ..Default::default()  // si implementa Default
        }
        // Si no implementa Default, construir explícitamente.
    }

    // ── Test 1: Roundtrip append + read ─────────────────────────────────

    #[test]
    fn test_append_and_read_roundtrip() {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("test.jsonl");

        let entry = make_entry("freeze_process");

        append_journal(&path, &entry).expect("append should succeed");

        let entries = read_journal(&path).expect("read should succeed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].action, "freeze_process");
    }

    // ── Test 2: Múltiples entries ────────────────────────────────────────

    #[test]
    fn test_append_multiple_entries() {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("test.jsonl");

        for i in 0..5 {
            let entry = make_entry(&format!("action_{}", i));
            append_journal(&path, &entry).expect("append");
        }

        let entries = read_journal(&path).expect("read");
        assert_eq!(entries.len(), 5);
    }

    // ── Test 3: read_journal en archivo inexistente devuelve Vec vacío ──

    #[test]
    fn test_read_nonexistent_returns_empty() {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("nonexistent.jsonl");

        let entries = read_journal(&path).expect("should not error on missing file");
        assert!(entries.is_empty());
    }

    // ── Test 4: Rotación cuando supera MAX_JOURNAL_BYTES ─────────────────

    #[test]
    fn test_rotation_triggers_at_size_limit() {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("big.jsonl");

        // Escribir un entry pequeño primero para crear el archivo
        let entry = make_entry("initial");
        append_journal(&path, &entry).expect("initial write");

        // Rellenar el archivo hasta superar MAX_JOURNAL_BYTES (10MB)
        // Usamos fs::write para sobreescribir con un archivo grande
        let big_content = "x".repeat(11 * 1024 * 1024); // 11 MB
        fs::write(&path, &big_content).expect("write big file");

        // El siguiente append debe activar la rotación
        let new_entry = make_entry("after_rotation");
        append_journal(&path, &new_entry).expect("append after threshold");

        // Verificar que se creó el archivo rotado
        let rotated = path.with_extension("jsonl.1");
        assert!(
            rotated.exists(),
            "Rotated file should exist at {:?}",
            rotated
        );

        // El archivo principal debe ser pequeño (solo el nuevo entry)
        let main_size = fs::metadata(&path).expect("metadata").len();
        assert!(
            main_size < 1024 * 1024,
            "After rotation, main journal should be small, got {} bytes",
            main_size
        );
    }

    // ── Test 5: Rechazo de symlink en el path ────────────────────────────

    #[test]
    fn test_append_rejects_symlink_path() {
        let dir = TempDir::new().expect("create temp dir");
        let real_file = dir.path().join("real.jsonl");
        let symlink_path = dir.path().join("symlink.jsonl");

        // Crear archivo real
        fs::write(&real_file, "").expect("create real file");

        // Crear symlink apuntando al archivo real
        std::os::unix::fs::symlink(&real_file, &symlink_path)
            .expect("create symlink");

        // append_journal debe rechazar escribir a través de un symlink
        let entry = make_entry("should_fail");
        let result = append_journal(&symlink_path, &entry);

        assert!(
            result.is_err(),
            "append_journal should reject symlink path"
        );
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("symlink"),
            "Error message should mention symlink, got: {}",
            err_msg
        );
    }

    // ── Test 6: Líneas malformadas en read_journal son ignoradas ─────────

    #[test]
    fn test_read_journal_ignores_malformed_lines() {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("mixed.jsonl");

        // Crear un archivo con una línea válida, una inválida, y otra válida
        let valid_entry = make_entry("valid_action");
        let valid_json = serde_json::to_string(&valid_entry).expect("serialize");

        let content = format!("{}\nnot valid json {{{{{\n{}\n", valid_json, valid_json);
        fs::write(&path, content).expect("write mixed file");

        // read_journal debe devolver solo las 2 líneas válidas (ignora malformadas)
        let entries = read_journal(&path).expect("read");
        assert_eq!(
            entries.len(),
            2,
            "Should have 2 valid entries, ignoring the malformed line"
        );
    }
}
```

> **Verificar antes de implementar**:
> ```bash
> # Campos de JournalEntry:
> grep -A15 "struct JournalEntry" src/engine/types.rs
>
> # ¿Tiene Default?
> grep -B2 "struct JournalEntry" src/engine/types.rs
> ```

---

## Commit B — Tests lock_ext.rs + capabilities.rs

**Commit message**: `test(lock_ext+capabilities): poison recovery and capability detection`

### Tests para `src/engine/lock_ext.rs`

Al final del archivo:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, RwLock};
    use std::thread;

    // ── LockRecover — mutex normal ──────────────────────────────────────

    #[test]
    fn test_lock_recover_non_poisoned_mutex() {
        let m = Mutex::new(42u32);
        let guard = m.lock_recover();
        assert_eq!(*guard, 42);
    }

    // ── LockRecover — mutex envenenado ──────────────────────────────────

    #[test]
    fn test_lock_recover_poisoned_mutex_does_not_panic() {
        let m = Arc::new(Mutex::new(100u32));
        let m_clone = Arc::clone(&m);

        // Envenenar el mutex: un thread hace panic mientras lo tiene bloqueado
        let handle = thread::spawn(move || {
            let _guard = m_clone.lock().expect("lock in spawned thread");
            panic!("intentional panic to poison the mutex");
        });

        // El join falla (el thread hizo panic) — eso es esperado
        let _ = handle.join();

        // El mutex está envenenado. lock_recover() debe recuperar el valor sin panic:
        let guard = m.lock_recover(); // ← NO debe hacer panic
        assert_eq!(*guard, 100, "Value should be preserved despite poison");
    }

    // ── RwLockRecover — read normal ──────────────────────────────────────

    #[test]
    fn test_read_recover_non_poisoned() {
        let rw = RwLock::new("hello");
        let guard = rw.read_recover();
        assert_eq!(*guard, "hello");
    }

    // ── RwLockRecover — write normal ─────────────────────────────────────

    #[test]
    fn test_write_recover_non_poisoned() {
        let rw = RwLock::new(vec![1u32, 2, 3]);
        {
            let mut guard = rw.write_recover();
            guard.push(4);
        }
        let guard = rw.read_recover();
        assert_eq!(*guard, vec![1, 2, 3, 4]);
    }

    // ── RwLockRecover — write envenenado ─────────────────────────────────

    #[test]
    fn test_write_recover_poisoned_rwlock_does_not_panic() {
        let rw = Arc::new(RwLock::new(0u32));
        let rw_clone = Arc::clone(&rw);

        // Envenenar con write lock
        let handle = thread::spawn(move || {
            let _guard = rw_clone.write().expect("write in spawned thread");
            panic!("intentional panic to poison rwlock");
        });

        let _ = handle.join();

        // write_recover() debe recuperar sin panic:
        let mut guard = rw.write_recover();
        *guard = 999;
        assert_eq!(*guard, 999);
    }
}
```

### Tests para `src/engine/capabilities.rs`

Al final del archivo:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // ── detect_capabilities() no debe hacer panic ────────────────────────

    #[test]
    fn test_detect_capabilities_does_not_panic() {
        // En macOS, siempre debe retornar sin panic
        let report = detect_capabilities();
        // Si llegamos aquí, no hubo panic
        let _ = report;
    }

    // ── En macOS, taskpolicy siempre disponible ───────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn test_can_taskpolicy_is_true_on_macos() {
        let report = detect_capabilities();
        assert!(
            report.can_taskpolicy,
            "taskpolicy should always be available on macOS"
        );
    }

    // ── sysctl debe estar disponible en macOS ─────────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn test_can_sysctl_is_true_on_macos() {
        let report = detect_capabilities();
        assert!(
            report.can_sysctl,
            "sysctl should be available on macOS (kern.ostype exists)"
        );
    }

    // ── /usr/bin/mdutil existe en macOS estándar ──────────────────────────

    #[test]
    #[cfg(target_os = "macos")]
    fn test_can_mdutil_on_standard_macos() {
        let report = detect_capabilities();
        assert!(
            report.can_mdutil,
            "/usr/bin/mdutil should exist on standard macOS"
        );
    }

    // ── El report siempre tiene todos los campos poblados ────────────────

    #[test]
    fn test_report_fields_are_populated() {
        let report = detect_capabilities();
        // Verificar que unavailable es un Vec (no None u otro error)
        // is_root es bool, no puede ser "no inicializado"
        // Solo verificamos que el struct se construyó correctamente
        let _: bool = report.is_root;
        let _: bool = report.can_taskpolicy;
        let _: bool = report.can_sysctl;
        let _: Vec<String> = report.unavailable;
    }

    // ── unavailable solo contiene features que realmente no están ────────

    #[test]
    #[cfg(target_os = "macos")]
    fn test_unavailable_does_not_contain_taskpolicy_on_macos() {
        let report = detect_capabilities();
        assert!(
            !report.unavailable.contains(&"taskpolicy".to_string()),
            "taskpolicy should not be in unavailable on macOS"
        );
    }
}
```

> **Verificar estructura de CapabilityReport**:
> ```bash
> grep -A15 "struct CapabilityReport" src/engine/types.rs
> ```

---

## PR #6 — Descripción completa

```markdown
## Summary

Agrega cobertura de tests a tres módulos de infraestructura crítica:
`journal.rs` (seguridad I/O), `lock_ext.rs` (resilencia del daemon),
`capabilities.rs` (detección de capacidades en runtime).

### Commit A — tempfile + journal tests (6 tests)
- Cargo.toml: agrega `tempfile = "3"` a [dev-dependencies]
- journal.rs: roundtrip append/read, múltiples entries, file inexistente,
  rotación a los 10MB, rechazo de symlinks, líneas malformadas ignoradas

### Commit B — lock_ext + capabilities tests (9 tests)
- lock_ext.rs: Mutex normal, Mutex envenenado (no panic), RwLock read/write,
  RwLock write envenenado (no panic)
- capabilities.rs: no panic, taskpolicy en macOS, sysctl en macOS,
  mdutil en macOS, campos poblados, unavailable correcto

### Por qué el test de symlink es importante
El journal escribe a `/var/lib/apollo/journal.jsonl` cuando corre como root.
Un atacante con acceso limitado podría crear un symlink en esa ruta y redirigir
las escrituras. El test verifica que `append_journal` detecta y rechaza esto.

### Por qué el test de envenenamiento es importante
Si cualquier thread del daemon hace panic mientras sostiene un lock (ej: un
bug en socket_handler), el mutex queda "envenenado". Sin `lock_recover()`,
la próxima adquisición del lock haría panic en el daemon principal también,
causando una caída en cascada. El test verifica que la recuperación funciona.

## Test plan
- [ ] `cargo test` — todos los 6+9 tests nuevos pasan
- [ ] `cargo test --lib` — sin regresiones
- [ ] El test de symlink crea y limpia el symlink correctamente (TempDir hace cleanup)
- [ ] El test de poison spawn un thread, lo deja crashear, y verifica recovery
```

---

## Checklist antes de mergear PR #6

- [ ] Verificar campos de `JournalEntry` con grep antes de `make_entry()`
- [ ] Verificar que `JournalEntry` tiene `action: String` o ajustar
- [ ] Test de rotación: verificar que `path.with_extension("jsonl.1")` es correcto para el nombre del rotated file (leer línea 42 de journal.rs: `let rotated = path.with_extension("jsonl.1")`)
- [ ] Test de symlink: solo funciona en Unix — ya está marcado para macOS implícitamente
- [ ] Poison test: el `join()` debe ser `let _ = handle.join()` (ignorar el error del panic)
- [ ] Capabilities tests marcados con `#[cfg(target_os = "macos")]` donde aplica
- [ ] `cargo test` sin regresiones
