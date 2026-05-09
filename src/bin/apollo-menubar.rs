//! apollo-menubar — Monitor de Apollo en la barra de menu de macOS.
// objc 0.2 usa cfg(cargo-clippy) (sintaxis antigua de clippy) en sus macros.
// En Rust moderno esto genera unexpected_cfg; suprimimos el warning aquí.
#![allow(unexpected_cfgs)]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use apollo_engine::engine::protocol::{DaemonRequest, DaemonResponse};
use apollo_engine::engine::types::{DaemonStatus, OptimizationProfile, SafetyPolicy};

// ── IPC ──

fn send_request(req: DaemonRequest) -> Option<DaemonResponse> {
    for &path in &[
        "/var/run/apollo-optimizer.sock",
        "/tmp/apollo-optimizer.sock",
    ] {
        if let Ok(mut stream) = UnixStream::connect(path) {
            let payload = serde_json::to_string(&req).ok()?;
            writeln!(stream, "{}", payload).ok()?;
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            use std::io::Read;
            reader
                .by_ref()
                .take(10 * 1024 * 1024)
                .read_line(&mut line)
                .ok()?;
            return serde_json::from_str(&line).ok();
        }
    }
    None
}

fn fetch_status() -> Option<DaemonStatus> {
    match send_request(DaemonRequest::GetStatus)? {
        DaemonResponse::Status(s) => Some(s),
        _ => None,
    }
}

/// Abre una conexion de suscripcion persistente al daemon y llama `on_push` por cada StatusPush.
/// Devuelve cuando la conexion se cierra. El caller debe reintentar con backoff.
fn subscribe_loop(on_push: impl Fn(DaemonStatus)) -> bool {
    for &path in &[
        "/var/run/apollo-optimizer.sock",
        "/tmp/apollo-optimizer.sock",
    ] {
        let Ok(stream) = UnixStream::connect(path) else {
            continue;
        };
        let Ok(read_stream) = stream.try_clone() else {
            continue;
        };
        let mut write_stream = stream;

        let payload = serde_json::to_string(&DaemonRequest::Subscribe).unwrap_or_default();
        if writeln!(write_stream, "{}", payload).is_err() {
            continue;
        }

        let mut reader = BufReader::new(read_stream);

        // Leer ack inicial (DaemonResponse::Ok)
        let mut ack = String::new();
        if reader.read_line(&mut ack).is_err() {
            continue;
        }

        // Leer StatusPush indefinidamente
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => return true, // desconexion limpia, reintentar
                Ok(_) => {
                    if let Ok(DaemonResponse::StatusPush(status)) = serde_json::from_str(&line) {
                        on_push(status);
                    }
                }
            }
        }
    }
    false // no habia socket disponible
}

// ── Helpers ──

fn profile_name(p: OptimizationProfile) -> &'static str {
    match p {
        OptimizationProfile::BalancedRoot => "Balanced",
        OptimizationProfile::AggressiveRoot => "Aggressive",
        OptimizationProfile::SafeRoot => "Safe",
    }
}

fn fmt_bytes(b: u64) -> String {
    if b >= 1_073_741_824 {
        format!("{:.1} GB", b as f64 / 1_073_741_824.0)
    } else if b >= 1_048_576 {
        format!("{:.0} MB", b as f64 / 1_048_576.0)
    } else {
        format!("{:.0} KB", b as f64 / 1024.0)
    }
}

fn fmt_num(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{},{:03}", n / 1000, n % 1000)
    } else {
        n.to_string()
    }
}

/// Barra de progreso unicode. 10 bloques.
fn bar(ratio: f64) -> String {
    let n = (ratio.clamp(0.0, 1.0) * 10.0).round() as usize;
    format!("{}{}", "\u{2593}".repeat(n), "\u{2591}".repeat(10 - n))
}

fn time_ago(secs: u64) -> String {
    if secs < 60 {
        format!("hace {}s", secs)
    } else {
        format!("hace {}m", secs / 60)
    }
}

// ── Titulo en la barra ──

fn bar_title(status: &Option<DaemonStatus>) -> String {
    let Some(s) = status else {
        return "\u{1F680} Apollo".to_string();
    };
    let alert = if s.metrics.thermal_state == "critical"
        || s.metrics.thermal_state == "serious"
        || s.metrics.memory_pressure >= 0.85
    {
        " \u{26A0}" // ⚠ si hay problema real
    } else if s.kill_switch {
        " \u{23F8}" // ⏸ si está pausado
    } else {
        ""
    };
    format!("\u{1F680} Apollo{}", alert)
}

// ── Menu ──

#[cfg(target_os = "macos")]
fn build_menu(status: &Option<DaemonStatus>, updated_secs_ago: u64) -> tray_icon::menu::Menu {
    use tray_icon::menu::{Menu, MenuItem, PredefinedMenuItem, Submenu};

    let menu = Menu::new();
    let info = |text: &str| MenuItem::new(text, false, None);
    let sep = || PredefinedMenuItem::separator();

    // Cabecera
    let _ = menu.append(&info("Apollo Optimizer"));
    let _ = menu.append(&info(&format!(
        "Actualizado {}",
        time_ago(updated_secs_ago)
    )));
    let _ = menu.append(&sep());

    let Some(s) = status else {
        let _ = menu.append(&info("Daemon no conectado"));
        let _ = menu.append(&info("Ejecuta: apollo-optimizerd"));
        let _ = menu.append(&sep());
        let _ = menu.append(&PredefinedMenuItem::quit(Some("Salir")));
        return menu;
    };

    let m = &s.metrics;

    // ── VEREDICTO ──
    let veredicto = if m.thermal_state == "critical" {
        "🔴  Emergencia termica — throttling activo"
    } else if m.memory_pressure > 0.85 {
        "🔴  Presion de memoria critica"
    } else if apollo_engine::engine::safety::survival_mode_active_total(
        m.memory_pressure,
        m.swap_used_bytes,
        m.swap_total_bytes,
    ) {
        "🟠  Modo supervivencia activo"
    } else if m.memory_pressure > 0.60 {
        "🟡  Presion moderada — monitoreando"
    } else if s.last_blockers.len() > 5 {
        "🟡  Multiples bloqueadores detectados"
    } else {
        "🟢  Sistema optimizado"
    };
    let _ = menu.append(&info(veredicto));
    let _ = menu.append(&sep());

    // ── METRICAS CLAVE ──
    let mem_pct = m.memory_pressure * 100.0;
    let _ = menu.append(&info(&format!(
        "💾  RAM   {}  {:.0}%",
        bar(m.memory_pressure),
        mem_pct
    )));

    let swap_trend = if m.swap_delta_bps > 100.0 {
        "  📈 subiendo"
    } else if m.swap_delta_bps < -100.0 {
        "  📉 bajando"
    } else {
        ""
    };
    let _ = menu.append(&info(&format!(
        "💿  Swap  {}{}",
        fmt_bytes(m.swap_used_bytes),
        swap_trend
    )));

    if let Some(temp) = m.iokit_p_cluster_temp {
        let pkg_w = m
            .energy_package_watts
            .unwrap_or_else(|| m.iokit_package_watts.map(|w| w as f64).unwrap_or(0.0));
        let watts_str = if pkg_w > 0.0 {
            format!("  {:.1}W", pkg_w)
        } else {
            String::new()
        };
        let _ = menu.append(&info(&format!(
            "🌡  Temp  {}  {:.0}C{}",
            bar(temp as f64 / 110.0),
            temp,
            watts_str
        )));
    }

    let thermal_label = match m.thermal_state.as_str() {
        "critical" => "Critica",
        "serious" => "Seria",
        "moderate" | "fair" => "Moderada",
        _ => "Nominal",
    };
    let _ = menu.append(&info(&format!(
        "📊  Thermal: {}   Score: {:.2}",
        thermal_label, m.last_pressure_score
    )));
    let _ = menu.append(&sep());

    // ── WORKLOAD + APP ──
    if !m.current_workload.is_empty() {
        let _ = menu.append(&info(&format!(
            "🧠  Carga:  {} ({:.0}%)",
            m.current_workload,
            m.ml_confidence * 100.0
        )));
    }
    if let Some(app) = &m.foreground_app {
        let idle = if m.foreground_idle { "  💤 idle" } else { "" };
        let _ = menu.append(&info(&format!("🖥  App:    {}{}", app.name, idle)));
    }
    let _ = menu.append(&sep());

    // ── PERFIL ──
    let profile_mode = if s.override_active {
        "override"
    } else if s.auto_profile_enabled {
        "auto"
    } else {
        "fijo"
    };
    let _ = menu.append(&info(&format!(
        "⚡  Perfil:  {} ({})   Ciclos: {}",
        profile_name(s.effective_profile),
        profile_mode,
        fmt_num(m.cycles)
    )));
    let _ = menu.append(&sep());

    // ── ACCIONES PRINCIPALES ──
    // Submenu cambiar perfil
    let sub = Submenu::new("Cambiar perfil", true);
    let mark = |p: OptimizationProfile| if s.effective_profile == p { "* " } else { "  " };
    let _ = sub.append(&MenuItem::with_id(
        "profile-balanced",
        &format!("{}Balanced", mark(OptimizationProfile::BalancedRoot)),
        true,
        None,
    ));
    let _ = sub.append(&MenuItem::with_id(
        "profile-aggressive",
        &format!(
            "{}Aggressive  (20 min)",
            mark(OptimizationProfile::AggressiveRoot)
        ),
        true,
        None,
    ));
    let _ = sub.append(&MenuItem::with_id(
        "profile-safe",
        &format!("{}Safe  (20 min)", mark(OptimizationProfile::SafeRoot)),
        true,
        None,
    ));
    let _ = sub.append(&PredefinedMenuItem::separator());
    let auto_label = if s.auto_profile_enabled {
        "Auto-perfil: Activo"
    } else {
        "Auto-perfil: Inactivo"
    };
    let auto_id = if s.auto_profile_enabled {
        "auto-off"
    } else {
        "auto-on"
    };
    let _ = sub.append(&MenuItem::with_id(auto_id, auto_label, true, None));
    let _ = menu.append(&sub);

    // Submenu detalle tecnico (para power users)
    let detail = Submenu::new("Detalle tecnico", true);
    let di = |text: &str| MenuItem::new(text, false, None);

    let policy = SafetyPolicy::for_profile(s.effective_profile);
    let b = &m.budgets;
    let _ = detail.append(&di("🎯  Acciones del ciclo"));
    let _ = detail.append(&di(&format!(
        "   Boosts {}/{}  Throttles {}/{}  Freezes {}/{}",
        b.cycle_boosts,
        policy.max_boosts_per_cycle,
        b.cycle_throttles,
        policy.max_throttles_per_cycle,
        b.cycle_freezes,
        policy.max_freezes_per_cycle,
    )));
    let _ = detail.append(&PredefinedMenuItem::separator());

    let _ = detail.append(&di("🔴  Bloqueadores"));
    if s.last_blockers.is_empty() {
        let _ = detail.append(&di("   Sin bloqueadores activos"));
    } else {
        for (i, blk) in s.last_blockers.iter().take(5).enumerate() {
            let _ = detail.append(&di(&format!(
                "   {}. {} ({})  {:.2}",
                i + 1,
                blk.name,
                blk.pid,
                blk.score
            )));
        }
    }
    let _ = detail.append(&PredefinedMenuItem::separator());

    let _ = detail.append(&di("⚡  Reactor"));
    let _ = detail.append(&di(&format!(
        "   Modo: {}  Salud: {}  Pulsos: {}",
        m.reactor_mode,
        m.reactor_health,
        fmt_num(m.reactor_pulses)
    )));
    let _ = detail.append(&di(&format!(
        "   Eventos: {} (Mem:{} Therm:{} Spawn:{} Power:{})",
        fmt_num(m.reactor_events_total),
        m.reactor_events_mem,
        m.reactor_events_thermal,
        m.reactor_events_spawn,
        m.reactor_events_power
    )));
    let sentinel = if m.resource_interrupt_active {
        "🚨 ACTIVO"
    } else {
        "idle"
    };
    let _ = detail.append(&di(&format!(
        "   Sentinel: {}  Fires: {}",
        sentinel,
        fmt_num(m.resource_interrupts_total)
    )));
    let _ = detail.append(&PredefinedMenuItem::separator());

    let _ = detail.append(&di("⚡  Energia"));
    let pkg_w = m
        .energy_package_watts
        .unwrap_or_else(|| m.iokit_package_watts.map(|w| w as f64).unwrap_or(0.0));
    let _ = detail.append(&di(&format!(
        "   Paquete: {:.1}W  CPU: {:.1}W  GPU: {:.1}W",
        pkg_w,
        m.energy_cpu_watts.unwrap_or(0.0),
        m.energy_gpu_watts.unwrap_or(0.0)
    )));
    let fallback_wh = m.throttles_applied as f64 * 0.003 + m.freezes_applied as f64 * 0.005;
    let wh = m.energy_savings_wh.unwrap_or(fallback_wh);
    let co2 = m.energy_co2_avoided_g.unwrap_or(wh * 0.075);
    let _ = detail.append(&di(&format!(
        "   🌱 Ahorrado: {:.2} Wh  CO2: {:.2}g",
        wh, co2
    )));
    let _ = detail.append(&PredefinedMenuItem::separator());

    let _ = detail.append(&di("📈  Sesion total"));
    let _ = detail.append(&di(&format!(
        "   Ciclos: {}  Switches: {}  🧟 Zombies: {}  Kills: {}",
        fmt_num(m.cycles),
        m.profile_switches,
        m.zombies_detected,
        m.kills_applied
    )));
    let _ = detail.append(&di(&format!(
        "   Boosts: {}  Throttles: {}  Freezes: {}",
        fmt_num(m.boosts_applied),
        fmt_num(m.throttles_applied),
        fmt_num(m.freezes_applied)
    )));

    if let Some(llm) = &s.llm {
        let _ = detail.append(&PredefinedMenuItem::separator());
        let estado = if llm.enabled && llm.training_active {
            "🟢 Activo"
        } else if llm.enabled {
            "🟡 Standby"
        } else {
            "⬜ Desactivado"
        };
        let _ = detail.append(&di(&format!(
            "🤖  LLM Teacher: {}  Budget: {}/{}",
            estado, llm.daily_budget_remaining, llm.daily_budget
        )));
        let _ = detail.append(&di(&format!(
            "   Patrones: {} inter / {} ruido",
            llm.learned_policy.interactive_patterns, llm.learned_policy.noise_patterns
        )));
    }

    let _ = menu.append(&detail);
    let _ = menu.append(&sep());

    // Pausar / Reanudar
    if s.kill_switch {
        let _ = menu.append(&MenuItem::with_id(
            "resume",
            "▶  Reanudar optimizacion",
            true,
            None,
        ));
    } else {
        let _ = menu.append(&MenuItem::with_id(
            "pause",
            "⏸  Pausar optimizacion",
            true,
            None,
        ));
    }
    let _ = menu.append(&sep());
    let _ = menu.append(&PredefinedMenuItem::quit(Some("Salir")));

    menu
}

// ── Clicks ──

fn handle_click(id: &str) {
    match id {
        "profile-balanced" => {
            send_request(DaemonRequest::SetProfile {
                profile: OptimizationProfile::BalancedRoot,
                ttl_minutes: None,
            });
        }
        "profile-aggressive" => {
            send_request(DaemonRequest::SetProfile {
                profile: OptimizationProfile::AggressiveRoot,
                ttl_minutes: Some(20),
            });
        }
        "profile-safe" => {
            send_request(DaemonRequest::SetProfile {
                profile: OptimizationProfile::SafeRoot,
                ttl_minutes: Some(20),
            });
        }
        "auto-on" => {
            send_request(DaemonRequest::SetAutoProfile { enabled: true });
        }
        "auto-off" => {
            send_request(DaemonRequest::SetAutoProfile { enabled: false });
        }
        "pause" => {
            for p in ["/var/run/apollo.disable", "/tmp/apollo.disable"] {
                let _ = std::fs::File::create(p);
            }
        }
        "resume" => {
            for p in ["/var/run/apollo.disable", "/tmp/apollo.disable"] {
                let _ = std::fs::remove_file(p);
            }
        }
        _ => {}
    }
}

/// Encuentra el NSStatusBarButton de nuestro item y elimina el espacio reservado para icono.
/// Usa la propiedad privada `_statusItems` via KVC — estable en macOS 10.12+.
#[cfg(target_os = "macos")]
unsafe fn compact_tray_button() {
    use cocoa::base::{id, nil};
    use cocoa::foundation::NSString;
    #[allow(unused_imports)]
    use objc::{class, msg_send, sel, sel_impl}; // sel/sel_impl son necesarios para que msg_send! expanda

    let bar: id = msg_send![class!(NSStatusBar), systemStatusBar];
    let key: id = NSString::alloc(nil).init_str("_statusItems");
    let items: id = msg_send![bar, valueForKey: key];
    if items.is_null() {
        return;
    }
    let count: usize = msg_send![items, count];
    for i in 0..count {
        // _statusItems es un NSPointerArray, se accede con pointerAtIndex:
        let item_ptr: *mut std::ffi::c_void = msg_send![items, pointerAtIndex: i];
        if item_ptr.is_null() {
            continue;
        }
        let item: id = item_ptr as id;
        let button: id = msg_send![item, button];
        if button.is_null() {
            continue;
        }
        // NSNoImage = 0: elimina el area reservada para el icono en el boton
        let _: () = msg_send![button, setImagePosition: 0usize];
    }
}

// ── Main ──

/// Evento personalizado para despertar el event loop desde el hilo de fondo.
#[derive(Debug)]
enum AppEvent {
    /// El hilo de fondo acaba de obtener datos frescos del daemon.
    StatusUpdated,
}

#[cfg(target_os = "macos")]
fn main() {
    use tray_icon::{menu::MenuEvent, TrayIconBuilder};
    use winit::{
        event::Event,
        event_loop::{ControlFlow, EventLoopBuilder},
        platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS},
    };

    // Single-instance lock: si ya hay una instancia corriendo, salir silenciosamente.
    let lock_path = "/tmp/apollo-menubar.lock";
    let lock_file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(lock_path)
        .expect("lock file");
    use std::os::unix::io::AsRawFd;
    let locked = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if locked != 0 {
        eprintln!("apollo-menubar: ya hay una instancia corriendo, saliendo.");
        return;
    }

    struct AppData {
        status: Option<DaemonStatus>,
        last_fetched: Instant,
    }

    let data = Arc::new(Mutex::new(AppData {
        status: fetch_status(),
        last_fetched: Instant::now(),
    }));
    let data_bg = Arc::clone(&data);

    // event_loop antes del hilo para poder crear el proxy
    let event_loop = EventLoopBuilder::<AppEvent>::with_user_event()
        .with_activation_policy(ActivationPolicy::Accessory)
        .build()
        .expect("event loop");

    // proxy: permite que el hilo de fondo despierte el event loop al tener datos nuevos
    let proxy = event_loop.create_proxy();

    // Hilo de suscripcion push: recibe StatusPush del daemon sin polling
    thread::spawn(move || {
        // Fetch inicial para tener datos antes de que llegue el primer push
        if let Some(s) = fetch_status() {
            if let Ok(mut d) = data_bg.lock() {
                d.status = Some(s);
                d.last_fetched = Instant::now();
            }
            let _ = proxy.send_event(AppEvent::StatusUpdated);
        }

        loop {
            let proxy2 = proxy.clone();
            let data2 = Arc::clone(&data_bg);
            let connected = subscribe_loop(move |status| {
                if let Ok(mut d) = data2.lock() {
                    d.status = Some(status);
                    d.last_fetched = Instant::now();
                }
                let _ = proxy2.send_event(AppEvent::StatusUpdated);
            });
            if !connected {
                // Daemon no disponible; reintentar cada 2s
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        }
    });

    let (init_status, init_secs) = {
        let d = data.lock().unwrap_or_else(|e| e.into_inner());
        (d.status.clone(), d.last_fetched.elapsed().as_secs())
    };

    let tray = TrayIconBuilder::new()
        .with_menu(Box::new(build_menu(&init_status, init_secs)))
        .with_title(&bar_title(&init_status))
        .with_tooltip("Apollo Optimizer")
        .build()
        .expect("tray icon");

    // Elimina el espacio invisible que NSStatusBarButton reserva para el icono
    unsafe {
        compact_tray_button();
    }

    // Momento en que el usuario abrio el menu (click en la barra).
    // Mientras el menu podria estar abierto, no llamamos set_menu() para no cerrarlo.
    let mut menu_opened_at: Option<Instant> = None;

    let _ = event_loop.run(move |event, elwt| {
        // ── Observer: datos nuevos listos ────────────────────────────────────
        if let Event::UserEvent(AppEvent::StatusUpdated) = &event {
            let d = data.lock().unwrap_or_else(|e| e.into_inner());
            let secs_ago = d.last_fetched.elapsed().as_secs();

            // El titulo se puede actualizar siempre — no cierra el menu
            let _ = tray.set_title(Some(&bar_title(&d.status)));

            // set_menu() cierra el popup si esta abierto; solo llamarlo cuando esta cerrado
            let menu_might_be_open = menu_opened_at
                .map(|t| t.elapsed() < Duration::from_secs(5))
                .unwrap_or(false);
            if !menu_might_be_open {
                let _ = tray.set_menu(Some(Box::new(build_menu(&d.status, secs_ago))));
            }
        }

        // Seguimos sondeando eventos de tray/menu cada 100 ms (son canales externos)
        elwt.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(100),
        ));

        // TrayIconEvent::Click = usuario abre el menu
        use tray_icon::TrayIconEvent;
        while TrayIconEvent::receiver().try_recv().is_ok() {
            menu_opened_at = Some(Instant::now());
        }

        // MenuEvent = usuario hizo click en un item → menu ya se cerro
        while let Ok(ev) = MenuEvent::receiver().try_recv() {
            handle_click(ev.id.0.as_str());
            menu_opened_at = None;
        }

        // Menu cerrado sin click (dismiss): detectar por timeout de 3s y reconstruir con datos frescos
        // Garantiza que la proxima apertura siempre muestre datos actuales
        if let Some(opened) = menu_opened_at {
            if opened.elapsed() > Duration::from_secs(3) {
                menu_opened_at = None;
                if let Some(s) = fetch_status() {
                    if let Ok(mut d) = data.lock() {
                        d.status = Some(s);
                        d.last_fetched = Instant::now();
                    }
                }
                let d = data.lock().unwrap_or_else(|e| e.into_inner());
                let _ = tray.set_title(Some(&bar_title(&d.status)));
                let _ = tray.set_menu(Some(Box::new(build_menu(&d.status, 0))));
            }
        }
    });
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("apollo-menubar solo esta disponible en macOS");
    std::process::exit(1);
}
