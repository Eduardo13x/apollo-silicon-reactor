use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};

mod dashboard;
mod perceptual_doctor;
mod whisper;

use anyhow::Context;
use apollo_engine::engine::protocol::{DaemonRequest, DaemonResponse, PROTOCOL_VERSION};
use apollo_engine::engine::types::{LatencyTarget, OptimizationProfile};
use clap::{Parser, Subcommand};

static VERSION_CHECKED: AtomicBool = AtomicBool::new(false);

fn socket_candidates() -> [&'static str; 2] {
    [
        "/var/run/apollo-optimizer.sock",
        "/tmp/apollo-optimizer.sock",
    ]
}

#[derive(Parser)]
#[command(name = "apollo-optimizerctl")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Visual dashboard with system status, gauges, and verdict
    Dashboard,
    Status,
    Metrics,
    TopBlockers,
    ProfileTimeline,
    Doctor,
    Capabilities,
    Restore,
    PanicRestore,
    /// Trigger an immediate maintenance purge through the daemon.
    /// Rate-limited to 5 minutes between successive invocations.
    Purge,
    /// Pause all optimization (creates kill switch file)
    Pause,
    /// Resume optimization (removes kill switch file)
    Resume,
    /// Check if Apollo is paused
    IsPaused,
    SetAutoProfile {
        #[arg(value_parser = ["on", "off"])]
        enabled: String,
    },
    ClearProfileOverride,
    SetProfile {
        #[arg(value_parser = ["balanced-root", "aggressive-root", "safe-root"])]
        profile: String,
        #[arg(long, default_value_t = 20)]
        ttl_minutes: u64,
    },
    SetLatencyTarget {
        #[arg(value_parser = ["low", "normal", "max"])]
        target: String,
    },
    DumpPolicy,
    /// Turn the MarkovPrewarm micro-canary on or off.
    ///
    /// Enabling it starts B: from the next eligible opportunity the daemon may
    /// withhold a real pre-warm as an experimental control. At most one in a
    /// hundred eligible opportunities, one open experiment at a time, and only
    /// MarkovPrewarm. Withholding a pre-warm removes speculative work rather
    /// than adding any.
    ///
    /// Disabling stops the sampling and lets open experiments drain, so an arm
    /// already withheld still reaches its pair.
    Canary {
        #[arg(value_parser = ["on", "off"])]
        state: String,
    },
    Feedback {
        #[arg(value_parser = ["good", "bad"])]
        rating: String,
        #[arg(long)]
        note: Option<String>,
    },
    Usage {
        #[command(subcommand)]
        command: UsageCommands,
    },
    /// Show reactive sysctl governor status
    SysctlGovernor,
    /// Revert all sysctl changes made by the daemon to their startup defaults
    RevertSysctls,
    /// Show daemon protocol version and build info
    Version,
    /// Show circuit breaker and degradation health summary
    Health,
    /// Sprint patch (2026-06-05) — S5. Single-glyph status snippet read
    /// directly from `runtime_metrics.json`. No daemon RPC; safe to embed
    /// in shell prompts. Silent on stale (>5s) snapshots.
    Whisper {
        #[arg(long)]
        always_on: bool,
    },
    /// Diagnose the Perceptual Interaction Layer hop by hop. Reports which
    /// component is responsible for missing data instead of a bare zero.
    PerceptualDoctor,
}

#[derive(Subcommand)]
enum UsageCommands {
    Top {
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    Explain {
        name: String,
    },
}

fn to_profile(s: &str) -> OptimizationProfile {
    match s {
        "aggressive-root" => OptimizationProfile::AggressiveRoot,
        "safe-root" => OptimizationProfile::SafeRoot,
        _ => OptimizationProfile::BalancedRoot,
    }
}

fn to_latency_target(s: &str) -> LatencyTarget {
    match s {
        "max" => LatencyTarget::Max,
        "low" => LatencyTarget::Low,
        _ => LatencyTarget::Normal,
    }
}

/// Low-level send: connects, sends request, reads one response line.
/// Does NOT trigger version checking — used by `check_version_once()` to avoid recursion.
fn send_raw(req: DaemonRequest) -> anyhow::Result<DaemonResponse> {
    let mut stream = None;
    for path in socket_candidates() {
        if let Ok(s) = UnixStream::connect(path) {
            stream = Some(s);
            break;
        }
    }
    let mut stream = stream.context("cannot connect to daemon socket")?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(5)))?;
    let payload = serde_json::to_string(&req)?;
    writeln!(stream, "{}", payload)?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    const MAX_RESPONSE_BYTES: u64 = 10 * 1024 * 1024; // 10MB limit for daemon response
    use std::io::Read;
    reader
        .by_ref()
        .take(MAX_RESPONSE_BYTES)
        .read_line(&mut line)?;
    let response = serde_json::from_str::<DaemonResponse>(&line)?;
    Ok(response)
}

/// Run a one-time protocol version check before the first RPC in a process lifetime.
/// Warns on mismatch but never exits — preserves backward compatibility.
fn check_version_once() {
    // Swap returns the previous value; if it was already true we've checked before.
    if VERSION_CHECKED.swap(true, Ordering::Relaxed) {
        return;
    }
    // Use send_raw to avoid recursive call through send_request.
    let resp = match send_raw(DaemonRequest::GetVersion) {
        Ok(r) => r,
        Err(_) => return, // Daemon unreachable — the real command will surface a better error.
    };
    if let DaemonResponse::VersionInfo { protocol, .. } = resp {
        if protocol != PROTOCOL_VERSION {
            eprintln!(
                "Warning: protocol version mismatch — daemon={protocol}, ctl={PROTOCOL_VERSION}"
            );
            if protocol > PROTOCOL_VERSION {
                eprintln!(
                    "  The daemon is newer. Some commands may fail. Consider updating apollo-optimizerctl."
                );
            }
        }
    }
}

fn send_request(req: DaemonRequest) -> anyhow::Result<DaemonResponse> {
    check_version_once();
    send_raw(req)
}

fn handle_dashboard() -> anyhow::Result<()> {
    let response = send_request(DaemonRequest::GetStatus)
        .context("No se pudo conectar al daemon. ¿Está corriendo apollo-optimizerd?")?;
    match response {
        DaemonResponse::Status(s) => {
            print!("{}", dashboard::render_dashboard_v2(&s));
            Ok(())
        }
        DaemonResponse::Error { message } => anyhow::bail!(message),
        _ => anyhow::bail!("respuesta inesperada del daemon"),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if matches!(&cli.command, Commands::Dashboard) {
        return handle_dashboard();
    }

    // Sprint patch (2026-06-05): the `whisper` snippet bypasses the daemon
    // socket entirely — it reads `runtime_metrics.json` directly so the
    // shell prompt never blocks on a stalled daemon.
    if let Commands::Whisper { always_on } = cli.command {
        return whisper::run(always_on);
    }

    if matches!(cli.command, Commands::PerceptualDoctor) {
        let response = send_request(DaemonRequest::GetMetrics)?;
        let DaemonResponse::Metrics(metrics) = response else {
            anyhow::bail!("daemon did not return metrics");
        };
        let report = perceptual_doctor::diagnose(&metrics);
        print!("{}", perceptual_doctor::render(&report));
        return Ok(());
    }

    let response = match cli.command {
        Commands::Dashboard => unreachable!(),
        Commands::Whisper { .. } => unreachable!(),
        Commands::PerceptualDoctor => unreachable!(),
        Commands::Status => send_request(DaemonRequest::GetStatus),
        Commands::Metrics => send_request(DaemonRequest::GetMetrics),
        Commands::TopBlockers => send_request(DaemonRequest::GetTopBlockers),
        Commands::ProfileTimeline => send_request(DaemonRequest::GetProfileTimeline),
        Commands::Doctor => send_request(DaemonRequest::Doctor),
        Commands::Capabilities => send_request(DaemonRequest::GetCapabilities),
        Commands::Restore => send_request(DaemonRequest::Restore),
        Commands::PanicRestore => send_request(DaemonRequest::PanicRestore),
        Commands::Purge => send_request(DaemonRequest::Purge),
        Commands::Pause => {
            let path = if unsafe { libc::geteuid() } == 0 {
                "/var/run/apollo.disable"
            } else {
                "/tmp/apollo.disable"
            };
            let _ = std::fs::File::create(path);
            println!("Apollo paused. Optimization suspended.");
            println!("To resume: apollo-optimizerctl resume");
            return Ok(());
        }
        Commands::Resume => {
            // Try both paths — the kill switch may have been created by root
            // even if we're running without sudo now
            let kill_paths = ["/var/run/apollo.disable", "/tmp/apollo.disable"];
            let mut removed = false;
            for path in &kill_paths {
                if std::path::Path::new(path).exists() {
                    match std::fs::remove_file(path) {
                        Ok(_) => {
                            println!("Removed kill switch: {}", path);
                            removed = true;
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                            eprintln!(
                                "Error: cannot remove {} — permission denied. Try: sudo apollo-optimizerctl resume",
                                path
                            );
                            std::process::exit(1);
                        }
                        Err(_) => {}
                    }
                }
            }
            if removed {
                println!("Apollo resumed. Optimization active.");
            } else {
                println!("Apollo was not paused (no kill switch found).");
            }
            return Ok(());
        }
        Commands::IsPaused => {
            let kill_paths = ["/var/run/apollo.disable", "/tmp/apollo.disable"];
            let mut paused = false;
            for path in &kill_paths {
                if std::path::Path::new(path).exists() {
                    println!("PAUSED — kill switch active: {}", path);
                    paused = true;
                }
            }
            if paused {
                println!("To resume: sudo apollo-optimizerctl resume");
            } else {
                println!("ACTIVE — optimization running normally");
            }
            return Ok(());
        }
        Commands::SetAutoProfile { enabled } => send_request(DaemonRequest::SetAutoProfile {
            enabled: enabled == "on",
        }),
        Commands::ClearProfileOverride => send_request(DaemonRequest::ClearProfileOverride),
        Commands::SetProfile {
            profile,
            ttl_minutes,
        } => send_request(DaemonRequest::SetProfile {
            profile: to_profile(&profile),
            ttl_minutes: Some(ttl_minutes),
        }),
        Commands::SetLatencyTarget { target } => send_request(DaemonRequest::SetLatencyTarget {
            target: to_latency_target(&target),
        }),
        Commands::DumpPolicy => send_request(DaemonRequest::GetLearnedPolicy),
        Commands::Canary { state } => send_request(DaemonRequest::SetCanaryEnabled {
            enabled: state == "on",
        }),
        Commands::Feedback { rating, note } => {
            send_request(DaemonRequest::Feedback { rating, note })
        }
        Commands::Usage { command } => match command {
            UsageCommands::Top { limit } => {
                send_request(DaemonRequest::UsageTop { limit: Some(limit) })
            }
            UsageCommands::Explain { name } => send_request(DaemonRequest::UsageExplain { name }),
        },
        Commands::SysctlGovernor => send_request(DaemonRequest::GetSysctlGovernor),
        Commands::RevertSysctls => send_request(DaemonRequest::RevertSysctls),
        Commands::Version => send_request(DaemonRequest::GetVersion),
        Commands::Health => send_request(DaemonRequest::GetHealth),
    }?;

    match response {
        DaemonResponse::Ok => println!("ok"),
        DaemonResponse::Status(s) => println!("{}", serde_json::to_string_pretty(&s)?),
        DaemonResponse::StatusPush(s) => println!("{}", serde_json::to_string_pretty(&s)?),
        DaemonResponse::Metrics(m) => println!("{}", serde_json::to_string_pretty(&m)?),
        DaemonResponse::TopBlockers(b) => println!("{}", serde_json::to_string_pretty(&b)?),
        DaemonResponse::ProfileTimeline(t) => println!("{}", serde_json::to_string_pretty(&t)?),
        DaemonResponse::Capabilities(c) => println!("{}", serde_json::to_string_pretty(&c)?),
        DaemonResponse::LearnedPolicy(p) => println!("{}", serde_json::to_string_pretty(&p)?),
        DaemonResponse::Usage(u) => println!("{}", serde_json::to_string_pretty(&u)?),
        DaemonResponse::SysctlGovernor(s) => println!("{}", serde_json::to_string_pretty(&s)?),
        DaemonResponse::VersionInfo { protocol, build } => {
            println!("apollo-optimizer v{build}  (protocol v{protocol})");
            if protocol != PROTOCOL_VERSION {
                eprintln!(
                    "warning: protocol mismatch — daemon uses v{protocol}, client uses v{PROTOCOL_VERSION}"
                );
            }
        }
        DaemonResponse::Doctor { checks } => {
            for c in checks {
                println!("{}", c);
            }
            let _ = fs::metadata("/var/run/apollo-optimizer.sock");
            let _ = fs::metadata("/tmp/apollo-optimizer.sock");
        }
        DaemonResponse::Health(h) => println!("{}", serde_json::to_string_pretty(&h)?),
        DaemonResponse::PurgeResult { fired, reason } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "fired": fired,
                    "reason": reason,
                }))?
            );
        }
        DaemonResponse::Error { message } => {
            anyhow::bail!(message);
        }
    }

    Ok(())
}
