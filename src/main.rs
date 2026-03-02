mod collector;
mod optimizer;
mod reactor;
mod sysctl_tuner;

use clap::{Parser, Subcommand};
use collector::SystemCollector;
use optimizer::OptimizerEngine;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "apollo-optimizer")]
#[command(about = "A 360-degree system optimizer for macOS", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Collects system metrics and saves a snapshot
    Snapshot {
        #[arg(short, long, default_value = "system_snapshot.json")]
        output: String,
    },
    /// Runs the optimization engine (CPU & RAM)
    Optimize,
    /// Runs disk cleanup
    Clean,
    /// Activates Turbo Mode (Faster than Native)
    Turbo,
    /// Runs in daemon mode (continuous optimization)
    Daemon,
    /// Configures Smart Startup (prevents app reopening)
    Startup,
    /// LLM Mode: Aggressive optimization for AI workloads
    Llm,
    /// Restores background services and reverses quarantines
    Restore,
}

fn configure_runtime_environment() {
    // launchd uses a reduced PATH by default; ensure common developer tool locations exist.
    let baseline = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin";
    let current = std::env::var("PATH").unwrap_or_default();
    let merged = if current.is_empty() {
        baseline.to_string()
    } else {
        format!("{baseline}:{current}")
    };
    std::env::set_var("PATH", merged);
}

fn main() {
    configure_runtime_environment();
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Turbo) => {
            let optimizer = OptimizerEngine::new();
            optimizer.apply_turbo_mode();
        }
        Some(Commands::Llm) => {
            println!("🔥 Engaging LLM Optimization Mode...");
            let optimizer = OptimizerEngine::new();
            optimizer.apply_llm_mode();
        }
        Some(Commands::Restore) => {
            let optimizer = OptimizerEngine::new();
            optimizer.restore_background_noise();
            println!("Background services restored.");
        }
        Some(Commands::Snapshot { output }) => {
            println!("Collecting system snapshot...");
            let mut collector = SystemCollector::new();
            let snapshot = collector.collect_snapshot();

            let json = match serde_json::to_string_pretty(&snapshot) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Failed to serialize snapshot: {}", e);
                    std::process::exit(1);
                }
            };
            let mut file = match File::create(output) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Failed to create output file '{}': {}", output, e);
                    std::process::exit(1);
                }
            };
            if let Err(e) = file.write_all(json.as_bytes()) {
                eprintln!("Failed to write snapshot file '{}': {}", output, e);
                std::process::exit(1);
            }

            println!("Snapshot saved to {}", output);
        }
        Some(Commands::Optimize) => {
            println!("Running optimization...");
            let mut collector = SystemCollector::new();
            let snapshot = collector.collect_snapshot();

            let optimizer = OptimizerEngine::new();
            optimizer.optimize(&snapshot);

            println!("Optimization complete.");
        }
        Some(Commands::Clean) => {
            println!("Running disk cleanup...");
            let optimizer = OptimizerEngine::new();
            optimizer.clean_disk();
            println!("Disk cleanup complete.");
        }
        Some(Commands::Startup) => {
            let optimizer = OptimizerEngine::new();
            optimizer.configure_startup();
            println!("Smart Startup configured.");
        }
        Some(Commands::Daemon) => {
            println!("Starting Apollo Optimizer Daemon...");
            let mut collector = SystemCollector::new();
            let optimizer = Arc::new(OptimizerEngine::new());

            // 1. Initial extreme tuning on startup
            // BUG 16 fix: boost optimizer QoS once here instead of every set_high_priority call.
            optimizer.boost_self_once();
            optimizer.configure_startup();
            // BUG 7 fix: DO NOT auto-apply turbo_mode on daemon startup.
            // Turbo mode makes invasive system changes (disable animations, etc).
            // User must explicitly run `apollo-optimizer turbo` to activate it.
            // optimizer.apply_turbo_mode(); // User must run this explicitly via CLI

            // 2. Register Signal Handler for safe shutdown
            let opt_clone = Arc::clone(&optimizer);
            if let Err(e) = ctrlc::set_handler(move || {
                opt_clone.cleanup();
                std::process::exit(0);
            }) {
                eprintln!("Warning: failed to register signal handler: {}", e);
            }

            // 3. Start Apollo System Nervous System (Reactive Nerves)
            let reactor = reactor::SystemReactor::new(Arc::clone(&optimizer));
            reactor.start();

            let mut next_tick = Instant::now();
            let mut last_cleanup = Instant::now();

            loop {
                // ZERO-POLLING: If we have reactive nerves, we can afford to pulse less frequently
                // This saves battery and CPU cycles on M1.
                let snapshot = collector.collect_snapshot();
                optimizer.optimize(&snapshot);

                // Periodic disk cleanup every 24h (real elapsed time).
                if last_cleanup.elapsed() >= Duration::from_secs(24 * 60 * 60) {
                    println!("⏰ Scheduled daily disk cleanup...");
                    optimizer.clean_disk();
                    last_cleanup = Instant::now();
                }

                // Adaptive Tick Rate. In Reactive Mode, healthy systems can sleep longer.
                // BUG 1 fix: `>= 15` made ALL modes sleep 300s (pro mode returns exactly 15).
                // Use `> 15` so pro mode (15s) executes at 15-second intervals.
                let mut sleep_secs = optimizer.get_tick_rate();
                if sleep_secs > 15 {
                    sleep_secs = 300;
                }

                next_tick += Duration::from_secs(sleep_secs);
                let now = Instant::now();
                if next_tick > now {
                    std::thread::sleep(next_tick - now);
                } else {
                    // If optimization work overran the interval, resync without accumulating drift.
                    next_tick = now;
                }
            }
        }
        None => {
            println!("No command specified. Use --help for usage information.");
        }
    }
}
