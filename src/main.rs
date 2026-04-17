// The CLI binary compiles its own copy of `collector` and `engine` because
// `apollo-optimizer` predates the library split. The daemon (`apollo-optimizerd`)
// and client (`apollo-optimizerctl`) consume the same code via the `apollo_optimizer`
// lib crate, so the CLI binary only exercises a narrow subset — most items here
// look dead from main.rs's perspective. Suppressing dead_code at the binary
// crate root keeps the warnings clean without duplicating the gate on every
// lib item. Removing this requires routing the CLI through the lib crate.
#![allow(dead_code)]

mod collector;
mod engine;

use clap::{Parser, Subcommand};
use collector::SystemCollector;
use std::fs::File;
use std::io::Write;

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
    /// [DEPRECATED] Use apollo-optimizerctl to interact with the daemon
    Optimize,
    /// [DEPRECATED] Use apollo-optimizerctl doctor
    Clean,
    /// [DEPRECATED] Use: apollo-optimizerctl profile set performance
    Turbo,
    /// [DEPRECATED] Use apollo-optimizerd directly
    Daemon,
    /// [DEPRECATED] Use: ./scripts/install-root-daemon.sh
    Startup,
    /// [DEPRECATED] Use: apollo-optimizerctl profile set llm-boost
    Llm,
    /// [DEPRECATED] Use: apollo-optimizerctl restore
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
        Some(Commands::Turbo) => {
            eprintln!("apollo-optimizer turbo esta deprecado.");
            eprintln!("Usa: apollo-optimizerctl profile set performance");
            std::process::exit(1);
        }
        Some(Commands::Llm) => {
            eprintln!("apollo-optimizer llm esta deprecado.");
            eprintln!("Usa: apollo-optimizerctl profile set llm-boost");
            std::process::exit(1);
        }
        Some(Commands::Restore) => {
            eprintln!("apollo-optimizer restore esta deprecado.");
            eprintln!("Usa: apollo-optimizerctl restore");
            std::process::exit(1);
        }
        Some(Commands::Optimize) => {
            eprintln!("apollo-optimizer optimize esta deprecado.");
            eprintln!("El daemon apollo-optimizerd optimiza continuamente.");
            eprintln!("Usa: apollo-optimizerctl status");
            std::process::exit(1);
        }
        Some(Commands::Clean) => {
            eprintln!("apollo-optimizer clean esta deprecado.");
            eprintln!("Usa: apollo-optimizerctl doctor");
            std::process::exit(1);
        }
        Some(Commands::Startup) => {
            eprintln!("apollo-optimizer startup esta deprecado.");
            eprintln!("Instala el daemon con: ./scripts/install-root-daemon.sh");
            std::process::exit(1);
        }
        Some(Commands::Daemon) => {
            eprintln!("Error: usa apollo-optimizerd directamente para el modo daemon.");
            eprintln!("Ejemplo: apollo-optimizerd daemon --profile balanced-root");
            std::process::exit(1);
        }
        None => {
            println!("No command specified. Use --help for usage information.");
        }
    }
}
