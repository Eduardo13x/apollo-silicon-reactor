use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

use anyhow::Context;
use apollo_optimizer::engine::protocol::{DaemonRequest, DaemonResponse};
use apollo_optimizer::engine::types::{LatencyTarget, OptimizationProfile};
use clap::{Parser, Subcommand};

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
    Status,
    Metrics,
    TopBlockers,
    ProfileTimeline,
    Doctor,
    Capabilities,
    Restore,
    PanicRestore,
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
    Llm {
        #[command(subcommand)]
        command: LlmCommands,
    },
    DumpPolicy,
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

#[derive(Subcommand)]
enum LlmCommands {
    Status,
    Disable,
    Test,
    SetKey {
        /// API key value. If omitted, reads from APOLLO_LLM_API_KEY.
        #[arg(long)]
        key: Option<String>,
        /// Training TTL (days). After TTL, daemon deletes key and stops using LLM.
        #[arg(long, default_value_t = 30)]
        ttl_days: u64,
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

fn send_request(req: DaemonRequest) -> anyhow::Result<DaemonResponse> {
    let mut stream = None;
    for path in socket_candidates() {
        if let Ok(s) = UnixStream::connect(path) {
            stream = Some(s);
            break;
        }
    }
    let mut stream = stream.context("cannot connect to daemon socket")?;
    let payload = serde_json::to_string(&req)?;
    writeln!(stream, "{}", payload)?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let response = serde_json::from_str::<DaemonResponse>(&line)?;
    Ok(response)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let response = match cli.command {
        Commands::Status => send_request(DaemonRequest::GetStatus),
        Commands::Metrics => send_request(DaemonRequest::GetMetrics),
        Commands::TopBlockers => send_request(DaemonRequest::GetTopBlockers),
        Commands::ProfileTimeline => send_request(DaemonRequest::GetProfileTimeline),
        Commands::Doctor => send_request(DaemonRequest::Doctor),
        Commands::Capabilities => send_request(DaemonRequest::GetCapabilities),
        Commands::Restore => send_request(DaemonRequest::Restore),
        Commands::PanicRestore => send_request(DaemonRequest::PanicRestore),
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
        Commands::Llm { command } => match command {
            LlmCommands::Status => send_request(DaemonRequest::GetLlmStatus),
            LlmCommands::Disable => send_request(DaemonRequest::LlmDisable),
            LlmCommands::Test => send_request(DaemonRequest::LlmTest),
            LlmCommands::SetKey { key, ttl_days } => {
                let key = key
                    .or_else(|| std::env::var("APOLLO_LLM_API_KEY").ok())
                    .context("missing API key: pass --key or set APOLLO_LLM_API_KEY")?;
                send_request(DaemonRequest::LlmSetKey {
                    api_key: key,
                    ttl_days,
                })
            }
        },
        Commands::DumpPolicy => send_request(DaemonRequest::GetLearnedPolicy),
        Commands::Feedback { rating, note } => {
            send_request(DaemonRequest::Feedback { rating, note })
        }
        Commands::Usage { command } => match command {
            UsageCommands::Top { limit } => {
                send_request(DaemonRequest::UsageTop { limit: Some(limit) })
            }
            UsageCommands::Explain { name } => send_request(DaemonRequest::UsageExplain { name }),
        },
    }?;

    match response {
        DaemonResponse::Ok => println!("ok"),
        DaemonResponse::Status(s) => println!("{}", serde_json::to_string_pretty(&s)?),
        DaemonResponse::Metrics(m) => println!("{}", serde_json::to_string_pretty(&m)?),
        DaemonResponse::TopBlockers(b) => println!("{}", serde_json::to_string_pretty(&b)?),
        DaemonResponse::ProfileTimeline(t) => println!("{}", serde_json::to_string_pretty(&t)?),
        DaemonResponse::Capabilities(c) => println!("{}", serde_json::to_string_pretty(&c)?),
        DaemonResponse::LlmStatus(s) => println!("{}", serde_json::to_string_pretty(&s)?),
        DaemonResponse::LlmTestResult {
            ok,
            http_status,
            error,
            suggestion,
        } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "ok": ok,
                    "http_status": http_status,
                    "error": error,
                    "suggestion": suggestion,
                }))?
            )
        }
        DaemonResponse::LearnedPolicy(p) => println!("{}", serde_json::to_string_pretty(&p)?),
        DaemonResponse::Usage(u) => println!("{}", serde_json::to_string_pretty(&u)?),
        DaemonResponse::Doctor { checks } => {
            for c in checks {
                println!("{}", c);
            }
            let _ = fs::metadata("/var/run/apollo-optimizer.sock");
            let _ = fs::metadata("/tmp/apollo-optimizer.sock");
        }
        DaemonResponse::Error { message } => {
            anyhow::bail!(message);
        }
    }

    Ok(())
}
