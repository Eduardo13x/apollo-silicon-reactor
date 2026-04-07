fn main() {
    // Only compile the C bridges on macOS.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let sdk_path = std::process::Command::new("xcrun")
            .args(["--show-sdk-path"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk".into());

        // ── IOReport bridge ──────────────────────────────────────────────
        cc::Build::new()
            .file("src/engine_c/ioreport_bridge.c")
            // Objective-C blocks runtime (required by IOReportIterate callback).
            // On macOS/Clang, -fblocks is supported natively; libSystem.dylib
            // provides the blocks runtime, so no extra -lBlocksRuntime needed.
            .flag("-fblocks")
            .flag("-O2")
            // CoreFoundation types are used in the bridge header.
            .flag("-isysroot")
            .flag(&sdk_path)
            .compile("ioreport_bridge");

        // Link the IOReport private dylib (present at /usr/lib/libIOReport.dylib
        // on macOS 12+ with Apple Silicon).
        println!("cargo:rustc-link-lib=dylib=IOReport");
        println!("cargo:rustc-link-search=native=/usr/lib");

        // CoreFoundation is required for CFStringRef, CFDictionaryRef, etc.
        println!("cargo:rustc-link-lib=framework=CoreFoundation");

        // Re-run if the bridge C file changes.
        println!("cargo:rerun-if-changed=src/engine_c/ioreport_bridge.c");

        // ── SMC bridge ───────────────────────────────────────────────────
        cc::Build::new()
            .file("src/engine_c/smc_bridge.c")
            .flag("-O2")
            .flag("-isysroot")
            .flag(&sdk_path)
            .compile("smc_bridge");

        // IOKit is already linked by ioreport_bridge (framework = IOKit).
        // Adding it here is harmless (linker deduplicates).
        println!("cargo:rustc-link-lib=framework=IOKit");
        println!("cargo:rerun-if-changed=src/engine_c/smc_bridge.c");
    }
}
