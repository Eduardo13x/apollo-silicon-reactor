use crate::collector::SystemCollector;
use crate::optimizer::OptimizerEngine;
use std::ffi::CString;
use std::sync::Arc;
use std::thread;

pub struct SystemReactor {
    optimizer: Arc<OptimizerEngine>,
}

impl SystemReactor {
    pub fn new(optimizer: Arc<OptimizerEngine>) -> Self {
        Self { optimizer }
    }

    pub fn start(&self) {
        println!("🧠 INITIALIZING APOLLO SYSTEM NERVOUS SYSTEM...");
        let opt = Arc::clone(&self.optimizer);

        thread::spawn(move || {
            unsafe {
                let kq = libc::kqueue();
                if kq == -1 {
                    println!("❌ [Reactor] kqueue initialization failed.");
                    return;
                }

                // --- NERVE 1: Memory Pressure (Direct Kernel Filter) ---
                let mem_kev = libc::kevent {
                    ident: 0,
                    filter: -12, // EVFILT_VM (Hardcoded for Darwin)
                    flags: libc::EV_ADD | libc::EV_ENABLE | libc::EV_CLEAR,
                    fflags: 0x80000000, // NOTE_VM_PRESSURE
                    data: 0,
                    udata: 1 as *mut libc::c_void, // ID 1 = Memory
                };
                if libc::kevent(kq, &mem_kev, 1, std::ptr::null_mut(), 0, std::ptr::null()) == -1 {
                    println!("⚠️ [Reactor] Failed to register memory-pressure nerve.");
                }

                // --- NERVE 2: Thermal Pressure (Darwin Notification -> FD) ---
                let mut thermal_fd = 0;
                let thermal_name = CString::new("com.apple.system.thermalpressurelevel")
                    .expect("static thermal notification string should not contain NUL");
                let reg_status = notify_register_file_descriptor(
                    thermal_name.as_ptr(),
                    &mut thermal_fd,
                    0,
                    &mut 0, // token output
                );

                if reg_status == 0 && thermal_fd > 0 {
                    let thermal_kev = libc::kevent {
                        ident: thermal_fd as usize,
                        filter: libc::EVFILT_READ,
                        flags: libc::EV_ADD | libc::EV_ENABLE,
                        fflags: 0,
                        data: 0,
                        udata: 2 as *mut libc::c_void, // ID 2 = Thermal
                    };
                    if libc::kevent(
                        kq,
                        &thermal_kev,
                        1,
                        std::ptr::null_mut(),
                        0,
                        std::ptr::null(),
                    ) == -1
                    {
                        println!("⚠️ [Reactor] Thermal nerve registration failed.");
                    } else {
                        println!("🔥 Thermal Nerve: Engaged.");
                    }
                } else {
                    println!("⚠️ Thermal Nerve failed to initialize.");
                }

                // --- NERVE 3: Lifecycle (App Launches) ---
                let mut launch_fd = 0;
                let launch_name = CString::new("com.apple.launchd.spawn")
                    .expect("static launch notification string should not contain NUL");
                let launch_status = notify_register_file_descriptor(
                    launch_name.as_ptr(),
                    &mut launch_fd,
                    0,
                    &mut 0,
                );

                if launch_status == 0 && launch_fd > 0 {
                    let launch_kev = libc::kevent {
                        ident: launch_fd as usize,
                        filter: libc::EVFILT_READ,
                        flags: libc::EV_ADD | libc::EV_ENABLE,
                        fflags: 0,
                        data: 0,
                        udata: 3 as *mut libc::c_void, // ID 3 = Lifecycle
                    };
                    if libc::kevent(
                        kq,
                        &launch_kev,
                        1,
                        std::ptr::null_mut(),
                        0,
                        std::ptr::null(),
                    ) == -1
                    {
                        println!("⚠️ [Reactor] Lifecycle nerve registration failed.");
                    } else {
                        println!("🚀 Lifecycle Nerve: Engaged (App launch detection).");
                    }
                }

                // --- NERVE 4: Power Source (Plug/Unplug) ---
                let mut power_fd = 0;
                let power_name = CString::new("com.apple.system.powersources.source")
                    .expect("static power notification string should not contain NUL");
                let power_status =
                    notify_register_file_descriptor(power_name.as_ptr(), &mut power_fd, 0, &mut 0);

                if power_status == 0 && power_fd > 0 {
                    let power_kev = libc::kevent {
                        ident: power_fd as usize,
                        filter: libc::EVFILT_READ,
                        flags: libc::EV_ADD | libc::EV_ENABLE,
                        fflags: 0,
                        data: 0,
                        udata: 4 as *mut libc::c_void, // ID 4 = Power
                    };
                    if libc::kevent(kq, &power_kev, 1, std::ptr::null_mut(), 0, std::ptr::null())
                        == -1
                    {
                        println!("⚠️ [Reactor] Power nerve registration failed.");
                    } else {
                        println!("🔌 Power Nerve: Engaged (Instant Eco-Mode switching).");
                    }
                }

                println!("✅ SYSTEM NERVOUS SYSTEM ACTIVE (Memory + Thermal + Lifecycle + Power Reactive).");

                let mut out_ev = std::mem::zeroed::<libc::kevent>();
                loop {
                    let n = libc::kevent(kq, std::ptr::null(), 0, &mut out_ev, 1, std::ptr::null());

                    #[allow(clippy::comparison_chain)]
                    if n > 0 {
                        let id = out_ev.udata as usize;
                        let mut collector = SystemCollector::new();
                        let snapshot = collector.collect_snapshot();

                        match id {
                            1 => {
                                println!("🚨 REACTIVE PULSE: Memory Pressure Change.");
                                opt.optimize(&snapshot);
                                if snapshot.memory.used_ram as f64
                                    / snapshot.memory.total_ram as f64
                                    > 0.85
                                {
                                    opt.clean_disk();
                                }
                            }
                            2 => {
                                // Drain the pipe
                                let mut dummy: i32 = 0;
                                libc::read(
                                    thermal_fd,
                                    &mut dummy as *mut _ as *mut libc::c_void,
                                    4,
                                );
                                println!("🌡️  REACTIVE PULSE: Thermal Signature Change.");
                                opt.optimize(&snapshot);
                            }
                            3 => {
                                // Drain the pipe
                                let mut dummy: i32 = 0;
                                libc::read(launch_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                                println!("⚡ REACTIVE PULSE: New Process Spawned.");

                                // Immediate re-scan and optimization
                                let mut collector = SystemCollector::new();
                                let snapshot = collector.collect_snapshot();
                                opt.optimize(&snapshot);
                            }
                            4 => {
                                // Drain the pipe
                                let mut dummy: i32 = 0;
                                libc::read(power_fd, &mut dummy as *mut _ as *mut libc::c_void, 4);
                                println!("🔌 REACTIVE PULSE: Power Source Change.");

                                // On power change, we immediately update the entire system state
                                let mut collector = SystemCollector::new();
                                let snapshot = collector.collect_snapshot();
                                opt.optimize(&snapshot);
                            }
                            _ => {}
                        }
                    } else if n < 0 {
                        break;
                    }
                }

                if thermal_fd > 0 {
                    libc::close(thermal_fd);
                }
                if launch_fd > 0 {
                    libc::close(launch_fd);
                }
                if power_fd > 0 {
                    libc::close(power_fd);
                }
                libc::close(kq);
            }
        });
    }
}

// Low-Level Darwin APIs
#[link(name = "System")]
extern "C" {
    fn notify_register_file_descriptor(
        name: *const libc::c_char,
        out_fd: *mut libc::c_int,
        flags: libc::c_int,
        out_token: *mut libc::c_int,
    ) -> u32;
}
