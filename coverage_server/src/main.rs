use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, Duration};
use std::error::Error;
use std::collections::HashMap;
use serde::Serialize;
use tungstenite::Message;
use tungstenite::server::accept;

#[repr(C)]
struct Shmem {
    fuzz_cases:     AtomicU64,
    coverage:       AtomicU64,
    coverage_freqs: [AtomicU64; 1024],
}

impl Clone for Shmem {
    fn clone(&self) -> Self {
        let tmp: Shmem = unsafe { core::mem::zeroed() };
        tmp.fuzz_cases.store(
            self.fuzz_cases.load(Ordering::Relaxed), Ordering::Relaxed);
        tmp.coverage.store(
            self.coverage.load(Ordering::Relaxed), Ordering::Relaxed);
        for (new, old) in tmp.coverage_freqs.iter()
                .zip(self.coverage_freqs.iter()) {
            new.store(old.load(Ordering::Relaxed), Ordering::Relaxed);
        }

        tmp
    }
}

fn get_shmem() -> &'static Shmem {
    use libc::*;

    unsafe {
        let shmfd = open(
            b"../afl_test/shared_memory.shm\0".as_ptr() as *const i8,
            O_RDWR | O_CREAT, 0o644);
        assert!(shmfd >= 0, "open() failed");

        // Set the file size
        ftruncate(shmfd, core::mem::size_of::<Shmem>() as i64);

        // Map the file
        let map = mmap(core::ptr::null_mut(), core::mem::size_of::<Shmem>(),
            PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0);
        assert!(map != MAP_FAILED, "Failed to map shared memory");

        &*(map as *const Shmem)
    }
}

#[derive(Serialize)]
struct NodeInfo {
    color:      u8,
    coverage:   u64,
    discovered: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Get access to shared memory
    let shmem = get_shmem();
    
    // Don't start until the first fuzz case
    while shmem.fuzz_cases.load(Ordering::Relaxed) == 0 {
        std::thread::sleep(Duration::from_millis(25));
    }

    print!("First fuzz case detected\n");

    // Start a timer
    let start = Instant::now();

    // Create the server
    let server = TcpListener::bind("127.0.0.1:9001")?;

    let mut data = HashMap::new();

    // Wait for connections
    for stream in server.incoming() {
        // Accept the connection
        let mut websocket = accept(stream?)?;

        print!("Accepted connection\n");

        // Handle messages forever
        let mut last_status = Instant::now();
        let mut handler = || -> Result<(), Box<dyn Error>> {
            let mut last_shmem = shmem.clone();
            loop {
                std::thread::sleep(Duration::from_millis(15));

                // Get elapsed time
                let elapsed = start.elapsed().as_secs_f64();
                let cases = shmem.fuzz_cases.load(Ordering::Relaxed);

                if last_status.elapsed() > Duration::from_millis(1000) {
                    print!("[{:10.3}] | cases {:10} [{:8.1}/sec] | \
                           coverage {:10}\n",
                           elapsed,
                           cases, cases as f64 / elapsed,
                           shmem.coverage.load(Ordering::Relaxed));
                    last_status = Instant::now();
                }

                for (ii, (old, new)) in last_shmem.coverage_freqs.iter()
                        .zip(shmem.coverage_freqs.iter()).enumerate() {
                    let new = new.load(Ordering::Relaxed);
                    let old = old.load(Ordering::Relaxed);
                    if new == 0 { continue; }

                    let ent = data.entry(format!("node{}", ii))
                        .or_insert(NodeInfo {
                            color: 100,
                            coverage: 0,
                            discovered: 0,
                        });

                    if new > old {
                        // Update color on new coverage
                        ent.color = 70;

                        if old == 0 {
                            ent.discovered = cases;
                        }
                    } else {
                        ent.color = core::cmp::min(ent.color + 1, 90);
                    }

                    // Update coverage
                    ent.coverage = new;
                }
            
                websocket.write_message(
                    Message::Text(serde_json::to_string(&data)?))?;

                // Update last shmem
                last_shmem = shmem.clone();
            }

            Ok(())
        };

        let _ = handler();
    }

    Ok(())
}

