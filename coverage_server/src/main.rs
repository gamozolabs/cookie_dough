use std::fs::File;
use std::io::Write;
use std::net::TcpListener;
use std::iter::once;
use std::path::Path;
use std::sync::{Mutex, Arc};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, Duration};
use std::error::Error;
use std::process::{Stdio, Command};
use std::collections::{BTreeSet, BTreeMap};
use rand::random;
use serde::Serialize;
use tungstenite::Message;
use tungstenite::server::accept;

const SHM_FILENAME: &'static str = "../afl_test/shared_memory.shm";

#[repr(C)]
struct Shmem {
    fuzz_cases:     AtomicU64,
    coverage:       AtomicU64,
    coverage_freqs: [AtomicU64; 1024],
    hit_on_case:    [AtomicU64; 1024],
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

fn get_shmem<P: AsRef<Path>>(filename: P) -> &'static Shmem {
    use libc::*;

    unsafe {
        let filename: Vec<u8> = filename.as_ref().to_str().unwrap().as_bytes()
            .iter().chain(once(&0u8)).cloned().collect();
        let shmfd = open(filename.as_ptr() as *const i8,
            O_RDWR | O_CREAT, 0o644);
        assert!(shmfd >= 0, "open() failed");

        // Set the file size
        ftruncate(shmfd, core::mem::size_of::<Shmem>() as i64);

        // Map the file
        let map = mmap(core::ptr::null_mut(), core::mem::size_of::<Shmem>(),
            PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0);
        assert!(map != MAP_FAILED, "Failed to map shared memory");

        // Close fd
        close(shmfd);

        &*(map as *const Shmem)
    }
}

#[derive(Serialize)]
struct NodeInfo {
    color:      u8,
    coverage:   u64,
    discovered: u64,
}

fn run_fuzzer(stat_prefix: &str, command: &'static [&'static str])
        -> Result<(), Box<dyn Error>> {
    // Delete old shared memory
    let _ = std::fs::remove_file(SHM_FILENAME);

    // Delete old outputs
    let _ = std::fs::remove_dir_all("../afl_test/outputs");
    
    // Get access to shared memory
    let shmem = get_shmem(SHM_FILENAME);

    // Keeps track of threads we spawn
    let mut threads = Vec::new();
    
    // Create a channel for signalling when to exit the fuzzer
    let (sender, receiver) = std::sync::mpsc::channel();

    // Create a new fuzzer instance
    threads.push(std::thread::spawn(move || {
        let mut process = Command::new(command[0])
            .current_dir("../afl_test")
            .args(&command[1..])
            .stderr(Stdio::null())
            .stdout(Stdio::null())
            .spawn().unwrap();

        loop {
            if let Ok(Some(_status)) = process.try_wait() {
                break;
            }

            if let Ok(_) = receiver.try_recv() {
                Command::new("killall").arg("-9").arg("a.out")
                    .status().unwrap();
            }

            std::thread::sleep(Duration::from_millis(10));
        }
    }));

    // Don't start until the first fuzz case
    while shmem.fuzz_cases.load(Ordering::Relaxed) == 0 {
        std::thread::sleep(Duration::from_millis(25));
    }

    // Create the data
    let global_data = Arc::new(Mutex::new(BTreeMap::new()));

    // Start a stats thread
    let data = global_data.clone();
    threads.push(std::thread::spawn(move || {
        // Start a timer
        let start = Instant::now();

        let mut last_status = Instant::now();
        loop {
            let last_shmem = shmem.clone();

            std::thread::sleep(Duration::from_millis(5));

            let mut data = data.lock().unwrap();

            // Get elapsed time
            let elapsed = start.elapsed().as_secs_f64();
            let cases = shmem.fuzz_cases.load(Ordering::Relaxed);

            if last_status.elapsed() > Duration::from_millis(25) {
                eprint!("\r[{:10.3}] | cases {:12} [{:12.1}/sec] | \
                       coverage {:10}",
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
                } else {
                    ent.color = core::cmp::min(ent.color + 1, 90);
                }

                // Update coverage
                ent.coverage = new;
                ent.discovered =
                    shmem.hit_on_case[ii].load(Ordering::Relaxed);
            }

            if cases > 1_000_000 { 
                // Request fuzzer to exit
                let _ = sender.send(true);
                return;
            }
        }
    }));

    /*
    // Create the server
    let server = TcpListener::bind("127.0.0.1:9001")?;

    // Wait for connections
    for stream in server.incoming() {
        // Accept the connection
        let mut websocket = accept(stream?)?;

        eprint!("Accepted connection\n");

        // Handle messages forever
        let mut handler = || -> Result<(), Box<dyn Error>> {
            loop {
                std::thread::sleep(Duration::from_millis(100));
 
                websocket.write_message(
                    Message::Text(serde_json::to_string(
                            &*global_data.lock().unwrap())?))?;
            }
        };

        let _ = handler();
    }*/

    // Wait for all threads to exit
    for thr in threads {
        thr.join().unwrap();
    }

    eprint!("\n");
    
    unsafe {
        libc::munmap(shmem as *const _ as *mut _, 0);
    }

    let data_fn = format!("data/{}_{:08x}.shm", stat_prefix, random::<u32>());
    std::fs::create_dir_all("data").unwrap();
    std::fs::copy(SHM_FILENAME, &data_fn).unwrap();

    Ok(())
}

/// A structure that holds stats about a fuzzers performance
#[derive(Default)]
struct FuzzerStats {
    /// Average number of cases to visit a given block
    block_find: BTreeMap<usize, f64>,

    /// Average number of times a block is hit
    block_freq: BTreeMap<usize, f64>,
}

fn process_data(prefix: &str) -> Result<FuzzerStats, Box<dyn Error>> {
    if !Path::new("data").is_dir() {
        return Ok(Default::default());
    }

    // Create a new stats structure
    let mut stats = FuzzerStats::default();

    // Create a mapping of blocks to the times to hit the block
    let mut time_to_block = BTreeMap::new();
    
    // Create a mapping of blocks to the number of times the block was hit
    let mut num_hits = BTreeMap::new();

    // Go through each data file
    for filename in std::fs::read_dir("data")? {
        // Filter based on the filename prefix we requested
        let filename = filename?.path();
        if !filename.file_name().unwrap().to_str().unwrap()
                .starts_with(prefix) {
            continue;
        }
       
        // Load the data file
        let shmem = get_shmem(filename);

        // Log the time it took to find coverage 
        for (ii, blocks) in shmem.hit_on_case.iter().enumerate() {
            let blocks = blocks.load(Ordering::Relaxed);
            if blocks != 0 {
                time_to_block.entry(ii)
                    .or_insert_with(|| Vec::new()).push(blocks as f64);
            }
        }
        
        // Log the number of hits per block
        for (ii, freq) in shmem.coverage_freqs.iter().enumerate() {
            let freq = freq.load(Ordering::Relaxed);
            if freq != 0 {
                num_hits.entry(ii).or_insert_with(|| Vec::new())
                    .push(freq as f64);
            }
        }
    }

    // Compute means for each block
    for (block, cases_to_hit) in time_to_block {
        let total_hits = &num_hits[&block];

        // Calcuate average cases to hit the block
        let to_hit = cases_to_hit.iter().copied().sum::<f64>() /
            cases_to_hit.len() as f64;

        // Calculate average number of hits for the block
        let num_hit = total_hits.iter().copied().sum::<f64>() /
            total_hits.len() as f64;

        // Update stat records
        stats.block_find.insert(block, to_hit);
        stats.block_freq.insert(block, num_hit);
    }
    
    Ok(stats)
}

fn main() -> Result<(), Box<dyn Error>> {
    // First, remove all data
    let _ = std::fs::remove_dir_all("./data");

    // Create a list of all fuzzers to test, and modes to use to test them
    let mut fuzz_modes = [
        ("internal-corrupt-1", &["./a.out", "internal", "1"][..], None),
        ("internal-corrupt-2", &["./a.out", "internal", "2"][..], None),
        ("internal-corrupt-4", &["./a.out", "internal", "4"][..], None),
        ("internal-corrupt-8", &["./a.out", "internal", "8"][..], None),
        ("internal-corrupt-16", &["./a.out", "internal", "16"][..], None),
        ("internal-corrupt-32", &["./a.out", "internal", "32"][..], None),
        ("internal-corrupt-64", &["./a.out", "internal", "64"][..], None),

        ("afl-explore", &["/home/pleb/AFLplusplus/afl-fuzz",
            "-p", "explore",
            "-d", "-i", "inputs", "-o", "outputs", "./a.out", "@@"][..], None),
        
        ("afl-fast", &["/home/pleb/AFLplusplus/afl-fuzz",
         "-p", "fast",
         "-d", "-i", "inputs", "-o", "outputs", "./a.out", "@@"][..], None),
    ];

    // All unique blocks seen between all fuzzers and runs
    let mut seen_blocks: BTreeSet<usize> = BTreeSet::new();

    // Go through and run each fuzzer :D
    for (prefix, use_internal, stats) in fuzz_modes.iter_mut() {
        // Run the fuzzer
        for ii in 0..20 {
            eprint!("Iter {} of {}\n", ii, prefix);
            run_fuzzer(prefix, *use_internal)?;
        }

        // Process the results
        let data = process_data(prefix)?;

        // Update the IDs of the blocks we have observed
        seen_blocks.extend(data.block_find.keys());

        // Update stats for this fuzzer
        *stats = Some(data);
    }

    let mut log = File::create("log.txt")?;
    for fuzzer in &fuzz_modes {
        write!(log, "\"{}\"\n", fuzzer.0)?;

        // Generate some stats for each block for each fuzzer
        for &block in &seen_blocks {
            // Get the find rate for this fuzzer
            let find = fuzzer.2.as_ref().unwrap().block_find.get(&block);
            if find.is_none() { continue; }
            let find = find.unwrap();

            write!(log, "{:10} {:15.3}\n", block, find)?;
        }

        write!(log, "\n\n")?;
    }

    Ok(())
}

