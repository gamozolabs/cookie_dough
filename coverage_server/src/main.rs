use std::net::TcpListener;
use std::iter::once;
use std::path::Path;
use std::sync::{Mutex, Arc};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, Duration};
use std::error::Error;
use std::process::Command;
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

fn run_fuzzer(use_internal: bool) -> Result<(), Box<dyn Error>> {
    // Delete old shared memory
    let _ = std::fs::remove_file(SHM_FILENAME);
    
    // Get access to shared memory
    let shmem = get_shmem(SHM_FILENAME);

    // Keeps track of threads we spawn
    let mut threads = Vec::new();
    
    // Create a channel for signalling when to exit the fuzzer
    let (sender, receiver) = std::sync::mpsc::channel();

    // Create a new fuzzer instance
    threads.push(std::thread::spawn(move || {
        let mut process = if use_internal {
            Command::new("./a.out")
                .current_dir("../afl_test")
                .arg("internal").spawn().unwrap()
        } else {
            Command::new("/home/pleb/AFLplusplus/afl-fuzz")
                .current_dir("../afl_test")
                /*.arg("-p")
                .arg("fast")*/
                .arg("-d")
                .arg("-i")
                .arg("inputs")
                .arg("-o")
                .arg("outputs")
                .arg("./a.out")
                .arg("@@")
                .spawn().unwrap()
        };

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

    print!("First fuzz case detected\n");
    
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

            std::thread::sleep(Duration::from_millis(100));

            let mut data = data.lock().unwrap();

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
                } else {
                    ent.color = core::cmp::min(ent.color + 1, 90);
                }

                // Update coverage
                ent.coverage = new;
                ent.discovered =
                    shmem.hit_on_case[ii].load(Ordering::Relaxed);
            }

            if cases > 500_000 { 
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

        print!("Accepted connection\n");

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
    
    unsafe {
        libc::munmap(shmem as *const _ as *mut _, 0);
    }

    let data_fn = if use_internal {
        format!("data/internal_{:08x}.shm",
            random::<u32>())
    } else {
        format!("data/afl_{:08x}.shm",
            random::<u32>())
    };
    std::fs::create_dir_all("data").unwrap();
    std::fs::copy(SHM_FILENAME, &data_fn).unwrap();

    Ok(())
}

fn process_data(prefix: &str) ->
        Result<(
            BTreeMap<usize, (f64, f64)>,
            BTreeMap<usize, (f64, f64)>
        ), Box<dyn Error>> {
    if !Path::new("data").is_dir() {
        return Ok(Default::default());
    }

    let mut time_to_bug = BTreeMap::new();
    let mut num_hits = BTreeMap::new();

    // Go through each data file
    for filename in std::fs::read_dir("data")? {
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
                time_to_bug.entry(ii).or_insert_with(|| Vec::new()).push(blocks);
            }
        }
        
        // Log the number of hits per block
        for (ii, freq) in shmem.coverage_freqs.iter().enumerate() {
            let freq = freq.load(Ordering::Relaxed);
            if freq != 0 {
                num_hits.entry(ii).or_insert_with(|| Vec::new()).push(freq);
            }
        }
    }
    
    let mut hits_mean_stddev = BTreeMap::new();
    for (&block_id, hits) in num_hits.iter() {
        // Compute the sum
        let sum = hits.iter().map(|&x| x as f64)
            .sum::<f64>();

        // Compute the sum of squares
        let sumx2 = hits.iter().map(|&x| x as f64 * x as f64)
            .sum::<f64>();

        // Compute the mean
        let mean = sum / hits.len() as f64;

        // Compute the stddev
        let stddev =
            ((sumx2 / hits.len() as f64) - (mean * mean)).sqrt();

        hits_mean_stddev.insert(block_id, (mean, stddev));
    }

    let mut block_time_mean_stddev = BTreeMap::new();
    for (&block_id, time_to_bug) in time_to_bug.iter() {
        // Compute the sum
        let sum = time_to_bug.iter().map(|&x| x as f64)
            .sum::<f64>();

        // Compute the sum of squares
        let sumx2 = time_to_bug.iter().map(|&x| x as f64 * x as f64)
            .sum::<f64>();

        // Compute the mean
        let mean = sum / time_to_bug.len() as f64;

        // Compute the stddev
        let stddev =
            ((sumx2 / time_to_bug.len() as f64) - (mean * mean)).sqrt();

        block_time_mean_stddev.insert(block_id, (mean, stddev));
    }

    Ok((block_time_mean_stddev, hits_mean_stddev))
}

fn main() -> Result<(), Box<dyn Error>> {
    let _ = std::fs::remove_dir_all("./data");

    for _ in 0..25 {
        run_fuzzer(false)?;
    }
    for _ in 0..25 {
        run_fuzzer(true)?;
    }

    {
        let (internal_ttbdb, internal_hitsdb) = process_data("internal_")?;
        let (afl_ttbdb, afl_hitsdb) = process_data("afl_")?;

        let mut dot = std::fs::read_to_string("foo.dot")?;

        let keys: BTreeSet<_> = internal_ttbdb.keys().chain(afl_ttbdb.keys()).collect();
        for &block_id in &keys {
            let internal_ttb = internal_ttbdb.get(&block_id)
                .unwrap_or(&(std::f64::MAX, std::f64::MAX));
            let afl_ttb = afl_ttbdb.get(&block_id)
                .unwrap_or(&(std::f64::MAX, std::f64::MAX));
            
            let internal_hits = internal_hitsdb.get(&block_id)
                .unwrap_or(&(std::f64::MAX, std::f64::MAX));
            let afl_hits = afl_hitsdb.get(&block_id)
                .unwrap_or(&(std::f64::MAX, std::f64::MAX));

            // Compute the AFL speedup, eg 2.0 means AFL took 1/2 the time to
            // find the block
            let afl_speedup = (internal_ttb.0 / afl_ttb.0).min(1000.);

            // Compute the AFL hit ratio (2.0 means AFL visited the block 2x
            // more than our fuzzer)
            let afl_rate = (afl_hits.0 / internal_hits.0).min(1000.);

            let ident = format!("id=\"node{}\"", block_id);

            let ttb_fill = if afl_ttbdb.get(&block_id).is_none() &&
                    internal_ttbdb.get(&block_id).is_some() {
                // AFL was unable to find the block, but we were able to
                "black".to_string()
            } else if afl_ttbdb.get(&block_id).is_some() &&
                    internal_ttbdb.get(&block_id).is_none() {
                // AFL found the block and we did not
                "blue".to_string()
            } else if afl_speedup < 1.0 {
                let intensity = ((1.0 - afl_speedup) * 1.5).min(1.0);
                let intensity = intensity.max(0.2);
                format!("0.000 {:5.3} 1.000", intensity)
            } else {
                let intensity = (afl_speedup / 3.).min(1.0);
                let intensity = intensity.max(0.2);
                format!("0.333 {:5.3} 1.000", intensity)
            };
            
            let hit_fill = if afl_rate < 1.0 {
                let intensity = ((1.0 - afl_rate) * 1.5).min(1.0);
                let intensity = intensity.max(0.2);
                format!("0.000 {:5.3} 1.000", intensity)
            } else {
                let intensity = (afl_rate / 3.).min(1.0);
                let intensity = intensity.max(0.2);
                format!("0.333 {:5.3} 1.000", intensity)
            };

            // Inner circle in the gradient is the ratio of time to hit
            // Outer edges of gradient is the ratio of number of hits
            let new = format!("id=\"node{}\",style=\"radial\",fillcolor=\"{}:{}\",label=\"{:.4} | {:.4}\"",
                    block_id, ttb_fill, ttb_fill, afl_speedup, afl_rate);

            dot = dot.replace(&ident, &new);
        }

        std::fs::write("foo_colored.dot", dot)?;

        assert!(Command::new("dot").args(&[
            "-Tsvg",
            "-ofoo_colored.svg",
            "foo_colored.dot",
        ]).status().unwrap().success());

        return Ok(());
    }
}

