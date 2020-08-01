use std::fs::File;
use std::io::Write;
use std::iter::once;
use std::path::Path;
use std::sync::{Mutex, Arc};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, Duration};
use std::error::Error;
use std::process::{Stdio, Command};
use std::collections::{BTreeMap, VecDeque};

/// Shared memory in use by the fuzzed target
#[repr(C)]
struct Shmem {
    fuzz_cases:     AtomicU64,
    coverage:       AtomicU64,
    start_time:     AtomicU64,
    coverage_freqs: [AtomicU64; 8192],
    hit_on_case:    [AtomicU64; 8192],
    hit_on_rdtsc:   [AtomicU64; 8192],
}

/// Get access to shared memory
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

/// A structure that holds stats about a fuzzers performance
#[derive(Default, Debug)]
struct FuzzerStats {
    /// Number of cases to visit a given block
    block_find: BTreeMap<usize, u64>,
    
    /// rdtsc time to visit a given block
    block_rdtsc: BTreeMap<usize, u64>,

    /// Number of times a block is hit
    block_freq: BTreeMap<usize, u64>,
}

fn process_data(shmem: &Shmem) -> Result<FuzzerStats, Box<dyn Error>> {
    // Create a new stats structure
    let mut stats = FuzzerStats::default();

    // Log the cases it took to find coverage 
    for (ii, blocks) in shmem.hit_on_case.iter().enumerate() {
        let blocks = blocks.load(Ordering::Relaxed);
        if blocks != 0 {
            assert!(stats.block_find.insert(ii, blocks).is_none());
        }
    }
    
    for (ii, blocks) in shmem.hit_on_rdtsc.iter().enumerate() {
        let blocks = blocks.load(Ordering::Relaxed);
        if blocks != 0 {
            assert!(stats.block_rdtsc.insert(ii, blocks).is_none());
        }
    }
    
    for (ii, blocks) in shmem.coverage_freqs.iter().enumerate() {
        let blocks = blocks.load(Ordering::Relaxed);
        if blocks != 0 {
            assert!(stats.block_freq.insert(ii, blocks).is_none());
        }
    }

    Ok(stats)
}

/// A structure which holds information about a fuzz runs progress and speed
struct FuzzerProgress {
    /// Number of cases for the fuzzer
    cases: u64,

    /// Amount of coverage the fuzzer has explored
    coverage: u64,

    /// The time when the fuzzer started running
    start_time: Option<Instant>,

    /// Upon successful execution of the fuzzer, this will be populated with
    /// the information about the fuzz run
    stats: Option<FuzzerStats>,
}

fn worker(name: String, id: usize, command: &[&str],
          progress: Arc<Mutex<FuzzerProgress>>) {
    // Create job directory
    let job_dir = Path::new("temps").join(&format!("{}-{}", name, id));
    std::fs::create_dir_all(&job_dir).unwrap();

    // Create paths
    let shm_path = job_dir.join("shared_memory.shm");
    let inputs   = job_dir.join("inputs");
    let outputs  = job_dir.join("outputs");
    let aout     = job_dir.join("a.out");
    let afl      = job_dir.join("afl-fuzz");
    std::fs::create_dir_all(&inputs).unwrap();
    std::fs::create_dir_all(&outputs).unwrap();
    
    // Get access to shared memory
    let shmem = get_shmem(&shm_path);
    if shmem.fuzz_cases.load(Ordering::Relaxed) > 0 {
        // Stats already present, we're just analyzing
        
        // Parse the stats
        let stats = process_data(shmem).unwrap();
        
        // Unmap shared memory
        unsafe {
            libc::munmap(shmem as *const _ as *mut _, 0);
        }

        // Save the stats
        progress.lock().unwrap().stats = Some(stats);
        return;
    }

    // Copy the binaries
    std::fs::copy("../afl_test/a.out", &aout).unwrap();
    std::fs::copy("../../AFLplusplus/afl-fuzz", &afl).unwrap();

    // Create a template input file for AFL to build on
    std::fs::write(inputs.join("test_input"), vec![0u8; 8192]).unwrap();
    
    unsafe {
        // For some reason we get an ETEXTBUSY if we don't wait a bit here
        // or sync out the writes of the a.out
        libc::sync();

        // Update start time for the fuzzer
        shmem.start_time.store(
            core::arch::x86_64::_rdtsc(), Ordering::SeqCst);
    }

    {
        // Fuzz start timer
        progress.lock().unwrap().start_time = Some(Instant::now());
    }

    // Create a new fuzzer instance
    let mut process = Command::new(command[0])
        .current_dir(job_dir)
        .env("AFL_NO_AFFINITY", "1")
        .env("AFL_SKIP_CPUFREQ", "1")
        .args(&command[1..])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .spawn().unwrap();

    loop {
        if let Some(_) = process.try_wait().unwrap() {
            break;
        }

        std::thread::sleep(Duration::from_millis(50));

        // Update progress of the fuzzer
        let mut prog = progress.lock().unwrap();
        prog.cases    = shmem.fuzz_cases.load(Ordering::Relaxed);
        prog.coverage = shmem.coverage.load(Ordering::Relaxed);
    }

    // Parse the stats
    let stats = process_data(shmem).unwrap();
    
    // Unmap shared memory
    unsafe {
        libc::munmap(shmem as *const _ as *mut _, 0);
    }

    // Save the stats
    progress.lock().unwrap().stats = Some(stats);
}

fn main() -> Result<(), Box<dyn Error>> {
    /// Number of times to run each fuzzer to average data over
    const NUM_AVERAGES: usize = 100;

    /// Number of fuzz cases to perform before exiting the fuzzer
    const NUM_CASES: &'static str = "5000000";
    
    // Remove all temporary directories
    //let _ = std::fs::remove_dir_all("./temps");
    
    // All the different fuzzers we want to explore
    // This holds the name of the fuzz job and the command to invoke to run
    // the fuzzer
    let fuzz_modes = [
        ("internal-corrupt-1", &["./a.out", NUM_CASES, "internal", "1"][..]),
        ("internal-corrupt-2", &["./a.out", NUM_CASES, "internal", "2"][..]),
        ("internal-corrupt-4", &["./a.out", NUM_CASES, "internal", "4"][..]),
        ("internal-corrupt-8", &["./a.out", NUM_CASES, "internal", "8"][..]),
        ("internal-corrupt-16", &["./a.out", NUM_CASES, "internal", "16"][..]),
        ("internal-corrupt-32", &["./a.out", NUM_CASES, "internal", "32"][..]),
        ("internal-corrupt-64", &["./a.out", NUM_CASES, "internal", "64"][..]),
        
        ("afl-explore", &["./afl-fuzz",
            "-p", "explore",
            "-d", "-E", NUM_CASES, "-i", "inputs", "-o", "outputs",
            "./a.out", NUM_CASES, "@@"][..]),
        
        ("afl-fast", &["./afl-fuzz",
         "-p", "fast",
         "-d", "-E", NUM_CASES, "-i", "inputs", "-o", "outputs", "./a.out",
         NUM_CASES, "@@"][..]),
    ];

    // Tracks which fuzzer should be run
    let mut to_run = VecDeque::new();
    for run_id in 0..NUM_AVERAGES {
        for (name, command) in &fuzz_modes {
            let progress = Arc::new(Mutex::new(FuzzerProgress {
                cases:      0,
                coverage:   0,
                start_time: None,
                stats:      None,
            }));
            to_run.push_back(
                (name.to_string(), run_id, *command, progress));
        }
    }

    // Tracks which fuzzers are running
    let mut running: Vec<(String, usize, &[&str], Arc<Mutex<FuzzerProgress>>)>=
        Vec::new();

    // Holds complete fuzz jobs
    let mut complete = Vec::new();
    let mut last_done = 0.;

    // Keep track of the status last print time
    let start_time = Instant::now();
    let mut last_print = Instant::now();
    'done_running: loop {
        // Wait for us to have some threads available
        while running.len() >= 250 || to_run.len() == 0 {
            std::thread::sleep(Duration::from_millis(50));

            // Remove any running process if it has completed, indicated by
            // the `stats` field being populated
            let print_status = if last_print.elapsed().as_secs_f64() > 1.0 {
                print!("\x1b[2J");
                last_print = Instant::now();
                true
            } else {
                false
            };

            let num_cases = NUM_CASES.parse::<f64>().unwrap();
            let mut done_cases = num_cases * complete.len() as f64;

            // Compute total number of cases we need to complete
            let total_cases = num_cases * fuzz_modes.len() as f64 *
                NUM_AVERAGES as f64;

            running.retain(|job| {
                let mut progress = job.3.lock().unwrap();
                if progress.stats.is_none() {
                    // Job has just barely started, process not spawned yet
                    if progress.start_time.is_none() { return true; }

                    // Get the time since we statred the job
                    let elapsed = progress.start_time.unwrap().elapsed()
                        .as_secs_f64();

                    // Job is still running
                    if print_status {
                        print!("Job {:25} {:4} | uptime {:6.1} | \
                                cases {:10} | cov {:5}\n",
                            job.0, job.1, elapsed, progress.cases,
                            progress.coverage);

                        // Update total number of cases
                        done_cases += progress.cases as f64;
                    }

                    true
                } else {
                    // Job is done, save the complete information and remove
                    // it from the running list
                    complete.push((job.0.clone(),
                        progress.stats.take().unwrap()));
                    false
                }
            });
                
            if print_status {
                let prog = done_cases / total_cases;
                let mps  = (done_cases - last_done) / 1e6;
                print!("[{:7.1}] | {:8.3} M of {:8.3} M [{:6.2} M/sec] \
                        [{:5.4}] | {:5.0} sec remain\n",
                    start_time.elapsed().as_secs_f64(),
                    done_cases / 1e6, total_cases / 1e6,
                    mps, prog, (total_cases - done_cases) / 1e6 / mps);

                last_done = done_cases;
            }

            if to_run.len() == 0 && running.len() == 0 {
                // All done with jobs
                break 'done_running;
            }
        }

        // Get the job job
        let next_schedule = to_run.pop_front().unwrap();

        // Run the job
        let name = next_schedule.0.clone();
        let id   = next_schedule.1;
        let cmd  = next_schedule.2;
        let prog = next_schedule.3.clone();
        std::thread::spawn(move || {
            worker(name, id, cmd, prog);
        });

        // Put the job into the running job list
        running.push(next_schedule);
    }

    #[derive(Default)]
    struct Stat {
        find:  BTreeMap<usize, Vec<u64>>,
        rdtsc: BTreeMap<usize, Vec<u64>>,
    }

    // All jobs complete, create summary buckets
    let mut summary = BTreeMap::new();
    for (name, stats) in complete {
        let stat = summary.entry(name.to_string())
            .or_insert_with(|| Stat::default());

        for (&blkid, &val) in &stats.block_find {
            let ent = stat.find.entry(blkid).or_insert(Vec::new());
            ent.push(val);
            ent.sort_by_key(|&x| x);
        }
        for (&blkid, &val) in &stats.block_rdtsc {
            let ent = stat.rdtsc.entry(blkid).or_insert(Vec::new());
            ent.push(val);
            ent.sort_by_key(|&x| x);
        }
    }

    // Create graph file
    let mut outfd = File::create("log.txt").unwrap();

    for (name, stat) in summary.iter_mut() {
        // Name header for data
        write!(outfd, "\n\n{}\n", name).unwrap();

        for block in stat.find.keys() {
            // Get the arrays of find and rdtsc times
            let find  = &stat.find[block];
            let rdtsc = &stat.rdtsc[block];

            let med_find  = find[find.len()   / 2];
            let med_rdtsc = rdtsc[rdtsc.len() / 2];

            let sum_find  = find.iter().map(|&x| x as f64).sum::<f64>();
            let sum2_find = find.iter().map(|&x|
                    x as f64 * x as f64).sum::<f64>();
            let mean_find = sum_find / find.len() as f64;
            let std_find  =
                ((sum2_find / find.len() as f64) -
                 (mean_find * mean_find)).sqrt();

            write!(outfd, "{:5} {:20} {:20.2} {:20.2} {:20} {:10}\n",
                   block,
                   med_find, mean_find, std_find,
                   med_rdtsc,
                   find.len()).unwrap();
        }
    }

    Ok(())
}

