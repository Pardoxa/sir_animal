use std::{sync::Mutex, ops::DerefMut};
use net_ensembles::{sampling::{WangLandauEnsemble, WangLandau, WangLandauHist}, rand::Rng, MarkovChain};
use rand_pcg::Pcg64;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use {
    crate::{
        simple_sample::DefaultOpts,
        misc::*
    },
    std::{
        time::Instant,
        num::*,
        fs::File,
        io::{BufWriter, Write, BufReader}
    },
    super::*,
    serde::de::DeserializeOwned,
    serde_json::Value,
    net_ensembles::{
        sampling::{
            histogram::*,
            WangLandau1T,
        },
        rand::SeedableRng
    }
};

pub fn execute_wl(def: DefaultOpts, start_time: Instant)
{
    let (param, json) = parse(def.json.as_ref());
    execute_wl_helper(
        param, 
        start_time, 
        def.num_threads,
        json
    )
}

pub fn execute_wl_continue(def: DefaultOpts, start_time: Instant)
{
    let (param, json) = parse(def.json.as_ref());

    wl_continue(param, start_time, json)
}

fn execute_wl_helper(
    opts: WlOpts,
    start_time: Instant,
    threads: Option<NonZeroUsize>,
    value: Value
)
{
    if let Some(num) = threads
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num.get())
            .build_global()
            .unwrap();
    }

    let base = opts.base_opts.construct();
    let ld_model = LdModel::new(base, opts.markov_seed, opts.max_time_steps);

    let hist = if let Some(i) = opts.interval
    {
        HistUsizeFast::new_inclusive(i.start as usize, i.end_inclusive as usize)
            .unwrap()
    } else {
        let end = ld_model.dual_graph.graph_2().vertex_count();
        HistUsizeFast::new_inclusive(0, end).unwrap()
    };

    let rng = rand_pcg::Pcg64::seed_from_u64(opts.wl_seed);

    let error_msg = "unable to build wl";

    let mut wl = if let Some(val) = opts.init_with_at_least
    {
        let right = ld_model.dual_graph.graph_2().vertex_count();
        let other_hist = HistUsizeFast::new_inclusive(val.get(), right)
            .unwrap();

        let mut wl = WangLandau1T::new(
            opts.log_f_threshold, 
            ld_model, 
            rng, 
            opts.markov_step_size.get(), 
            other_hist, 
            5000
        ).expect(error_msg);

        wl.init_greedy_heuristic(
            |model| Some(model.calc_c()), 
            None
        ).expect("unable to init");

        let (ensemble, _, rng) = wl.into_inner();

        WangLandau1T::new(
            opts.log_f_threshold, 
            ensemble, 
            rng, 
            opts.markov_step_size.get(), 
            hist, 
            5000
        ).expect(error_msg)
    } else {
        WangLandau1T::new(
            opts.log_f_threshold, 
            ld_model, 
            rng, 
            opts.markov_step_size.get(), 
            hist, 
            5000
        ).expect(error_msg)
    };

    wl.init_greedy_heuristic(
        |model| Some(model.calc_c()), 
        None
    ).expect("unable to init");

    if wl.hist().left() == 0 {
        let mut initial = wl.log_density().clone();
        initial[0] = 20000.0;
        initial[1] = 2000.0;
        initial[2] = 200.0;
        initial[3] = 100.0;
        initial[4] = 50.0;
        initial[5] = 40.0;
        initial[6] = 30.0;
    
        wl = wl.set_initial_probability_guess(initial, 1.0)
            .expect("unknown error");
    
        wl.init_greedy_heuristic(
            |model| Some(model.calc_c()), 
            None
        ).expect("unable to init");
    }


    println!("finished greedy build after {}", humantime::format_duration(start_time.elapsed()));

    unsafe{
        wl.ensemble_mut().unfinished_sim_counter = 0;
        wl.ensemble_mut().total_sim_counter = 0;
    }

    let allowed = opts.time.in_seconds();

    wl_helper(wl, start_time, allowed, opts, vec![value])

}

fn wl_helper<Q>(
    mut wl: WL, 
    start_time: Instant, 
    allowed: u64,
    quick_name: Q,
    json_vec: Vec<Value>
) where Q: QuickName
{
    unsafe{
        wl.wang_landau_while_unsafe(
            |model| Some(model.calc_c()), 
            |_| start_time.elapsed().as_secs() < allowed
        )
    }

    let unfinished_count = wl.ensemble().unfinished_sim_counter;
    let total_sim_count = wl.ensemble().total_sim_counter;

    let unfinished_frac = unfinished_count as f64 / total_sim_count as f64;

    let num_of_continuation = json_vec.len();

    let name = quick_name.quick_name_with_ending(&format!("_{num_of_continuation}.dat"));
    println!("creating {name}");

    let density = wl.log_density_base10();
    let file = File::create(name)
        .unwrap();

    let mut buf = BufWriter::new(file);

    write_commands(&mut buf).unwrap();
    for v in json_vec.iter()
    {
        write_json(&mut buf, v);
    }
    
    writeln!(buf, "#steps: {}, log_f {}", wl.step_counter(), wl.log_f()).unwrap();
    wl.write_log(&mut buf).unwrap();

    writeln!(buf, "#Unfinished Sim: {unfinished_count} total: {total_sim_count}, unfinished_frac {unfinished_frac}")
        .unwrap();

    let hist = wl.hist();
    for (bin, density) in hist.bin_iter().zip(density)
    {
        writeln!(buf, "{bin} {:e}", density).unwrap();
    }

    let save_name = quick_name.quick_name_with_ending(&format!("_{num_of_continuation}.bincode"));
    println!("creating {save_name}");
    let file = File::create(save_name).unwrap();
    let buf = BufWriter::new(file);

    let json_vec: Vec<String> = json_vec
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    bincode::serialize_into(buf, &(wl, json_vec))
        .expect("Bincode serialization issue")
}

pub type WL = WangLandau1T<HistogramFast<usize>, Pcg64, LdModel, MarkovStep, (), usize>;
fn wl_continue(
    opts: WlContinueOpts, 
    start_time: Instant,
    json: Value
)
{
    let allowed = opts.time.in_seconds();

    
    let (wl, jsons): (WL, Vec<String>) = generic_deserialize_from_file(&opts.file_name);

    let mut jsons: Vec<Value> = jsons.into_iter()
        .map(|s| serde_json::from_str(&s).unwrap())
        .collect();

    let copy = jsons[0].clone();
    let wl_opts: WlOpts = serde_json::from_value(copy).unwrap();
    jsons.push(json);
    wl_helper(wl, start_time, allowed, wl_opts, jsons)

}

pub fn generic_deserialize_from_file<T>(filename: &str) -> T
where T: DeserializeOwned
{
    let file = File::open(filename).expect("unable to open bincode file");
    let buf = BufReader::new(file);

    let res: Result<T, _> =  bincode::deserialize_from(buf);

        res.expect("unable to parse binary file")
}

pub fn execute_markov_ss(opts: DefaultOpts)
{
    let (param, json) = parse(opts.json.as_ref());
    let num_threads = opts.num_threads.unwrap_or(NonZeroUsize::new(1).unwrap());
    execute_markov_ss_helper(&param, num_threads, json)
}


fn execute_markov_ss_helper(options: &LdSimpleOpts, num_threads: NonZeroUsize, json: Value)
{
    let base = options.base_opts.construct();

    let mut rng = Pcg64::seed_from_u64(options.markov_seed);
    let seed = rng.gen_range(0..u64::MAX);
    let ld_model = LdModel::new(base, seed, options.max_time_steps);

    let right = ld_model.dual_graph.graph_2().vertex_count();

    let hist_c = AtomicHistUsize::new_inclusive(0, right, right + 1)
        .unwrap();

    let samples_per_thread = options.samples.get() / num_threads.get();
    let rest = options.samples.get() - samples_per_thread * num_threads.get();
    if rest != 0 {
        println!("Skipping {rest} samples for easy parallel processing");
    }

    let bar = indication_bar((samples_per_thread * num_threads.get()) as u64);

    let lock = Mutex::new(rng);

    (0..num_threads.get())
        .into_par_iter()
        .for_each(
            |_|
            {
                let mut model = ld_model.clone();
                let mut locked = lock.lock().unwrap();
                let rng = Pcg64::from_rng(locked.deref_mut()).unwrap();
                drop(locked);
                model.re_randomize(rng);
                let mut steps = Vec::new();
                for i in 0..samples_per_thread
                {
                    model.m_steps(options.markov_step_size.get(), &mut steps);
                    let c = model.calc_c();
                    hist_c.increment_quiet(c);

                    if i % 10000 == 0 {
                        bar.inc(10000);
                    }
                }
            }
        );
    bar.finish();

    let name = options.quick_name_with_ending(".hist");

    let hist_c = hist_c.into();

    hist_to_file(&hist_c, name, &json)

}


pub fn hist_to_file(hist: &HistUsize, file_name: String, json: &Value)
{
    let normed = norm_hist(hist);

    println!("Creating {}", &file_name);
    let file = File::create(file_name)
        .unwrap();
    let mut buf = BufWriter::new(file);
    crate::misc::write_commands(&mut buf).unwrap();
    write!(buf, "#").unwrap();
    serde_json::to_writer(&mut buf, json)
        .unwrap();
    writeln!(buf).unwrap();
    writeln!(buf, "#bin_center log10_prob hits bin_left bin_right").unwrap();

    hist.bin_hits_iter()
        .zip(normed)
        .for_each(
            |((bin, hits), log_prob)|
            {
                let center = (bin[0] + bin[1]) as f64 / 2.0;
                writeln!(buf, "{} {} {} {} {}", center, log_prob, hits, bin[0], bin[1]).unwrap()
            }
        );
}

pub fn norm_hist(hist: &HistUsize) -> Vec<f64>
{
    let mut density: Vec<_> = hist.hist()
        .iter()
        .map(|&hits| (hits as f64).log10())
        .collect();

    subtract_max(density.as_mut_slice());
    let int = integrate_log(density.as_slice(), hist.hist().len());
    let sub = int.log10();
    density.iter_mut()
        .for_each(|v| *v -= sub);
    density
}

pub fn subtract_max(slice: &mut[f64])
{
    let mut max = std::f64::NEG_INFINITY;
    slice.iter()
        .for_each(
            |v|
            {
                if *v > max {
                    max = *v;
                }
            }
        );
    slice.iter_mut()
        .for_each(|v| *v -= max);
}

pub fn integrate_log(curve: &[f64], n: usize) -> f64
{
    let delta = 1.0 / n as f64;
    curve.iter()
        .map(|&val| delta * 10_f64.powf(val))
        .sum()
}