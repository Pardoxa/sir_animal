use net_ensembles::sampling::{WangLandauEnsemble, WangLandau, WangLandauHist};
use rand_pcg::Pcg64;

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

    let mut wl = WangLandau1T::new(
        opts.log_f_threshold, 
        ld_model, 
        rng, 
        opts.markov_step_size.get(), 
        hist, 
        5000
    ).expect("unable to build wl");

    wl.init_greedy_heuristic(
        |model| Some(model.calc_c()), 
        None
    ).expect("unable to init");

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
