use net_ensembles::sampling::{WangLandauEnsemble, WangLandau, WangLandauHist};

use {
    crate::{
        simple_sample::DefaultOpts,
        misc::*
    },
    std::{
        time::Instant,
        num::*,
        fs::File,
        io::{BufWriter, Write}
    },
    super::*,
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

    unsafe{
        wl.wang_landau_while_unsafe(
            |model| Some(model.calc_c()), 
            |_| start_time.elapsed().as_secs() < allowed
        )
    }

    let unfinished_count = wl.ensemble().unfinished_sim_counter;
    let total_sim_count = wl.ensemble().total_sim_counter;

    let unfinished_frac = unfinished_count as f64 / total_sim_count as f64;

    let name = opts.quick_name_with_ending(".dat");
    println!("creating {name}");

    let density = wl.log_density_base10();
    let file = File::create(name)
        .unwrap();

    let mut buf = BufWriter::new(file);

    write_commands(&mut buf).unwrap();
    write_json(&mut buf, &value);
    writeln!(buf, "#steps: {}, log_f {}", wl.step_counter(), wl.log_f()).unwrap();
    wl.write_log(&mut buf).unwrap();

    writeln!(buf, "#Unfinished Sim: {unfinished_count} total: {total_sim_count}, unfinished_frac {unfinished_frac}")
        .unwrap();

    let hist = wl.hist();
    for (bin, density) in hist.bin_iter().zip(density)
    {
        writeln!(buf, "{bin} {:e}", density).unwrap();
    }

    let save_name = opts.quick_name_with_ending(".bincode");
    println!("creating {save_name}");
    let file = File::create(save_name).unwrap();
    let buf = BufWriter::new(file);

    bincode::serialize_into(buf, &wl)
        .expect("Bincode serialization issue")

}