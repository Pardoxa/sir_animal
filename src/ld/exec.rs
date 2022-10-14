use std::{sync::Mutex, ops::DerefMut};
use net_ensembles::{
    sampling::{
        WangLandauEnsemble, 
        WangLandau, 
        WangLandauHist, 
        EntropicSampling, Entropic, HeatmapUsize, GnuplotSettings,
        GnuplotAxis, EntropicEnsemble, Rewl, RewlBuilder, Rees
    }, 
    rand::Rng, MarkovChain
};
use rand_pcg::Pcg64;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;
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

pub fn execute_rewl(def: DefaultOpts, start_time: Instant)
{
    let (param, json) = parse(def.json.as_ref());
    execute_rewl_helper(
        param, 
        start_time, 
        def.num_threads,
        json
    )
}

fn execute_rewl_helper(
    mut opts: RewlOpts,
    start_time: Instant,
    threads: Option<NonZeroUsize>,
    value: Value
)
{
    opts.sort_interval();
    if let Some(num) = threads
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num.get())
            .build_global()
            .unwrap();
    }

    let base = opts.base_opts.construct();
    let ld_model = LdModel::new(base, opts.markov_seed, opts.max_time_steps);

    let hists: Vec<_> = opts.interval.iter()
        .map(|interval| HistUsizeFast::new_inclusive(interval.start as usize, interval.end_inclusive as usize).expect("Hist error"))
        .collect();

    let mut rng = rand_pcg::Pcg64::seed_from_u64(opts.wl_seed);


    let mut ensembles: Vec<_> = (0..hists.len()-1)
        .map(
            |_|
            {
                let mut clone = ld_model.clone();
                let rng = Pcg64::from_rng(&mut rng).unwrap();
                clone.re_randomize(rng);
                clone
            }
        ).collect();
    
    ensembles.push(ld_model);

    let rewl = RewlBuilder::from_ensemble_vec(
        ensembles, 
        hists, 
        opts.markov_step_size.get(), 
        NonZeroUsize::new(1600).unwrap(), 
        NonZeroUsize::new(1).unwrap(), 
        opts.log_f_threshold
    ).expect("unable to create rewl");
    
    let mut rewl = 
        rewl.greedy_build(|model| Some(model.calc_c()));

    println!("finished greedy build after {}", humantime::format_duration(start_time.elapsed()));

    unsafe{
        rewl.ensemble_iter_mut()
            .for_each(
                |ensemble|
                {
                    ensemble.unfinished_sim_counter = 0;
                    ensemble.total_sim_counter = 0;
                    ensemble.stats.reset();
                }
            );
    }

    let _ = rewl.change_sweep_size_of_interval(0, NonZeroUsize::new(10000).unwrap());

    let allowed = opts.time.in_seconds();

    rewl_helper(rewl, start_time, allowed, opts, vec![value])

}

fn rewl_helper(
    mut rewl: REWL, 
    start_time: Instant, 
    allowed: u64,
    quick_name: RewlOpts,
    json_vec: Vec<Value>
)
{
    
    rewl.simulate_while(
        |model| Some(model.calc_c()), 
        |_| start_time.elapsed().as_secs() < allowed
    );
    

    let min_roundtrips = rewl.roundtrip_iter().min().unwrap();
    println!("min roundtrips {min_roundtrips}");

    //let log_name = quick_name.quick_name(0);

    let unfinished_count = 
    rewl.ensemble_iter()
        .fold(
            0, 
            |acc, m| m.unfinished_sim_counter + acc
        );

    let total_simulations_count = 
    rewl.ensemble_iter()
        .fold(
            0, 
            |acc, m| m.total_sim_counter + acc
        );

    let rewl = Arc::new(rewl);
    let rewl_clone = rewl.clone();
    let unfinished_frac = unfinished_count as f64 / total_simulations_count as f64;
    
    let times_repeated = json_vec.len();
    let mut name = quick_name.quick_name(None, ReplicaMode::REWL, times_repeated);
    name.push_str(".stats");
    println!("creating: {name}");
    let handle = std::thread::spawn(move || {

        let file = File::create(&name)
            .expect("unable to create file");
        let mut buf = BufWriter::new(file);
    
        let _ = crate::misc::write_commands(&mut buf);
    
        rewl_clone
            .ensemble_iter()
            .enumerate()
            .for_each(
                |(index, ensemble)|
                {
                    let _ = writeln!(buf, "#ensemble: {index}");
                    ensemble.stats.log(&mut buf);
                }
            );
    });

    rewl.walkers()
        .iter()
        .enumerate()
        .for_each(
            |(index, walker)|
            {
                let name = quick_name.quick_name(Some(index), ReplicaMode::REWL, times_repeated);

                let name = format!("{name}.dat");
                println!("creating {name}");
                let file = File::create(name).unwrap();
                let mut buf = BufWriter::new(file);

                let density = walker.log10_density();
                let _ = crate::misc::write_commands(&mut buf);
                for json in json_vec.iter()
                {
                    crate::misc::write_json(&mut buf, json);
                }

                writeln!(buf, "#hist log10").unwrap();
                writeln!(buf, "#walker {index}, steps: {}", walker.step_count())
                    .unwrap();
                let log_f = walker.log_f();
                writeln!(buf, "# log_f: {log_f}").unwrap();

                writeln!(
                    buf, 
                    "# Replica_exchanges {}, proposed_replica_exchanges {} acceptance_rate {}",
                    walker.replica_exchanges(),
                    walker.proposed_replica_exchanges(),
                    walker.replica_exchange_frac()
                ).unwrap();

                writeln!(
                    buf,
                    "# Acceptance_rate_markov: {}",
                    walker.acceptance_rate_markov()
                ).unwrap();

                write!(buf, "#rewl roundtrips of all walkers:").unwrap();

                for roundtrip in rewl.roundtrip_iter()
                {
                    write!(buf, " {roundtrip}").unwrap();
                }
                writeln!(buf).unwrap();
                writeln!(buf, "#All walker: Unfinished Sim: {unfinished_count} total: {total_simulations_count}, unfinished_frac {unfinished_frac}")
                    .unwrap();

                let hist = walker.hist();
                for (bin, density) in hist.bin_iter().zip(density)
                {
                    writeln!(buf, "{bin} {:e}", density).unwrap();
                }
            }
        );
    
    handle.join().unwrap();
    let rewl = match Arc::try_unwrap(rewl){
        Ok(rewl) => rewl,
        Err(_) => {
            unreachable!()
        } 
    };

    let mut save_name = quick_name.quick_name(None, ReplicaMode::REWL, times_repeated);
    save_name.push_str(".bincode");
    println!("creating {save_name}");

    let file = File::create(save_name)
        .expect("unable to create file");
    let buf = BufWriter::new(file);
    
    let json_vec: Vec<String> = json_vec
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    bincode::serialize_into(buf, &(rewl, json_vec))
        .expect("Bincode serialization issue")

        
}

#[allow(clippy::upper_case_acronyms)]
pub type REES = Rees<(), LdModel, Pcg64, HistogramFast<usize>, usize, MarkovStep, ()>;

pub struct ReesExtra
{
    pub sir_writer_humans: SirWriter,
    pub sir_writer_dogs: SirWriter,
    pub other_info: ZippingWriter,
    pub every: NonZeroU64,
    pub layer_helper: LayerHelper
}

impl ReesExtra
{
    pub fn new(
        base_name: &str, 
        index: usize, 
        every: NonZeroU64,
        layer_helper: LayerHelper
    ) -> Self 
    {
        let humans = format!("{base_name}H");
        let sir_writer_humans = SirWriter::new(&humans, index);
        let dogs = format!("{base_name}D");
        let sir_writer_dogs = SirWriter::new(&dogs, index);
        let other_info = format!("{base_name}_{index}.other");
        let other_info = ZippingWriter::new(other_info);
        Self { sir_writer_humans, sir_writer_dogs, other_info, every, layer_helper }
    }
}

pub fn exec_rees_beginning(def: DefaultOpts, start_time: Instant)
{
    let (param, json) = parse(def.json.as_ref());

    rees_beginning(param, start_time, json)
}

fn rees_beginning(
    opts: BeginEntropicOpts,
    start_time: Instant,
    json: Value
)
{
    let (rewl, jsons): (REWL, Vec<String>) = generic_deserialize_from_file(&opts.file_name);

    let mut jsons = into_jsons(jsons);
    jsons.push(json);

    let copy = jsons[0].clone();
    let rewl_opts: RewlOpts = serde_json::from_value(copy).unwrap();

    let allowed = opts.time.unwrap().in_seconds();

    let rees = rewl.into_rees();

    let print_samples = opts.target_samples;
    let print_samples = print_samples.try_into().unwrap();

    rees_helper(
        rees, 
        start_time, 
        allowed, 
        rewl_opts, 
        jsons, 
        print_samples
    )
}

fn rees_helper(
    rees: REES, 
    start_time: Instant, 
    allowed: u64,
    quick_name: RewlOpts,
    json_vec: Vec<Value>,
    rees_print_samples: NonZeroU64
)
{
    let times_repeated = json_vec.len();

    let ensemble = rees.get_ensemble(0).unwrap();
    let animal_size = ensemble.dual_graph.graph_1().vertex_count();
    let human_size = ensemble.dual_graph.graph_2().vertex_count();
    drop(ensemble);
    
    let extra: Vec<_> = rees.walkers().iter().enumerate()
        .map(
            |(index, walker)|
            {
                let threshold = walker.step_threshold();
                let num = (threshold as f64 / rees_print_samples.get() as f64).floor();
                let every = NonZeroU64::new(num as u64)
                    .unwrap_or(NonZeroU64::new(1).unwrap());
                let name = quick_name.quick_name(
                    Some(index), 
                    ReplicaMode::REES, 
                    times_repeated
                );
                let layer_helper = LayerHelper::new(human_size, animal_size);
                ReesExtra::new(&name, index, every, layer_helper)
            }
        ).collect();

    let mut rees = match rees.add_extra(extra){
        Ok(rees) => rees,
        Err(_) => unreachable!()
    };



    rees.simulate_while(
        |model| Some(model.calc_c()),
        |_| start_time.elapsed().as_secs() < allowed, 
        |walker, ensemble, extra|
        {
            if walker.step_count() % extra.every == 0
            {
                let e = walker.energy_copy();
                ensemble.entropic_writer(
                    &mut extra.layer_helper, 
                    &mut extra.sir_writer_humans, 
                    &mut extra.sir_writer_dogs, 
                    e
                );

                let (res_dogs, res_humans) = extra
                    .layer_helper
                    .calc_layer_res(0.1, &ensemble.dual_graph);

                let dog_c = ensemble.current_c_dogs();
                let humans_infected_by_dogs = extra.layer_helper.humans_infected_by_dogs;
                let dogs_infected_by_humans = extra.layer_helper.dogs_infected_by_humans;
                let _ = write!(extra.other_info, "{e} {dog_c} {humans_infected_by_dogs} {dogs_infected_by_humans}");

                let nan = " NaN NaN NaN NaN";
                let _ = if let Some(res) = res_dogs
                {
                    write!(
                        extra.other_info, 
                        " {} {} {} {}",
                        res.max_index,
                        res.max_count,
                        res.layer,
                        res.gamma
                    )
                } else {
                    write!(extra.other_info, "{nan}")
                };

                let _ = if let Some(res) = res_humans
                {
                    writeln!(
                        extra.other_info, 
                        " {} {} {} {}",
                        res.max_index,
                        res.max_count,
                        res.layer,
                        res.gamma
                    )
                } else {
                    writeln!(extra.other_info, "{nan}")
                };
            }
        }
    );
    
    let rees = Arc::new(rees);
    let rees_clone = rees.clone();

    let mut name = quick_name.quick_name(None, ReplicaMode::REES, times_repeated);
    name.push_str(".stats");
    println!("creating: {name}");
    let handle = std::thread::spawn(move || {

        let file = File::create(&name)
            .expect("unable to create file");
        let mut buf = BufWriter::new(file);
    
        let _ = crate::misc::write_commands(&mut buf);
    
        rees_clone
            .ensemble_iter()
            .enumerate()
            .for_each(
                |(index, ensemble)|
                {
                    let _ = writeln!(buf, "#ensemble: {index}");
                    ensemble.stats.log(&mut buf);
                }
            );
    });

    let unfinished_count = 
    rees.ensemble_iter()
        .fold(
            0, 
            |acc, m| m.unfinished_sim_counter + acc
        );

    let total_simulations_count = 
    rees.ensemble_iter()
        .fold(
            0, 
            |acc, m| m.total_sim_counter + acc
        );

    let unfinished_frac = unfinished_count as f64 / total_simulations_count as f64;

    let min_roundtrips = rees.rees_roundtrip_iter().min().unwrap();
    println!("min roundtrips {min_roundtrips}");


    rees.walkers()
        .iter()
        .enumerate()
        .for_each(
            |(index, walker)|
            {
                let name = quick_name.quick_name(Some(index), ReplicaMode::REES, times_repeated);

                let name = format!("{name}.dat");
                println!("creating {name}");
                let file = File::create(name).unwrap();
                let mut buf = BufWriter::new(file);

                let density = walker.log10_density();
                let _ = crate::misc::write_commands(&mut buf);
                for json in json_vec.iter()
                {
                    crate::misc::write_json(&mut buf, json);
                }

                writeln!(buf, "#hist log10").unwrap();
                writeln!(buf, "#walker {index}, steps: {}", walker.step_count())
                    .unwrap();
                let step_count = walker.step_count();
                let step_goal = walker.step_threshold();
                let frac = step_count as f64 / step_goal as f64;
                writeln!(buf, "# frac: {frac}, steps_done: {step_count} step_goal: {step_goal}").unwrap();

                writeln!(
                    buf, 
                    "# Replica_exchanges {}, proposed_replica_exchanges {} acceptance_rate {}",
                    walker.replica_exchanges(),
                    walker.proposed_replica_exchanges(),
                    walker.replica_exchange_frac()
                ).unwrap();

                writeln!(
                    buf,
                    "# Acceptance_rate_markov: {}",
                    walker.acceptance_rate_markov()
                ).unwrap();

                write!(buf, "#rewl roundtrips of all walkers:").unwrap();
                for roundtrip in rees.rewl_roundtrip_iter()
                {
                    write!(buf, " {roundtrip}").unwrap();
                }
                writeln!(buf).unwrap();
                write!(buf, "#rees roundtrips of all walkers:").unwrap();
                for roundtrip in rees.rees_roundtrip_iter()
                {
                    write!(buf, " {roundtrip}").unwrap();
                }
                writeln!(buf).unwrap();
                writeln!(buf, "#All walker: Unfinished Sim: {unfinished_count} total: {total_simulations_count}, unfinished_frac {unfinished_frac}")
                    .unwrap();

                let hist = walker.hist();
                for (bin, density) in hist.bin_iter().zip(density)
                {
                    writeln!(buf, "{bin} {:e}", density).unwrap();
                }
            }
        );
    
    handle.join().unwrap();
    let rees = match Arc::try_unwrap(rees){
        Ok(rees) => rees,
        Err(_) => {
            unreachable!()
        } 
    };
    let (rees, _) = rees.unpack_extra();

    let mut save_name = quick_name.quick_name(None, ReplicaMode::REES, times_repeated);
    save_name.push_str(".bincode");
    println!("creating {save_name}");

    let file = File::create(save_name)
        .expect("unable to create file");
    let buf = BufWriter::new(file);
    
    let json_vec: Vec<String> = json_vec
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    bincode::serialize_into(buf, &(rees, json_vec))
        .expect("Bincode serialization issue")

        
}

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
            50000
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
            50000
        ).expect(error_msg)
    } else {
        WangLandau1T::new(
            opts.log_f_threshold, 
            ld_model, 
            rng, 
            opts.markov_step_size.get(), 
            hist, 
            50000
        ).expect(error_msg)
    };

    wl.init_greedy_heuristic(
        |model| Some(model.calc_c()), 
        None
    ).expect("unable to init");

    /*if wl.hist().left() == 0 {
        let mut initial = wl.log_density().clone();
        //initial[0] = 20000.0;
        //initial[1] = 2000.0;
        //initial[2] = 200.0;
        //initial[3] = 100.0;
        //initial[4] = 50.0;
        //initial[5] = 40.0;
        //initial[6] = 30.0;
    
        wl = wl.set_initial_probability_guess(initial, 0.1)
            .expect("unknown error");
    
        wl.init_greedy_heuristic(
            |model| Some(model.calc_c()), 
            None
        ).expect("unable to init");
    }*/


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

    wl.ensemble().stats.log(std::io::stdout());

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

fn into_jsons(jsons: Vec<String>) -> Vec<Value>
{
    jsons.into_iter()
        .map(|s| serde_json::from_str(&s).unwrap())
        .collect()
}
#[allow(clippy::upper_case_acronyms)]
pub type REWL = Rewl<LdModel, Pcg64, HistogramFast<usize>, usize, MarkovStep, ()>;

pub type WL = WangLandau1T<HistogramFast<usize>, Pcg64, LdModel, MarkovStep, (), usize>;
fn wl_continue(
    opts: WlContinueOpts, 
    start_time: Instant,
    json: Value
)
{
    let allowed = opts.time.in_seconds();

    
    let (mut wl, jsons): (WL, Vec<String>) = generic_deserialize_from_file(&opts.file_name);

    unsafe{wl.ensemble_mut().stats = MarkovStats::default()};

    let mut jsons = into_jsons(jsons);

    let copy = jsons[0].clone();
    let wl_opts: WlOpts = serde_json::from_value(copy).unwrap();
    jsons.push(json);
    wl_helper(wl, start_time, allowed, wl_opts, jsons)

}

pub fn exec_entropic_beginning(def: DefaultOpts, start_time: Instant)
{
    let (param, json) = parse(def.json.as_ref());

    entropic_beginning(param, start_time, json)
}

fn entropic_beginning(
    opts: BeginEntropicOpts,
    start_time: Instant,
    json: Value
)
{
    let (wl, jsons): (WL, Vec<String>) = generic_deserialize_from_file(&opts.file_name);

    let mut jsons = into_jsons(jsons);
    jsons.push(json);

    let mut entropic = EntropicSampling::from_wl(wl)
        .unwrap();

    let copy = jsons[0].clone();
    let wl_opts: WlOpts = serde_json::from_value(copy).unwrap();
    let wrapped: WlOptsEntropicWrapper = wl_opts.into();

    let base_name = wrapped.quick_name();

    let dog_c_name = format!("{base_name}.c_dogs");
    let mut dog_writer = ZippingWriter::new(dog_c_name);

    let total_steps = entropic.step_goal();

    let every = total_steps as f64 / opts.target_samples.get() as f64;
    let every = every.floor() as usize;
    let every = NonZeroUsize::new(every).unwrap_or(NonZeroUsize::new(1).unwrap());

    let hist = HistUsizeFast::new_inclusive(0, entropic.ensemble().dual_graph.graph_2().vertex_count())
        .unwrap();
    let gamma_hist = HistF64::new(-8.0, 8.0, 1000).unwrap();

    let mut heatmap_dogs = HeatmapUsize::new(gamma_hist, hist);
    let mut heatmap_humans = heatmap_dogs.clone();

    let sir_human_name = format!("{base_name}H");
    let sir_dog_name = format!("{base_name}D");

    let mut sir_writer_humans = SirWriter::new(&sir_human_name, 0);
    let mut sir_writer_animals = SirWriter::new(&sir_dog_name, 0);

    let human_size = entropic.ensemble().dual_graph.graph_2().vertex_count();
    let animal_size = entropic.ensemble().dual_graph.graph_1().vertex_count();

    let mut layer_helper = LayerHelper::new(human_size, animal_size);

    let name = format!("{base_name}HbD");
    let mut humans_by_dogs = ZippingWriter::new(name);
    let name = format!("{base_name}DbH");
    let mut dogs_by_humans = ZippingWriter::new(name);

    if let Some(time) = opts.time
    {
        let allowed = time.in_seconds();

        unsafe{
            entropic.entropic_sampling_while_unsafe(
                |model| Some(model.calc_c()),  
                |model| {
                    let e = *model.energy();

                    if model.steps_total() % every == 0 {
                        heatmap_humans
                            .count_multiple(model.ensemble().humans_gamma_iter(), e)
                            .unwrap();
                        heatmap_dogs
                            .count_multiple(model.ensemble().dogs_gamma_iter(), e)
                            .unwrap();
                        let c = model.ensemble_mut().calc_c();
                        assert_eq!(c, e);
                        model.ensemble_mut().entropic_writer(
                            &mut layer_helper, 
                            &mut sir_writer_humans, 
                            &mut sir_writer_animals, 
                            e
                        );
                        let _ = writeln!(humans_by_dogs, "{e} {}", layer_helper.humans_infected_by_dogs);
                        let _ = writeln!(dogs_by_humans, "{e} {}", layer_helper.dogs_infected_by_humans);
                        let dog_c = model.ensemble().current_c_dogs();
                        let _ = writeln!(dog_writer, "{e} {dog_c}");
                        let c = model.ensemble().dual_graph.graph_2().contained_iter().filter(|node| !node.is_susceptible()).count();
                        println!("C: {c} dogs {dog_c}");
                        assert_eq!(c, e);
                    }
                }, 
                |_| start_time.elapsed().as_secs() < allowed
            );
        };

    } else{
        unimplemented!()
    }

    let heat_name = format!("{base_name}HU.gp");
    print_heatmap(heatmap_humans, heat_name);

    let heat_name = format!("{base_name}D.gp");
    print_heatmap(heatmap_dogs, heat_name);

}

fn print_heatmap(heatmap: HeatmapUsize<HistogramFloat<f64>, HistogramFast<usize>>, name: String)
{
    let heatmap = heatmap.transpose_inplace();
    let heatmap = heatmap.into_heatmap_normalized_columns();
   
    println!("creating {name}");

    let heat_file = File::create(name)
        .unwrap();
    let buf = BufWriter::new(heat_file);

    let mut settings = GnuplotSettings::new();
    settings.x_label("C")
        .y_label("Gamma")
        .x_axis(GnuplotAxis::new(0.0, 1.0, 5))
        .y_axis(GnuplotAxis::new(-8.0, 8.0, 5));

    let _ = heatmap.gnuplot(buf, settings);
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