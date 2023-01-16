use {
    crate::{
        simple_sample::DefaultOpts,
        misc::*,
        sir_nodes::{TransFun, fun_choose}
    },
    std::{
        time::Instant,
        num::*,
        fs::File,
        io::{BufWriter, Write}
    },
    super::*,
    serde_json::Value,
    serde::Serialize,
    net_ensembles::{
        sampling::{
            WangLandau1T,
            WangLandauEnsemble, 
            WangLandau, 
            WangLandauHist,
        },
        rand::SeedableRng
    },
    rand_distr::{Uniform, Distribution},
    rand_pcg::Pcg64,
    rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator}
};

pub fn execute_mutation_scan(def: DefaultOpts, start_time: Instant)
{
    let (param, json): (JumpScan, _) = parse(def.json.as_ref());

    fun_choose!(
        mutation_scan_exec, 
        param.base_opts.fun, 
        (
            param, 
            start_time, 
            def.num_threads,
            json
        )
    )
}


fn mutation_scan_exec<T>(
    opts: JumpScan,
    start_time: Instant,
    threads: Option<NonZeroUsize>,
    value: Value
)where T: Default + Serialize + Clone + 'static + TransFun
{
    if let Some(num) = threads
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num.get())
            .build_global()
            .unwrap();
    }



    let hist = {
        let end = 1;
        let interval = Interval{start: 0, end_inclusive: end};
        interval.get_hist(opts.neg_bins.get())
    };

    let mut rng = rand_pcg::Pcg64::seed_from_u64(opts.wl_seeding_seed);
    let mut markov_rng = rand_pcg::Pcg64::seed_from_u64(opts.markov_seeding_seed);

    let uniform = Uniform::new_inclusive(0, u64::MAX);

    let sample_param: Vec<_> = (0..opts.mutation_samples.get())
        .map(
            |i|
            {
                let new_wl_rng = Pcg64::from_rng(&mut rng).unwrap();
                
                let markov_seed = uniform.sample(&mut markov_rng);
                let mutation = if i == opts.mutation_samples.get() -1 {
                    opts.mutation_end
                } else {
                    let mutation_start = opts.base_opts.sigma;
                    let diff = (opts.mutation_end - mutation_start) / (opts.mutation_samples.get() as f64);
                    mutation_start + diff * i as f64
                };
                (new_wl_rng, markov_seed, mutation)
            } 
        ).collect();

    let error_msg = "unable to build wl";

    let allowed = opts.time.in_seconds();
    
    let mut res = Vec::new();

    sample_param.into_par_iter()
        .enumerate()
        .map(
            |(i, (rng, markov_seed, mutation))|
            {
                let mut base_opts = opts.base_opts.clone();
                base_opts.sigma = mutation;
                let base = base_opts.construct::<T>();

                let ld_model = LdModel::new(base, markov_seed, opts.max_time_steps, opts.neg_bins.get());

                let mut wl = 
                    WangLandau1T::new(
                        opts.log_f_threshold, 
                        ld_model, 
                        rng, 
                        opts.markov_step_size.get(), 
                        hist.clone(), 
                        50000
                    ).expect(error_msg);

                wl.init_greedy_heuristic(
                    |model| Some(model.calc_c().min(1)), 
                    None
                ).expect("unable to init");
            
                println!("finished greedy build of {i} after {}", humantime::format_duration(start_time.elapsed()));

                unsafe{
                    wl.ensemble_mut().unfinished_sim_counter = 0;
                    wl.ensemble_mut().total_sim_counter = 0;
                }

                let is_finished = wl.is_finished();
                let (sigma, prob) =  mutation_scan_helper(wl, start_time, allowed, &opts, vec![value.clone()]);
                (sigma, prob, is_finished)
            }
        ).collect_into_vec(&mut res);

    let file = File::create("result.dat").unwrap();
    let mut buf = BufWriter::new(file);

    write_commands(&mut buf).unwrap();
    write_json(&mut buf, &value);

    writeln!(buf, "#sigma probability finished").unwrap();
    for r in res 
    {
        let b = if r.2 {
            "true"
        } else {
            "false"
        };
        writeln!(buf, "{:e} {:e} {b}", r.0, r.1).unwrap()
    }


}

fn mutation_scan_helper<T>(
    mut wl: WL<T>, 
    start_time: Instant, 
    allowed: u64,
    opts: &JumpScan,
    json_vec: Vec<Value>
) -> (f64, f64)
where 
    T: Serialize + Clone + Default + TransFun
{

    unsafe{
        wl.wang_landau_while_unsafe(
            |model| Some(model.calc_c().min(1)), 
            |_| start_time.elapsed().as_secs() < allowed
        )
    }

    wl.ensemble().stats.log(std::io::stdout());

    let unfinished_count = wl.ensemble().unfinished_sim_counter;
    let total_sim_count = wl.ensemble().total_sim_counter;

    let unfinished_frac = unfinished_count as f64 / total_sim_count as f64;

    let num_of_continuation = json_vec.len();

    let sigma = wl.ensemble().sigma;

    let name = opts.quick_name_from_start(sigma);
    let name = format!("{name}_{num_of_continuation}.dat");
    println!("creating {name}");

    let mut density = wl.log_density_base10();
    net_ensembles::sampling::glue::norm_log10_sum_to_1(&mut density);
    
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
    for (bin, density) in hist.bin_iter().zip(density.iter())
    {
        writeln!(buf, "{bin} {:e}", density).unwrap();
    }

    let  name = opts.quick_name_from_start(sigma);
    let name = format!("{name}_{num_of_continuation}_sum.dat");
    println!("creating {name}");
    
    let file = File::create(name).expect("unable to create file");
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
    
    let mut sum = 0.0;
    for (bin, density) in hist.bin_iter().zip(density)
    {
        if bin <= 0 {
            sum += 10_f64.powf(density);
            if bin == 0{
                let val = sum.log10();
                writeln!(buf, "{bin} {:e}", val).unwrap();
            }
        } else {
            writeln!(buf, "{bin} {:e}", density).unwrap();
        }
    }

    
    let  name = opts.quick_name_from_start(sigma);
    let save_name = format!("{name}_{num_of_continuation}.bincode");
    println!("creating {save_name}");
    let file = File::create(save_name).unwrap();
    let buf = BufWriter::new(file);

    let json_vec: Vec<String> = json_vec
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    bincode::serialize_into(buf, &(wl, json_vec))
        .expect("Bincode serialization issue");
    (sigma, 1.0 - sum)
}