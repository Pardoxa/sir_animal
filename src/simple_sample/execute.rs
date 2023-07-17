use super::{DefaultOpts, SimpleSample, SimpleSampleScan, SimpleSampleSpecific, SimpleSampleCheck};
use crate::misc::{parse, write_json, write_commands};
use crate::sir_nodes::{SirFun, TransFun};
use indicatif::ProgressIterator;
use net_ensembles::Node;
use serde_json::Value;
use std::sync::atomic::*;
use std::{num::*, sync::Mutex, ops::DerefMut};
use net_ensembles::sampling::histogram::*;
use rand_pcg::Pcg64;
use net_ensembles::rand::SeedableRng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

pub fn execute_ss_check(param: DefaultOpts)
{
    let threads = param.num_threads.unwrap_or(NonZeroUsize::new(1).unwrap());
    let (s_param, json): (SimpleSampleCheck, _) = parse(param.json.as_ref());

    crate::sir_nodes::fun_choose!(
        ss_check_run,
        s_param.opts.base_opts.fun,
        (s_param, json, threads)
    );


}

fn ss_check_run<T>
(
    param: SimpleSampleCheck,
    json: Value,
    threads: NonZeroUsize 
) where T: Default + Clone + Send + Sync + 'static + TransFun,
SirFun<T>: Node
{

    let samples_per_threads =  param.opts.samples.get() / threads.get();

    let name = &param.output_name;
    let file = File::create(name).unwrap();
    let mut buf = BufWriter::new(file);
    let _ = write_commands(&mut buf);
    write_json(&mut buf, &json);

    let sir_rng = Pcg64::seed_from_u64(param.opts.base_opts.sir_seed);
    let lock = Mutex::new(sir_rng);

    let base = param.opts.base_opts;
    let model = base.construct::<T>();
    
    let opportunity_sample_count = AtomicU64::new(0);
    let opportunity_counter_sum = AtomicU64::new(0);

    let lambda_sum = Mutex::new(0.0);

    let infected_next_to_humans_count = AtomicU64::new(0);
    let c_animal_sum = AtomicU64::new(0);


    (0..threads.get())
        .into_par_iter()
        .for_each(
            |_|
            {
                let mut model = model.clone();
                let mut rng_lock = lock.lock().unwrap();
                let rng = Pcg64::from_rng(rng_lock.deref_mut()).unwrap();
                drop(rng_lock);
                model.sir_rng = rng;
                for _ in 0..samples_per_threads
                {
                    let res = model.checking_iterate();
                    if let Some(lambda) = res.average_lambda_of_animals_that_have_potential_to_infect_first_human
                    {
                        let mut lock = lambda_sum.lock().unwrap();
                        *lock += lambda;
                        drop(lock);
                        opportunity_counter_sum.fetch_add(res.count_of_animals_that_have_opportunity_to_infect_human, Ordering::Relaxed);
                        opportunity_sample_count.fetch_add(1, Ordering::Relaxed);
                    }
                    infected_next_to_humans_count.fetch_add(res.count_of_infected_animals_next_to_humans, Ordering::Relaxed);
                    c_animal_sum.fetch_add(res.c_animals, Ordering::Relaxed);
                }
            }
        );
    let opportunity_sample_count = opportunity_sample_count.into_inner();
    let opportunity_counter_sum = opportunity_counter_sum.into_inner();
    let infected_next_to_humans_count = infected_next_to_humans_count.into_inner();
    let c_animal_sum = c_animal_sum.into_inner();

    let av_c_animals = c_animal_sum as f64 / (samples_per_threads * threads.get()) as f64;

    let av_lambda = lambda_sum.into_inner().unwrap() / opportunity_sample_count as f64;
    let av_opportunities = opportunity_counter_sum as f64 / opportunity_sample_count as f64;

    let av_next_to_counter = infected_next_to_humans_count as f64 / (samples_per_threads * threads.get()) as f64;

    writeln!(buf, "#av_lambda av_opportunities av_next_to_counter av_c_animals").unwrap();

    writeln!(
        buf,
        "{av_lambda} {av_opportunities} {av_next_to_counter} {av_c_animals}"
    ).unwrap();
}

pub fn execute_simple_sample_specific(param: DefaultOpts)
{
    let (s_param, json): (SimpleSampleSpecific, _) = parse(param.json.as_ref());
    crate::sir_nodes::fun_choose!(
        simple_sample_specific,
        s_param.opts.base_opts.fun,
        (s_param, json, param.num_threads)
    )
}

fn simple_sample_specific<T>
(
    mut param: SimpleSampleSpecific,
    json: Value,
    threads: Option<NonZeroUsize> 
) where T: Default + Clone + Send + Sync + 'static + TransFun,
SirFun<T>: Node
{
    param.sigma_list.sort_unstable_by(|a,b| a.total_cmp(b));
    let threads = threads.unwrap_or(NonZeroUsize::new(1).unwrap());
    let samples_per_threads =  param.opts.samples.get() / threads.get();
    let rest = param.opts.samples.get() - samples_per_threads * threads.get();
    if rest != 0 {
        println!("Skipping {rest} samples for easy parallel processing");
    }
    let total_samples = param.opts.samples.get() - rest;
    let total_samples = total_samples as f64;

    let name = "specific.dat";
    let file = File::create(name).unwrap();
    println!("creating {name}");
    let mut buf = BufWriter::new(file);
    write_commands(&mut buf).unwrap();
    write_json(&mut buf, &json);

    let name = "specific_C.dat";
    let file = File::create(name).unwrap();
    println!("creating {name}");
    let mut buf_c = BufWriter::new(file);
    write_commands(&mut buf_c).unwrap();
    write_json(&mut buf_c, &json);

    let sir_rng = Pcg64::seed_from_u64(param.opts.base_opts.sir_seed);
    let lock = Mutex::new(sir_rng);

    let bar = indicatif::ProgressBar::new(param.sigma_list.len() as u64)
        .with_style(indicatif::ProgressStyle::with_template("[{elapsed_precise}] - [{eta}] - [{duration}] {bar:40.cyan/blue} {pos}/{len}").unwrap());

    for &sigma in param.sigma_list.iter().progress_with(bar)
    {

        let mut base = param.opts.base_opts.clone();
        base.sigma = sigma;
        let model = base.construct::<T>();
        let hits = AtomicU64::new(0);
        let c_sum = AtomicUsize::new(0);
        let c_sq_sum = AtomicUsize::new(0);

        (0..threads.get())
            .into_par_iter()
            .for_each(
                |_|
                {
                    let mut model = model.clone();
                    let mut rng_lock = lock.lock().unwrap();
                    let rng = Pcg64::from_rng(rng_lock.deref_mut()).unwrap();
                    drop(rng_lock);
                    model.sir_rng = rng;
                    for _ in 0..samples_per_threads
                    {
                        model.iterate_until_extinction();
                        let c = model.count_c_humans();
                        if c > 0 {
                            hits.fetch_add(1, Ordering::Relaxed);
                        }
                        c_sum.fetch_add(c, Ordering::Relaxed);
                        c_sq_sum.fetch_add(c*c, Ordering::Relaxed);
                    }
                }
            );

        let hits = hits.into_inner();
        let prob = hits as f64 / total_samples;

        writeln!(buf, "{sigma} {prob}").unwrap();

        let c_sum = c_sum.into_inner();
        let c_sq_sum = c_sq_sum.into_inner();
        let average_c = c_sum as f64 / total_samples;
        let var = c_sq_sum as f64 / total_samples - average_c * average_c;
        writeln!(buf_c, "{sigma} {average_c} {var}").unwrap();
    }
    
}

pub fn execute_simple_sample_scan(param: DefaultOpts)
{
    let (s_param, json): (SimpleSampleScan, _) = parse(param.json.as_ref());
    crate::sir_nodes::fun_choose!(
        simple_sample_scan,
        s_param.opts.base_opts.fun,
        (s_param, json, param.num_threads)
    )
}

fn simple_sample_scan<T>
(
    param: SimpleSampleScan,
    json: Value,
    threads: Option<NonZeroUsize> 
) where T: Default + Clone + Send + Sync + 'static + TransFun,
SirFun<T>: Node
{

    let threads = threads.unwrap_or(NonZeroUsize::new(1).unwrap());
    let samples_per_threads =  param.opts.samples.get() / threads.get();
    let rest = param.opts.samples.get() - samples_per_threads * threads.get();
    if rest != 0 {
        println!("Skipping {rest} samples for easy parallel processing");
    }
    let total_samples = param.opts.samples.get() - rest;
    let total_samples = total_samples as f64;

    let name = format!("scan_grid_prob{}.dat", param.mut_samples);
    let file = File::create(&name).unwrap();
    println!("creating {name}");
    let mut buf = BufWriter::new(file);
    write_commands(&mut buf).unwrap();
    write_json(&mut buf, &json);

    let name = format!("scan_grid_C{}.dat", param.mut_samples);
    let file = File::create(&name).unwrap();
    println!("creating {name}");
    let mut buf_c = BufWriter::new(file);
    write_commands(&mut buf_c).unwrap();
    write_json(&mut buf_c, &json);

    let name = format!("scan_grid_prob{}_matrix.dat", param.mut_samples);
    let file = File::create(&name).unwrap();
    println!("creating {name}");
    let mut buf_matr = BufWriter::new(file);
    write_commands(&mut buf_matr).unwrap();
    write_json(&mut buf_matr, &json);

    let name = format!("scan_grid_C{}_matrix.dat", param.mut_samples);
    let file = File::create(&name).unwrap();
    println!("creating {name}");
    let mut buf_c_matr = BufWriter::new(file);
    write_commands(&mut buf_c_matr).unwrap();
    write_json(&mut buf_c_matr, &json);

    let sir_rng = Pcg64::seed_from_u64(param.opts.base_opts.sir_seed);
    let lock = Mutex::new(sir_rng);

    let bar = indicatif::ProgressBar::new(param.mut_samples.get() as u64)
        .with_style(indicatif::ProgressStyle::with_template("[{elapsed_precise}] - [{eta}] - [{duration}] {bar:40.cyan/blue} {pos}/{len}").unwrap());

    for j in (0..param.mut_samples.get()).progress_with(bar)
    {
        let gamma = if j == param.mut_samples.get() -1 {
            param.gamma_end
        } else {
            let gamma_start = param.opts.base_opts.initial_gamma;
            let diff = (param.gamma_end - gamma_start) / ((param.mut_samples.get() - 1) as f64);
            gamma_start + diff * j as f64
        };

        for i in 0..param.mut_samples.get()
        {
            let mutation = if i == param.mut_samples.get() -1 {
                param.sigma_end
            } else {
                let mutation_start = param.opts.base_opts.sigma;
                let diff = (param.sigma_end - mutation_start) / ((param.mut_samples.get() - 1) as f64);
                mutation_start + diff * i as f64
            };
    
            let mut base = param.opts.base_opts.clone();
            base.sigma = mutation;
            base.initial_gamma = gamma;
            let model = base.construct::<T>();
            let hits = AtomicU64::new(0);

            let c_sum = AtomicUsize::new(0);
            let c_sq_sum = AtomicUsize::new(0);
    
            (0..threads.get())
                .into_par_iter()
                .for_each(
                    |_|
                    {
                        let mut model = model.clone();
                        let mut rng_lock = lock.lock().unwrap();
                        let rng = Pcg64::from_rng(rng_lock.deref_mut()).unwrap();
                        drop(rng_lock);
                        model.sir_rng = rng;
                        for _ in 0..samples_per_threads
                        {
                            model.iterate_until_extinction();
                            let c = model.count_c_humans();
                            if c > 0 {
                                hits.fetch_add(1, Ordering::Relaxed);
                            }
                            c_sum.fetch_add(c, Ordering::Relaxed);
                            c_sq_sum.fetch_add(c*c, Ordering::Relaxed);
                        }
                    }
                );
    
            let hits = hits.into_inner();
            let prob = hits as f64 / total_samples;
    
            writeln!(buf, "{mutation} {gamma} {prob}").unwrap();
            write!(buf_matr, "{prob} ").unwrap();
    
            let c_sum = c_sum.into_inner();
            let c_sq_sum = c_sq_sum.into_inner();

            let average_c = c_sum as f64 / total_samples;
            let var = c_sq_sum as f64 / total_samples - average_c * average_c;

            writeln!(buf_c, "{mutation} {gamma} {average_c} {var}").unwrap();
            write!(buf_c_matr, "{} ", average_c / param.opts.base_opts.system_size_humans.get() as f64).unwrap();

        }
        writeln!(buf_c_matr).unwrap();
        writeln!(buf_matr).unwrap();
        writeln!(buf).unwrap();
        writeln!(buf_c).unwrap();
    }
    
}

pub fn execute_simple_sample(param: DefaultOpts)
{
    let (s_param, json): (SimpleSample, _) = parse(param.json.as_ref());
    crate::sir_nodes::fun_choose!(
        simple_sample,
        s_param.base_opts.fun,
        (s_param, json, param.num_threads)
    )
}

fn simple_sample<T>(
    param: SimpleSample,
    json: Value,
    threads: Option<NonZeroUsize>
)where T: Default + Clone + Send + Sync + 'static + TransFun,
    SirFun<T>: Node
{
    let mut model = param.base_opts.construct::<T>();

    let max_c = model.dual_graph.graph_2().vertex_count();

    let bins = param.bins.map_or(max_c + 1, NonZeroUsize::get);

    let hist_c = AtomicHistUsize::new_inclusive(0, max_c, bins).unwrap();

    let hist_c_dogs = AtomicHistUsize::new_inclusive(0, model.dual_graph.graph_1().vertex_count(), model.dual_graph.graph_1().vertex_count() + 1).unwrap();

    let hist_gamma_animals = AtomicHistF64::new(-8.0, 8.0, 4000)
        .unwrap();

    let hist_gamma_humans = AtomicHistF64::new(-8.0, 8.0, 4000)
        .unwrap();

    let threads = threads.unwrap_or(NonZeroUsize::new(1).unwrap());

    let samples_per_threads =  param.samples.get() / threads.get();
    let rest = param.samples.get() - samples_per_threads * threads.get();
    if rest != 0 {
        println!("Skipping {rest} samples for easy parallel processing");
    }
    let sir_rng = Pcg64::from_rng(&mut model.sir_rng)
        .unwrap();
    let lock = Mutex::new(sir_rng);

    let bar = crate::misc::indication_bar((threads.get()*samples_per_threads) as u64);

    (0..threads.get())
        .into_par_iter()
        .for_each(
            |_|
            {
                let mut model = model.clone();
                let mut rng_lock = lock.lock().unwrap();
                let rng = Pcg64::from_rng(rng_lock.deref_mut()).unwrap();
                drop(rng_lock);
                model.sir_rng = rng;
                for i in 0..samples_per_threads
                {
                    model.iterate_until_extinction();
                    let c = model.count_c_humans();
                    hist_c.increment_quiet(c);
                    let c_dog = model.count_c_dogs();
                    hist_c_dogs.increment_quiet(c_dog);
                    for gamma in model.gamma_iter_humans(){
                        let _ = hist_gamma_humans.increment(gamma);
                    }
                    for gamma in model.gamma_iter_animals(){
                        let _ = hist_gamma_animals.increment(gamma);
                    }
                    if i % 10000 == 0 {
                        bar.inc(10000)
                    }
                }
            }
        );
    bar.finish();

    let name = param.quick_name();
    let hist_name = format!("{name}.hist");
    hist_to_file(&hist_c.into(), hist_name, &json);

    let hist_name = format!("{name}_animal.hist");
    hist_to_file(&hist_c_dogs.into(), hist_name, &json);

    let name_animals = format!("{name}_animal.Ghist");
    hist_float_to_file(&(hist_gamma_animals.into()), name_animals, &json);

    let name_humans = format!("{name}_human.Ghist");
    hist_float_to_file(&(hist_gamma_humans.into()), name_humans, &json);


}

pub fn hist_to_file(hist: &HistUsize, file_name: String, json: &Value)
{
    let normed = norm_hist(hist);

    println!("Creating {}", &file_name);
    let file = File::create(file_name)
        .unwrap();
    let mut buf = BufWriter::new(file);
    crate::misc::write_commands(&mut buf).unwrap();
    crate::misc::write_json(&mut buf, json);
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


pub fn hist_float_to_file(hist: &HistF64, file_name: String, json: &Value)
{
    let normed = norm_hist_float(hist);

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
                let center = (bin[0] + bin[1]) / 2.0;
                writeln!(buf, "{} {} {} {} {}", center, log_prob, hits, bin[0], bin[1]).unwrap()
            }
        );
}

pub fn norm_hist_float(hist: &HistF64) -> Vec<f64>
{
    let mut density: Vec<_> = hist.hist()
        .iter()
        .map(|&hits| (hits as f64).log10())
        .collect();

    subtract_max(density.as_mut_slice());
    let int = integrate_log_float(density.as_slice(), hist.bin_iter());
    let sub = int.log10();
    density.iter_mut()
        .for_each(|v| *v -= sub);
    density
}


pub fn integrate_log_float<'a, I: Iterator<Item=&'a [f64;2]>>(curve: &[f64], bins: I) -> f64
{
    curve.iter()
        .zip(bins)
        .map(|(&val, bin)| (bin[1]-bin[0]) * 10_f64.powf(val))
        .sum()
}