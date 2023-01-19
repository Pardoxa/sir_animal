use super::{DefaultOpts, SimpleSample, SimpleSampleScan};
use crate::misc::{parse, write_json, write_commands};
use crate::sir_nodes::{SirFun, TransFun};
use indicatif::ProgressIterator;
use net_ensembles::Node;
use serde_json::Value;
use std::sync::atomic::AtomicU64;
use std::{num::*, sync::Mutex, ops::DerefMut};
use net_ensembles::sampling::histogram::*;
use rand_pcg::Pcg64;
use net_ensembles::rand::SeedableRng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

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

    let file = File::create("scan_grid.dat").unwrap();
    println!("creating scan.dat");
    let mut buf = BufWriter::new(file);
    write_commands(&mut buf).unwrap();
    write_json(&mut buf, &json);

    for j in 0..param.mut_samples.get()
    {
        let gamma = if j == param.mut_samples.get() -1 {
            param.gamma_end
        } else {
            let gamma_start = param.opts.base_opts.initial_gamma;
            let diff = (param.gamma_end - gamma_start) / ((param.mut_samples.get() - 1) as f64);
            gamma_start + diff * j as f64
        };

        for i in (0..param.mut_samples.get()).into_iter().progress_count(param.mut_samples.get() as u64)
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
            let mut model = base.construct::<T>();
            let hits = AtomicU64::new(0);
    
            let sir_rng = Pcg64::from_rng(&mut model.sir_rng)
                .unwrap();
            let lock = Mutex::new(sir_rng);
    
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
                                hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            
                        }
                    }
                );
    
            let hits = hits.into_inner();
            let prob = hits as f64 / total_samples;
    
            writeln!(buf, "{mutation} {gamma} {prob}").unwrap();
    
        }
        writeln!(buf).unwrap();
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