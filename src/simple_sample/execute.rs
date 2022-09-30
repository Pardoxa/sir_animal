use super::{DefaultOpts, SimpleSample};
use crate::misc::parse;
use serde_json::Value;
use std::{num::*, sync::Mutex, ops::DerefMut};
use net_ensembles::sampling::histogram::*;
use rand_pcg::Pcg64;
use net_ensembles::rand::SeedableRng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

pub fn execute_simple_sample(param: DefaultOpts)
{
    let (s_param, json) = parse(param.json.as_ref());
    simple_sample(s_param, json, param.num_threads)
}

fn simple_sample(
    param: SimpleSample,
    json: Value,
    threads: Option<NonZeroUsize>
)
{
    let mut model = param.base_opts.construct();

    let max_c = model.dual_graph.graph_2().vertex_count();

    let bins = param.bins.map_or(max_c + 1, NonZeroUsize::get);

    let hist_c = AtomicHistUsize::new_inclusive(0, max_c, bins).unwrap();

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
                    let c = model.count_c();
                    hist_c.increment_quiet(c);
                    if i % 1000 == 0 {
                        bar.inc(1000)
                    }
                }
            }
        );
    bar.finish();

    let name = param.quick_name();
    let hist_name = format!("{name}.hist");

    hist_to_file(&hist_c.into(), hist_name, &json)


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