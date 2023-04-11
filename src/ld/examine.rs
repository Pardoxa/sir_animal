use std::{path::{PathBuf, Path}, io::{BufReader, BufWriter, Write}, fs::File, collections::BTreeMap, str::FromStr};
use bincode_helper::DeserializeAnyhow;
use net_ensembles::sampling::{HeatmapU, HistogramVal, Histogram, HistUsizeFast, HistF64, GnuplotSettings, GnuplotAxis};
use structopt::*;
use super::*;


#[derive(StructOpt, Debug, Clone)]
pub struct ExamineOptions
{
    /// Globbing to the bh files
    #[structopt(long, short)]
    pub glob: String,
}

#[derive(Debug, Clone, Copy)]
pub enum WhichMode
{
    Heatmap,
    Unknown
}


pub fn examine(opts: ExamineOptions)
{
    println!("You will now examine");
    for file in glob::glob(&opts.glob).unwrap()
    {
        println!("{file:?}");
    }
    println!("How would you like to examine them?");
    println!("0 for heatmap");
    let mut map = BTreeMap::new();
    map.insert(0, WhichMode::Heatmap);
    map.insert(1, WhichMode::Unknown);
    
    let mode = loop{
        for (key, val) in map.iter()
        {
            println!("press {key} for {val:?}");
        }
        let num = get_number();
        match map.get(&num){
            None => {
                eprintln!("invalid number, try again");
            },
            Some(entry) => {
                break *entry;
            }
        }
    };

    match mode{
        WhichMode::Heatmap => {
            heatmap_examiner(&opts.glob)
        },
        _ => unimplemented!()
    }


}

fn get_number<T>() -> T
where T: FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display
{
    loop{
        println!("input number");
        let mut buffer = String::new();
        let line = std::io::stdin().read_line(&mut buffer);
        let input = &buffer[..buffer.len()-1];
        match line {
            Ok(_) => {
                match input.parse(){
                    Ok(num) => {
                        return num
                    },
                    Err(e) => {
                        eprintln!("error during parsing stdin: {e} -  try again"); 
                        eprintln!("input was {input}");
                    }
                }
            },
            Err(error) => {
                eprintln!("error during reading stdin: {error} -  try again");
                eprintln!("input was {input}");
            }
        }
    };
    
}

pub fn heatmap_examiner(glob: &str){
    println!("creating heatmap for C values. input left number.");
    let left: usize = get_number();
    println!("input right number");
    let right: usize = loop {
        let right = get_number();
        if right > left {
            break right;
        }
        println!("right has to be smaller than left, try again");
    };
    let hist = HistUsizeFast::new_inclusive(left, right)
        .expect("unable to create the hist");

    println!("create val heatmap. Input left");
    let left: f64 = get_number();
    println!("input right number");
    let right: f64 = loop {
        let right = get_number();
        if right > left {
            break right;
        }
        println!("right has to be smaller than left, try again");
    };
    println!("how many intervals?");
    let num_intervals = loop{
        let num = get_number();
        if num < 2{
            println!("Has to be bigger than 2, try again")
        } else {
            break num;
        }
    };
    let hist_f = HistF64::new(left, right, num_intervals)
        .expect("unable to create hist");
    let mut heatmap = HeatmapU::new(hist, hist_f);
    #[allow(clippy::complexity)]
    let mut fun_map: BTreeMap<u8, (&str, fn ((usize, InfoGraph)) -> (usize, f64))> = BTreeMap::new();
    fun_map.insert(0, ("average_human_gamma", c_and_average_human_gamma));
    fun_map.insert(1, ("max gamma human", c_and_max_human_gamma));
    fun_map.insert(2, ("the average human lambda of the humans", c_and_average_human_human_lambda));
    fun_map.insert(3, ("max human to human lambda", c_and_max_human_human_lambda));
    fun_map.insert(4, ("median human to human lambda", c_and_median_human_human_lambda));
    fun_map.insert(100, ("animal max gamma", c_and_max_animal_gamma));
    fun_map.insert(101, ("animal average gamma", c_and_average_animal_gamma));
    fun_map.insert(102, ("the average animal lambda of the animals", c_and_average_animal_animal_lambda));
    fun_map.insert(103, ("the max animal lambda of the animals", c_and_max_animal_animal_lambda));
    fun_map.insert(104, ("max lambda from animal to human", c_and_max_from_animal_to_human_lambda));
    fun_map.insert(105, ("median animal to human lambda", c_and_median_animal_to_human_lambda));
    fun_map.insert(106, ("average animal to human lambda", c_and_average_animal_to_human_lambda));
    
    println!("choose function");
    let (fun, label) = loop{
        for (key, val) in fun_map.iter()
        {
            println!("for {} choose {key}", val.0);
        }
        let num = get_number();
        match fun_map.get(&num){
            None => {
                println!("invalid number, try again");
            },
            Some((label, fun)) => break (fun, label)
        }
    };

    println!("generating heatmap");

    for file in glob::glob(glob).unwrap()
    {
        let file = file.unwrap();
        println!("file {file:?}");
        let mut analyzer = TopAnalyzer::new(file);
        heatmap_count(&mut heatmap, analyzer.iter_all_info(), fun);
    }
    println!("Success");

    let out_file = loop{
        println!("input name of output file");
        let mut buffer = String::new();
        let line = std::io::stdin().read_line(&mut buffer);
        let input = &buffer[..buffer.len()-1];
        match line {
            Err(e) => println!("there was the error: {e:?}"),
            Ok(_) => {
                let mut opts = File::options();
                match opts.create_new(true).write(true).open(input) {
                    Err(e) => {
                        match e.kind(){
                            std::io::ErrorKind::AlreadyExists => {
                                println!("the file already exists. Do you want to overwrite it? y/n");
                                let mut buffer = String::new();
                                let line = std::io::stdin().read_line(&mut buffer);
                                let new_input = &buffer[..buffer.len()-1];
                                match line{
                                    Ok(_) => {
                                        if new_input.eq_ignore_ascii_case("y"){
                                            opts.create_new(false)
                                                .truncate(true);
                                            match opts.open(input) {
                                                Err(e) => eprintln!("error during file creation {e:?}"),
                                                Ok(f) => break f
                                            }
                                        }
                                    },
                                    Err(e) => {
                                        panic!("input error {e:?}");
                                    }
                                }
                            },
                            _ => eprintln!("unable to create file due to {e}\ntry again")
                        }
                    },
                    Ok(f) => break f,
                }
            }
        }
    };
    let buf = BufWriter::new(out_file);
    let heat = heatmap.heatmap_normalized_columns();

    let x_axis = GnuplotAxis::new(heat.width_hist().left() as f64, heat.width_hist().right() as f64, 10);

    let y_axis = GnuplotAxis::new(heat.height_hist().first_border(), *heat.height_hist().borders().last().unwrap(), 10);

    let mut gs = GnuplotSettings::new();
    gs.x_label("C")
        .x_axis(x_axis)
        .y_axis(y_axis)
        .y_label(*label);
    let _ = heat.gnuplot(buf, gs);

}



pub struct TopAnalyzer
{
    pub path: PathBuf,
    pub topology: TopologyGraph,
    pub de: DeserializeAnyhow<BufReader<File>>,
    pub read_counter: usize
}

impl TopAnalyzer{
    pub fn new<P>(path: P) -> Self
    where P: AsRef<Path>
    {
        let file = File::open(path.as_ref())
            .unwrap();
        let buf = BufReader::new(file);
        let mut de = DeserializeAnyhow::new(buf);

        let topology: TopologyGraph = de.deserialize().unwrap();
        Self{
            path: path.as_ref().to_path_buf(),
            topology,
            de,
            read_counter: 0
        }
    }

    pub fn iter_all(&mut self) -> impl Iterator<Item = (&TopologyGraph, (usize, CondensedInfo))>
    {
        let topology = &self.topology;
        let deserializer = &mut self.de;
        let counter = &mut self.read_counter;
        std::iter::from_fn(
            move ||
            {
                let compat: Option<(usize, CondensedInfo)> = deserializer.deserialize().ok();
                if compat.is_some(){
                    *counter += 1;
                }
                let compat = compat?;
                Some((topology, compat))
            }
        )
    }

    pub fn iter_all_info(&'_ mut self) -> impl Iterator<Item = (usize, InfoGraph)> + '_
    {
        let deserializer = &mut self.de;
        let counter = &mut self.read_counter;
        std::iter::from_fn(
            move ||
            {
                let compat: Option<(usize, CondensedInfo)> = deserializer.deserialize().ok();
                if compat.is_some(){
                    *counter += 1;
                }
                let compat = compat?;
                Some((compat.0, compat.1.to_info_graph()))
            }
        )
    }

    pub fn test_run(&mut self) 
    {
        let file = File::create("tmp_test.dat")
            .unwrap();
        let mut buf = BufWriter::new(file);

        writeln!(buf, "#C number_of_jumps prior_dogs_first_jump max_mutation_first_jump average_mut_first_jump")
            .unwrap();

        self.iter_all_info()
            .for_each(
                |(C, info)|
                {
                    let mutation = info.dog_mutations();
                    let prior = if let Some(pri) = mutation.dogs_prior_to_jump
                    {
                        format!("{pri}")
                    } else {
                        "NaN".to_owned()
                    };
                    writeln!(
                        buf, 
                        "{C} {} {} {} {}", 
                        mutation.number_of_jumps,
                        prior,
                        mutation.max_mutation,
                        mutation.average_mutation_on_first_infected_path
                    ).unwrap()
                }
            )
    }
}

fn c_and_average_animal_gamma(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut average = 0.0;
    let mut count = 0_u32;
    for gamma in item.1.animal_gamma_iter()
    {
        count += 1;
        average += gamma;
    }
    (item.0, average / count as f64)
}

// the average animal lambda of the animals
fn c_and_average_animal_animal_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut average = 0.0;
    let mut count = 0_u32;
    for gt in item.1.animal_gamma_trans_iter()
    {
        let lambda = gt.trans_animal;
        count += 1;
        average += lambda;
    }
    (item.0, average / count as f64)
}

// the average human lambda of the animals
fn c_and_average_animal_to_human_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut average = 0.0;
    let mut count = 0_u32;
    for gt in item.1.animal_gamma_trans_iter()
    {
        let lambda = gt.trans_human;
        count += 1;
        average += lambda;
    }
    (item.0, average / count as f64)
}

fn c_and_max_animal_gamma(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_gamma = f64::NEG_INFINITY;
    for gamma in item.1.animal_gamma_iter()
    {
        if max_gamma < gamma {
            max_gamma = gamma;
        }
    }
    if max_gamma == f64::NEG_INFINITY {
        max_gamma = f64::NAN;
    }
    (item.0, max_gamma)
}

fn c_and_max_animal_animal_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_lambda = f64::NEG_INFINITY;
    for gt in item.1.animal_gamma_trans_iter()
    {
        let lambda = gt.trans_animal;
        if max_lambda < lambda {
            max_lambda = lambda;
        }
    }
    if max_lambda == f64::NEG_INFINITY {
        max_lambda = f64::NAN;
    }
    (item.0, max_lambda)
}

fn c_and_max_from_animal_to_human_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_lambda = f64::NEG_INFINITY;
    for gt in item.1.animal_gamma_trans_iter()
    {
        let lambda = gt.trans_human;
        if max_lambda < lambda {
            max_lambda = lambda;
        }
    }
    if max_lambda == f64::NEG_INFINITY {
        max_lambda = f64::NAN;
    }
    (item.0, max_lambda)
}

fn c_and_median_animal_to_human_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut median_list: Vec<_> = item.1.animal_gamma_trans_iter()
        .map(|val| val.trans_human)
        .collect();
    median_list.sort_unstable_by(f64::total_cmp);
    let median = if median_list.is_empty()
    {
        f64::NAN
    } else {
        let len = median_list.len();
        median_list[len / 2]
    };
    (item.0, median)
}

fn c_and_average_human_gamma(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut average = 0.0;
    let mut count = 0_u32;
    for gamma in item.1.human_gamma_iter()
    {
        count += 1;
        average += gamma;
    }
    (item.0, average / count as f64)
}

fn c_and_median_human_human_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut median_list: Vec<_> = item.1.human_gamma_trans_iter()
        .map(|val| val.trans_human)
        .collect();
    median_list.sort_unstable_by(f64::total_cmp);
    let median = if median_list.is_empty()
    {
        f64::NAN
    } else {
        let len = median_list.len();
        median_list[len / 2]
    };
    (item.0, median)
}

fn c_and_average_human_human_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut average = 0.0;
    let mut count = 0_u32;
    for gt in item.1.human_gamma_trans_iter()
    {
        let lambda = gt.trans_human;
        count += 1;
        average += lambda;
    }
    (item.0, average / count as f64)
}

fn c_and_max_human_gamma(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_gamma = f64::NEG_INFINITY;
    for gamma in item.1.human_gamma_iter()
    {
        if max_gamma < gamma {
            max_gamma = gamma;
        }
    }
    if max_gamma == f64::NEG_INFINITY {
        max_gamma = f64::NAN;
    }
    (item.0, max_gamma)
}

fn c_and_max_human_human_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_lambda = f64::NEG_INFINITY;
    for gt in item.1.human_gamma_trans_iter()
    {
        let lambda = gt.trans_human;
        if max_lambda < lambda {
            max_lambda = lambda;
        }
    }
    if max_lambda == f64::NEG_INFINITY {
        max_lambda = f64::NAN;
    }
    (item.0, max_lambda)
}



pub fn heatmap_count<Hw, Hh, It, I, F>(
    heatmap: &mut HeatmapU<Hw, Hh>, 
    iter: It,
    fun: F
)
where It: Iterator<Item = I>,
    F: Fn (I) -> (usize, f64),
    Hw: HistogramVal<usize> + Histogram,
    Hh: HistogramVal<f64> + Histogram
{
    for i in iter {
        let (c, val) = fun(i);
        let _ = heatmap.count(c, val);
    }
}