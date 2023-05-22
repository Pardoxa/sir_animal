use std::{path::{PathBuf, Path}, io::{BufReader, BufWriter, Write}, fs::File, collections::BTreeMap, str::FromStr, num::NonZeroUsize};
use bincode_helper::DeserializeAnyhow;
use net_ensembles::sampling::{*};
use num::ToPrimitive;
use structopt::*;
use super::*;
use net_ensembles::traits::*;

#[derive(StructOpt, Debug, Clone)]
pub struct ExamineOptions
{
    /// Globbing to the bh files
    #[structopt(long, short)]
    pub glob: String,

    #[structopt(short = "x", long = "x_tics", default_value = "11")]
    pub x_tics: NonZeroUsize,

    #[structopt(short = "y", long = "y_tics", default_value = "11")]
    pub y_tics: NonZeroUsize
}


pub fn examine(opts: ExamineOptions)
{
    println!("You will now examine");
    for file in glob::glob(&opts.glob).unwrap()
    {
        println!("{file:?}");
    }
    heatmap_examiner(opts)

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

fn get_yes_or_no() -> bool
{
    let mut buffer = String::new();
    let line = std::io::stdin().read_line(&mut buffer);
    let new_input = &buffer[..buffer.len()-1];
    match line{
        Ok(_) => {
            new_input.eq_ignore_ascii_case("y") || new_input.eq_ignore_ascii_case("yes")
        },
        Err(e) => {
            panic!("input error {e:?}");
        }
    }
}
    
#[allow(clippy::enum_variant_names)]
enum FunctionInputChooser {
    WithTopology,
    WithoutTopology,
    MultipleInputsWithoutTopology,
    WithoutTopologyWithN(u16),
    WithoutTopologyWithFloatAndN(u16, f64)
}

fn get_function_input_chooser() -> FunctionInputChooser {
    loop {
        println!("Please choose an: ");
        println!("1. With topology (w/ topology)");
        println!("2. Without topology (w/o topology)");
        println!("3. Multiple inputs without topology (multi w/o topology)");
        println!("4. With N (w n <number>)");
        println!("5. With float and N (w n <number> <float>)");

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        let input = input.trim().to_lowercase();

        match input.as_str() {
            "1" | "w" | "with_topology" => return FunctionInputChooser::WithTopology,
            "2" | "w/o" | "without_topology" => return FunctionInputChooser::WithoutTopology,
            "3" | "multi" | "multiple_inputs_without_topology" => return FunctionInputChooser::MultipleInputsWithoutTopology,
            input_str if input_str.starts_with("w n") => {
                let input_parts: Vec<_> = input_str.split_whitespace().collect();
                match input_parts.len() {
                    3 => {
                        if let Ok(n) = input_parts[2].parse::<u16>() {
                            return FunctionInputChooser::WithoutTopologyWithN(n);
                        }
                    },
                    4 => {
                        if let Ok(n) = input_parts[2].parse::<u16>() {
                            if let Ok(f) = input_parts[3].parse::<f64>() {
                                return FunctionInputChooser::WithoutTopologyWithFloatAndN(n, f);
                            }
                        }
                    },
                    _ => (),
                }
            }
            _ => (),
        }
        println!("Invalid input. Please try again.")
    }
}


type MyHeatmap = HeatmapUsize<HistUsize, HistogramFloat<f64>>;

pub fn without_global_topology(heatmap_mean: &mut HeatmapAndMean<MyHeatmap>, opts: &ExamineOptions) -> String
{
    #[allow(clippy::complexity)]
    let mut fun_map: BTreeMap<u8, (&str, fn ((usize, InfoGraph)) -> (usize, f64))> = BTreeMap::new();
    fun_map.insert(0, ("average_human_gamma", c_and_average_human_gamma));
    fun_map.insert(1, ("max gamma human", c_and_max_human_gamma));
    fun_map.insert(2, ("the average human lambda of the humans", c_and_average_human_human_lambda));
    fun_map.insert(3, ("max human to human lambda", c_and_max_human_human_lambda));
    fun_map.insert(4, ("median human to human lambda", c_and_median_human_human_lambda));
    fun_map.insert(5, ("max previous dogs in infection chain - humans", c_and_previous_dogs_of_humans));
    fun_map.insert(6, ("average previous dogs - humans infected by animals", c_and_average_prev_dogs_of_humans_infected_by_animals));
    fun_map.insert(7, ("average mutation - humans only", c_and_average_mutation_humans));
    fun_map.insert(8, ("average positive mutation - humans only", c_and_average_positive_mutation_humans));
    fun_map.insert(9, ("average negative mutation - humans only", c_and_average_negative_mutation_humans));
    fun_map.insert(10, ("max child count of humans infected by animals", c_and_max_children_of_humans_infected_by_animals));
    fun_map.insert(11, ("max child count of humans infected by animals/C", c_and_frac_max_children_of_humans_infected_by_animals));
    fun_map.insert(12, ("average lambda change human <-> human", av_lambda_change_human_human_trans));
    fun_map.insert(13, ("frac negative lambda changes human <-> human", frac_negative_lambda_change_human_human_trans));
    fun_map.insert(14, ("av negative lambda changes human <-> human", av_negative_lambda_change_human_human_trans));
    fun_map.insert(15, ("av positive lambda changes human <-> human", av_positive_lambda_change_human_human_trans));
    fun_map.insert(16, ("frac max tree width - counting humans only", max_tree_width_div_total_humans_only));
    fun_map.insert(17, ("second largest child count of humans infected by animals", c_and_second_largest_children_of_humans_infected_by_animals));
    fun_map.insert(18, ("second largest child count of humans infected by animals STRICT", c_and_second_largest_children_of_humans_infected_by_animals_strict));    
    
    fun_map.insert(100, ("animal max gamma", c_and_max_animal_gamma));
    fun_map.insert(101, ("animal average gamma", c_and_average_animal_gamma));
    fun_map.insert(102, ("the average animal lambda of the animals", c_and_average_animal_animal_lambda));
    fun_map.insert(103, ("the max animal lambda of the animals", c_and_max_animal_animal_lambda));
    fun_map.insert(104, ("max lambda from animal to human", c_and_max_from_animal_to_human_lambda));
    fun_map.insert(105, ("median animal to human lambda", c_and_median_animal_to_human_lambda));
    fun_map.insert(106, ("average animal to human lambda", c_and_average_animal_to_human_lambda));
    fun_map.insert(107, ("average mutation only animals", c_and_average_mutation_animals));
    fun_map.insert(108, ("max mutation only animals", c_and_max_mutation_animals));
    fun_map.insert(109, ("average gamma of dogs infecting humans", c_and_average_gamma_of_dogs_infecting_humans));
    fun_map.insert(110, ("max gamma of dogs infecting humans", c_and_max_gamma_of_dogs_infecting_humans));
    fun_map.insert(111, ("number of dogs infecting humans", c_and_number_of_dogs_infecting_humans));
    fun_map.insert(112, ("average negative mutation animals", c_and_average_negative_mutation_animals));
    fun_map.insert(113, ("average positive mutation animals", c_and_average_positive_mutation_animals));
    fun_map.insert(114, ("average recovery time animals", c_and_average_recovery_time_animals));
    fun_map.insert(115, ("max recovery time animals", c_and_max_recovery_time_animals));
    fun_map.insert(116, ("average recovery time animals that infected humans", c_and_average_recovery_time_dogs_infecting_humans));
    fun_map.insert(117, ("max recovery time animals that infected humans", c_and_max_recovery_time_dogs_infecting_humans));
    fun_map.insert(118, ("recovery time of animal that infected the first human", c_recovery_time_of_first_dog_infecting_humans));
    fun_map.insert(119, ("av recovery time of animals on path to first human", c_and_average_recovery_duration_animals_on_path_to_first_human));
    fun_map.insert(120, ("av mutation of animals on path to first human", c_and_average_mutation_animals_on_path_to_first_human));
    fun_map.insert(121, ("av lambda change on path to human with most descendants", c_and_average_lambda_change_animals_on_path_to_human_with_most_children));
    fun_map.insert(122, ("frac of negative lambda change on path to human with most descendants", c_and_frac_of_negative_lambda_change_animals_on_path_to_human_with_most_children));
    fun_map.insert(123, ("av gamma change on path to human with most descendants", c_and_average_gamma_change_animals_on_path_to_human_with_most_children));
    fun_map.insert(124, ("av path len of path to human with most descendants", c_and_path_len_to_human_with_most_children));
    fun_map.insert(125, ("gamma of human with most descendants", c_and_gamma_of_human_with_most_children));
    fun_map.insert(126, ("lambda of human with most descendants", c_and_lambda_of_human_with_most_children));
    fun_map.insert(127, ("lambda of animal before human with most descendants", c_and_lambda_of_animal_before_human_with_most_children));
    fun_map.insert(128, ("C animals", c_and_total_animals));

    
    
    fun_map.insert(200, ("maximum of all mutations", total_mutation_max));
    fun_map.insert(201, ("average of all mutations", total_mutation_average));
    fun_map.insert(202, ("sum of all mutations", total_mutation_sum));
    fun_map.insert(203, ("sum of abs of all mutations", total_mutation_sum_abs));
    fun_map.insert(204, ("abs of all mutations / nodes", total_mutation_per_node_abs));
    fun_map.insert(205, ("fraction negative mutations TOTAL", frac_neg_mutations));
    fun_map.insert(206, ("average of all mutations that are negative", average_neg_mutations));
    fun_map.insert(207, ("average of all mutations that are positive", average_pos_mutations));
    fun_map.insert(208, ("average recovery time", c_and_average_recovery_time));
    fun_map.insert(209, ("max recovery time", c_and_max_recovery_time));
    fun_map.insert(210, ("average recovery time leafs", c_and_average_recovery_time_leafs));
    fun_map.insert(211, ("longest outbreak path", c_and_max_outbreak_path_length));
    fun_map.insert(212, ("average path length leafs", c_and_average_path_length_leafs));
    fun_map.insert(213, ("max_path_length / average path length leafs", c_and_max_outbreak_path_dif_average_path_leafs));
    fun_map.insert(214, ("Mittlerer verzweigungsgrad von nicht-blättern", c_and_mittlerer_verzweigungsgrad));
    fun_map.insert(215, ("Tree diameter", tree_diameter));
    fun_map.insert(216, ("average descendant count", average_descendant_count));
    fun_map.insert(217, ("max tree width", max_tree_width));
    fun_map.insert(218, ("max tree width frac", max_tree_width_div_total));
    fun_map.insert(219, ("layer hight of max width in tree", hight_of_layer_with_max_width));
    fun_map.insert(220, ("frac of leafs in tree", fraction_of_leafs_vs_all_infected));
    fun_map.insert(221, ("modded ladder length", modified_ladder_length_of_tree));
    fun_map.insert(222, ("ladder length", ladder_length_of_tree));
    fun_map.insert(223, ("ladder length without divide", ladder_length_of_tree_no_divide));
    fun_map.insert(224, ("frac il nodes", frac_il_nodes));
    fun_map.insert(225, ("il nodes", il_nodes));
    fun_map.insert(226, ("cherry count", cherry_count));
    fun_map.insert(227, ("relative cherry count", relative_cherry_count));
    fun_map.insert(228, ("max width over max depth tree", max_tree_width_div_height));
    fun_map.insert(229, ("max width difference", max_width_difference));
    fun_map.insert(230, ("max width increase", max_width_increase));
    fun_map.insert(231, ("max relative width increase", max_relative_width_increase));
    fun_map.insert(232, ("max closeness", max_closeness));
    fun_map.insert(233, ("min closeness", min_closeness));
    fun_map.insert(234, ("max load", max_load));
    fun_map.insert(235, ("max load normed", max_load_normed));
    fun_map.insert(236, ("ln(max tree width) / max depth", ln_max_tree_width_div_lin_height));
    
    
    println!("choose function");
    let (fun, label) = loop{
        for (key, val) in fun_map.iter()
        {
            println!("{key}. -> {}", val.0);
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

    for file in glob::glob(&opts.glob).unwrap()
    {
        let file = file.unwrap();
        println!("file {file:?}");
        let mut analyzer = TopAnalyzer::new(file);
        heatmap_count(heatmap_mean, analyzer.iter_all_info(), fun);
    }
    println!("Success");
    (*label).to_string()
}

pub fn without_global_topology_with_n(heatmap_mean: &mut HeatmapAndMean<MyHeatmap>, opts: &ExamineOptions, n: u16) -> String
{
    #[allow(clippy::complexity)]
    let mut fun_map: BTreeMap<u8, (&str, fn ((usize, InfoGraph), u16) -> (usize, f64))> = BTreeMap::new();
    fun_map.insert(0, ("average recovery time of nodes with at least <number> children", c_and_average_recovery_time_minimal_children));
    fun_map.insert(1, ("average recovery time of nodes with exactly <number> children", c_and_average_recovery_time_exact_children));
    fun_map.insert(10, ("number of nodes with at least <number> children", c_and_frac_infected_nodes_with_at_least_n_children));
    fun_map.insert(11, ("number of nodes with exactly <number> children", c_and_frac_infected_nodes_with_exactly_n_children));
    
    
    println!("choose function");
    let (fun, label) = loop{
        for (key, val) in fun_map.iter()
        {
            println!("type {key} for {}", val.0);
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

    for file in glob::glob(&opts.glob).unwrap()
    {
        let file = file.unwrap();
        println!("file {file:?}");
        let mut analyzer = TopAnalyzer::new(file);
        heatmap_count_with_n(heatmap_mean, analyzer.iter_all_info(), fun, n);
    }
    println!("Success");
    format!("{label}_{n}")
}

pub fn without_global_topology_with_n_and_float(
    heatmap_mean: &mut HeatmapAndMean<MyHeatmap>, 
    opts: &ExamineOptions, 
    n: u16,
    float: f64
) -> String
{
    println!("n: {n}, float: {float}");

    #[allow(clippy::complexity)]
    let mut fun_map: BTreeMap<u8, (&str, fn ((usize, InfoGraph), u16, f64) -> (usize, f64))> = BTreeMap::new();
    fun_map.insert(0, ("max children given max mutation difference", c_and_max_children_give_mutation));
    fun_map.insert(1, ("max children of a human given max mutation difference", c_and_max_children_given_mutation_human));
    fun_map.insert(2, ("frac max children given max mutation difference", c_and_frac_max_children_given_mutation));
    
    println!("choose function");
    let (fun, label) = loop{
        for (key, val) in fun_map.iter()
        {
            println!("type {key} for {}", val.0);
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

    for file in glob::glob(&opts.glob).unwrap()
    {
        let file = file.unwrap();
        println!("file {file:?}");
        let mut analyzer = TopAnalyzer::new(file);
        heatmap_count_with_n_and_float(heatmap_mean, analyzer.iter_all_info(), fun, n, float);
    }
    println!("Success");
    format!("{label}_{n}")
}

pub fn multiple_values_per_sample_no_topology(heatmap_mean: &mut HeatmapAndMean<MyHeatmap>, opts: &ExamineOptions) -> String
{
    #[allow(clippy::complexity)]
    let mut fun_map: BTreeMap<u8, &(&str, fn (&'_ (usize, InfoGraph)) -> (usize, Box<dyn Iterator<Item=f64> + '_>))> = BTreeMap::new();
    fun_map.insert(0, &("all mutations of humans", c_and_all_mutations_humans));
    fun_map.insert(1, &("all mutations animals on path to first human", c_and_all_mutations_path_to_first_human));

    println!("choose function");
    let (fun, label) = loop{
        for (key, val) in fun_map.iter()
        {
            println!("type {key} for {}", val.0);
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

    for file in glob::glob(&opts.glob).unwrap()
    {
        let file = file.unwrap();
        println!("file {file:?}");
        let mut analyzer = TopAnalyzer::new(file);
        heatmap_count_multiple(heatmap_mean, analyzer.iter_all_info(), fun);
    }
    println!("Success");
    label.to_string()
}

pub fn with_global_topology(heatmap_mean: &mut HeatmapAndMean<MyHeatmap>, opts: &ExamineOptions) -> String
{
    #[allow(clippy::complexity)]
    let mut fun_map: BTreeMap<u8, (&str, fn ((usize, InfoGraph, &TopologyGraph)) -> (usize, f64))> = BTreeMap::new();
    fun_map.insert(0, ("average degree of humans infected by animals", c_and_average_degree_of_humans_infected_by_animals));
    
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

    for file in glob::glob(&opts.glob).unwrap()
    {
        let file = file.unwrap();
        println!("file {file:?}");
        let mut analyzer = TopAnalyzer::new(file);
        heatmap_count(heatmap_mean, analyzer.iter_all_info_with_topology(), fun);
    }
    println!("Success");
    label.to_string()
}

pub fn heatmap_examiner(opts: ExamineOptions){
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
    
    let hist = 
    loop{
        println!("Bin_size:");
        let bin_size: usize = get_number();
        let diff = right - left + 1;
        let rest = diff % bin_size;
        if rest > 0{
            println!("rest is {rest} - try again. Note: we have {diff} bins");
        }else {
            let bins = diff / bin_size;
            break HistUsize::new_inclusive(left, right, bins).expect("unable to create hist")
        }
    };
    HistUsizeFast::new_inclusive(left, right)
        .expect("unable to create the hist");

    println!("create val heatmap. Input left");
    let left_float: f64 = get_number();
    println!("input right number");
    let right_float: f64 = loop {
        let right_float = get_number();
        if right_float > left_float {
            break right_float;
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
    let hist_f = HistF64::new(left_float, right_float, num_intervals)
        .expect("unable to create hist");
    let x_width = hist.bin_count();

    let bin_mids = hist.bin_iter()
        .map(|bins| (bins[0]+bins[1]-1) as f64 / 2.0)
        .collect();
    
    let heatmap = HeatmapU::new(hist, hist_f);


    let mut heatmap_mean = HeatmapAndMean::new(heatmap, x_width, bin_mids);

     
    let choice = get_function_input_chooser();

    let label = match choice{
        FunctionInputChooser::WithTopology => {
            with_global_topology(&mut heatmap_mean, &opts)
        },
        FunctionInputChooser::WithoutTopology => {
            without_global_topology(&mut heatmap_mean, &opts)
        },
        FunctionInputChooser::MultipleInputsWithoutTopology => {
            multiple_values_per_sample_no_topology(&mut heatmap_mean, &opts)
        },
        FunctionInputChooser::WithoutTopologyWithN(n) => {
            without_global_topology_with_n(&mut heatmap_mean, &opts, n)
        },
        FunctionInputChooser::WithoutTopologyWithFloatAndN(n, float) => {
            without_global_topology_with_n_and_float(&mut heatmap_mean, &opts, n, float)
        }
    };

    let (heat_file, av_file) = loop{
        println!("input name of output file");
        let mut buffer = String::new();
        let line = std::io::stdin().read_line(&mut buffer);
        let input_no_file_ending = &buffer[..buffer.len()-1];
        let input = format!("{input_no_file_ending}.gp");
        match line {
            Err(e) => println!("there was the error: {e:?}"),
            Ok(_) => {
                let mut opts = File::options();
                match opts.create_new(true).write(true).open(&input) {
                    Err(e) => {
                        match e.kind(){
                            std::io::ErrorKind::AlreadyExists => {
                                println!("the file already exists. Do you want to overwrite it? y/n");
                                let yes = get_yes_or_no();
                                if yes {
                                    opts.create_new(false)
                                        .truncate(true);
                                    match opts.open(input) {
                                        Err(e) => eprintln!("error during file creation {e:?}"),
                                        Ok(f) => {
                                            let av_file = File::create(format!("{input_no_file_ending}.dat")).unwrap();
                                            break (f, av_file)
                                        }
                                    }
                                }
                            },
                            _ => eprintln!("unable to create file due to {e}\ntry again")
                        }
                    },
                    Ok(f) => {
                        let av_file = File::create(format!("{input_no_file_ending}.dat")).unwrap();
                        break (f, av_file)
                    },
                }
            }
        }
    };

    let buf = BufWriter::new(heat_file);
    let heat = heatmap_mean.heatmap.heatmap_normalized_columns();

    let x_axis = GnuplotAxis::new(
        left as f64, 
        right as f64, 
        opts.x_tics.get()
    );

    let y_axis = GnuplotAxis::new(
        heat.height_hist().first_border(), 
        *heat.height_hist().borders().last().unwrap(), 
        opts.y_tics.get()
    );

    let mut gs = GnuplotSettings::new();
    gs.x_label("C")
        .x_axis(x_axis)
        .y_axis(y_axis)
        .y_label(&label);
    let _ = heat.gnuplot(buf, gs);

    heatmap_mean.write_av(&label, av_file);
    let misses = heatmap_mean.heatmap.total_misses();
    let total = heatmap_mean.heatmap.total();

    println!("misses: {misses} total: {total} fraction {}", misses as f64 / total as f64);

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
        let buf = BufReader::with_capacity(1024*16, file);
        let mut de = DeserializeAnyhow::new(buf);

        let topology: TopologyGraph = de.deserialize().unwrap();
        Self{
            path: path.as_ref().to_path_buf(),
            topology,
            de,
            read_counter: 0
        }
    }

    #[allow(dead_code)]
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

    pub fn iter_all_info_with_topology(&'_ mut self) -> impl Iterator<Item = (usize, InfoGraph, &TopologyGraph)> + '_
    {
        let deserializer = &mut self.de;
        let counter = &mut self.read_counter;
        let top = &self.topology;
        std::iter::from_fn(
            move ||
            {
                let compat: Option<(usize, CondensedInfo)> = deserializer.deserialize().ok();
                if compat.is_some(){
                    *counter += 1;
                }
                let compat = compat?;
                Some((compat.0, compat.1.to_info_graph(), top))
            }
        )
    }

    #[allow(dead_code)]
    pub fn test_run(&mut self) 
    {
        let file = File::create("tmp_test.dat")
            .unwrap();
        let mut buf = BufWriter::new(file);

        writeln!(buf, "#C number_of_jumps prior_dogs_first_jump max_mutation_first_jump average_mut_first_jump")
            .unwrap();

        self.iter_all_info()
            .for_each(
                |(c, info)|
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
                        "{c} {} {} {} {}", 
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

fn c_and_average_mutation_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for mutation in item.1.animal_mutation_iter()
    {
        sum += mutation;
        count += 1;
    }
    (item.0, sum / count as f64)
}

fn c_and_average_positive_mutation_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for mutation in item.1.animal_mutation_iter()
    {
        if mutation > 0.0 {
            sum += mutation;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

fn c_and_average_negative_mutation_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for mutation in item.1.animal_mutation_iter()
    {
        if mutation < 0.0 {
            sum += mutation;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

fn c_and_max_mutation_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max = f64::NEG_INFINITY;

    for mutation in item.1.animal_mutation_iter()
    {
        if mutation > max {
            max = mutation;
        }
    }
    (item.0, max)
}

fn c_and_average_recovery_time_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;
    for node in item.1.info.contained_iter().take(item.1.dog_count)
    {
        if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
        {
            sum += (recovery.get() - infection.get()) as f64;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

fn c_and_max_recovery_time_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max = f64::NEG_INFINITY;
    for node in item.1.info.contained_iter().take(item.1.dog_count)
    {
        if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
        {
            let duration = (recovery.get() - infection.get()) as f64;
            if duration > max 
            {
                max = duration
            }
        }
    }
    (item.0, max)
}

fn c_and_average_recovery_time_dogs_infecting_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;
    for node in item.1.animals_infecting_humans_node_iter()
    {
        if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
        {
            sum += (recovery.get() - infection.get()) as f64;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

fn c_and_max_recovery_time_dogs_infecting_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max = f64::NEG_INFINITY;
    for node in item.1.animals_infecting_humans_node_iter()
    {
        if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
        {
            let duration = (recovery.get() - infection.get()) as f64;
            if duration > max {
                max = duration;
            }
        }
    }
    (item.0, max)
}

fn c_recovery_time_of_first_dog_infecting_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    if let Some(node) = item.1.first_animal_infecting_a_human()
    {
        if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
        {

            let duration = (recovery.get() - infection.get()) as f64;
            return (item.0, duration);
        }
    }
    (item.0, f64::NAN)
}

fn c_and_average_recovery_duration_animals_on_path_to_first_human(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut counter = 0_u16;

    let average = match item.1.path_from_first_animal_infecting_human_to_root()
    {
        None => f64::NAN,
        Some(it) => {
            for node in it {
                if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
                {
        
                    let duration = (recovery.get() - infection.get()) as f64;
                    sum += duration;
                    counter += 1;
                }
            }
            sum / counter as f64
        }
    };
    (item.0, average)
}

pub fn c_and_average_mutation_animals_on_path_to_first_human(item: (usize, InfoGraph)) -> (usize, f64) {
    let mut sum = 0.0;
    let mut counter = 0_u32;
    for mutation in item.1.path_from_first_animal_infecting_human_to_root_mutation_iter() 
    {
        sum += mutation;
        counter += 1;
    }
    (item.0, sum / counter as f64)
}

fn c_and_all_mutations_path_to_first_human(item: &'_ (usize, InfoGraph)) -> (usize, Box<dyn Iterator<Item=f64> + '_>)
{
    let iter = item.1.path_from_first_animal_infecting_human_to_root_mutation_iter();
    (item.0, Box::new(iter))
}


fn c_and_average_gamma_of_dogs_infecting_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0;
    for node in item.1.animals_infecting_humans_node_iter()
    {
        let gamma = node.get_gamma();
        sum += gamma;
        count += 1;
    }
    (item.0, sum / count as f64)
}

fn c_and_max_gamma_of_dogs_infecting_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max = f64::NEG_INFINITY;
    for node in item.1.animals_infecting_humans_node_iter()
    {
        let gamma = node.get_gamma();
        if max < gamma 
        {
            max = gamma;
        }
    }
    if max.is_infinite()
    {
        max = f64::NAN;
    }
    (item.0, max)
}

fn c_and_number_of_dogs_infecting_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut count = 0;
    for _ in item.1.animals_infecting_humans_node_iter()
    {
        count += 1;
    }
    (item.0, count as f64)
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

fn c_and_average_positive_mutation_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for mutation in item.1.human_mutation_iter()
    {
        if mutation > 0.0 {
            sum += mutation;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

fn c_and_average_negative_mutation_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for mutation in item.1.human_mutation_iter()
    {
        if mutation < 0.0 {
            sum += mutation;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

fn c_and_average_mutation_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for mutation in item.1.human_mutation_iter()
    {
        
        sum += mutation;
        count += 1;
        
    }
    (item.0, sum / count as f64)
}

fn c_and_all_mutations_humans(item: &'_ (usize, InfoGraph)) -> (usize, Box<dyn Iterator<Item=f64> + '_>)
{
    let iter = item.1.human_mutation_iter();
    (item.0, Box::new(iter))
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

// maybe create a integer heatmap for this one if the results look interesting
fn c_and_previous_dogs_of_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut prev = 0;
    for human_node in item.1.human_node_iter()
    {
        if prev < human_node.prev_dogs{
            prev = human_node.prev_dogs;
        }
    }
    (item.0, prev as f64)
}


pub fn total_mutation_max(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_mut = 0.0;
    for this in item.1.total_mutation_iter()
    {
        if max_mut < this {
            max_mut = this;
        }
    }
    (item.0, max_mut)
}

pub fn total_mutation_average(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut av_mut = 0.0;
    let mut count = 0_u32;
    for this in item.1.total_mutation_iter()
    {
        count += 1;
        av_mut += this;
    }
    (item.0, av_mut / count as f64)
}


pub fn total_mutation_sum(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum_mut = 0.0;
    for this in item.1.total_mutation_iter()
    {
        sum_mut += this;
    }
    (item.0, sum_mut)
}

pub fn total_mutation_sum_abs(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum_mut = 0.0;
    for this in item.1.total_mutation_iter()
    {
        sum_mut += this.abs();
    }
    (item.0, sum_mut)
}

pub fn total_mutation_per_node_abs(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum_mut = 0.0;
    let mut total = 0;
    for this in item.1.total_mutation_iter()
    {
        total += 1;
        sum_mut += this.abs();
    }
    (item.0, sum_mut / total as f64)
}

pub fn frac_neg_mutations(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut counter_neg = 0_u32;
    let mut total = 0_u32;
    for this in item.1.total_mutation_iter()
    {
        total += 1;
        if this < 0.0 {
            counter_neg += 1;
        }
    }
    (item.0, counter_neg as f64 / total as f64)
}

/// Averaging over all the negative mutations, only counting negative ones
pub fn average_neg_mutations(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut counter_neg = 0_u32;
    let mut sum = 0.0;
    for this in item.1.total_mutation_iter()
    {
        if this < 0.0 {
            counter_neg += 1;
            sum += this;
        }
    }
    (item.0, sum / counter_neg as f64)
}

/// Averaging over all the positive mutations, only counting positive ones
pub fn average_pos_mutations(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut counter_pos = 0_u32;
    let mut sum = 0.0;
    for this in item.1.total_mutation_iter()
    {
        if this > 0.0 {
            counter_pos += 1;
            sum += this;
        }
    }
    (item.0, sum / counter_pos as f64)
}

fn c_and_average_degree_of_humans_infected_by_animals(item: (usize, InfoGraph, &TopologyGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut counter = 0_u32;
    for (_, global_node) in item.1.humans_infected_by_animals_info_node_and_global_node(item.2)
    {
        let degree = global_node.degree();
        sum += degree as f64;
        counter += 1;
    }
    (item.0, sum / counter as f64)
}

fn c_and_average_prev_dogs_of_humans_infected_by_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0;
    for human_node in item.1.humans_infected_by_animals()
    {
        let prev = human_node.prev_dogs;
        sum += prev as f64;
        count += 1;
    }
    (item.0, sum / count as f64)
}

fn c_and_average_recovery_time(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0;
    for node in item.1.info.contained_iter()
    {
        if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
        {
            sum += (recovery.get() - infection.get()) as f64;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

fn c_and_average_recovery_time_leafs(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;
    for node in item.1.leaf_node_iter()
    {
        if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
        {
            sum += (recovery.get() - infection.get()) as f64;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

fn c_and_max_recovery_time(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max = f64::NEG_INFINITY;
    for node in item.1.info.contained_iter()
    {
        if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
        {
            let recovery_time = (recovery.get() - infection.get()) as f64;
            if recovery_time > max {
                max = recovery_time;
            }
        }
    }
    (item.0, max)
}

fn c_and_max_outbreak_path_length(item: (usize, InfoGraph)) -> (usize, f64)
{
    let initial = item.1.initial_infection[0];
    let len = item.1.info.longest_shortest_path_from_index(initial).unwrap();
    
    (item.0, len as f64)
}

fn c_and_average_path_length_leafs(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0;
    let mut counter = 0_u32;
    let initial = item.1.initial_infection[0];
    for (index, _, depth) in item.1.info.bfs_index_depth(initial)
    {
        if item.1.info.degree(index).unwrap() == 1 {
            counter += 1;
            sum += depth;
        }
    }
    (item.0, sum as f64 / counter as f64)
}

fn c_and_max_outbreak_path_dif_average_path_leafs(item: (usize, InfoGraph)) -> (usize, f64)
{
    let initial = item.1.initial_infection[0];
    let max_len = item.1.info.longest_shortest_path_from_index(initial).unwrap();

    let (c, av_len) = c_and_average_path_length_leafs(item);
    (c, max_len as f64 / av_len)
}

fn c_and_average_recovery_time_minimal_children(item: (usize, InfoGraph), minimal_children: u16) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;
    for node in item.1.nodes_with_at_least_n_children(minimal_children)
    {
        if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
        {
            sum += (recovery.get() - infection.get()) as f64;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

fn c_and_average_recovery_time_exact_children(item: (usize, InfoGraph), desired_children: u16) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;
    for (child_count, node) in item.1.nodes_with_child_count_iter()
    {
        if child_count == desired_children{
            if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
            {
                sum += (recovery.get() - infection.get()) as f64;
                count += 1;
            }
        }

    }
    (item.0, sum / count as f64)
}

fn c_and_frac_infected_nodes_with_exactly_n_children(item: (usize, InfoGraph), desired_children: u16) -> (usize, f64)
{
    let mut counter = 0_u32;
    let mut total = 0_u32;

    for (count, _) in item.1.nodes_with_child_count_iter()
    {
        if count == desired_children
        {
            counter += 1;
        }
        total += 1;
    }
    (item.0, counter as f64 / total as f64)
}

fn c_and_frac_infected_nodes_with_at_least_n_children(item: (usize, InfoGraph), desired_children: u16) -> (usize, f64)
{
    let mut counter = 0_u32;
    let mut total = 0_u32;

    for (count, _) in item.1.nodes_with_child_count_iter()
    {
        if count >= desired_children
        {
            counter += 1;
        }
        total += 1;
    }
    (item.0, counter as f64 / total as f64)
}

fn c_and_max_children_give_mutation(item: (usize, InfoGraph), _: u16, mutation: f64) -> (usize, f64)
{
    let mut max_count = 0;
    for (count, _) in item.1.iter_nodes_and_mutation_child_count(mutation)
    {
        if count > max_count{
            max_count = count;
        }
    }
    (item.0, max_count as f64)
}

fn c_and_frac_max_children_given_mutation(item: (usize, InfoGraph), _: u16, mutation: f64) -> (usize, f64)
{
    let mut max_count = 0;
    let mut total = 0;
    for (count, _) in item.1.iter_nodes_and_mutation_child_count(mutation)
    {
        total += 1;
        if count > max_count{
            max_count = count;
        }
    }
    (item.0, max_count as f64 / total as f64)
}

fn c_and_max_children_given_mutation_human(item: (usize, InfoGraph), _: u16, mutation: f64) -> (usize, f64)
{
    let mut max_count = 0;
    let dog_count = item.1.dog_count;
    let iter = item.1
        .iter_nodes_and_mutation_child_count_unfiltered(mutation)
        .skip(dog_count);
    for (count, _) in iter
    {
        if count > max_count{
            max_count = count;
        }
    }
    (item.0, max_count as f64)
}


fn c_and_max_children_of_humans_infected_by_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_child_count = 0;
    for (count, node) in item.1.iter_human_nodes_and_child_count()
    {
        if let InfectedBy::By(by) = node.infected_by
        {
            if (by as usize) < item.1.dog_count && count > max_child_count
            {
                max_child_count = count;
            }
        }
    }
    (item.0, max_child_count as f64)
}

fn c_and_second_largest_children_of_humans_infected_by_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut child_count = Vec::new();
    for (count, node) in item.1.iter_human_nodes_and_child_count()
    {
        if let InfectedBy::By(by) = node.infected_by
        {
            if (by as usize) < item.1.dog_count
            {
                child_count.push(count);
            }
        }
    }

    let count = if child_count.len() >= 2 {
        child_count.sort_unstable_by_key(|item| std::cmp::Reverse(*item));
        child_count[1]
    } else {
        0
    };

    (item.0, count as f64)
}

fn c_and_second_largest_children_of_humans_infected_by_animals_strict(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut child_count = Vec::new();
    for (count, node) in item.1.iter_human_nodes_and_child_count_of_first_infected_humans()
    {
        if let InfectedBy::By(by) = node.infected_by
        {
            assert!((by as usize) < item.1.dog_count);
            child_count.push(count);
        }
    }

    let count = if child_count.len() >= 2 {
        child_count.sort_unstable_by_key(|item| std::cmp::Reverse(*item));
        //dbg!(&child_count);
        child_count[1]
    } else {
        0
    };

    (item.0, count as f64)
}

fn c_and_frac_max_children_of_humans_infected_by_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_child_count = 0;
    for (count, node) in item.1.iter_human_nodes_and_child_count()
    {
        if let InfectedBy::By(by) = node.infected_by
        {
            if (by as usize) < item.1.dog_count && count > max_child_count
            {
                max_child_count = count;
            }
        }
    }
    (item.0, max_child_count as f64/ item.0 as f64)
}

fn c_and_mittlerer_verzweigungsgrad(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0;
    let mut count = 0_u32;

    for degree in item.1.info.degree_iter()
    {
        if degree > 1 {
            sum += degree;
            count += 1;
        }
    }
    (item.0, sum as f64 / count as f64)
}

// see https://doi.org/10.1093/emph/eou018
fn ladder_length_of_tree(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut leaf_children_count = vec![0u16; item.1.info.vertex_count()];
    let degrees = item.1.info.degree_vec();
    for (index, node) in item.1.info.container_iter().enumerate()
    {
        let adj = node.edges();
        if adj.is_empty() {
            continue;
        }
        for &i in adj {
            if degrees[i] == 1 {
                leaf_children_count[index] += 1;
            }
        }
    }

    let mut max_ladder = 0_u16;
    let mut leaf_count = 0;
    for (index, &degree) in degrees.iter().enumerate()
    {
        if degree == 1 {
            leaf_count += 1;
            let mut ladder_count = 0;
            let mut current_node = item.1.info.at(index);
            while let InfectedBy::By(by) = current_node.infected_by
            {
                current_node = item.1.info.at(by as usize);
                if leaf_children_count[by as usize] != 1 {
                    break;
                }
                ladder_count += 1;
            }
            if ladder_count > max_ladder{
                max_ladder = ladder_count;
            }
        }
    }
    (item.0, max_ladder as f64 / leaf_count as f64)
}

// see https://doi.org/10.1093/emph/eou018
fn ladder_length_of_tree_no_divide(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut leaf_children_count = vec![0u16; item.1.info.vertex_count()];
    let degrees = item.1.info.degree_vec();
    for (index, node) in item.1.info.container_iter().enumerate()
    {
        let adj = node.edges();
        if adj.is_empty() {
            continue;
        }
        for &i in adj {
            if degrees[i] == 1 {
                leaf_children_count[index] += 1;
            }
        }
    }

    let mut max_ladder = 0_u16;
    for (index, &degree) in degrees.iter().enumerate()
    {
        if degree == 1 {
            let mut ladder_count = 0;
            let mut current_node = item.1.info.at(index);
            while let InfectedBy::By(by) = current_node.infected_by
            {
                current_node = item.1.info.at(by as usize);
                if leaf_children_count[by as usize] != 1 {
                    break;
                }
                ladder_count += 1;
            }
            if ladder_count > max_ladder{
                max_ladder = ladder_count;
            }
        }
    }
    (item.0, max_ladder as f64)
}

// see https://doi.org/10.1093/emph/eou018
// Note: I modify it because I think the original measure was used for binary trees
fn modified_ladder_length_of_tree(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut leaf_children_count = vec![0u16; item.1.info.vertex_count()];
    let degrees = item.1.info.degree_vec();
    for (index, node) in item.1.info.container_iter().enumerate()
    {
        let adj = node.edges();
        if adj.is_empty() {
            continue;
        }
        for &i in adj {
            if degrees[i] == 1 {
                leaf_children_count[index] += 1;
            }
        }
    }

    let mut max_ladder = 0_u16;
    let mut leaf_count = 0;
    for (index, &degree) in degrees.iter().enumerate()
    {
        if degree == 1 {
            leaf_count += 1;
            let mut ladder_count = 0;
            let mut current_node = item.1.info.at(index);
            while let InfectedBy::By(by) = current_node.infected_by
            {
                current_node = item.1.info.at(by as usize);
                if leaf_children_count[by as usize] != 1 || degrees[by as usize] > 3 {
                    break;
                }
                ladder_count += 1;
            }
            if ladder_count > max_ladder{
                max_ladder = ladder_count;
            }
        }
    }
    (item.0, max_ladder as f64 / leaf_count as f64)
}

// see https://doi.org/10.1093/emph/eou018
fn frac_il_nodes(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut leaf_children_count = vec![0u16; item.1.info.vertex_count()];
    let degrees = item.1.info.degree_vec();
    for (index, node) in item.1.info.container_iter().enumerate()
    {
        let adj = node.edges();
        if adj.is_empty() {
            continue;
        }
        for &i in adj {
            if degrees[i] == 1 {
                leaf_children_count[index] += 1;
            }
        }
    }

    let mut il_count = 0_u16;
    let mut total_count = 0;
    for (index, &degree) in degrees.iter().enumerate()
    {
        if degree == 0 {
            continue;
        }
        total_count += 1;
        if degree == 1 {
            if let InfectedBy::By(by) = item.1.info.at(index).infected_by
            {
                if leaf_children_count[by as usize] == 1 || degrees[by as usize] == 2 {
                    il_count += 1;
                }
            }
        }
        
    }
    (item.0, il_count as f64 / total_count as f64)
}

// see https://doi.org/10.1093/emph/eou018
fn il_nodes(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut leaf_children_count = vec![0u16; item.1.info.vertex_count()];
    let degrees = item.1.info.degree_vec();
    for (index, node) in item.1.info.container_iter().enumerate()
    {
        let adj = node.edges();
        if adj.is_empty() {
            continue;
        }
        for &i in adj {
            if degrees[i] == 1 {
                leaf_children_count[index] += 1;
            }
        }
    }

    let mut il_count = 0_u16;
    for (index, &degree) in degrees.iter().enumerate()
    {
        if degree == 0 {
            continue;
        }
        if degree == 1 {
            if let InfectedBy::By(by) = item.1.info.at(index).infected_by
            {
                if leaf_children_count[by as usize] == 1 || degrees[by as usize] == 2 {
                    il_count += 1;
                }
            }
        }
        
    }
    (item.0, il_count as f64)
}

fn cherry_count(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut cherry_count = 0;
    'a: for node in item.1.info.container_iter()
    {
        let adj = node.edges();
        if adj.len() != 3 {
            continue;
        }
        let parent = if let InfectedBy::By(by) = node.contained().infected_by {
            by
        } else {
            continue
        };
        for other in adj {
            if *other == parent as usize {
                continue;
            }
            let container = item.1.info.container(*other);
            if container.degree() != 1 {
                continue 'a;
            }
        }
        cherry_count += 1;
    }
    (item.0, cherry_count as f64)
}

fn relative_cherry_count(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut cherry_count = 0;
    let mut total = 0;
    'a: for node in item.1.info.container_iter()
    {
        let adj = node.edges();
        if !adj.is_empty()
        {
            total += 1;
        }
        if adj.len() != 3 {
            continue;
        }
        let parent = if let InfectedBy::By(by) = node.contained().infected_by {
            by
        } else {
            continue
        };
        for other in adj {
            if *other == parent as usize {
                continue;
            }
            let container = item.1.info.container(*other);
            if container.degree() != 1 {
                continue 'a;
            }
        }
        cherry_count += 1;
    }
    let max_possible_cherries = (total - 1) as f64 / 3.0;
    (item.0, cherry_count as f64 / max_possible_cherries)
}

fn c_and_average_gamma_change_animals_on_path_to_human_with_most_children(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;
    let average = match item.1.iter_gamma_change_from_animal_that_infects_human_with_most_children_to_root()
    {
        None => f64::NAN,
        Some(iter) => {
            for gamma_change in iter 
            {
                sum += gamma_change;
                count += 1;
            }
            sum / count as f64
        }
    };
    (item.0, average)
}

fn c_and_path_len_to_human_with_most_children(item: (usize, InfoGraph)) -> (usize, f64)
{
    let path_len = match item.1.path_from_human_with_most_children_to_root()
    {
        None => f64::NAN,
        Some(iter) =>  iter.count() as f64
    };
    (item.0, path_len)
}

fn c_and_gamma_of_human_with_most_children(item: (usize, InfoGraph)) -> (usize, f64)
{
    let human_with_most_children = item.1.human_with_most_children();
    let gamma = match human_with_most_children{
        None => f64::NAN,
        Some(index) => {
            item.1.info.at(index).get_gamma()
        }
    };
    (item.0, gamma)
}

fn c_and_lambda_of_human_with_most_children(item: (usize, InfoGraph)) -> (usize, f64)
{
    let human_with_most_children = item.1.human_with_most_children();
    let lambda = match human_with_most_children{
        None => f64::NAN,
        Some(index) => {
            item.1.info.at(index).get_lambda_human()
        }
    };
    (item.0, lambda)
}

fn c_and_lambda_of_animal_before_human_with_most_children(item: (usize, InfoGraph)) -> (usize, f64)
{
    let human_with_most_children = item.1.human_with_most_children();
    let lambda = match human_with_most_children{
        None => f64::NAN,
        Some(index) => {
            if let InfectedBy::By(by) = item.1.info.at(index).infected_by
            {
                item.1.info.at(by as usize).get_lambda_human()
            } else {
                unreachable!()
            }
        }
    };
    (item.0, lambda)
}

fn c_and_average_lambda_change_animals_on_path_to_human_with_most_children(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;
    let average = match item.1.iter_lambda_change_from_animal_that_infects_human_with_most_children_to_root()
    {
        None => f64::NAN,
        Some(iter) => {
            for lambda_change in iter 
            {
                sum += lambda_change;
                count += 1;
            }
            sum / count as f64
        }
    };
    (item.0, average)
}

fn c_and_frac_of_negative_lambda_change_animals_on_path_to_human_with_most_children(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut count = 0_u32;
    let mut count_negative = 0_u32;
    let average = match item.1.iter_lambda_change_from_animal_that_infects_human_with_most_children_to_root()
    {
        None => f64::NAN,
        Some(iter) => {
            for lambda_change in iter 
            {
                count += 1;
                if lambda_change < 0.0 {
                    count_negative += 1;
                }
            }
            count_negative as f64 / count as f64
        }
    };
    (item.0, average)
}

fn av_lambda_change_human_human_trans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for lambda in item.1.lambda_changes_human_human_transmission()
    {
        sum += lambda;
        count += 1;
    }
    (item.0, sum / count as f64)
}

fn frac_negative_lambda_change_human_human_trans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut count_negatives = 0_u32;
    let mut count = 0_u32;

    for lambda in item.1.lambda_changes_human_human_transmission()
    {
        count += 1;
        if lambda < 0.0 {
            count_negatives += 1;
        }
    }
    (item.0, count_negatives as f64 / count as f64)
}

fn av_negative_lambda_change_human_human_trans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum_negatives = 0.0;
    let mut count = 0_u32;

    for lambda in item.1.lambda_changes_human_human_transmission()
    {
        if lambda < 0.0 {
            count += 1;
            sum_negatives += lambda;
        }
    }
    (item.0, sum_negatives / count as f64)
}

fn av_positive_lambda_change_human_human_trans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum_positives = 0.0;
    let mut count = 0_u32;

    for lambda in item.1.lambda_changes_human_human_transmission()
    {
        if lambda > 0.0 {
            count += 1;
            sum_positives += lambda;
        }
    }
    (item.0, sum_positives / count as f64)
}

fn tree_diameter(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut diameter = 0;
    for (index, degree) in item.1.info.degree_iter().enumerate()
    {
        if degree == 1 {
            let longest_shortest_path = item.1.info.longest_shortest_path_from_index(index).unwrap();
            if longest_shortest_path > diameter{
                diameter = longest_shortest_path;
            }
        }
    }
    (item.0, diameter as f64)
}

fn average_descendant_count(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0;
    let mut n = 0_u16;
    for (count, _) in item.1.nodes_with_child_count_iter()
    {
        if count > 0 {
            sum += count as u64;
            n += 1;
        }
    }
    (item.0, sum as f64 / n as f64)
}

fn max_tree_width(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut current_width = 0_u32;
    let mut current_depth = 0;
    let mut max_width = 0;
    let root = item.1.initial_infection[0];
    for (_, _, depth) in item.1.info.bfs_index_depth(root)
    {
        if depth == current_depth{
            current_width += 1;
        } else {
            if max_width < current_width {
                max_width = current_width;
            }
            current_depth = depth;
            current_width = 1;
        }
    }
    if current_width > max_width {
        max_width = current_width;
    }
    (item.0, max_width as f64)
}

fn max_tree_width_div_total(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut current_width = 0_u32;
    let mut current_depth = 0;
    let mut max_width = 0;
    let root = item.1.initial_infection[0];
    let mut total_number_of_nodes = 0_u32;
    for (_, _, depth) in item.1.info.bfs_index_depth(root)
    {
        if depth == current_depth{
            current_width += 1;
        } else {
            if max_width < current_width {
                max_width = current_width;
            }
            current_depth = depth;
            current_width = 1;
        }
        total_number_of_nodes += 1;
    }
    if current_width > max_width {
        max_width = current_width;
    }
    (item.0, max_width as f64 / total_number_of_nodes as f64)
}

fn max_tree_width_div_height(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut current_width = 0_u32;
    let mut current_depth = 0;
    let mut max_width = 0;
    let root = item.1.initial_infection[0];
    for (_, _, depth) in item.1.info.bfs_index_depth(root)
    {
        if depth == current_depth{
            current_width += 1;
        } else {
            if max_width < current_width {
                max_width = current_width;
            }
            current_depth = depth;
            current_width = 1;
        }
    }
    if current_width > max_width {
        max_width = current_width;
    }
    (item.0, max_width as f64 / current_depth as f64)
}

fn ln_max_tree_width_div_lin_height(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut current_width = 0_u32;
    let mut current_depth = 0;
    let mut max_width = 0;
    let root = item.1.initial_infection[0];
    for (_, _, depth) in item.1.info.bfs_index_depth(root)
    {
        if depth == current_depth{
            current_width += 1;
        } else {
            if max_width < current_width {
                max_width = current_width;
            }
            current_depth = depth;
            current_width = 1;
        }
    }
    if current_width > max_width {
        max_width = current_width;
    }
    (item.0, (max_width as f64).ln() / current_depth as f64)
}

fn max_width_difference(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut last_width = 0;
    let mut current_width = 0_i32;
    let mut current_depth = 0;
    let mut max_width_difference = 0;
    let root = item.1.initial_infection[0];
    for (_, _, depth) in item.1.info.bfs_index_depth(root)
    {
        if depth == current_depth{
            current_width += 1;
        } else {
            let difference = (current_width - last_width).abs();
            if max_width_difference < difference {
                max_width_difference = difference;
            }
            current_depth = depth;
            last_width = current_width;
            current_width = 1;
        }
    }
    let difference = (current_width - last_width).abs();
    if max_width_difference < difference {
        max_width_difference = difference;
    }
    (item.0, max_width_difference as f64)
}

fn max_width_increase(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut last_width = 0;
    let mut current_width = 0_i32;
    let mut current_depth = 0;
    let mut max_width_difference = 0;
    let root = item.1.initial_infection[0];
    for (_, _, depth) in item.1.info.bfs_index_depth(root)
    {
        if depth == current_depth{
            current_width += 1;
        } else {
            let difference = current_width - last_width;
            if max_width_difference < difference {
                max_width_difference = difference;
            }
            current_depth = depth;
            last_width = current_width;
            current_width = 1;
        }
    }
    let difference = current_width - last_width;
    if max_width_difference < difference {
        max_width_difference = difference;
    }
    (item.0, max_width_difference as f64)
}

fn max_relative_width_increase(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut last_width = 1;
    let mut current_width = 0_i32;
    let mut current_depth = 0;
    let mut max_width_difference = 0.0;
    let root = item.1.initial_infection[0];
    for (_, _, depth) in item.1.info.bfs_index_depth(root)
    {
        if depth == current_depth{
            current_width += 1;
        } else {
            let difference = (current_width - last_width) as f64 / last_width as f64;
            if max_width_difference < difference {
                max_width_difference = difference;
            }
            current_depth = depth;
            last_width = current_width;
            current_width = 1;
        }
    }
    let difference = (current_width - last_width) as f64 / last_width as f64;
    if max_width_difference < difference {
        max_width_difference = difference;
    }
    (item.0, max_width_difference)
}

fn max_tree_width_div_total_humans_only(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut current_width = 0_u32;
    let mut current_depth = 0;
    let mut max_width = 0;
    let root = item.1.initial_infection[0];
    let mut total_number_of_nodes = 0_u32;
    for (index, _, depth) in item.1.info.bfs_index_depth(root)
    {
        if index < item.1.dog_count{
            continue;
        }
        if depth == current_depth{
            current_width += 1;
        } else {
            if max_width < current_width {
                max_width = current_width;
            }
            current_depth = depth;
            current_width = 1;
        }
        total_number_of_nodes += 1;
    }
    if current_width > max_width {
        max_width = current_width;
    }
    (item.0, max_width as f64 / total_number_of_nodes as f64)
}

fn hight_of_layer_with_max_width(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut current_width = 0_u32;
    let mut current_depth = 0;
    let mut depth_of_max_width = 0;
    let mut max_width = 0;
    let root = item.1.initial_infection[0];
    for (_, _, depth) in item.1.info.bfs_index_depth(root)
    {
        if depth == current_depth{
            current_width += 1;
        } else {
            if max_width < current_width {
                max_width = current_width;
            }
            depth_of_max_width = current_depth;
            current_depth = depth;
            current_width = 1;
        }
    }
    if current_width > max_width {
        depth_of_max_width = current_depth;
    }
    (item.0, depth_of_max_width as f64)
}

fn fraction_of_leafs_vs_all_infected(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut leafs = 0;
    let mut total = 0;
    for degree in item.1.info.degree_iter()
    {
        if degree == 0 {
            continue;
        }
        else if degree == 1 {
            leafs += 1;
        }
        total += 1;
    }
    (item.0, leafs as f64 / total as f64)
}

fn min_closeness(item: (usize, InfoGraph)) -> (usize, f64)
{
    let closeness = item.1.info.closeness_centrality();
    let mut min = f64::INFINITY;
    closeness.iter()
        .for_each(|&val| 
            {
                if val > 0.0 && val < min {
                    min = val;
                }
            }
        );
    (item.0, min)
}

fn max_closeness(item: (usize, InfoGraph)) -> (usize, f64)
{
    let closeness = item.1.info.closeness_centrality();
    let mut max = f64::NEG_INFINITY;
    closeness.iter()
        .for_each(|&val| 
            {
                if val.is_finite() && val > max {
                    max = val;
                }
            }
        );
    (item.0, max)
}

fn max_load(item: (usize, InfoGraph)) -> (usize, f64)
{
    let load = item.1.info.vertex_load(false);
    let mut max = f64::NEG_INFINITY;
    load.iter()
        .for_each(|&val| 
            {
                if val.is_finite() && val > max {
                    max = val;
                }
            }
        );
    (item.0, max)
}

// something is fishy with the normalization - do I calculate the vertex load correctly?
fn max_load_normed(item: (usize, InfoGraph)) -> (usize, f64)
{
    let load = item.1.info.vertex_load(true);
    let mut max = f64::NEG_INFINITY;
    let total = item.1.info.contained_iter()
        .filter(|entry| entry.was_infected())
        .count();
    load.iter()
        .for_each(|&val| 
            {
                if val.is_finite() && val > max {
                    max = val;
                }
            }
        );
    let total_bigint: num::bigint::BigInt = (total - 1).into();
    let binom = num::integer::binomial(total_bigint, 2.into());
    
    let normalization = binom.to_f64().unwrap();
    //println!("max: {max} total: {total} binom: {binom} normalization: {normalization} res: {}", max / normalization);
    (item.0, max / normalization)
}
fn c_and_total_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut total = 0;
    for node in item.1.info.contained_iter().take(item.1.dog_count)
    {
        if node.was_infected()
        {
            total += 1;
        }
    }
    (item.0, total as f64)
}

pub struct HeatmapAndMean<H>
{
    pub heatmap: H,
    pub count: Vec<usize>,
    pub sum: Vec<f64>,
    pub sum_sq: Vec<f64>,
    pub max: Vec<f64>,
    pub min: Vec<f64>,
    pub median: Vec<Vec<f64>>,
    pub bin_mids: Vec<f64>
}

impl<H> HeatmapAndMean<H>{
    pub fn new(heatmap: H, size: usize, bin_mids: Vec<f64>) -> Self
    {
        let median = (0..size)
            .map(|_| Vec::new())
            .collect();

        Self { 
            heatmap, 
            count: vec![0; size], 
            sum: vec![0.0; size], 
            sum_sq: vec![0.0; size],
            max: vec![f64::NEG_INFINITY; size],
            min: vec![f64::INFINITY;size],
            median,
            bin_mids
        }
    }

    pub fn write_av(&mut self, label: &str, file: File)
    {
        let mut buf = BufWriter::new(file);
        writeln!(buf, "# {label}").unwrap();
        writeln!(buf, "#index sum sample_count average variance min max median").unwrap();
        let len = self.sum.len();

        for index in 0..len
        {
            let sum = self.sum[index];
            let count = self.count[index];
            let bin_mid = self.bin_mids[index];
            let sum_sq = self.sum_sq[index];
            let average = sum / count as f64;
            let variance = sum_sq / count as f64 - average * average;
            self.median[index].sort_unstable_by(f64::total_cmp);
            let len = self.median[index].len();
            let median = self.median[index].get(len/2).copied().unwrap_or(f64::NAN);
            writeln!(buf, "{bin_mid} {sum} {count} {average} {variance} {} {} {median}", self.min[index], self.max[index]).unwrap();
        }
    }
}

pub fn heatmap_count<'a, 'b, Hw, Hh, It, I, F>(
    heatmap_mean: &mut HeatmapAndMean<HeatmapU<Hw, Hh>>, 
    iter: It,
    fun: F
)
where It: Iterator<Item = I> + 'a,
    I: 'a,
    F: Fn (I) -> (usize, f64) + 'b,
    Hw: HistogramVal<usize> + Histogram,
    Hh: HistogramVal<f64> + Histogram
{
    for i in iter {
        let (c, val) = fun(i);
        let result = heatmap_mean.heatmap.count(c, val);
        if let Ok((x, _)) = result
        {
            heatmap_mean.count[x] += 1;
            heatmap_mean.sum[x] += val;
            heatmap_mean.sum_sq[x] += val * val;
            heatmap_mean.median[x].push(val);
            if heatmap_mean.min[x] > val {
                heatmap_mean.min[x] = val;
            }
            if heatmap_mean.max[x] < val {
                heatmap_mean.max[x] = val;
            }
        }
    }
}

pub fn heatmap_count_with_n<'a, 'b, Hw, Hh, It, I, F>(
    heatmap_mean: &mut HeatmapAndMean<HeatmapU<Hw, Hh>>, 
    iter: It,
    fun: F,
    n: u16
)
where It: Iterator<Item = I> + 'a,
    I: 'a,
    F: Fn (I, u16) -> (usize, f64) + 'b,
    Hw: HistogramVal<usize> + Histogram,
    Hh: HistogramVal<f64> + Histogram
{
    for i in iter {
        let (c, val) = fun(i, n);
        let result = heatmap_mean.heatmap.count(c, val);
        if let Ok((x, _)) = result
        {
            heatmap_mean.count[x] += 1;
            heatmap_mean.sum[x] += val;
            heatmap_mean.sum_sq[x] += val * val;
            heatmap_mean.median[x].push(val);
            if heatmap_mean.min[x] > val {
                heatmap_mean.min[x] = val;
            }
            if heatmap_mean.max[x] < val {
                heatmap_mean.max[x] = val;
            }
        }
    }
}

pub fn heatmap_count_with_n_and_float<'a, 'b, Hw, Hh, It, I, F>(
    heatmap_mean: &mut HeatmapAndMean<HeatmapU<Hw, Hh>>, 
    iter: It,
    fun: F,
    n: u16,
    float: f64
)
where It: Iterator<Item = I> + 'a,
    I: 'a,
    F: Fn (I, u16, f64) -> (usize, f64) + 'b,
    Hw: HistogramVal<usize> + Histogram,
    Hh: HistogramVal<f64> + Histogram
{
    for i in iter {
        let (c, val) = fun(i, n, float);
        let result = heatmap_mean.heatmap.count(c, val);
        if let Ok((x, _)) = result
        {
            heatmap_mean.count[x] += 1;
            heatmap_mean.sum[x] += val;
            heatmap_mean.sum_sq[x] += val * val;
            heatmap_mean.median[x].push(val);
            if heatmap_mean.min[x] > val {
                heatmap_mean.min[x] = val;
            }
            if heatmap_mean.max[x] < val {
                heatmap_mean.max[x] = val;
            }
        }
    }
}


pub fn heatmap_count_multiple<'a, 'b, Hw, Hh, It, I, F>(
    heatmap_mean: &mut HeatmapAndMean<HeatmapU<Hw, Hh>>, 
    iter: It,
    fun: F
)
where It: Iterator<Item = I> + 'a,
    I: 'a,
    F: Fn (&'_ I) -> (usize, Box<dyn Iterator<Item = f64> + '_>),
    Hw: HistogramVal<usize> + Histogram,
    Hh: HistogramVal<f64> + Histogram
{
    for i in iter {
        
        let (c, many_value_iter) = fun(&i);
        
        for val in many_value_iter{
            let result = heatmap_mean.heatmap.count(c, val);
            if let Ok((x, _)) = result
            {
                heatmap_mean.count[x] += 1;
                heatmap_mean.sum[x] += val;
                heatmap_mean.sum_sq[x] += val * val;
                heatmap_mean.median[x].push(val);
                if heatmap_mean.min[x] > val {
                    heatmap_mean.min[x] = val;
                }
                if heatmap_mean.max[x] < val {
                    heatmap_mean.max[x] = val;
                }
            }
        }
        
    }
}
