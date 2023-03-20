use std::{
    fs::File, 
    path::PathBuf, 
    io::{BufReader, BufWriter}, 
    str::FromStr, 
    fmt::Write, 
    sync::RwLock, 
    ops::DerefMut,
    num::*
};
use bincode_helper::*;
use structopt::StructOpt;

use super::{CondensedInfo, InfoNode, InfoGraph, TopologyGraph, InfectedBy};

static GLOBAL_TOPOLOGY: RwLock<Option<TopologyGraph>> = RwLock::new(None); 

#[derive(Debug, Clone)]
pub enum WhichCol
{
    Time, 
    LambdaHuman,
    LambdaAnimal,
    Gamma,
    RecoveryTime,
    Duration,
    Children,
    GlobalDegree,
    GlobalDegreeTimesDuration,
    Mutation
}

impl FromStr for WhichCol
{
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();

        match lower.as_str()
        {
            "mutation" | "m" => Ok(Self::Mutation),
            "children" | "c" => Ok(Self::Children),
            "globaldegree" => Ok(Self::GlobalDegree),
            "gdxd" => Ok(Self::GlobalDegreeTimesDuration),
            "duration" => Ok(Self::Duration),
            "r" | "recovery" => Ok(WhichCol::RecoveryTime),
            "time" | "t" => Ok(WhichCol::Time),
            "lambda_human" | "lambdahuman" | "human" | "h" => Ok(WhichCol::LambdaHuman),
            "lambda_dog" | "lambda_animal" | "animal" | "dog" | "d" => Ok(WhichCol::LambdaAnimal),
            "gamma" | "g" => Ok(WhichCol::Gamma),
            _ => {
                let matching = r#""mutation" | "m" => Ok(Self::Mutation),
"children" | "c" => Ok(Self::Children),
"globaldegree" => Ok(Self::GlobalDegree),
"gdxd" => Ok(Self::GlobalDegreeTimesDuration),
"duration" => Ok(Self::Duration),
"r" | "recovery" => Ok(WhichCol::RecoveryTime),
"time" | "t" => Ok(WhichCol::Time),
"lambda_human" | "lambdahuman" | "human" | "h" => Ok(WhichCol::LambdaHuman),
"lambda_dog" | "lambda_animal" | "animal" | "dog" | "d" => Ok(WhichCol::LambdaAnimal),
"gamma" | "g" => Ok(WhichCol::Gamma),"#;
                eprintln!("{matching}");
                Err("Unknown patter") 
            }
        }
    }
}


#[derive(Debug, Clone)]
pub enum Label
{
    Extra(String),
    Time,
    Id,
    Layer,
    Gamma,
    LambdaHuman,
    LambdaAnimal,
    Children,
    Duration,
    GlobalDegree,
    GlobalDegreeTimesDuration,
    Mutation
}

impl Label
{
    pub fn push_str(&self, index: usize, info: &InfoGraph, s: &mut String, float_precision: Option<NonZeroUsize>)
    {
        let node = info.info.at(index);

        let format_floats = |s: &mut String, to_write: f64|
        {
            if let Some(p) = float_precision
            {
                let _ = s.write_fmt(format_args!("{:.*}", p.get(), to_write));
            } else {
                let _ = s.write_fmt(format_args!("{to_write}"));
            }
        };
        match self
        {
            Self::Extra(e) => {
                let _ = s.write_fmt(format_args!("{e}"));
            },
            Self::Id => {
                let _ = s.write_fmt(format_args!("{index}"));
            },
            Self::Time => {
                let _ = s.write_fmt(format_args!("{}", node.time_step.unwrap()));
            },
            Self::Layer => {
                let _ = s.write_fmt(format_args!("{}", node.layer.unwrap()));
            },
            Self::Gamma => {
                format_floats(s, node.get_gamma());
            },
            Self::LambdaHuman => {
                format_floats(s, node.get_lambda_human());
            },
            Self::LambdaAnimal => {
                format_floats(s, node.get_lambda_dog());
            },
            Self::Children => {
                let _ = s.write_fmt(format_args!("{}", info.disease_children_count[index]));
            },
            Self::Duration =>
            {
                let a = node.get_time();
                let b = node.get_recovery_time();
                let dur = b - a;
                let _ = s.write_fmt(format_args!("{dur}"));
            },
            Self::GlobalDegree => {
                let top = GLOBAL_TOPOLOGY.read()
                    .unwrap();
                let degree = top.as_ref().unwrap().degree(index).unwrap();
                let _ = s.write_fmt(format_args!("{degree}"));

            },
            Self::GlobalDegreeTimesDuration =>
            {
                let top = GLOBAL_TOPOLOGY.read()
                    .unwrap();
                let degree = top.as_ref().unwrap().degree(index).unwrap();
                let a = node.get_time();
                let b = node.get_recovery_time();
                let dur = b - a;
                let degree_times_duration = dur * degree as f64;
                let _ = s.write_fmt(format_args!("{degree_times_duration}"));
            },
            Self::Mutation => {
                if let InfectedBy::By(by) = node.infected_by
                {
                    let vorfahre = info.info.at(by as usize);
                    let gamma_now = node.get_gamma();
                    let old_gamma = vorfahre.get_gamma();
                    let mutation = (old_gamma -gamma_now).abs();
                    format_floats(s, mutation);
                } else {
                    s.push_str("init");
                }
                
            }
        }
    }
}

pub fn push_all(
    info: &InfoGraph, 
    index: usize, 
    to_do: &[Label],
    float_precision: Option<NonZeroUsize>
) -> String 
{
    let mut s = "".to_owned();
    let mut iter = to_do.iter();
    if let Some(t) = iter.next()
    {
        t.push_str(index, info, &mut s, float_precision)
    }
    for n in iter {
        s.push('\n');
        n.push_str(index, info, &mut s, float_precision);
    }
    s
}

impl FromStr for Label
{
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s
        {
            "mutation" | "m" => Ok(Self::Mutation),
            "children" | "c" => Ok(Self::Children),
            "time" | "t" => Ok(Self::Time),
            "id" | "index" => Ok(Self::Id),
            "layer" => Ok(Self::Layer),
            "gamma" => Ok(Self::Gamma),
            "lambda_human" | "lambdahuman" | "human" | "h" => Ok(Self::LambdaHuman),
            "lambda_dog" | "lambda_animal" | "animal" | "dog" | "d" => Ok(Self::LambdaAnimal),
            "duration" => Ok(Self::Duration),
            "globaldegree" => Ok(Self::GlobalDegree),
            "gdxd" => Ok(Self::GlobalDegreeTimesDuration),
            o => Ok(Self::Extra(o.to_owned())) 
        }
    }
}

#[derive(Debug, StructOpt, Clone)]
pub struct PrintOpts
{
    /// File to read
    #[structopt(long, short)]
    pub file: PathBuf,

    /// Print specific energy if possible
    #[structopt(long, short)]
    pub energy: Option<usize>,

    #[structopt(long)]
    /// if provided this will be the output name. Otherwise output will be printed to console
    pub out: Option<PathBuf>,

    /// Debug info
    #[structopt(long, short)]
    pub debug: bool,

    #[structopt(long, default_value="1.0")]
    pub scaling_factor: f64,

    #[structopt(long, default_value="0.0")]
    pub subtract_beginning: f64,

    #[structopt(long, short)]
    pub invert_color: bool,

    #[structopt(long, short, default_value="0.0")]
    pub threshold: f64,

    #[structopt(long)]
    /// Which data to look at: m - mutation, c - children, globaldegree, gdxd - globaldegreetimesduration
    /// duration, r: recovery, time, lambda_human, lambda_dog, gamma
    pub col: WhichCol,

    #[structopt(long)]
    pub other_color: bool,

    #[structopt(long)]
    pub label: Vec<Label>,

    #[structopt(long)]
    /// Print help for labels
    pub label_help: bool,

    #[structopt(long)]
    /// directly create dot file
    pub dot: bool,

    #[structopt(long)]
    float_precision: Option<NonZeroUsize>,

    /// print info from index
    #[structopt(long)]
    pub index: Option<usize>,

    #[structopt(long)]
    /// Just count the number of entries, ignore everything else
    pub count: bool,
}


impl PrintOpts
{
    pub fn execute(&self) 
    {
        if self.label_help{
            
let help = r#""mutation" | "m" => Ok(Self::Mutation),
"children" | "c" => Ok(Self::Children),
"time" | "t" => Ok(Self::Time),
"id" | "index" => Ok(Self::Id),
"layer" => Ok(Self::Layer),
"gamma" => Ok(Self::Gamma),
"lambda_human" | "lambdahuman" | "human" | "h" => Ok(Self::LambdaHuman),
"lambda_dog" | "lambda_animal" | "animal" | "dog" | "d" => Ok(Self::LambdaAnimal),
"duration" => Ok(Self::Duration),
"globaldegree" => Ok(Self::GlobalDegree),
"gdxd" => Ok(Self::GlobalDegreeTimesDuration),
o => Ok(Self::Extra(o.to_owned()))"#;
            eprintln!("Add as many labels as you want. The labels are defined via \n{help}");
            return;
        }

        
        let file = File::open(&self.file)
            .unwrap();
        let buf = BufReader::new(file);
        let mut de = DeserializeAnyhow::new(buf);

        let topology: TopologyGraph = de.deserialize().unwrap();
        let mut lock = GLOBAL_TOPOLOGY.write().unwrap();
        let t = lock.deref_mut();
        *t = Some(topology);
        drop(lock);
        if self.count{
            let mut counter = 0;
            let mut _compat: (usize, CondensedInfo);
            loop{
                _compat = match de.deserialize(){
                    Ok(v) => v,
                    Err(_) => {
                        println!("Contains {counter} elements");
                        return;
                    }
                };
                counter += 1;
            }
        }

        let compat: (usize, CondensedInfo) = match self.energy
        {
            None => {
                if let Some(index) = self.index{
                    let mut compat: (usize, CondensedInfo);
                    let mut counter = 0;
                    loop{
                        compat = de.deserialize().unwrap();
                        if counter == index{
                            break compat;
                        }
                        counter += 1;
                    }
                } else {
                    de.deserialize().unwrap()
                }
            },
            Some(e) => {
                loop{
                    let compat: (usize, CondensedInfo) = de.deserialize().unwrap();
                    if self.debug
                    {
                        println!("{}", compat.0);
                    }
                    if compat.0 == e {
                        break compat;
                    }
                }
            }
        };

        let mut chooser: Option<fn(&InfoNode) -> f64> = match self.col{
            WhichCol::Time          => Some(InfoNode::get_time),
            WhichCol::LambdaHuman   => Some(InfoNode::get_lambda_human),
            WhichCol::LambdaAnimal  => Some(InfoNode::get_lambda_dog),
            WhichCol::Gamma         => Some(InfoNode::get_gamma),
            WhichCol::RecoveryTime  => Some(InfoNode::get_recovery_time),
            WhichCol::Duration      => Some(InfoNode::get_time_difference),
            _                       => None
        };
        
        #[allow(clippy::type_complexity)]
        let mut which: Box<dyn FnMut (&InfoGraph, usize) -> f64> = if let Some(which) = &mut chooser
        {
            Box::new(
                |info: &InfoGraph, index: usize|
                {
                    let node = info.info.at(index);
                    which(node)
                }
            )
        } else {
            match self.col
            {
                WhichCol::Mutation => {
                    Box::new(
                        |info: &InfoGraph, index: usize|
                        {
                            let node = info.info.at(index);
                            if let InfectedBy::By(by) = node.infected_by
                            {
                                let vorfahre = info.info.at(by as usize);
                                let gamma_now = node.get_gamma();
                                let old_gamma = vorfahre.get_gamma();
                                (old_gamma -gamma_now).abs() 
                            } else {
                                0.0
                            }
                        }
                    )
                }
                WhichCol::Children => {
                    Box::new(
                        |info: &InfoGraph, index: usize|
                        {
                            info.disease_children_count[index] as f64
                        }
                    )
                },
                WhichCol::GlobalDegree => {
                    Box::new(
                        |_: &InfoGraph, index: usize|
                        {
                            let lock = GLOBAL_TOPOLOGY.read()
                                .unwrap();
                            let degree = lock.as_ref().unwrap().degree(index).unwrap();
                            degree as f64
                        }
                    )
                },
                WhichCol::GlobalDegreeTimesDuration => {
                    Box::new(
                        |info: &InfoGraph, index: usize|
                        {
                            let lock = GLOBAL_TOPOLOGY.read()
                                .unwrap();
                            let degree = lock.as_ref().unwrap().degree(index).unwrap();
                            let duration = info.info.at(index).get_time_difference();
                            degree as f64 * duration
                        }
                    )
                },
                _ => unreachable!()
            }
            
        };

        let scaling = |mut val: f64| 
        {
            val -= self.subtract_beginning;
            val *= self.scaling_factor;
            if self.invert_color{
                1.0-val
            } else {
                val
            }
        };

        let mut infos = compat.1.to_info_graph();

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        let color = if self.other_color
        {
            super::color
        } else {
            super::color2
        };

        let fun = |info: &InfoGraph, _human_or_dog: super::HumanOrDog, index|
        {
            let quantity_of_interest = which(info, index);
            let val = scaling(quantity_of_interest);
            min = min.min(val);
            max = max.max(val);
            color(val)
        };


        let label_fun = |node: &InfoGraph, _human_or_dog: super::HumanOrDog, index|
        {
            push_all(node, index, &self.label, self.float_precision)
        };

        if let Some(outname) = &self.out
        {
            let file = File::create(outname)
                .expect("unable to create file");
            let buf = BufWriter::new(file);
            super::write_dot(
                &mut infos, 
                buf,
                fun,
                label_fun
            );
        } else {
            super::write_dot(
                &mut infos, 
                std::io::stdout(),
                fun,
                label_fun
            );
        }

        if self.debug
        {
            println!("min: {min}");
            println!("max: {max}");
        }

        if self.dot
        {
            let path = self.out.as_ref().unwrap().as_os_str();

            let tpdf= "-Tpdf".as_ref();

            let mut pdf = self.out.as_ref().unwrap().clone();
            pdf.set_extension("pdf");

            let pdf = pdf.as_os_str();

            let o = "-o".as_ref();

            let out = std::process::Command::new("dot")
                .args([path, tpdf, o, pdf])
                .output();

            match out
            {
                Ok(o) => println!("out: {} err: {}", std::str::from_utf8(&o.stdout).unwrap(), std::str::from_utf8(&o.stderr).unwrap()),
                Err(e) => eprintln!("err: {e}")
            }
        }
        
    }
}