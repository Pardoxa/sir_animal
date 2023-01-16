use serde::{Serialize, Deserialize};

use crate::misc::*;
use crate::simple_sample::BaseOpts;
use crate::sir_nodes::FunChooser;
use std::num::{NonZeroUsize, NonZeroI32};

#[derive(Serialize, Deserialize, Clone)]
pub struct BeginEntropicOpts
{
    pub file_name: String,
    pub time: Option<RequestedTime>,
    pub target_samples: NonZeroUsize,
    pub fun_type: FunChooser
}

impl Default for BeginEntropicOpts
{
    fn default() -> Self {
        Self{
            file_name: "".to_owned(),
            time: Some(RequestedTime::default()),
            target_samples: NonZeroUsize::new(200000).unwrap(),
            fun_type: FunChooser::default()
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WlContinueOpts
{
    pub file_name: String,
    pub time: RequestedTime,
    pub fun_type: FunChooser
}

impl Default for WlContinueOpts
{
    fn default() -> Self {
        Self{
            time: RequestedTime::default(),
            file_name: "".to_owned(),
            fun_type: FunChooser::default()
        }
    }
}

#[derive(Clone)]
pub struct WlOptsEntropicWrapper
{
    item: WlOpts
}

impl QuickName for WlOptsEntropicWrapper
{
    fn quick_name(&self) -> String
    {
        let old_name = self.item.base_opts.quick_name();

        format!(
            "{old_name}ENTROPIC{}",
            if let Some(val) = self.item.interval{
                format!("I{}-{}", val.start, val.end_inclusive)
            } else {
                "".to_owned()
            }
        )
    }
}

impl From<WlOpts> for WlOptsEntropicWrapper
{
    fn from(o: WlOpts) -> Self
    {
        WlOptsEntropicWrapper { item: o }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RewlOpts
{
    pub time: RequestedTime,
    pub base_opts: BaseOpts,
    pub max_time_steps: NonZeroUsize,
    pub markov_seed: u64,
    pub markov_step_size: NonZeroUsize,
    pub log_f_threshold: f64,
    pub interval: Vec<Interval>,
    pub init_with_at_least: Option<NonZeroI32>,
    pub wl_seed: u64,
    pub which_fun: FunChooser,
    pub biased_dog_mutation: Option<f64>,
    pub neg_bins: i32
}


impl Default for RewlOpts
{
    fn default() -> Self {
        Self{
            time: RequestedTime::default(),
            base_opts: BaseOpts::default(),
            markov_seed: 2384720,
            markov_step_size: NonZeroUsize::new(2000).unwrap(),
            log_f_threshold: 1e-6,
            interval: vec![Interval{start: 0, end_inclusive: 1}],
            init_with_at_least: None,
            max_time_steps: NonZeroUsize::new(1000).unwrap(),
            wl_seed: 28934624,
            which_fun: FunChooser::default(),
            biased_dog_mutation: None,
            neg_bins: -100
        }    
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy)]
pub enum ReplicaMode
{
    REES,
    REWL
}

impl ReplicaMode
{
    pub fn to_name(self) -> &'static str
    {
        match self {
            Self::REES => "REES",
            Self::REWL => "REWL"
        }
    }
}

impl RewlOpts
{
    pub fn sort_interval(&mut self)
    {
        self.interval.sort_unstable_by_key(|item| item.start);
    }

    pub fn quick_name(&self, index: Option<usize>, mode: ReplicaMode, times_repeated: usize) -> String
    {
        let old_name = self.base_opts.quick_name();

        let interval = if let Some(index) = index{
            let interval = self.interval[index];
            format!("I{}-{}", interval.start, interval.end_inclusive)
        } else {
            "".to_owned()
        };

        format!(
            "{old_name}{}_{interval}x{times_repeated}",
            mode.to_name()
        )
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct JumpScan
{
    pub time: RequestedTime,
    pub base_opts: BaseOpts,
    pub max_time_steps: NonZeroUsize,
    pub markov_seeding_seed: u64,
    pub markov_step_size: NonZeroUsize,
    pub log_f_threshold: f64,
    pub interval: Option<Interval>,
    pub wl_seeding_seed: u64,
    pub mutation_end: f64,
    pub mutation_samples: NonZeroUsize,
    pub neg_bins: NonZeroI32
}

impl JumpScan
{
    pub fn quick_name_from_start(&self, current_mutation: f64) -> String
    {
        let old_name = self.base_opts.quick_name();

        format!(
            "{old_name}WL{current_mutation}{}",
            if let Some(val) = self.interval{
                format!("I{}-{}", val.start, val.end_inclusive)
            } else {
                "".to_owned()
            }
        )
    }
}

impl Default for JumpScan
{
    fn default() -> Self {
        Self{
            time: RequestedTime::default(),
            base_opts: BaseOpts::default(),
            markov_seeding_seed: 2384720,
            markov_step_size: NonZeroUsize::new(2000).unwrap(),
            log_f_threshold: 1e-6,
            interval: Some(Interval{start: 0, end_inclusive: 1}),
            max_time_steps: NonZeroUsize::new(1000).unwrap(),
            wl_seeding_seed: 218934624,
            neg_bins: NonZeroI32::new(-100).unwrap(),
            mutation_end: 0.5,
            mutation_samples: NonZeroUsize::new(4).unwrap()
        }    
    }
    
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WlOpts
{
    pub time: RequestedTime,
    pub base_opts: BaseOpts,
    pub max_time_steps: NonZeroUsize,
    pub markov_seed: u64,
    pub markov_step_size: NonZeroUsize,
    pub log_f_threshold: f64,
    pub interval: Option<Interval>,
    pub init_with_at_least: Option<NonZeroI32>,
    pub wl_seed: u64,
    pub neg_bins: i32
}

impl QuickName for WlOpts
{
    fn quick_name(&self) -> String
    {
        let old_name = self.base_opts.quick_name();

        format!(
            "{old_name}WL{}",
            if let Some(val) = self.interval{
                format!("I{}-{}", val.start, val.end_inclusive)
            } else {
                "".to_owned()
            }
        )
    }
}

impl Default for WlOpts
{
    fn default() -> Self {
        Self{
            time: RequestedTime::default(),
            base_opts: BaseOpts::default(),
            markov_seed: 2384720,
            markov_step_size: NonZeroUsize::new(2000).unwrap(),
            log_f_threshold: 1e-6,
            interval: Some(Interval{start: 0, end_inclusive: 1}),
            init_with_at_least: None,
            max_time_steps: NonZeroUsize::new(1000).unwrap(),
            wl_seed: 28934624,
            neg_bins: -100
        }    
    }
    
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LdSimpleOpts
{
    pub time: RequestedTime,
    pub base_opts: BaseOpts,
    pub max_time_steps: NonZeroUsize,
    pub markov_seed: u64,
    pub markov_step_size: NonZeroUsize,
    pub samples: NonZeroUsize,
    pub neg_bins: i32
}

impl Default for LdSimpleOpts
{
    fn default() -> Self {
        Self { 
            time: RequestedTime::default(), 
            base_opts: BaseOpts::default(), 
            max_time_steps: NonZeroUsize::new(800).unwrap(), 
            markov_seed: 124870, 
            markov_step_size: NonZeroUsize::new(2000).unwrap(),
            samples: NonZeroUsize::new(1000).unwrap(),
            neg_bins: -100
        }
    }
}

impl QuickName for LdSimpleOpts
{
    fn quick_name(&self) -> String {
        let old_name = self.base_opts.quick_name();

        format!(
            "{old_name}MS{}S",
            self.markov_seed
        )
    }
}