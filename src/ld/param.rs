use serde::{Serialize, Deserialize};

use crate::misc::*;
use crate::simple_sample::BaseOpts;
use std::num::NonZeroUsize;

#[derive(Serialize, Deserialize, Clone)]
pub struct BeginEntropicOpts
{
    pub file_name: String,
    pub time: Option<RequestedTime>,
    pub target_samples: NonZeroUsize
}

impl Default for BeginEntropicOpts
{
    fn default() -> Self {
        Self{
            file_name: "".to_owned(),
            time: Some(RequestedTime::default()),
            target_samples: NonZeroUsize::new(200000).unwrap()
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WlContinueOpts
{
    pub file_name: String,
    pub time: RequestedTime
}

impl Default for WlContinueOpts
{
    fn default() -> Self {
        Self{
            time: RequestedTime::default(),
            file_name: "".to_owned()
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
pub struct WlOpts
{
    pub time: RequestedTime,
    pub base_opts: BaseOpts,
    pub max_time_steps: NonZeroUsize,
    pub markov_seed: u64,
    pub markov_step_size: NonZeroUsize,
    pub log_f_threshold: f64,
    pub interval: Option<Interval>,
    pub init_with_at_least: Option<NonZeroUsize>,
    pub wl_seed: u64
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
    pub samples: NonZeroUsize
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
            samples: NonZeroUsize::new(1000).unwrap()
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