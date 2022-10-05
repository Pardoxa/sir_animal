use serde::{Serialize, Deserialize};

use crate::misc::*;
use crate::simple_sample::BaseOpts;
use std::num::NonZeroUsize;

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
    pub wl_seed: u64
}

pub trait QuickName
{
    fn quick_name(&self) -> String;
    fn quick_name_with_ending(&self, ending: &str) -> String
    {
        format!("{}{ending}", self.quick_name())
    }
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
            max_time_steps: NonZeroUsize::new(1000).unwrap(),
            wl_seed: 28934624
        }    
    }
    
}