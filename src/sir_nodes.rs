use serde::*;
use net_ensembles::Node;

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
pub struct GammaTrans
{
    pub gamma: f64,
    pub trans_animal: f64,
    pub trans_human: f64
}


#[derive(Serialize, Deserialize, Copy, Clone)]
pub enum SirState
{
    S,
    I,
    R,
    Transitioning
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct CurrentInfectionProb{
    pub num_i: u64,
    pub product: f64
}


#[derive(Serialize, Deserialize, Copy, Clone)]
pub enum GTHelper{
    G(GammaTrans),
    O(CurrentInfectionProb)
}

#[derive(Clone, Copy)]
pub union GT {
    gamma: GammaTrans,
    other: CurrentInfectionProb
}

impl From<GTHelper> for GT
{
    fn from(other: GTHelper) -> Self {
        match other{
            GTHelper::G(g) => {
                GT{gamma: g}
            },
            GTHelper::O(o) => {
                GT{other: o}
            }
        }
    }
}



// function:
// f(x) = (cos(x*5)+2)/3*exp(-x**2)
#[derive(Clone, Copy)]
pub struct SirFun{
    fun_state: GT,
    sir: SirState
}

impl Default for SirFun{
    fn default() -> Self
    {
        Self{
            sir: SirState::S,
            fun_state: GT{other: CurrentInfectionProb{num_i: 0, product: 0.0}}
        }
    }
}

impl Node for SirFun{
    fn new_from_index(_: usize) -> Self
    {
        SirFun::default()
    }
}

impl From<SirFunHelper> for SirFun {
    fn from(other: SirFunHelper) -> Self {
        let gt = other.fun_state.into();
        Self { fun_state: gt, sir: other.sir }
    }
}

impl Serialize for SirFun{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer {
        
        let g = match self.sir{
            SirState::I => {
                GTHelper::O(unsafe{self.fun_state.other})
            },
            _ => {
                GTHelper::G(unsafe{self.fun_state.gamma})
            }
        };
        let o = SirFunHelper{
            fun_state: g,
            sir: self.sir
        };
        o.serialize(serializer)
    }
}

impl<'a> Deserialize<'a> for SirFun{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'a> {
        let other: SirFunHelper = SirFunHelper::deserialize(deserializer)?;
        Ok(other.into())
    }
}

// function:
// f(x) = (cos(x*5)+2)/3*exp(-x**2)
#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct SirFunHelper{
    fun_state: GTHelper,
    sir: SirState
}

#[inline]
pub fn trans_fun(gamma: f64, max_lambda: f64) -> GammaTrans
{
    let other_gamma = gamma-3.0;
    let g2 = -gamma * gamma;
    let other_g2 = -other_gamma*other_gamma;
    let g10 = gamma * 10.0;
    let other_g10 = other_gamma * 10.0;
    let trans = (g10.cos()+2.0)/3.0*g2.exp()* max_lambda;
    let other_trans = (other_g10.cos()+2.0)/3.0*other_g2.exp()* max_lambda;
    GammaTrans{
        gamma,
        trans_animal: trans,
        trans_human: other_trans,
    }
}



impl SirFun
{

    #[inline]
    pub fn is_susceptible(&self) -> bool
    {
        matches!(self.sir, SirState::S)
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_infected(&self) -> bool
    {
        matches!(self.sir, SirState::I)
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get_sus_state(&self) -> CurrentInfectionProb
    {
        unsafe{self.fun_state.other}
    }

    #[inline]
    pub fn get_gamma_trans(&self) -> GammaTrans
    {
        unsafe{self.fun_state.gamma}
    }

    #[allow(dead_code)]
    pub fn get_infectiouse_neighbor_count(&self) -> u64
    {
        unsafe{
            self.fun_state.other.num_i
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get_gamma(&self) -> f64
    {
        unsafe{self.fun_state.gamma.gamma}
    }

    #[inline]
    pub fn set_s(&mut self)
    {
        self.sir = SirState::S;
    }

    #[inline]
    #[allow(dead_code)]
    pub fn set_s_ld(&mut self)
    {
        self.sir = SirState::S;
        self.fun_state.other = CurrentInfectionProb{
            num_i: 0,
            product: 1.0
        };
    }

    #[inline]
    #[allow(dead_code)]
    pub fn add_to_s(&mut self, other_trans: f64)
    {
        let counter_prob = 1.0 - other_trans;
        unsafe{ 
            self.fun_state.other.product *= counter_prob;
            self.fun_state.other.num_i += 1;
        }

    }

    #[inline]
    #[allow(dead_code)]
    pub fn subtract_from_s(&mut self, other_trans: f64)
    {
        let counter_prob = 1.0 - other_trans;
        unsafe{
            self.fun_state.other.product /= counter_prob;
            self.fun_state.other.num_i -= 1;
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn set_gt_and_transition(&mut self, gamma: f64, max_lambda: f64)
    {
        self.fun_state.gamma = trans_fun(gamma, max_lambda);
        self.sir = SirState::Transitioning;
    }

    #[inline]
    #[allow(dead_code)]
    pub fn transition_to_i(&mut self)
    {
        self.sir = SirState::I;
    }

    #[inline]
    pub fn progress_to_i(&mut self, gamma: f64, max_lambda: f64)
    {
        self.sir = SirState::I;
        self.fun_state.gamma = trans_fun(gamma, max_lambda);
    }

    #[inline]
    pub fn progress_to_i_with_gt(&mut self, gamma_trans: GammaTrans)
    {
        self.sir = SirState::I;
        self.fun_state.gamma = gamma_trans;
    }

    #[inline]
    pub fn progress_to_r(&mut self)
    {
        self.sir = SirState::R;
    }
}