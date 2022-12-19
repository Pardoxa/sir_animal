use std::any::TypeId;
use serde::*;
use net_ensembles::Node;

macro_rules! fun_choose {
    ($a: ident, $b: expr, $tree: tt) => {
        match $b {
            crate::sir_nodes::FunChooser::Beta => {
                $a::<crate::sir_nodes::BetaFun>$tree
            },
            crate::sir_nodes::FunChooser::FirstTest => {
                $a::<crate::sir_nodes::FirstFun>$tree
            }
            _ => {todo!()}
        }
    };
}
pub(crate) use fun_choose;

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

impl SirState
{
    #[inline]
    pub fn was_ever_infected(self) -> bool
    {
        matches!(self, SirState::I | SirState::R)
    }
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
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
    pub gamma: GammaTrans,
    pub other: CurrentInfectionProb
}

impl GT {
    pub unsafe fn get_gamma(&self) -> f64
    {
        self.gamma.gamma
    }

    pub unsafe fn get_trans_human(&self) -> f64
    {
        self.gamma.trans_human
    }
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

pub trait TransFun
{
    fn trans_fun(gamma: f64, max_lambda: f64) -> GammaTrans;
}

#[derive(Clone, Copy, Default, Serialize, Deserialize, Debug)]
pub enum FunChooser{
    #[default]
    Beta,
    Gauß,
    FirstTest
}

impl FunChooser {
    pub fn get_str(self) -> &'static str
    {
        match self
        {
            Self::Beta => "Beta",
            Self::Gauß => "Gauß",
            Self::FirstTest => "fTest"
        }
    }

    pub fn get_type_id(self) -> TypeId
    {
        match self{
            Self::Beta => TypeId::of::<BetaFun>(),
            _ => todo!()
        }
    }
}

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct GaußFun{}

impl TransFun for GaußFun
{
    #[inline]
    fn trans_fun(_: f64, _: f64) -> GammaTrans {
        todo!()
    }
}

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct BetaFun{}

impl TransFun for BetaFun
{
    #[inline]
    fn trans_fun(gamma: f64, max_lambda: f64) -> GammaTrans {
        let other_gamma = gamma-2.0*std::f64::consts::SQRT_2+0.40693138353594516;
        let sq = -gamma*gamma;
        
        let g5 = gamma*5.0;
        let mut dog_trans = (sq+2.0)*(g5.cos()+2.0)/6.0;

        let human_sq = -other_gamma*other_gamma;
        let human_g5 = other_gamma*5.0;
        let mut human_trans = (human_sq+2.0)*(human_g5.cos()+2.0)/6.0;

        if human_trans <= 0.0 {
            human_trans = 0.0;
        } else {
            human_trans *= max_lambda;
        }
        if dog_trans <= 0.0 {
            dog_trans = 0.0;
        } else {
            dog_trans *= max_lambda;
        }

        GammaTrans { 
            gamma, 
            trans_animal: dog_trans,
            trans_human: human_trans
        }
    }
}

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct FirstFun{}

impl TransFun for FirstFun
{
    #[inline]
    fn trans_fun(gamma: f64, max_lambda: f64) -> GammaTrans {
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
}

// function:
// f(x) = (cos(x*5)+2)/3*exp(-x**2)
#[derive(Clone, Copy)]
pub struct SirFun<Trans>{
    pub fun_state: GT,
    pub sir: SirState,
    trans: Trans
}

impl<T> Default for SirFun<T>
where T: Default{
    fn default() -> Self
    {
        Self{
            sir: SirState::S,
            fun_state: GT{other: CurrentInfectionProb{num_i: 0, product: 0.0}},
            trans: T::default()
        }
    }
}

impl<T> Node for SirFun<T>
where T: Clone+ Default + Serialize{
    fn new_from_index(_: usize) -> Self
    {
        SirFun::default()
    }
}

impl<T> From<SirFunHelper<T>> for SirFun<T> {
    fn from(other: SirFunHelper<T>) -> Self {
        let gt = other.fun_state.into();
        Self { fun_state: gt, sir: other.sir, trans: other.trans }
    }
}

impl<T> Serialize for SirFun<T>
where T: Serialize + Clone
{
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
            sir: self.sir,
            trans: self.trans.clone()
        };
        o.serialize(serializer)
    }
}

impl<'a, T> Deserialize<'a> for SirFun<T>
where T: Deserialize<'a>{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'a> {
        let other: SirFunHelper<T> = SirFunHelper::<T>::deserialize(deserializer)?;
        Ok(other.into())
    }
}

// function:
// f(x) = (cos(x*5)+2)/3*exp(-x**2)
#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct SirFunHelper<T>{
    fun_state: GTHelper,
    sir: SirState,
    trans: T
}



impl<T> SirFun<T>
where T: TransFun
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
    pub fn was_ever_infected(&self) -> bool
    {
        self.sir.was_ever_infected()
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
    pub fn set_gt_and_transition(&mut self, gamma: f64, max_lambda: f64)
    {
        self.fun_state.gamma = T::trans_fun(gamma, max_lambda);
        self.sir = SirState::Transitioning;
    }

    #[inline]
    pub fn transition_to_i(&mut self)
    {
        self.sir = SirState::I;
    }

    #[inline]
    pub fn progress_to_i(&mut self, gamma: f64, max_lambda: f64)
    {
        self.sir = SirState::I;
        self.fun_state.gamma = T::trans_fun(gamma, max_lambda);
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