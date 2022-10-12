use crate::{sir_nodes::*, simple_sample::{BaseModel, PATIENTS_USIZE}};
use net_ensembles::{dual_graph::*, rand::{SeedableRng, seq::SliceRandom, Rng}, MarkovChain};
use rand_distr::{Uniform, StandardNormal, Distribution, Binomial};
use rand_pcg::Pcg64;
use serde::{Serialize, Deserialize};
use std::{num::*, io::Write, ops::Add};
use net_ensembles::{AdjList, AdjContainer};

use super::SirWriter;

const ROTATE: f64 = 0.01;
const PATIENT_MOVE: f64 = 0.03;
const P0_RAND: f64 = 0.04;
const MUTATION_MOVE: f64 = 0.11;
const BY_WHOM: f64 = 0.12; 
const ALEX_MOVE: f64 = 0.14;
const TIME_MOVE: f64 = 0.15;



#[derive(Clone, Serialize, Deserialize)]
pub struct Mutation
{
    pub mut_vec: Vec<f64>
}

impl Mutation
{

    #[inline]
    pub fn get(&self, index: usize) -> f64
    {
        //let add = index + self.offset;
        //if add < self.mut_vec.len()
        //{
        //    self.mut_vec[add]
        //} else {
        //    self.mut_vec[add - self.mut_vec.len()]
        //}
        unsafe{*self.mut_vec.get_unchecked(index)}
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> &mut f64
    {
        //let add = index + self.offset;
        //let len = self.mut_vec.len();
        //if add < len
        //{
        //    &mut self.mut_vec[add]
        //} else {
        //    &mut self.mut_vec[add - len]
        //}
        unsafe{self.mut_vec.get_unchecked_mut(index)}
    }

    pub fn mutation_swap(&mut self, step: &mut Vec<StepEntry>, rng: &mut Pcg64, amount: usize)
    {
        step
            .extend(
                self.mut_vec.iter()
                    .map(
                        |val| StepEntry{float: *val}
                    )
            );
        let index_uniform = Uniform::new(0, self.mut_vec.len());

        let num = rng.gen_range(1..amount);
        for _ in 0..num {
            let i1 = index_uniform.sample(&mut *rng);
            let i2 = index_uniform.sample(&mut *rng);
            self.mut_vec.swap(i1, i2);
        }
//        self.mut_vec.shuffle(rng)
    }

    pub fn unshuffle(&mut self, step: &[StepEntry])
    {
        self.mut_vec.iter_mut()
            .zip(step.iter())
            .for_each(
                |(s, o)|
                {
                    *s = unsafe{o.float};
                }
            );
    }
}


#[derive(Clone, Copy, Serialize, Deserialize, Default, Debug)]
pub struct Stats
{
    pub rejected: usize,
    pub accepted: usize
}

impl Stats
{
    #[inline]
    pub fn accept(&mut self)
    {
        self.accepted += 1;
    }

    #[inline]
    pub fn reject(&mut self)
    {
        self.rejected += 1;
    }

    pub fn total(&self) -> usize
    {
        self.rejected + self.accepted
    }

    pub fn acception_rate(&self) -> f64
    {
        self.accepted  as f64 / self.total() as f64
    }

    pub fn rejection_rate(&self) -> f64
    {
        self.rejected as f64 / self.total() as f64
    }
}

impl Add for Stats 
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self{
            accepted: self.accepted + rhs.accepted,
            rejected: self.rejected + rhs.rejected
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Default, Debug)]
pub struct MarkovStats
{
    pub rotation_both: Stats,
    pub rotation_animal: Stats,
    pub rotation_humans: Stats,
    pub patient_move: Stats,
    pub alex_move: Stats,
    pub time_move: Stats,
    pub by_whom: Stats,
    pub trans_rec: Stats,
    pub mutation_humans: Stats,
    pub mutation_animals: Stats,
    pub mutation_both: Stats,
    pub mutation_dogs_from_humans: Stats,
    pub mutation_humans_from_dogs: Stats,
    pub mutation_change: Stats,
    pub dfh_swap: Stats
}

impl MarkovStats
{
    pub fn log<W: Write>(&self, mut writer: W)
    where W: Write
    {

        let sum = self.rotation_both 
            + self.rotation_animal
            + self.rotation_humans
            + self.patient_move
            + self.alex_move
            + self.time_move
            + self.by_whom
            + self.trans_rec
            + self.mutation_humans
            + self.mutation_animals
            + self.mutation_both
            + self.mutation_dogs_from_humans
            + self.mutation_humans_from_dogs
            + self.mutation_change
            + self.dfh_swap;

        writeln!(writer, "#total:").unwrap();
        writeln!(writer, "#\ttotal:{}", sum.total()).unwrap();
        writeln!(writer, "#\taccepted: {} rate {}", sum.accepted, sum.acception_rate()).unwrap();
        writeln!(writer, "#\trejected: {} rate {}", sum.rejected, sum.rejection_rate()).unwrap();

        let mut logger = |stats: Stats, name| 
        {
            writeln!(writer, "#{name}:").unwrap();
            writeln!(writer, "#\ttotal:{}", stats.total()).unwrap();
            writeln!(writer, "#\taccepted: {} rate {}", stats.accepted, stats.acception_rate()).unwrap();
            writeln!(writer, "#\trejected: {} rate {}", stats.rejected, stats.rejection_rate()).unwrap();
        };

        macro_rules! log {
            ($t: ident) => {
                logger(self.$t, stringify!($t))                
            };
        }

        log!(rotation_both);
        log!(rotation_animal);
        log!(rotation_humans);
        log!(patient_move);
        log!(alex_move);
        log!(time_move);
        log!(by_whom);
        log!(trans_rec);
        log!(mutation_humans);
        log!(mutation_animals);
        log!(mutation_both);
        log!(mutation_dogs_from_humans);
        log!(mutation_humans_from_dogs);
        log!(mutation_change);
        log!(dfh_swap);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LdModel
{
    pub dual_graph: DefaultSDG<SirFun, SirFun>,
    pub reset_gamma: f64,
    pub markov_rng: Pcg64,
    pub recover_prob: f64,
    pub max_lambda: f64,
    pub sigma: f64,
    pub initial_gt: GammaTrans,
    pub offset_humans: Offset,
    pub offset_dogs: Offset,
    pub infected_by_whom_dogs: Vec<f64>,
    pub infected_by_whom_humans: Vec<f64>,
    pub mutation_vec_dogs: Mutation,
    pub mutation_vec_humans: Mutation,
    pub mutation_humans_from_dogs: Mutation,
    pub mutation_dogs_from_humans: Mutation,
    pub total_sim_counter: usize,
    pub unfinished_sim_counter: usize,
    pub max_degree_dogs: NonZeroUsize,
    pub max_time_steps: NonZeroUsize,
    pub new_infections_list_humans: Vec<usize>,
    pub new_infections_list_dogs: Vec<usize>,
    pub infected_list_humans: Vec<usize>,
    pub infected_list_dogs: Vec<usize>,
    pub trans_rand_vec_humans: Vec<f64>,
    pub trans_rand_vec_dogs: Vec<f64>,
    pub recovery_rand_vec_humans: Vec<f64>,
    pub recovery_rand_vec_dogs: Vec<f64>,
    pub initial_patients: [usize;PATIENTS_USIZE],
    pub last_extinction: usize,
    pub stats: MarkovStats
}



#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct PatientMove
{
    pub index_in_patient_vec: usize,
    pub old_node: usize
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ExchangeInfo
{
    pub index: usize,
    pub old_val: f64
}

#[derive(Clone, Copy)]
pub struct CorrelatedSwap
{
    pub index_a: u32,
    pub index_b: u32,
    pub time_step_a: u32,
    pub time_step_b: u32,
}

#[derive(Clone, Copy)]
pub union StepEntry
{
    exchange: ExchangeInfo,
    patient: PatientMove,
    float: f64,
    cor: CorrelatedSwap
}

pub struct MarkovStep
{
    pub which: WhichMove,
    pub list_animals_trans: Vec<StepEntry>,
    pub list_humans_trans: Vec<StepEntry>,
    pub list_animals_rec: Vec<StepEntry>,
    pub list_humans_rec: Vec<StepEntry>,
}

impl Default for MarkovStep
{
    fn default() -> Self {
        Self { 
            which: WhichMove::ByWhom, 
            list_animals_trans: Vec::new(), 
            list_humans_trans: Vec::new(),
            list_animals_rec: Vec::new(),
            list_humans_rec: Vec::new(),
        }
    }
}

impl Serialize for MarkovStep
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer {
        self.which.serialize(serializer)
    }
}

impl<'a> Deserialize<'a> for MarkovStep {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'a> {
        let which: WhichMove = WhichMove::deserialize(deserializer)?;
        Ok(
            Self{
                which,
                list_animals_trans: Vec::new(),
                list_humans_trans: Vec::new(),
                list_animals_rec: Vec::new(),
                list_humans_rec: Vec::new(),
            }
        )
    }
}

impl MarkovStep
{
    pub fn clear(&mut self)
    {
        self.list_animals_trans.clear();
        self.list_humans_trans.clear();
        self.list_animals_rec.clear();
        self.list_humans_rec.clear();
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum MutationHow 
{
    Humans,
    Animals,
    Both,
    DfH,
    DfHSWAP,
    HfD
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Direction
{
    Left,
    Right
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Rotation
{
    Animal(Direction),
    Human(Direction),
    Both(Direction)
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum WhichMove
{
    Rotate(Rotation),
    PatientMove,
    MutationChange,
    ByWhom,
    TransRec,
    MutationSwap(MutationHow),
    AlexMove,
    TimeMove(usize)
}

impl MarkovChain<MarkovStep, ()> for LdModel
{
    fn m_step(&mut self) -> MarkovStep {
        unimplemented!()
    }

    #[inline]
    fn undo_step(&mut self, step: &MarkovStep) {
        self.undo_step_quiet(step)
    }

    fn m_step_acc<Acc, AccFn>(&mut self, _: &mut Acc, _: AccFn) -> MarkovStep
        where AccFn: FnMut(&Self, &MarkovStep, &mut Acc) {
        unimplemented!()
    }

    fn m_steps_acc<Acc, AccFn>
        (
            &mut self,
            _: usize,
            _: &mut Vec<MarkovStep>,
            _: &mut Acc,
            _: AccFn
        )
        where AccFn: FnMut(&Self, &MarkovStep, &mut Acc) {
        unimplemented!()
    }

    fn m_steps_acc_quiet<Acc, AccFn>(
        &mut self, 
        _: usize, 
        _: &mut Acc, 
        _: AccFn
    )
        where AccFn: FnMut(&Self, &MarkovStep, &mut Acc) {
        unimplemented!()
    }

    fn m_steps_quiet(&mut self, _: usize) {
        unimplemented!()
    }

    #[inline]
    fn steps_accepted(&mut self, steps: &[MarkovStep]) {
        match steps[0].which
        {
            WhichMove::TransRec => {
                &mut self.stats.trans_rec
            },
            WhichMove::AlexMove => {
                &mut self.stats.alex_move
            },
            WhichMove::ByWhom => {
                &mut self.stats.by_whom
            },
            WhichMove::TimeMove(_) => {
                &mut self.stats.time_move
            },
            WhichMove::MutationChange => {
                &mut self.stats.mutation_change
            },
            WhichMove::Rotate(r) => {
                match r{
                    Rotation::Animal(_) =>  &mut self.stats.rotation_animal,
                    Rotation::Human(_) =>  &mut self.stats.rotation_humans,
                    Rotation::Both(_) =>  &mut self.stats.rotation_both,
                }
            },
            WhichMove::PatientMove => {
                &mut self.stats.patient_move
            },
            WhichMove::MutationSwap(how) => {
                match how {
                    MutationHow::Animals => &mut self.stats.mutation_animals,
                    MutationHow::Both => &mut self.stats.mutation_both,
                    MutationHow::Humans => &mut self.stats.mutation_humans,
                    MutationHow::DfH =>  &mut self.stats.mutation_dogs_from_humans,
                    MutationHow::HfD =>  &mut self.stats.mutation_humans_from_dogs,
                    MutationHow::DfHSWAP => &mut self.stats.dfh_swap
                }
            }
        }.accept();
    }

    #[inline]
    fn steps_rejected(&mut self, steps: &[MarkovStep]) {
        match steps[0].which
        {
            WhichMove::TransRec => {
                &mut self.stats.trans_rec
            },
            WhichMove::AlexMove => {
                &mut self.stats.alex_move
            },
            WhichMove::ByWhom => {
                &mut self.stats.by_whom
            },
            WhichMove::TimeMove(_) => {
                &mut self.stats.time_move
            },
            WhichMove::MutationChange => {
                &mut self.stats.mutation_change
            },
            WhichMove::Rotate(r) => {
                match r{
                    Rotation::Animal(_) =>  &mut self.stats.rotation_animal,
                    Rotation::Human(_) =>  &mut self.stats.rotation_humans,
                    Rotation::Both(_) =>  &mut self.stats.rotation_both,
                }
            },
            WhichMove::PatientMove => {
                &mut self.stats.patient_move
            },
            WhichMove::MutationSwap(how) => {
                match how {
                    MutationHow::Animals => &mut self.stats.mutation_animals,
                    MutationHow::Both => &mut self.stats.mutation_both,
                    MutationHow::Humans => &mut self.stats.mutation_humans,
                    MutationHow::DfH =>  &mut self.stats.mutation_dogs_from_humans,
                    MutationHow::HfD =>  &mut self.stats.mutation_humans_from_dogs,
                    MutationHow::DfHSWAP => &mut self.stats.dfh_swap
                }
            }
        }.reject();
    }

    fn undo_steps(&mut self, steps: &[MarkovStep], _: &mut Vec<()>) {
        assert!(steps.len() == 1);
        self.undo_step(&steps[0]);
    }

    fn undo_steps_quiet(&mut self, steps: &[MarkovStep]) {
        assert!(steps.len() == 1);
        self.undo_step(&steps[0]);
    }

    fn m_steps(&mut self, count: usize, steps: &mut Vec<MarkovStep>) {
        let step = if steps.len() == 1 
        {
            &mut steps[0]
        } else if steps.is_empty()
        {
            steps.push(MarkovStep::default());
            &mut steps[0]
        } else {
            unreachable!()
        };

        step.clear();

        let uniform = Uniform::new(0.0, 1.0);

        let which = uniform.sample(&mut self.markov_rng);
       
        if which < ROTATE
        {
            let which_direction = uniform.sample(&mut self.markov_rng);
            let direction = if which_direction < 0.5 
            {
                Direction::Left
            } else {
                Direction::Right
            };
            let which_rotation = uniform.sample(&mut self.markov_rng);
            if which_rotation < 1.0/3.0 
            {
                // only animal
                match direction {
                    Direction::Left => {
                        self.offset_dogs.plus_1();
                        step.which = WhichMove::Rotate(Rotation::Animal(Direction::Left));
                    },
                    Direction::Right => {
                        self.offset_dogs.minus_1();
                        step.which = WhichMove::Rotate(Rotation::Animal(Direction::Right));
                    }
                }
            } else if which_rotation < 2.0 / 3.0 
            {
                // only human
                match direction
                {
                    Direction::Left => {
                        self.offset_humans.plus_1();
                        step.which = WhichMove::Rotate(Rotation::Human(Direction::Left));
                    },
                    Direction::Right => {
                        self.offset_humans.minus_1();
                        step.which = WhichMove::Rotate(Rotation::Human(Direction::Right));
                    }
                }
            } else {
                // both
                match direction
                {
                    Direction::Left => {
                        self.offset_dogs.plus_1();
                        self.offset_humans.plus_1();
                        step.which = WhichMove::Rotate(Rotation::Both(Direction::Left));
                    }, 
                    Direction::Right => {
                        self.offset_dogs.minus_1();
                        self.offset_humans.minus_1();
                        step.which = WhichMove::Rotate(Rotation::Both(Direction::Right));
                    }
                }
            }
            
        } else if which < PATIENT_MOVE
        {   
            step.which = WhichMove::PatientMove;
            let which = uniform.sample(&mut self.markov_rng);
            let patient_index = self.markov_rng.gen_range(0..self.initial_patients.len());
            if which < 0.5 {
                let mut old = 0.0;
                // neighbor patient move
                let f = 1.0 / self.max_degree_dogs.get() as f64;
                

                let p0 = self.initial_patients[patient_index];
                let decision = uniform.sample(&mut self.markov_rng);

                for n in self.dual_graph.graph_1().container(p0).edges()
                {
                    let new_prob = old + f;
                    if (old..new_prob).contains(&decision)
                    {
                        if self.initial_patients.contains(n)
                        {
                            break;
                        }
                        step.list_animals_trans.push(StepEntry{patient: PatientMove{
                            index_in_patient_vec: patient_index,
                            old_node: p0
                        }});
                        self.initial_patients[patient_index] = *n;
                        return;
                    }
                    old = new_prob;
                }
            } else {
                let patient_dist = Uniform::new(0, self.dual_graph.graph_1().vertex_count());

                for patient in patient_dist.sample_iter(&mut self.markov_rng)
                {
                    if !self.initial_patients.contains(&patient)
                    {
                        let p0 = &mut self.initial_patients[patient_index];
                        step.list_animals_trans.push(StepEntry{patient: PatientMove{
                            index_in_patient_vec: patient_index,
                            old_node: *p0
                        }});
                        *p0 = patient;
                        return;
                    }
                }
                unreachable!()
            }
        } else if which < P0_RAND
        {
            step.which = WhichMove::TransRec;
            self.offset_dogs.set_time(0);
            self.offset_humans.set_time(0);

            for patient in self.initial_patients.iter()
            {
                let iter = std::iter::once(patient)
                    .chain(
                        self.dual_graph
                            .graph_1()
                            .container(*patient)
                            .edges()
                    );

                for &dog in iter {
                    let rand_index = self.offset_dogs.lookup_index(dog);
                    let old_trans_val = std::mem::replace(
                        &mut self.trans_rand_vec_dogs[rand_index], 
                        uniform.sample(&mut self.markov_rng)
                    );
                    let old_rec_val = std::mem::replace(
                        &mut self.recovery_rand_vec_dogs[rand_index],
                        uniform.sample(&mut self.markov_rng) 
                    );
                    step.list_animals_trans.push(
                        StepEntry{
                            exchange: ExchangeInfo { index: rand_index, old_val: old_trans_val }
                        }
                    );
                    step.list_animals_rec.push(
                        StepEntry{
                            exchange: ExchangeInfo { index: rand_index, old_val: old_rec_val }
                        }
                    );
                }

                let list = self.dual_graph.adj_1()[*patient].slice();
                for &human in list {
                    let rand_index = self.offset_humans.lookup_index(human);

                    let old_trans_val = std::mem::replace(
                        &mut self.trans_rand_vec_humans[rand_index], 
                        uniform.sample(&mut self.markov_rng)
                    );
                    let old_rec_val = std::mem::replace(
                        &mut self.recovery_rand_vec_humans[rand_index], 
                        uniform.sample(&mut self.markov_rng)
                    );
                    step.list_humans_trans.push(
                        StepEntry { exchange: ExchangeInfo { index: rand_index, old_val: old_trans_val } }      
                    );
                    step.list_humans_rec.push(
                        StepEntry { exchange:  ExchangeInfo { index: rand_index, old_val: old_rec_val } }
                       
                    );
                }
            }
        } else if which < MUTATION_MOVE
        {
            let which_which = uniform.sample(&mut self.markov_rng);
            if which_which < 0.5 {
                step.which = WhichMove::MutationChange;
                let humans = self.dual_graph.graph_2().vertex_count();
                let animals = self.dual_graph.graph_1().vertex_count();
                let amount = animals / 6;
                let how_many = self.markov_rng.gen_range(1..amount);
                for _ in 0..how_many{
                    let index = self.markov_rng.gen_range(0..humans+animals);
                    let val: f64 = StandardNormal.sample(&mut self.markov_rng);
                    let mut mutation = val * self.sigma;
                    if index < humans
                    {
                        std::mem::swap(&mut mutation, self.mutation_vec_humans.get_mut(index));
                        step.list_humans_rec.push(
                            StepEntry { exchange: ExchangeInfo { index, old_val: mutation } }
                            
                        );
                    } else {
                        let index = index - humans;
                        std::mem::swap(&mut mutation, self.mutation_vec_dogs.get_mut(index));
                        step.list_animals_rec.push(
                            StepEntry { exchange: ExchangeInfo { index, old_val: mutation } }
                            
                        );
                    }
                }
                
            } else {
                let decision = uniform.sample(&mut self.markov_rng);
                if decision < 1.0 / 6.0  {
                    let dog_amount: usize = 1_usize.max(self.mutation_vec_dogs.mut_vec.len() / 10);
                    self.mutation_vec_dogs.mutation_swap(&mut step.list_animals_trans, &mut self.markov_rng, dog_amount);
                    
                    step.which = WhichMove::MutationSwap(MutationHow::Animals);
                } else if decision < 2.0 / 6.0  {
                    let human_amount = (self.mutation_vec_humans.mut_vec.len() / 100).max(5);
                    self.mutation_vec_humans.mutation_swap(&mut step.list_humans_trans, &mut self.markov_rng, human_amount);
                    
                    step.which = WhichMove::MutationSwap(MutationHow::Humans);
                } else if decision < 3.0 / 6.0{
                    let human_amount = (self.mutation_vec_humans.mut_vec.len() / 100).max(5);
                    let dog_amount: usize = 1_usize.max(self.mutation_vec_dogs.mut_vec.len() / 10);
                    self.mutation_vec_humans.mutation_swap(&mut step.list_humans_trans, &mut self.markov_rng, human_amount);
                    self.mutation_vec_dogs.mutation_swap(&mut step.list_animals_trans, &mut self.markov_rng, dog_amount);

                    step.which = WhichMove::MutationSwap(MutationHow::Both);
                } else if decision < 4.0 / 6.0 
                {
                    step.which = WhichMove::MutationSwap(MutationHow::HfD);

                    step.list_humans_trans
                        .extend(
                            self.mutation_humans_from_dogs
                                .mut_vec
                                .iter()
                                .map(|&val| StepEntry{float: val})
                        );
                    

                    self.mutation_humans_from_dogs
                        .mut_vec
                        .iter_mut()
                        .for_each(
                            |val|
                            {
                                *val = StandardNormal.sample(&mut self.markov_rng);
                                *val *= self.sigma;
                            }
                        )
                    
                } else if decision < 5.0 / 6.0{
                    step.which = WhichMove::MutationSwap(MutationHow::DfH);

                    for _ in 0..10 {
                        let index = self.markov_rng.gen_range(0..self.mutation_dogs_from_humans.mut_vec.len());
                        let val: f64 = StandardNormal.sample(&mut self.markov_rng);
                        let mutation = val * self.sigma;
    
                        let old_val = self.mutation_dogs_from_humans.get(index);
                        *self.mutation_dogs_from_humans.get_mut(index) = mutation;
                        step.list_humans_trans.push(
                            StepEntry { exchange: ExchangeInfo{old_val, index} }
                        );
                    }

                } else {
                    step.which = WhichMove::MutationSwap(MutationHow::DfHSWAP);
                    let human_amount = (self.mutation_humans_from_dogs.mut_vec.len() / 5).max(2);
                    self.mutation_humans_from_dogs.mutation_swap(&mut step.list_humans_trans, &mut self.markov_rng, human_amount);
                }
            }
            
        } else if which < BY_WHOM
        {
            step.which = WhichMove::ByWhom;
            let humans = self.dual_graph.graph_2().vertex_count();
            let animals = self.dual_graph.graph_1().vertex_count();
            let index = self.markov_rng.gen_range(0..humans+animals);
            let mut by_whom = uniform.sample(&mut self.markov_rng);
            if index < humans
            {
                std::mem::swap(&mut by_whom, &mut self.infected_by_whom_humans[index]);
                step.list_humans_rec.push(
                    StepEntry { 
                        exchange: ExchangeInfo{
                            index,
                            old_val: by_whom
                        } 
                    }
                    
                );
            } else {
                let index = index - humans;
                std::mem::swap(&mut by_whom, &mut self.infected_by_whom_dogs[index]);
                step.list_animals_rec.push(
                    StepEntry { exchange: ExchangeInfo { index, old_val: by_whom } }
                );
            }
        } else if which < ALEX_MOVE
        {
            step.which = WhichMove::AlexMove;
            let time_uniform = Uniform::new(0, self.max_time_steps.get());
            let index_uniform = Uniform::new(0, self.dual_graph.graph_1().vertex_count());
            let num = self.markov_rng.gen_range(1..3);
            for _ in 0..num {
                let (index_dog_1, index_human_1) = loop{
                    let i = index_uniform.sample(&mut self.markov_rng);
                    let slice = self.dual_graph.adj_1()[i].slice();
                    if !slice.is_empty() {
                        break (i, slice[0]);
                    }
                };
                let (index_dog_2, index_human_2) = loop{
                    let i = index_uniform.sample(&mut self.markov_rng);
                    let slice = self.dual_graph.adj_1()[i].slice();
                    if !slice.is_empty() {
                        break (i, slice[0]);
                    }
                };
                if index_dog_2 == index_dog_1{
                    continue;
                }
                let time_step_1 = time_uniform.sample(&mut self.markov_rng);
                let time_step_2 = time_uniform.sample(&mut self.markov_rng);

                self.mutation_vec_dogs.mut_vec.swap(index_dog_1, index_dog_2);
                self.mutation_humans_from_dogs.mut_vec.swap(index_dog_1, index_dog_2);

                self.offset_dogs.set_time(time_step_1);
                let a = self.offset_dogs.lookup_index(index_dog_1);
                self.offset_dogs.set_time(time_step_2);
                let b = self.offset_dogs.lookup_index(index_dog_2);
                self.trans_rand_vec_dogs.swap(a, b);

                let mut time_humans_1 = time_step_1 + 1;
                if time_humans_1 == self.max_time_steps.get()
                {
                    time_humans_1 = 0;
                }
                self.offset_humans.set_time(time_humans_1);
                let a = self.offset_humans.lookup_index(index_human_1);
                let mut time_humans_2 = time_step_2 + 1;
                if time_humans_2 == self.max_time_steps.get()
                {
                    time_humans_2 = 0;
                }
                self.offset_humans.set_time(time_humans_2);
                let b = self.offset_humans.lookup_index(index_human_2);
                self.trans_rand_vec_humans.swap(a, b);

                step.list_animals_trans.push(
                    StepEntry{cor: CorrelatedSwap{index_a: index_dog_1 as u32, index_b: index_dog_2 as u32, time_step_a: time_step_1 as u32, time_step_b: time_step_2 as u32}}
                )
            }
        } else if which < TIME_MOVE
        {
            //println!("TIME MOVE");
            let min = self.max_time_steps.get().min(30);
            let time = self.markov_rng.gen_range(0..min);
            self.offset_humans.set_time(time);
            self.offset_dogs.set_time(time);

            step.which = WhichMove::TimeMove(time);

            let slice = self.offset_humans.get_slice_mut(&mut self.trans_rand_vec_humans);

            step.list_humans_trans
                .extend(
                    slice.iter()
                        .map(
                            |entry| StepEntry{float: *entry}
                        )
                );

            slice.iter_mut()
                .zip((&uniform).sample_iter(&mut self.markov_rng))
                .for_each(
                    |(s, o)| *s = o
                );
            
            let slice = self.offset_humans.get_slice_mut(&mut self.recovery_rand_vec_humans);
            step.list_humans_rec
                .extend(
                    slice.iter()
                        .map(
                            |entry| StepEntry{float: *entry}
                        )
                );

            slice.iter_mut()
                .zip((&uniform).sample_iter(&mut self.markov_rng))
                .for_each(
                    |(s, o)| *s = o
                );

            let slice = self.offset_dogs.get_slice_mut(&mut self.trans_rand_vec_dogs);

            step.list_animals_trans
                .extend(
                    slice.iter()
                        .map(
                            |entry| StepEntry{float: *entry}
                        )
                );

            slice.iter_mut()
                .zip((&uniform).sample_iter(&mut self.markov_rng))
                .for_each(
                    |(s, o)| *s = o
                );

            let slice = self.offset_dogs.get_slice_mut(&mut self.recovery_rand_vec_dogs);

            step.list_animals_rec
                .extend(
                    slice.iter()
                        .map(
                            |entry| StepEntry{float: *entry}
                        )
                );

            slice.iter_mut()
                .zip((&uniform).sample_iter(&mut self.markov_rng))
                .for_each(
                    |(s, o)| *s = o
                );


        }
        else {
            step.which = WhichMove::TransRec;

            let total_dogs = self.dual_graph.graph_1().vertex_count();
            let total_humans = self.dual_graph.graph_2().vertex_count();

            let dog_to_human = total_dogs as f64 / total_humans as f64;

            let dog_changes = Binomial::new(count as u64, dog_to_human).unwrap().sample(&mut self.markov_rng);

            {
                // dogs
                let index_uniform = Uniform::new(0, self.trans_rand_vec_dogs.len());
                let amount = Binomial::new(dog_changes, 0.5).unwrap().sample(&mut self.markov_rng);

                step.list_animals_trans.extend(
                    (0..amount)
                        .map(
                            |_|
                            {
                                let index = index_uniform.sample(&mut self.markov_rng);
                                
                                let old_transmission = std::mem::replace(
                                    &mut self.trans_rand_vec_dogs[index], 
                                    uniform.sample(&mut self.markov_rng)
                                );

                                StepEntry{
                                    exchange: ExchangeInfo { index, old_val: old_transmission }
                                }
                            }
                        )
                );
                step.list_animals_rec.extend(
                    (0..dog_changes-amount)
                        .map(
                            |_|
                            {
                                let index = index_uniform.sample(&mut self.markov_rng);
                                
                                let old_rec = std::mem::replace(
                                    &mut self.recovery_rand_vec_dogs[index], 
                                    uniform.sample(&mut self.markov_rng)
                                );
                                StepEntry{exchange: ExchangeInfo { index, old_val: old_rec }}   
                            }
                        )
                );

            }
            {
                // humans
                let human_changes = count as u64 - dog_changes;
                let index_uniform = Uniform::new(0, self.trans_rand_vec_humans.len());
                let amount = Binomial::new(human_changes, 0.5).unwrap().sample(&mut self.markov_rng);

                step.list_humans_trans.extend(
                    (0..amount)
                        .map(
                            |_|
                            {
                                let index = index_uniform.sample(&mut self.markov_rng);

                                let old_trans = std::mem::replace(
                                    &mut self.trans_rand_vec_humans[index], 
                                    uniform.sample(&mut self.markov_rng)
                                );
                                StepEntry{
                                    exchange: ExchangeInfo{index, old_val: old_trans}
                                }
                               
                            }
                        )
                );

                step.list_humans_rec.extend(
                    (0..human_changes-amount)
                        .map(
                            |_|
                            {
                                let index = index_uniform.sample(&mut self.markov_rng);

                                let old_trans = std::mem::replace(
                                    &mut self.recovery_rand_vec_humans[index], 
                                    uniform.sample(&mut self.markov_rng)
                                );

                                StepEntry{exchange: ExchangeInfo{index, old_val: old_trans}}
                            }
                        )   
                );
            }
        }

    }

    fn undo_step_quiet(&mut self, step: &MarkovStep) {
        match step.which
        {
            WhichMove::Rotate(r) => {
                match r {
                    Rotation::Animal(direction) => {
                        match direction
                        {
                            Direction::Left => self.offset_dogs.minus_1(),
                            Direction::Right => self.offset_dogs.plus_1()
                        }
                    },
                    Rotation::Both(direction) => {
                        match direction
                        {
                            Direction::Left => {
                                self.offset_dogs.minus_1();
                                self.offset_humans.minus_1()
                            },
                            Direction::Right => {
                                self.offset_dogs.plus_1();
                                self.offset_humans.plus_1();
                            }
                        }
                    },
                    Rotation::Human(direction) => {
                        match direction 
                        {
                            Direction::Left => self.offset_humans.minus_1(),
                            Direction::Right => self.offset_humans.plus_1()
                        }
                    }
                }
            },
            WhichMove::PatientMove => {
                step.list_animals_trans
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            let patient_move = unsafe{entry.patient};
                            self.initial_patients[patient_move.index_in_patient_vec] = patient_move.old_node;
                        }
                    );
            },
            WhichMove::MutationChange => {
                step.list_animals_rec
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            let entry = unsafe{entry.exchange};
                            *self.mutation_vec_dogs.get_mut(entry.index) = entry.old_val;
                        }
                    );
                step.list_humans_rec
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            let entry = unsafe{entry.exchange};
                            *self.mutation_vec_humans.get_mut(entry.index) = entry.old_val;
                        }
                    );
            },
            WhichMove::ByWhom => {
                step.list_animals_rec
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            let entry = unsafe{entry.exchange};
                            self.infected_by_whom_dogs[entry.index] = entry.old_val;
                        }
                    );
                step.list_humans_rec
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            let entry = unsafe{entry.exchange};
                            self.infected_by_whom_humans[entry.index] = entry.old_val;
                        }
                    );
            },
            WhichMove::TransRec => {
                step.list_animals_trans
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            let info = unsafe{entry.exchange};
                            self.trans_rand_vec_dogs[info.index] = info.old_val;
                        }
                    );
                step
                    .list_animals_rec
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            let entry = unsafe{entry.exchange};
                            self.recovery_rand_vec_dogs[entry.index] = entry.old_val
                        }
                    );
                step.list_humans_trans
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            let entry = unsafe{entry.exchange};
                            self.trans_rand_vec_humans[entry.index] = entry.old_val
                        }
                    );
                step.list_humans_rec
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            let entry = unsafe{entry.exchange};
                            self.recovery_rand_vec_humans[entry.index] = entry.old_val;
                        }
                    );
            },
            WhichMove::MutationSwap(how) => {

                match how{
                    MutationHow::Humans => self.mutation_vec_humans.unshuffle(&step.list_humans_trans),
                    MutationHow::Animals => self.mutation_vec_dogs.unshuffle(&step.list_animals_trans),
                    MutationHow::Both => {
                        self.mutation_vec_humans.unshuffle(&step.list_humans_trans);
                        self.mutation_vec_dogs.unshuffle(&step.list_animals_trans)
                    },
                    MutationHow::HfD => {
                        
                    self.mutation_humans_from_dogs
                        .mut_vec
                        .iter_mut()
                        .zip(step.list_humans_trans.iter())
                        .for_each(|(mutation, list)| *mutation = unsafe{list.float});
                    },
                    MutationHow::DfH => {
                        step.list_humans_trans.iter().rev()
                            .for_each(
                                |ex|
                                {
                                    let exchange = unsafe{ex.exchange};
                                    self.mutation_dogs_from_humans.mut_vec[exchange.index] = exchange.old_val;
                                }
                            )
                    },
                    MutationHow::DfHSWAP => {
                        self.mutation_humans_from_dogs.unshuffle(&step.list_humans_trans)
                    }
                }
            },
            WhichMove::AlexMove =>
            {
                for i in step.list_animals_trans.iter().rev()
                {
                    let entry = unsafe{i.cor};
                    let index_dogs_1 = entry.index_a as usize;
                    let index_dogs_2 = entry.index_b as usize;

                    let list_1 = self.dual_graph.adj_1()[index_dogs_1].slice();

                    let index_human_1 = if list_1.is_empty()
                    {
                        unreachable!()
                    } else {
                        list_1[0]
                    };

                    let list_2 = self.dual_graph.adj_1()[index_dogs_2].slice();

                    let index_human_2 = if list_2.is_empty()
                    {
                        unreachable!()
                    } else {
                        list_2[0]
                    };
                    self.mutation_vec_dogs.mut_vec.swap(index_dogs_1, index_dogs_2);
                    self.mutation_humans_from_dogs.mut_vec.swap(index_dogs_1, index_dogs_2);

                    self.offset_dogs.set_time(entry.time_step_a as usize);
                    let a = self.offset_dogs.lookup_index(index_dogs_1);
                    self.offset_dogs.set_time(entry.time_step_b as usize);
                    let b = self.offset_dogs.lookup_index(index_dogs_2);
                    self.trans_rand_vec_dogs.swap(a, b);

                    let mut time_humans = entry.time_step_a as usize + 1;
                    if time_humans == self.max_time_steps.get()
                    {
                        time_humans = 0;
                    }
                    self.offset_humans.set_time(time_humans);
                    let a = self.offset_humans.lookup_index(index_human_1);
                    let mut time_humans = entry.time_step_b as usize + 1;
                    if time_humans == self.max_time_steps.get()
                    {
                        time_humans = 0;
                    }
                    self.offset_humans.set_time(time_humans);
                    let b = self.offset_humans.lookup_index(index_human_2);
                    self.trans_rand_vec_humans.swap(a, b);
                }
            },
            WhichMove::TimeMove(time) => {
                self.offset_humans.set_time(time);
                self.offset_dogs.set_time(time);
    
                let slice = self.offset_humans.get_slice_mut(&mut self.trans_rand_vec_humans);
    
                step.list_humans_trans
                    .iter()
                    .zip(
                        slice.iter_mut()
                    ).for_each(
                        |(o, s)| *s = unsafe{o.float}
                    );

                let slice = self.offset_humans.get_slice_mut(&mut self.recovery_rand_vec_humans);

                step.list_humans_rec
                    .iter()
                    .zip(
                        slice.iter_mut()
                    ).for_each(
                        |(o, s)| *s = unsafe{o.float}
                    );
    
                let slice = self.offset_dogs.get_slice_mut(&mut self.trans_rand_vec_dogs);

                step.list_animals_trans
                    .iter()
                    .zip(
                        slice.iter_mut()
                    ).for_each(
                        |(o, s)| *s = unsafe{o.float}
                    );

                let slice = self.offset_dogs.get_slice_mut(&mut self.recovery_rand_vec_dogs);

                step.list_animals_rec
                    .iter()
                    .zip(
                        slice.iter_mut()
                    ).for_each(
                        |(o, s)| *s = unsafe{o.float}
                    );
            }
        }
    }
}

impl LdModel
{
    pub fn re_randomize(&mut self, mut rng: Pcg64)
    {
        

        let uniform_exclusive = Uniform::new(0.0, 1.0);
        let mut uniform_iter = uniform_exclusive.sample_iter(&mut rng);
        let mut randomizer = |v: &mut [f64]|
        {
            v.iter_mut().zip(&mut uniform_iter).for_each
                (
                    |(s, rand)| *s = rand
                );
        };

        randomizer(&mut self.infected_by_whom_dogs);
        randomizer(&mut self.infected_by_whom_humans);
        randomizer(&mut self.trans_rand_vec_dogs);
        randomizer(&mut self.trans_rand_vec_humans);
        randomizer(&mut self.recovery_rand_vec_dogs);
        randomizer(&mut self.recovery_rand_vec_humans);

        let mut initial_patients: Vec<_> = (0..self.dual_graph.graph_1().vertex_count())
            .collect();

        initial_patients.shuffle(&mut rng);

        self.initial_patients
            .iter_mut()
            .zip(initial_patients)
            .for_each(|(s, o)| *s = o);

        
        let mut s_normal = |v: &mut [f64]|
        {
            v.iter_mut()
                .zip(rand_distr::Distribution::<f64>::sample_iter(StandardNormal, &mut rng))
                .for_each(|(s, val)| *s = val * self.sigma);
        };

        s_normal(&mut self.mutation_vec_dogs.mut_vec);
        s_normal(&mut self.mutation_vec_humans.mut_vec);

        self.markov_rng = rng;
        
    }

    pub fn new(mut base: BaseModel, markov_seed: u64, max_sir_steps: NonZeroUsize) -> Self
    {
        let mut markov_rng = Pcg64::seed_from_u64(markov_seed);

        let uniform_exclusive = Uniform::new(0.0, 1.0);
        let mut uniform_iter = uniform_exclusive.sample_iter(&mut markov_rng);
        let mut collector = |n|
        {
            let mut v = Vec::with_capacity(n);
            v.extend(
                (&mut uniform_iter)
                .take(n)
            );
            v
        };

        let n_dogs = base.dual_graph.graph_1().vertex_count();
        let infected_by_whom_dogs: Vec<f64> = collector(n_dogs);

        let n_humans = base.dual_graph.graph_2().vertex_count();
        let infected_by_whom_humans = collector(n_humans);


        let trans_rand_vec_humans = collector(n_humans * max_sir_steps.get());
        let recovery_rand_vec_humans = collector(trans_rand_vec_humans.len());

        let trans_rand_vec_dogs = collector(n_dogs * max_sir_steps.get());
        let recovery_rand_vec_dogs = collector(trans_rand_vec_dogs.len());

        let mut s_normal = |n|
        {
            let mut v: Vec<f64> = Vec::with_capacity(n);
            v.extend(
                rand_distr::Distribution::<f64>::sample_iter(StandardNormal, &mut markov_rng)
                    .map(|val| val * base.sigma)
                    .take(n)
            );
            v
        };

        let mutation_vec_dogs = s_normal(n_dogs);
        let mutation_vec_dogs = Mutation { mut_vec: mutation_vec_dogs};
        let mutation_vec_humans = s_normal(n_humans);
        let mutation_vec_humans = Mutation { mut_vec: mutation_vec_humans};

        let humans_from_dogs = s_normal(n_dogs);
        let dogs_from_humans = s_normal(n_dogs);
        let humans_from_dogs = Mutation{mut_vec: humans_from_dogs};
        let dogs_from_humans = Mutation{mut_vec: dogs_from_humans};

        let max_degree_dogs = base
            .dual_graph
            .graph_1()
            .degree_iter()
            .max()
            .unwrap();

        let mut initial_patients = [0; PATIENTS_USIZE];

        base.possible_patients.shuffle(&mut markov_rng);

        initial_patients.as_mut_slice()
            .iter_mut()
            .zip(
                base.possible_patients
            ).for_each(
                |(new, old)| *new = old
            );

        Self { 
            initial_patients,
            dual_graph: base.dual_graph, 
            reset_gamma: base.reset_gamma, 
            markov_rng, 
            recover_prob: base.recovery_prob, 
            max_lambda: base.max_lambda, 
            sigma: base.sigma, 
            initial_gt: base.initial_gt, 
            offset_humans: Offset::new(max_sir_steps.get(), n_humans), 
            offset_dogs: Offset::new(max_sir_steps.get(), n_dogs), 
            infected_by_whom_dogs, 
            infected_by_whom_humans, 
            total_sim_counter: 0, 
            unfinished_sim_counter: 0, 
            max_degree_dogs: NonZeroUsize::new(max_degree_dogs).unwrap(), 
            max_time_steps: max_sir_steps, 
            new_infections_list_dogs: Vec::new(), 
            new_infections_list_humans: Vec::new(), 
            infected_list_dogs: Vec::new(),
            infected_list_humans: Vec::new(),
            trans_rand_vec_humans, 
            trans_rand_vec_dogs, 
            recovery_rand_vec_humans, 
            recovery_rand_vec_dogs,
            mutation_vec_dogs,
            mutation_vec_humans,
            last_extinction: usize::MAX,
            stats: MarkovStats::default(),
            mutation_humans_from_dogs: humans_from_dogs,
            mutation_dogs_from_humans: dogs_from_humans,
        }

    }

    pub fn reset_and_infect(&mut self)
    {
        self.dual_graph
            .graph_1_mut()
            .contained_iter_mut()
            .for_each(
                |state| state.set_s_ld()
            );

        self.dual_graph
            .graph_2_mut()
            .contained_iter_mut()
            .for_each(|state| state.set_s_ld());

        let trans_humans = self.initial_gt.trans_human;
        let trans_animal = self.initial_gt.trans_animal;

        for &i in self.initial_patients.iter()
        {
            self.dual_graph
                .graph_1_mut()
                .at_mut(i)
                .progress_to_i_with_gt(self.initial_gt);

            for s_node in self.dual_graph.graph_1_mut().contained_iter_neighbors_mut(i)
            {
                if s_node.is_susceptible()
                {
                    s_node.add_to_s(trans_animal);
                }
            }
            for s_node in self.dual_graph.graph_1_contained_iter_neighbors_in_other_graph_mut(i)
            {
                s_node.add_to_s(trans_humans);
            }
        }
        self.new_infections_list_humans.clear();
        self.new_infections_list_dogs.clear();
        self.infected_list_humans.clear();
        self.infected_list_dogs.clear();
        self.infected_list_dogs.extend_from_slice(&self.initial_patients);
    }

    fn iterate_once(&mut self)
    {
        // decide which nodes will become infected
        for (index, sir_fun) in self.dual_graph.graph_1_mut().contained_iter_mut().enumerate()
        {
            if sir_fun.is_susceptible()
            {
                let state = sir_fun.get_sus_state();
                if self.trans_rand_vec_dogs[self.offset_dogs.lookup_index(index)] >= state.product {
                    self.new_infections_list_dogs.push(index);
                }
            }
        }
        for (index, sir_fun) in self.dual_graph.graph_2_mut().contained_iter_mut().enumerate()
        {
            if sir_fun.is_susceptible()
            {
                let state = sir_fun.get_sus_state();
                if self.trans_rand_vec_humans[self.offset_humans.lookup_index(index)] >= state.product {
                    self.new_infections_list_humans.push(index);
                }
            }
        }

        // set new transmission values etc
        for &index in self.new_infections_list_dogs.iter()
        {
            let container = self.dual_graph.graph_1().container(index);
            if container.contained().get_infectiouse_neighbor_count() == 1 {
                // workaround. loop can be removed with rust 1.65.0 as the respective scope feature will become stable
                'scope: loop {
                    let neighbors = container.edges();
                    for &idx in neighbors
                    {
                        let node = self.dual_graph.graph_1().at(idx);
                        if node.is_infected()
                        {
                            let gamma = node.get_gamma();
                            let new_gamma = gamma + self.mutation_vec_dogs.get(index);
                            let node_to_transition = self.dual_graph.graph_1_mut().at_mut(index);
                            node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                            break 'scope;
                        }
                    }
                    let slice = self.dual_graph.adj_1()[index].slice();
                    for &idx in slice
                    {
                        let node = self.dual_graph.graph_2().at(idx);
                        if node.is_infected(){
                            let gamma = node.get_gamma();
                            let new_gamma = gamma + self.mutation_dogs_from_humans.get(index);
                            let node_to_transition = self.dual_graph.graph_1_mut().at_mut(index);
                            node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                            break 'scope;
                        }
                    }
                    unreachable!()
                }
            } else {
                let mut sum = 0.0;
                for node in self.dual_graph.graph_1_contained_iter(index)
                {
                    if node.is_infected()
                    {
                        sum += node.get_gamma_trans().trans_animal;
                    }
                }
            
                let which = self.infected_by_whom_dogs[index] * sum;
                sum = 0.0;
                
                'outer: loop{
                    let iter = self.dual_graph.graph_1_mut().contained_iter_neighbors_mut(index);
                    
                    for node in iter
                    {
                        if node.is_infected()
                        {
                            sum += node.get_gamma_trans().trans_animal;
                            if which <= sum {
                                let gamma = node.get_gamma();
                                let new_gamma = gamma + self.mutation_vec_dogs.get(index);
                                
                                let node_to_transition = self.dual_graph.graph_1_mut().at_mut(index);
                                node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                                break 'outer;
                            }
                        }
                    }
                    let iter = self.dual_graph.adj_1()[index].slice();
                    for &idx in iter
                    {
                        let node = self.dual_graph.graph_2().at(idx);
                        if node.is_infected(){
                            sum += node.get_gamma_trans().trans_animal;
                            if which <= sum {
                                let gamma = node.get_gamma();
                                let new_gamma = gamma + self.mutation_dogs_from_humans.get(index);
                                let node_to_transition = self.dual_graph.graph_1_mut().at_mut(index);
                                node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                                break 'outer;
                            }
                            
                        }
                    }
                    unreachable!()
                }
                
            }
        }

        for &index in self.new_infections_list_humans.iter()
        {
            let container = self.dual_graph.graph_2().container(index);
            if container.contained().get_infectiouse_neighbor_count() == 1 {
                // workaround. loop can be removed with rust 1.65.0 as the respective scope feature will become stable
                'scope: loop {
                    let neighbors = container.edges();
                    for &idx in neighbors
                    {
                        let node = self.dual_graph.graph_2().at(idx);
                        if node.is_infected()
                        {
                            let gamma = node.get_gamma();
                            let new_gamma = gamma + self.mutation_vec_humans.get(index);
                            let node_to_transition = self.dual_graph.graph_2_mut().at_mut(index);
                            node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                            break 'scope;
                        }
                    }
                    let slice = self.dual_graph.adj_2()[index].slice();
                    for &idx in slice
                    {
                        let node = self.dual_graph.graph_1().at(idx);
                        if node.is_infected(){
                            let gamma = node.get_gamma();
                            let lambda = node.get_gamma_trans().trans_human;
                            let new_gamma = gamma + self.mutation_humans_from_dogs.get(idx);
                            let node_to_transition = self.dual_graph.graph_2_mut().at_mut(index);
                            node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                            
                            let lambda2 = node_to_transition.get_gamma_trans().trans_human;
                            println!("infecting human {index} - {lambda} {lambda2}");
                            break 'scope;
                        }
                    }
                    unreachable!()
                }
                
            } else {
                let mut sum = 0.0;
                for node in self.dual_graph.graph_2_contained_iter(index)
                {
                    if node.is_infected()
                    {
                        sum += node.get_gamma_trans().trans_human;
                    }
                }
            
                let which = self.infected_by_whom_humans[index] * sum;
                let mut sum = 0.0;
                
                'outer: loop{
                    let mut iter = self.dual_graph.graph_2().contained_iter_neighbors(index);
                    for node in &mut iter
                    {
                        if node.is_infected()
                        {
                            sum += node.get_gamma_trans().trans_human;
                            if which <= sum {
                                let gamma = node.get_gamma();
                                let new_gamma = gamma + self.mutation_vec_humans.get(index);
                                let node = self.dual_graph.graph_2_mut().at_mut(index);
                                node.set_gt_and_transition(new_gamma, self.max_lambda);
                                break 'outer;
                            }
                        }
                    }

                    let iter = self.dual_graph.adj_2()[index].slice();
                    for &idx in iter
                    {
                        let node = self.dual_graph.graph_1().at(idx);
                        if node.is_infected()
                        {
                            sum += node.get_gamma_trans().trans_human;
                            if which <= sum {
                                
                                //println!("other {sum} vs {old_sum}");
                                let gamma = node.get_gamma();
                                let new_gamma = gamma + self.mutation_humans_from_dogs.get(idx);
                                let lambda_old = node.get_gamma_trans().trans_human;
                                let node = self.dual_graph.graph_2_mut().at_mut(index);
                                node.set_gt_and_transition(new_gamma, self.max_lambda);
                                let lambda_new = node.get_gamma_trans().trans_human;
                                println!("infecting another human {index} - {lambda_old} {lambda_new}");
                                break 'outer;
                            }
                        }
                    }
                    unreachable!()
                }
            }
        }

        // remove old infected nodes at random
        for id in (0..self.infected_list_dogs.len()).rev()
        {
            let index = self.infected_list_dogs[id];
            if self.recovery_rand_vec_dogs[self.offset_dogs.lookup_index(index)] < self.recover_prob
            {
                self.infected_list_dogs.swap_remove(id);
                let contained = self.dual_graph.graph_1_mut().at_mut(index);
                let gt = contained.get_gamma_trans();
                contained.progress_to_r();

                // dog neighbors:
                for neighbor in self.dual_graph.graph_1_mut().contained_iter_neighbors_mut(index)
                {
                    if neighbor.is_susceptible()
                    {
                        neighbor.subtract_from_s(gt.trans_animal)
                    }
                }

                // human neighbors
                for neighbor in self.dual_graph.graph_1_contained_iter_neighbors_in_other_graph_mut(index)
                {
                    if neighbor.is_susceptible()
                    {
                        neighbor.subtract_from_s(gt.trans_human);
                    }
                }
            }
        }


        // remove old infected nodes at random
        for id in (0..self.infected_list_humans.len()).rev()
        {
            let index = self.infected_list_humans[id];
            if self.recovery_rand_vec_humans[self.offset_humans.lookup_index(index)] < self.recover_prob
            {
                self.infected_list_humans.swap_remove(id);
                let contained = self.dual_graph.graph_2_mut().at_mut(index);
                let gt = contained.get_gamma_trans();
                contained.progress_to_r();

                // human neighbors:
                for neighbor in self.dual_graph.graph_2_mut().contained_iter_neighbors_mut(index)
                {
                    if neighbor.is_susceptible()
                    {
                        neighbor.subtract_from_s(gt.trans_human)
                    }
                }

                // dog neighbors
                for neighbor in self.dual_graph.graph_2_contained_iter_neighbors_in_other_graph_mut(index)
                {
                    if neighbor.is_susceptible()
                    {
                        neighbor.subtract_from_s(gt.trans_animal);
                    }
                }

            }
        }

        // transition to I dogs
        for &index in self.new_infections_list_dogs.iter()
        {
            let node = self.dual_graph.graph_1_mut().at_mut(index);
            node.transition_to_i();
            let gt = node.get_gamma_trans();

            for dog in self.dual_graph.graph_1_mut().contained_iter_neighbors_mut(index)
            {
                if dog.is_susceptible()
                {
                    dog.add_to_s(gt.trans_animal);
                }
            }

            for human in self.dual_graph.graph_1_contained_iter_neighbors_in_other_graph_mut(index)
            {
                if human.is_susceptible()
                {
                    human.add_to_s(gt.trans_human);
                }
            }
        }

        // transition to I humans
        for &index in self.new_infections_list_humans.iter()
        {
            let node = self.dual_graph.graph_2_mut().at_mut(index);
            node.transition_to_i();
            let gt = node.get_gamma_trans();

            for human in self.dual_graph.graph_2_mut().contained_iter_neighbors_mut(index)
            {
                if human.is_susceptible()
                {
                    human.add_to_s(gt.trans_human);
                }
            }

            for dog in self.dual_graph.graph_2_contained_iter_neighbors_in_other_graph_mut(index)
            {
                if dog.is_susceptible()
                {
                    dog.add_to_s(gt.trans_animal);
                }
            }
        }

        self.infected_list_dogs.append(&mut self.new_infections_list_dogs);
        self.infected_list_humans.append(&mut self.new_infections_list_humans);
        
    }


    pub fn entropic_writer(
        &mut self, 
        infection_helper: &mut LayerHelper, 
        writer_humans: &mut SirWriter,
        writer_animals: &mut SirWriter,
        last_energy: usize
    )
    {
        self.reset_and_infect();
        infection_helper.reset(&self.initial_patients);
        
        //let _ = writer_humans.write_energy(last_energy, self.last_extinction);
        //let _ = writer_animals.write_energy(last_energy, self.last_extinction);
        //let _ = writer_humans.write_current(self.dual_graph.graph_2());
        //let _ = writer_animals.write_current(self.dual_graph.graph_1());
        
        for i in 0..self.max_time_steps.get()
        {
            self.offset_set_time(i);
            self.iterate_once_writing(infection_helper);
            //let _ = writer_humans.write_current(self.dual_graph.graph_2());
            //let _ = writer_animals.write_current(self.dual_graph.graph_1());
            if self.infections_empty()
            {
                break;
            }
        }

        assert_eq!(
            last_energy,
            self.dual_graph
            .graph_2()
            .contained_iter()
            .filter(|node| !node.is_susceptible())
            .count()
        );
        //let _ = writer_humans.write_line();
        //let _ = writer_animals.write_line();
    }

    fn iterate_once_writing(
        &mut self,
        infection_helper: &mut LayerHelper
    )
    {

        #[inline]
        fn add_1(val: Option<NonZeroUsize>) -> Option<NonZeroUsize>
        {
            val.unwrap().checked_add(1)
        }

        //#[inline]
        //fn add_1(val: Option<NonZeroUsize>) -> Option<NonZeroUsize>
        //{
        //    unsafe{
        //        Some(
        //            NonZeroUsize::new_unchecked(
        //                val.unwrap_unchecked().get() + 1
        //            )
        //        )
        //    }
        //}

        // decide which nodes will become infected
        for (index, sir_fun) in self.dual_graph.graph_1_mut().contained_iter_mut().enumerate()
        {
            if sir_fun.is_susceptible()
            {
                let state = sir_fun.get_sus_state();
                if self.trans_rand_vec_dogs[self.offset_dogs.lookup_index(index)] >= state.product {
                    self.new_infections_list_dogs.push(index);
                }
            }
        }
        for (index, sir_fun) in self.dual_graph.graph_2_mut().contained_iter_mut().enumerate()
        {
            if sir_fun.is_susceptible()
            {
                let state = sir_fun.get_sus_state();
                if self.trans_rand_vec_humans[self.offset_humans.lookup_index(index)] >= state.product {
                    self.new_infections_list_humans.push(index);
                }
            }
        }

        // set new transmission values etc
        for &index in self.new_infections_list_dogs.iter()
        {
            let container = self.dual_graph.graph_1().container(index);
            if container.contained().get_infectiouse_neighbor_count() == 1 {
                // workaround. loop can be removed with rust 1.65.0 as the respective scope feature will become stable
                'scope: loop {
                    let neighbors = container.edges();
                    for &idx in neighbors
                    {
                        let node = self.dual_graph.graph_1().at(idx);
                        if node.is_infected()
                        {
                            infection_helper.animals_infected_by[index] = WhichGraph::Graph1(idx);
                            infection_helper.layer_dogs[index] = add_1(infection_helper.layer_dogs[idx]);
                            let gamma = node.get_gamma();
                            let new_gamma = gamma + self.mutation_vec_dogs.get(index);
                            let node_to_transition = self.dual_graph.graph_1_mut().at_mut(index);
                            node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                            break 'scope;
                        }
                    }
                    let slice = self.dual_graph.adj_1()[index].slice();
                    for &idx in slice
                    {
                        let node = self.dual_graph.graph_2().at(idx);
                        if node.is_infected(){
                            infection_helper.dogs_infected_by_humans_p1(index);
                            infection_helper.animals_infected_by[index] = WhichGraph::Graph2(idx);
                            infection_helper.layer_dogs[index] = add_1(infection_helper.layer_humans[idx]);
                            let gamma = node.get_gamma();
                            let new_gamma = gamma + self.mutation_dogs_from_humans.get(index);
                            let node_to_transition = self.dual_graph.graph_1_mut().at_mut(index);
                            node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                            break 'scope;
                        }
                    }
                    unreachable!()
                }
            } else {
                let mut sum = 0.0;
                for node in self.dual_graph.graph_1_contained_iter(index)
                {
                    if node.is_infected()
                    {
                        sum += node.get_gamma_trans().trans_animal;
                    }
                }
            
                let which = self.infected_by_whom_dogs[index] * sum;
                sum = 0.0;
                
                'outer: loop{

                    let iter = self.dual_graph.graph_1().contained_iter_neighbors_with_index(index);

                    for (idx, node) in iter 
                    {
                        if node.is_infected()
                        {
                            sum += node.get_gamma_trans().trans_animal;
                            if which <= sum {
                                infection_helper.animals_infected_by[index] = WhichGraph::Graph1(idx);
                                infection_helper.layer_dogs[index] = add_1(infection_helper.layer_dogs[idx]);
                                let gamma = node.get_gamma();
                                let new_gamma = gamma + self.mutation_vec_dogs.get(index);
                                let node_to_transition = self.dual_graph.graph_1_mut().at_mut(index);
                                node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                                break 'outer;
                            }
                        }
                    }

                    let other_iter = self.dual_graph.adj_1()[index].slice();

                    for &idx in other_iter
                    {
                        let node = self.dual_graph.graph_2().at(idx);
                        if node.is_infected()
                        {
                            sum += node.get_gamma_trans().trans_animal;
                            if which <= sum {
                                infection_helper.dogs_infected_by_humans_p1(index);
                                infection_helper.animals_infected_by[index] = WhichGraph::Graph2(idx);
                                infection_helper.layer_dogs[index] = add_1(infection_helper.layer_humans[idx]);
                                let gamma = node.get_gamma();
                                let new_gamma = gamma + self.mutation_dogs_from_humans.get(index);
                                let node_to_transition = self.dual_graph.graph_1_mut().at_mut(index);
                                node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                                break 'outer;
                            }
                        }
                    }
                    unreachable!()
                }
            }
        }

        for &index in self.new_infections_list_humans.iter()
        {
            let container = self.dual_graph.graph_2().container(index);
            if container.contained().get_infectiouse_neighbor_count() == 1 {
                // workaround. loop can be removed with rust 1.65.0 as the respective scope feature will become stable
                'scope: loop {
                    let neighbors = container.edges();
                    for &idx in neighbors
                    {
                        let node = self.dual_graph.graph_2().at(idx);
                        if node.is_infected()
                        {
                            infection_helper.humans_infected_by[index] = WhichGraph::Graph2(idx);
                            infection_helper.layer_humans[index] = add_1(infection_helper.layer_humans[idx]);
                            let gamma = node.get_gamma();
                            let new_gamma = gamma + self.mutation_vec_humans.get(index);
                            let node_to_transition = self.dual_graph.graph_2_mut().at_mut(index);
                            node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                            break 'scope;
                        }
                    }
                    let slice = self.dual_graph.adj_2()[index].slice();
                    for &idx in slice
                    {
                        let node = self.dual_graph.graph_1().at(idx);
                        if node.is_infected(){
                            infection_helper.humans_infected_by_dogs_p1(index);
                            infection_helper.humans_infected_by[index] = WhichGraph::Graph1(idx);
                            infection_helper.layer_humans[index] = add_1(infection_helper.layer_dogs[idx]);
                            let gamma = node.get_gamma();
                            let new_gamma = gamma + self.mutation_humans_from_dogs.get(idx);
                            let node_to_transition = self.dual_graph.graph_2_mut().at_mut(index);
                            node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                            break 'scope;
                        }
                    }
                    unreachable!()
                }
                
            } else {
                let mut sum = 0.0;
                for node in self.dual_graph.graph_2_contained_iter(index)
                {
                    if node.is_infected()
                    {
                        sum += node.get_gamma_trans().trans_human;
                    }
                }
            
                let which = self.infected_by_whom_humans[index] * sum;
                let mut sum = 0.0;
                
                'outer: loop{

                    let iter = self.dual_graph.graph_2().contained_iter_neighbors_with_index(index);

                    for (idx, node) in iter 
                    {
                        if node.is_infected()
                        {
                            sum += node.get_gamma_trans().trans_human;
                            if which <= sum {
                                infection_helper.humans_infected_by[index] = WhichGraph::Graph2(idx);
                                infection_helper.layer_humans[index] = add_1(infection_helper.layer_humans[idx]);
                                let gamma = node.get_gamma();
                                let new_gamma = gamma + self.mutation_vec_humans.get(index);
                                let node = self.dual_graph.graph_2_mut().at_mut(index);
                                node.set_gt_and_transition(new_gamma, self.max_lambda);
                                break 'outer;
                            }
                        }
                    }

                    let other_iter = self.dual_graph.adj_2()[index].slice();

                    for &idx in other_iter
                    {
                        let node = self.dual_graph.graph_1().at(idx);
                        if node.is_infected()
                        {
                            sum += node.get_gamma_trans().trans_human;
                            if which <= sum {
                                infection_helper.humans_infected_by_dogs_p1(index);
                                infection_helper.humans_infected_by[index] = WhichGraph::Graph1(idx);
                                infection_helper.layer_humans[index] = add_1(infection_helper.layer_dogs[idx]);
                                let gamma = node.get_gamma();
                                let new_gamma = gamma + self.mutation_humans_from_dogs.get(idx);
                                let node = self.dual_graph.graph_2_mut().at_mut(index);
                                node.set_gt_and_transition(new_gamma, self.max_lambda);
                                break 'outer;
                            }
                        }
                    }
                    unreachable!()
                }
            }
        }

        // remove old infected nodes at random
        for id in (0..self.infected_list_dogs.len()).rev()
        {
            let index = self.infected_list_dogs[id];
            if self.recovery_rand_vec_dogs[self.offset_dogs.lookup_index(index)] < self.recover_prob
            {
                self.infected_list_dogs.swap_remove(id);
                let contained = self.dual_graph.graph_1_mut().at_mut(index);
                let gt = contained.get_gamma_trans();
                contained.progress_to_r();

                // dog neighbors:
                for neighbor in self.dual_graph.graph_1_mut().contained_iter_neighbors_mut(index)
                {
                    if neighbor.is_susceptible()
                    {
                        neighbor.subtract_from_s(gt.trans_animal)
                    }
                }

                // human neighbors
                for neighbor in self.dual_graph.graph_1_contained_iter_neighbors_in_other_graph_mut(index)
                {
                    if neighbor.is_susceptible()
                    {
                        neighbor.subtract_from_s(gt.trans_human);
                    }
                }
            }
        }


        // remove old infected nodes at random
        for id in (0..self.infected_list_humans.len()).rev()
        {
            let index = self.infected_list_humans[id];
            if self.recovery_rand_vec_humans[self.offset_humans.lookup_index(index)] < self.recover_prob
            {
                self.infected_list_humans.swap_remove(id);
                let contained = self.dual_graph.graph_2_mut().at_mut(index);
                let gt = contained.get_gamma_trans();
                contained.progress_to_r();

                // human neighbors:
                for neighbor in self.dual_graph.graph_2_mut().contained_iter_neighbors_mut(index)
                {
                    if neighbor.is_susceptible()
                    {
                        neighbor.subtract_from_s(gt.trans_human)
                    }
                }

                // dog neighbors
                for neighbor in self.dual_graph.graph_2_contained_iter_neighbors_in_other_graph_mut(index)
                {
                    if neighbor.is_susceptible()
                    {
                        neighbor.subtract_from_s(gt.trans_animal);
                    }
                }

            }
        }

        // transition to I dogs
        for &index in self.new_infections_list_dogs.iter()
        {
            let node = self.dual_graph.graph_1_mut().at_mut(index);
            node.transition_to_i();
            let gt = node.get_gamma_trans();

            for dog in self.dual_graph.graph_1_mut().contained_iter_neighbors_mut(index)
            {
                if dog.is_susceptible()
                {
                    dog.add_to_s(gt.trans_animal);
                }
            }

            for human in self.dual_graph.graph_1_contained_iter_neighbors_in_other_graph_mut(index)
            {
                if human.is_susceptible()
                {
                    human.add_to_s(gt.trans_human);
                }
            }
        }

        // transition to I humans
        for &index in self.new_infections_list_humans.iter()
        {
            let node = self.dual_graph.graph_2_mut().at_mut(index);
            node.transition_to_i();
            let gt = node.get_gamma_trans();

            for human in self.dual_graph.graph_2_mut().contained_iter_neighbors_mut(index)
            {
                if human.is_susceptible()
                {
                    human.add_to_s(gt.trans_human);
                }
            }

            for dog in self.dual_graph.graph_2_contained_iter_neighbors_in_other_graph_mut(index)
            {
                if dog.is_susceptible()
                {
                    dog.add_to_s(gt.trans_animal);
                }
            }
        }

        self.infected_list_dogs.append(&mut self.new_infections_list_dogs);
        self.infected_list_humans.append(&mut self.new_infections_list_humans);
        
    }

    #[inline]
    pub fn offset_set_time(&mut self, time: usize)
    {
        self.offset_dogs.set_time(time);
        self.offset_humans.set_time(time);
    }

    #[inline]
    pub fn infections_empty(&self) -> bool
    {
        self.infected_list_dogs.is_empty() && self.infected_list_humans.is_empty()
    }

    pub fn current_c_dogs(&self) -> usize 
    {
        self.dual_graph
            .graph_1()
            .contained_iter()
            .filter(|node| !node.is_susceptible())
            .count()
    }

    #[inline]
    pub fn dogs_gamma_iter(&'_ self) -> impl Iterator<Item=f64> + '_
    {
        self.dual_graph
            .graph_1()
            .contained_iter()
            .filter_map(
                |sir|
                if !sir.is_susceptible(){
                    Some(sir.get_gamma())
                } else {
                    None
                }
            )
    }

    #[inline]
    pub fn humans_gamma_iter(&'_ self) -> impl Iterator<Item=f64> + '_
    {
        self.dual_graph
            .graph_2()
            .contained_iter()
            .filter_map(
                |sir|
                if !sir.is_susceptible(){
                    Some(sir.get_gamma())
                } else {
                    None
                }
            )
    }

    #[allow(clippy::never_loop)]
    pub fn calc_c(&mut self) -> usize
    {
        self.reset_and_infect();
        self.total_sim_counter += 1;
        
        'scope: loop{
            for i in 0..self.max_time_steps.get()
            {
                self.offset_set_time(i);
                self.iterate_once();
                if self.infections_empty()
                {
                    self.last_extinction = i + 1;
                    break 'scope;
                }
            }
            self.last_extinction = self.max_time_steps.get() + 1;
            self.unfinished_sim_counter += 1;
            break 'scope;
        }

        let c = self.dual_graph
            .graph_2()
            .contained_iter()
            .filter(|node| !node.is_susceptible())
            .count();

        let dogs = self.dual_graph.graph_1()
            .contained_iter()
            .filter(|node| !node.is_susceptible())
            .count();
        println!("markov c_human {c} vs dogs {dogs}");
        c
    }

}



/// # Index Mapping for RNG Vectors
/// * This does the `rotation` of the rng vecs without actually 
/// copying the whole (and very large) list - this is much more efficient
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Offset{
    offset: usize,
    n: usize,
    bound: NonZeroUsize,
    time_with_offset: usize
}

impl Offset {

    /// # Set the current time
    /// * has to be called before `self.lookup_index`
    /// * This is an optimization - It was once part of the `lookup_index`
    ///  function - the way it is now has to be calculated only once per timestep, not for 
    /// each node
    #[inline(always)]
    pub fn set_time(&mut self, time: usize) {
        self.time_with_offset = time + self.offset;
        if self.time_with_offset >= self.bound.get() {
            self.time_with_offset -= self.bound.get();
        }

        // I believe, if a panic happens, which would be fixed by the following, then
        // something else is wrong in the programm and therefore, the following fix
        // should never be used!
        // 
        // self.time_with_offset = self.time_with_offset % self.bound;
    }

    /// Increase offset by 1 - i.e., Rotate RNG Vectors 
    /// * Wraps around bounds
    pub fn plus_1(&mut self){
        self.offset += 1;
        if self.offset >= self.bound.get() {
            self.offset = 0;
        }
    }

    /// Decrease offset by 1 - i.e., Rotate RNG Vectors in oposite direction
    pub fn minus_1(&mut self){
        self.offset = self.offset.checked_sub(1)
            .unwrap_or(self.bound.get() - 1);
    }

    pub fn new(bound: usize, n: usize) -> Self{
        Self{
            offset: 0,
            n,
            bound: unsafe{NonZeroUsize::new_unchecked(bound.max(1))},
            time_with_offset: 0
        }
    }

    /// You have to call `set_time` to set the correct time.
    /// If the offset changed, `set_time` has to be called again,
    /// to guarantee correct results.
    #[inline]
    pub fn lookup_index(&self, index: usize) -> usize
    {
        self.time_with_offset * self.n + index
    }

    #[inline]
    pub fn get_slice_mut<'a, 'b, T>(&'a self, slice: &'b mut [T]) -> &'b mut [T]
    {
        let start = self.lookup_index(0);
        let n = self.n;
        &mut slice[start..start+n]
    }
}

pub struct LayerHelper
{
    pub layer_dogs: Vec<Option<NonZeroUsize>>,
    pub layer_humans: Vec<Option<NonZeroUsize>>,
    pub humans_infected_by: Vec<DualIndex>,
    pub animals_infected_by: Vec<DualIndex>,
    pub dogs_infected_by_humans: usize,
    pub humans_infected_by_dogs: usize
}

impl LayerHelper
{
    pub fn reset(&mut self, initial_patients: &[usize])
    {
        self.layer_dogs
            .iter_mut()
            .for_each(|val| *val = None);
        self.layer_humans
            .iter_mut()
            .for_each(|val| *val = None);
        self.dogs_infected_by_humans = 0;
        self.humans_infected_by_dogs = 0;

        for &index in initial_patients
        {
            self.layer_dogs[index] = NonZeroUsize::new(1)
        }
    }

    #[inline]
    pub fn dogs_infected_by_humans_p1(&mut self, index_dog: usize)
    {
        println!("dog inf by: {index_dog}");
        self.dogs_infected_by_humans += 1;
    }

    #[inline]
    pub fn humans_infected_by_dogs_p1(&mut self, index_human: usize)
    {
        println!("human inf by: {index_human}");
        self.humans_infected_by_dogs += 1;
    }

    pub fn new(size_humans: usize, size_dogs: usize) -> Self
    {
        Self{
            layer_dogs: vec![None; size_dogs],
            layer_humans: vec![None; size_humans],
            dogs_infected_by_humans: 0,
            humans_infected_by_dogs: 0,
            humans_infected_by: vec![DualIndex::Graph1(usize::MAX); size_humans],
            animals_infected_by: vec![DualIndex::Graph1(usize::MAX); size_dogs]
        }
    }
}