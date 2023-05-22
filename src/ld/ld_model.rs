use crate::{sir_nodes::*, simple_sample::{BaseModel, PATIENTS_USIZE}};
use net_ensembles::{dual_graph::*, rand::{SeedableRng, seq::SliceRandom, Rng}, MarkovChain, HasRng, Node, Graph, EmptyNode};
use rand_distr::{Uniform, Distribution, Binomial, OpenClosed01};
use rand_pcg::Pcg64;
use serde::{Serialize, Deserialize};
use std::{num::*, io::Write, ops::Add};
use net_ensembles::{AdjList, AdjContainer};
use super::{SirWriter, InfoGraph};

const ROTATE: f64 = 0.01;
const PATIENT_MOVE: f64 = 0.03;
const P0_RAND: f64 = 0.04;
const MUTATION_MOVE: f64 = 0.11;
const BY_WHOM: f64 = 0.12; 
const ALEX_MOVE: f64 = 0.14;
const TIME_MOVE: f64 = 0.15;

const EPS: [f64; 8] = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7];



#[inline]
pub fn box_müller(u1: f64, u2: f64) -> f64
{
    let f = (-2.0 * u1.ln()).sqrt();
    let inner = std::f64::consts::TAU * u2;
    inner.cos() * f
}


#[derive(Clone, Serialize, Deserialize)]
pub struct Mutation
{
    mutation_source: Vec<f64>,
    actual_mutation: Vec<f64>
}

impl Mutation
{

    pub fn new<R: Rng>(size: usize, mut rng: R, sigma: f64) -> Self
    {
        let noise_iter = <OpenClosed01 as rand_distr::Distribution<f64>>::sample_iter(OpenClosed01, &mut rng)
            .take(size * 2);

        let mutation_source: Vec<_> = noise_iter.collect();

        let actual_mutation: Vec<_> = mutation_source
            .chunks_exact(2)
            .map(
                |chunk|
                {
                    box_müller(chunk[0], chunk[1]) * sigma
                }
            ).collect();
        
        Self { mutation_source, actual_mutation }
        
    }

    pub fn re_randomize<R: Rng>(&mut self, mut rng: R, sigma: f64)
    {
        let noise_iter = <OpenClosed01 as rand_distr::Distribution<f64>>::sample_iter(OpenClosed01, &mut rng);
        self.mutation_source.iter_mut()
            .zip(noise_iter)
            .for_each(|(val, new_val)| *val = new_val);

        self.actual_mutation.iter_mut()
            .zip(self.mutation_source.chunks_exact(2))
            .for_each(
                |(val, chunk)|
                {
                    *val = box_müller(chunk[0], chunk[1]) * sigma
                }
            );
    }


    #[inline]
    pub fn get(&self, index: usize) -> f64
    {
        unsafe{*self.actual_mutation.get_unchecked(index)}
    }

    #[inline]
    pub fn draw_new<R: Rng>(&mut self, index: usize, mut rng: R, sigma: f64) -> [f64; 2]
    {
        let start = index * 2;
        let slice = &mut self.mutation_source[start..=start+1];
        let old = [slice[0], slice[1]];

        let noise_iter = <OpenClosed01 as rand_distr::Distribution<f64>>::sample_iter(OpenClosed01, &mut rng);

        slice.iter_mut()
            .zip(noise_iter)
            .for_each(
                |(to_change, new_val)| *to_change = new_val
            );

        let new_mutation = box_müller(slice[0], slice[1]) * sigma;
        self.actual_mutation[index] = new_mutation;

        old
    }

    #[inline]
    pub fn slight_change<R: Rng>(
        &mut self, 
        index: usize, 
        mut rng: R, 
        sigma: f64, 
        eps: f64
    ) -> [f64; 2]
    {
        let uniform = Uniform::new_inclusive(-1.0_f64, 1.0);
        let start = index * 2;
        let slice = &mut self.mutation_source[start..=start+1];
        let old = [slice[0], slice[1]];

        let decision: f64 = rng.gen();

        if decision < 1.0/3.0 {
            slice[0] = uniform.sample(&mut rng).mul_add(eps, old[0]);
            if slice[0] > 1.0 || slice[0] <= 0.0 {
                slice[0] = old[0];
            }
        } else if decision < 2.0/3.0{
            slice[1] = uniform.sample(&mut rng).mul_add(eps, old[1]);
            if slice[1] > 1.0 || slice[1] <= 0.0 {
                slice[1] = old[1];
            }
        } else {
            slice[0] = uniform.sample(&mut rng).mul_add(eps, old[0]);
            slice[1] = uniform.sample(&mut rng).mul_add(eps, old[1]);
            if slice[0] > 1.0 || slice[0] <= 0.0 {
                slice[0] = old[0];
            }
            if slice[1] > 1.0 || slice[1] <= 0.0 {
                slice[1] = old[1];
            }
        }

        let new_mutation = box_müller(slice[0], slice[1]) * sigma;
        self.actual_mutation[index] = new_mutation;

        old
    }

    #[inline]
    pub fn undo(&mut self, index: usize, floats: [f64; 2], sigma: f64)
    { 
        let start = index * 2;
        let slice = &mut self.mutation_source[start..=start+1];
        slice[0] = floats[0];
        slice[1] = floats[1];

        let old_mutation = box_müller(floats[0], floats[1]) * sigma;

        self.actual_mutation[index] = old_mutation;

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
    pub mutation_change: Stats,
    pub slight_mutation_change: Stats,
    pub dfh_swap: Stats
}

impl MarkovStats
{
    pub fn reset(&mut self)
    {
        *self = Self::default();
    }

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
            + self.slight_mutation_change
            + self.mutation_change
            + self.dfh_swap;

        writeln!(writer, "#total:").unwrap();
        writeln!(writer, "#\ttotal:{}", sum.total()).unwrap();
        writeln!(writer, "#\taccepted: {} rate {}", sum.accepted, sum.acception_rate()).unwrap();
        writeln!(writer, "#\trejected: {} rate {}", sum.rejected, sum.rejection_rate()).unwrap();

        let sum_f64 = sum.total() as f64;

        let mut logger = |stats: Stats, name| 
        {
            let total = stats.total();
            let fraction = total as f64 / sum_f64;
            writeln!(writer, "#{name}:").unwrap();
            writeln!(writer, "#\ttotal: {total} fraction {fraction}").unwrap();
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
        log!(slight_mutation_change);
        log!(mutation_change);
        log!(dfh_swap);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LdModel<T: Clone>
{
    pub dual_graph: DefaultSDG<SirFun<T>, SirFun<T>>,
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
    pub prev_last_extinction: usize,
    pub stats: MarkovStats,
    pub neg_bins: i32
}

impl<T> LdModel<T>
where T: Clone
{
    pub fn get_topology(&self) -> Graph<EmptyNode>
    {
        let dogs = self.dual_graph.graph_1().vertex_count();
        let humans = self.dual_graph.graph_2().vertex_count();
        let mut graph = Graph::new(dogs + humans);
        for (index, this) in self.dual_graph.graph_1().get_vertices().iter().enumerate()
        {
            for &neighbor in this.edges()
            {
                if index < neighbor
                {
                    let _ = graph.add_edge(neighbor, index);
                }
            }
        }

        for (index, human) in self.dual_graph.adj_1().iter().enumerate()
        {
            for other in human.slice().iter()
            {
                let human_idx = *other + dogs;
                let _ = graph.add_edge(human_idx, index);
            }
        }


        for (mut index, this) in self.dual_graph.graph_2().get_vertices().iter().enumerate()
        {
            index += dogs;
            for &neighbor in this.edges()
            {
                let neighbor = neighbor + dogs;
                if index < neighbor
                {
                    let _ = graph.add_edge(neighbor, index);
                }
            }
        }

        for (mut index, human) in self.dual_graph.adj_1().iter().enumerate()
        {
            index += dogs;
            for &other in human.slice().iter()
            {
                let _ = graph.add_edge(other, index);
            }
        }
        graph
    }


    pub fn epidemic_threshold(&self) -> f64
    {
        let size_1 = self.dual_graph.graph_1().vertex_count();
        let size_2 = self.dual_graph.graph_2().vertex_count();

        let mut k = 0;
        let mut k2 = 0;

        for i in 0..size_1
        {
            let degree = self.dual_graph.degree_1(i);
            k += degree;
            k2 += degree * degree;
        }
        for i in 0..size_2
        {
            let degree = self.dual_graph.degree_2(i);
            k += degree;
            k2 += degree * degree;
        }

        let average_k = k as f64 / (size_1 + size_2) as f64;
        let average_k2 = k2 as f64 / (size_1 + size_2) as f64;

        average_k / average_k2
    }
}

impl<T> HasRng<Pcg64> for LdModel<T>
where T: Clone
{
    fn rng(&mut self) -> &mut Pcg64 {
        &mut self.markov_rng
    }

    fn swap_rng(&mut self, rng: &mut Pcg64) {
        std::mem::swap(rng, self.rng())
    }
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

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Index
{
    pub index: usize
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct TwoFloats
{
    floats: [f64;2]
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
    two_floats: TwoFloats,
    index: Index
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
    SlightMutationChange,
    AlexMove,
    TimeMove(usize)
}

impl<T> MarkovChain<MarkovStep, ()> for LdModel<T>
where T: Clone
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
            WhichMove::SlightMutationChange => {
                &mut self.stats.slight_mutation_change
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
            WhichMove::SlightMutationChange => {
                &mut self.stats.slight_mutation_change
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

    #[allow(unreachable_code)]
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

        macro_rules! disabled {
            () => {
                return self.m_steps(count, steps);
            };
        }
       
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
                disabled!();
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
            disabled!();
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

                let mut rng = Pcg64::from_rng(&mut self.markov_rng).unwrap();

                let mut do_step = |list: &mut Vec<StepEntry>, mutation: &mut Mutation, index: usize|
                {
                    let old = mutation.draw_new(index, &mut rng, self.sigma);
                    list.push(
                        StepEntry{index: Index { index }}
                    );
                    list.push(StepEntry{ two_floats: TwoFloats { floats: old }});
                };

                for _ in 0..how_many{
                    let index = self.markov_rng.gen_range(0..humans+animals*3);
                    if index < humans
                    {
                        do_step(
                            &mut step.list_humans_rec,
                            &mut self.mutation_vec_humans,
                            index
                        );
                    } else if index < humans + animals {
                        let index = index - humans;
                        do_step(
                            &mut step.list_animals_rec,
                            &mut self.mutation_vec_dogs,
                            index
                        );
                    } else if index < humans + 2* animals
                    {
                        let index = index - humans - animals;
                        do_step(
                            &mut step.list_animals_trans,
                            &mut self.mutation_dogs_from_humans,
                            index
                        );
                    } else {
                        let index = index - humans - animals * 2;
                        do_step(
                            &mut step.list_humans_trans,
                            &mut self.mutation_humans_from_dogs,
                            index
                        );
                    }
                }
                
            } else {
                step.which = WhichMove::SlightMutationChange;

                let eps = *EPS.choose(&mut self.markov_rng).unwrap();
                

                let humans = self.dual_graph.graph_2().vertex_count();
                let animals = self.dual_graph.graph_1().vertex_count();
                let amount = animals / 3;
                let how_many = self.markov_rng.gen_range(2..amount);


                let mut rng = Pcg64::from_rng(&mut self.markov_rng).unwrap();

                let mut do_step = |list: &mut Vec<StepEntry>, mutation: &mut Mutation, index: usize|
                {
                    let old = mutation.slight_change(index, &mut rng, self.sigma, eps);
                    list.push(
                        StepEntry{index: Index { index }}
                    );
                    list.push(StepEntry{ two_floats: TwoFloats { floats: old }});
                };

                for _ in 0..how_many{
                    let index = self.markov_rng.gen_range(0..humans+animals*3);
                    if index < humans
                    {
                        do_step(
                            &mut step.list_humans_rec,
                            &mut self.mutation_vec_humans,
                            index
                        );
                    } else if index < humans + animals {
                        let index = index - humans;
                        do_step(
                            &mut step.list_animals_rec,
                            &mut self.mutation_vec_dogs,
                            index
                        );
                    } else if index < humans + 2* animals
                    {
                        let index = index - humans - animals;
                        do_step(
                            &mut step.list_animals_trans,
                            &mut self.mutation_dogs_from_humans,
                            index
                        );
                    } else {
                        let index = index - humans - animals * 2;
                        do_step(
                            &mut step.list_humans_trans,
                            &mut self.mutation_humans_from_dogs,
                            index
                        );
                    }
                }
            }
            
        } else if which < BY_WHOM
        {
            step.which = WhichMove::ByWhom;
            let humans = self.dual_graph.graph_2().vertex_count();
            let animals = self.dual_graph.graph_1().vertex_count();
            let index = self.markov_rng.gen_range(0..humans+animals);
            for _ in 0..animals {
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
            }
            
        } else if which < ALEX_MOVE
        {
            disabled!();
        } else if which < TIME_MOVE
        {
            //time move
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
        self.last_extinction = self.prev_last_extinction;
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
            WhichMove::MutationChange | WhichMove::SlightMutationChange => {

                let undo = |list: &[StepEntry], mutation: &mut Mutation|
                {
                    list
                        .chunks_exact(2)
                        .rev()
                        .for_each(
                            |chunk|
                            {
                                let index = unsafe{ chunk[0].index }.index;
                                let floats = unsafe{ chunk[1].two_floats}.floats;
                                mutation.undo(index, floats, self.sigma);
                            }
                        );
                };

                undo(&step.list_humans_rec, &mut self.mutation_vec_humans);
                undo(&step.list_animals_rec, &mut self.mutation_vec_dogs);
                undo(&step.list_animals_trans, &mut self.mutation_dogs_from_humans);
                undo(&step.list_humans_trans, &mut self.mutation_humans_from_dogs);
                
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
            WhichMove::AlexMove =>
            {
                unimplemented!();
                /*
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
                */
               
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

pub struct LambdaRes
{
    pub max_lambda_reached: f64,
    pub mean_lambda: f64
}

impl<T> LdModel<T>
where T: Clone + TransFun
{
    pub fn re_randomize(&mut self, mut rng: Pcg64)
    {
        self.new_infections_list_dogs.clear();
        self.new_infections_list_humans.clear();
        self.infected_list_dogs.clear();
        self.infected_list_humans.clear();

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

        
        self.mutation_vec_dogs.re_randomize(&mut rng, self.sigma);
        self.mutation_vec_humans.re_randomize(&mut rng, self.sigma);
        self.mutation_dogs_from_humans.re_randomize(&mut rng, self.sigma);
        self.mutation_humans_from_dogs.re_randomize(&mut rng, self.sigma);

        self.markov_rng = rng;
        
    }

    pub fn calculate_max_lambda_reached_humans(&self) -> LambdaRes
    where T: Default + Clone + Serialize + TransFun
    {
        let mut max_lambda = f64::NEG_INFINITY;
        let mut sum = 0_u32;
        let mut lambda_sum = 0.0;
        for contained in self.dual_graph.graph_2()
            .contained_iter()
        {
            if !contained.is_susceptible()
            {
                sum += 1;
                let lambda = contained.get_gamma_trans().trans_human;
                lambda_sum += lambda;
                if lambda > max_lambda{
                    max_lambda = lambda;
                }
            }
        }
        if sum == 0 {
            LambdaRes{
                max_lambda_reached: f64::NAN,
                mean_lambda: f64::NAN
            }
        } else {
            LambdaRes{
                max_lambda_reached: max_lambda,
                mean_lambda: lambda_sum / sum as f64
            }
        }
    }

    pub fn new(mut base: BaseModel<T>, markov_seed: u64, max_sir_steps: NonZeroUsize, neg_bins: i32) -> Self
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
        
        let mutation_vec_dogs = Mutation::new(n_dogs, &mut markov_rng, base.sigma);
        let mutation_vec_humans = Mutation::new(n_humans, &mut markov_rng, base.sigma);

        let humans_from_dogs = Mutation::new(n_dogs, &mut markov_rng, base.sigma);
        let dogs_from_humans = Mutation::new(n_dogs, &mut markov_rng, base.sigma);

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
            neg_bins,
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
            prev_last_extinction: usize::MAX,
            stats: MarkovStats::default(),
            mutation_humans_from_dogs: humans_from_dogs,
            mutation_dogs_from_humans: dogs_from_humans,
        }

    }

    pub fn reset_and_infect(&mut self)
    where SirFun<T>: Node
    {
        self.prev_last_extinction = self.last_extinction;
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
    where SirFun<T>: Node
    {
        // decide which nodes will become infected
        for (index, sir_fun) in self.dual_graph.graph_1_mut().contained_iter_mut().enumerate()
        {
            if sir_fun.is_susceptible()
            {
                let state = sir_fun.get_sus_state();
                if self.trans_rand_vec_dogs[self.offset_dogs.lookup_index(index)] > state.product {
                    self.new_infections_list_dogs.push(index);
                }
            }
        }
        for (index, sir_fun) in self.dual_graph.graph_2_mut().contained_iter_mut().enumerate()
        {
            if sir_fun.is_susceptible()
            {
                let state = sir_fun.get_sus_state();
                if self.trans_rand_vec_humans[self.offset_humans.lookup_index(index)] > state.product {
                    self.new_infections_list_humans.push(index);
                }
            }
        }

        // set new transmission values etc
        for &index in self.new_infections_list_dogs.iter()
        {
            let container = self.dual_graph.graph_1().container(index);
            if container.contained().get_infectiouse_neighbor_count() == 1 {
                
                'scope: {
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
                
                'outer: {
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
                    let state = self.dual_graph.graph_1().at(index).get_sus_state();
                    dbg!(state);
                    println!("sum: {sum} which: {which}");
                    unreachable!()
                }
                
            }
        }

        for &index in self.new_infections_list_humans.iter()
        {
            let container = self.dual_graph.graph_2().container(index);
            if container.contained().get_infectiouse_neighbor_count() == 1 {
                
                'scope: {
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
                            let new_gamma = gamma + self.mutation_humans_from_dogs.get(idx);
                            let node_to_transition = self.dual_graph.graph_2_mut().at_mut(index);
                            node_to_transition.set_gt_and_transition(new_gamma, self.max_lambda);
                            //println!("infecting human {index} - {lambda} {lambda2}");
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
                
                'outer: {
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
                                let node = self.dual_graph.graph_2_mut().at_mut(index);
                                node.set_gt_and_transition(new_gamma, self.max_lambda);
                                //println!("infecting another human {index} - {lambda_old} {lambda_new}");
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
        last_energy: i32
    )
    where T: Clone + Default + Serialize
    {
        let last_energy = if last_energy <= 0 
        {
            0
        } else {
            last_energy as usize
        };
        self.reset_and_infect();
        infection_helper.reset(&self.initial_patients);
        
        let _ = writer_humans.write_energy(last_energy, self.last_extinction);
        let _ = writer_animals.write_energy(last_energy, self.last_extinction);
        let _ = writer_humans.write_current(self.dual_graph.graph_2());
        let _ = writer_animals.write_current(self.dual_graph.graph_1());
        
        let max_time = self.max_time_steps.get() as u16;
        for i in 1..=max_time
        {
            let time = unsafe{NonZeroU16::new_unchecked(i)};
            let index_time = (i - 1) as usize;
            self.offset_set_time(index_time);
            self.iterate_once_writing(infection_helper, time);
            let _ = writer_humans.write_current(self.dual_graph.graph_2());
            let _ = writer_animals.write_current(self.dual_graph.graph_1());
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
        let _ = writer_humans.write_line();
        let _ = writer_animals.write_line();
    }

    fn iterate_once_writing(
        &mut self,
        infection_helper: &mut LayerHelper,
        time: NonZeroU16
    )
    where SirFun<T>: Node
    {

        //#[inline]
        //fn add_1(val: Option<NonZeroUsize>) -> Option<NonZeroUsize>
        //{
        //    val.unwrap().checked_add(1)
        //}

        #[inline]
        fn add_1(val: Option<NonZeroUsize>) -> Option<NonZeroUsize>
        {
            unsafe{
                Some(
                    NonZeroUsize::new_unchecked(
                        val.unwrap_unchecked().get() + 1
                    )
                )
            }
        }

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
                'scope: {
                    let neighbors = container.edges();
                    for &idx in neighbors
                    {
                        let node = self.dual_graph.graph_1().at(idx);
                        if node.is_infected()
                        {
                            infection_helper.graph.a_infects_b(idx, index, time);
                            infection_helper.animals_infected_by[index] = Some(WhichGraph::Graph1(idx));
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
                            let a = infection_helper.graph.get_human_index(idx);
                            infection_helper.graph.a_infects_b(a, index, time);
                            infection_helper.dogs_infected_by_humans_p1(index);
                            infection_helper.animals_infected_by[index] = Some(WhichGraph::Graph2(idx));
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
                
                'outer: {

                    let iter = self.dual_graph.graph_1().contained_iter_neighbors_with_index(index);

                    for (idx, node) in iter 
                    {
                        if node.is_infected()
                        {
                            sum += node.get_gamma_trans().trans_animal;
                            if which <= sum {
                                infection_helper.graph.a_infects_b(idx, index, time);
                                infection_helper.animals_infected_by[index] = Some(WhichGraph::Graph1(idx));
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
                                let a = infection_helper.graph.get_human_index(idx);
                                infection_helper.graph.a_infects_b(a, index, time);
                                infection_helper.dogs_infected_by_humans_p1(index);
                                infection_helper.animals_infected_by[index] = Some(WhichGraph::Graph2(idx));
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
            let new_infected_human = infection_helper.graph.get_human_index(index);
            let container = self.dual_graph.graph_2().container(index);
            if container.contained().get_infectiouse_neighbor_count() == 1 {
                
                'scope: {
                    let neighbors = container.edges();
                    for &idx in neighbors
                    {
                        let node = self.dual_graph.graph_2().at(idx);
                        if node.is_infected()
                        {
                            let a = infection_helper.graph.get_human_index(idx);
                            infection_helper.graph.a_infects_b(a, new_infected_human, time);
                            infection_helper.humans_infected_by[index] = Some(WhichGraph::Graph2(idx));
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
                            infection_helper.graph.a_infects_b(idx, new_infected_human, time);
                            infection_helper.humans_infected_by_dogs_p1(index);
                            infection_helper.humans_infected_by[index] = Some(WhichGraph::Graph1(idx));
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
                
                'outer: {
                    let iter = self.dual_graph.graph_2().contained_iter_neighbors_with_index(index);

                    for (idx, node) in iter 
                    {
                        if node.is_infected()
                        {
                            sum += node.get_gamma_trans().trans_human;
                            if which <= sum {
                                let a = infection_helper.graph.get_human_index(idx);
                                infection_helper.graph.a_infects_b(a, new_infected_human, time);
                                infection_helper.humans_infected_by[index] = Some(WhichGraph::Graph2(idx));
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
                                infection_helper.graph.a_infects_b(idx, new_infected_human, time);
                                infection_helper.humans_infected_by_dogs_p1(index);
                                infection_helper.humans_infected_by[index] = Some(WhichGraph::Graph1(idx));
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
                infection_helper.graph.recover_dog(index, time);
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
                infection_helper.graph.recover_human(index, time);
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
    where SirFun<T>: Node
    {
        self.dual_graph
            .graph_1()
            .contained_iter()
            .filter(|node| !node.is_susceptible())
            .count()
    }

    #[inline]
    pub fn dogs_gamma_iter(&'_ self) -> impl Iterator<Item=f64> + '_
    where SirFun<T>: Node
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
    where SirFun<T>: Node
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

    
    pub fn calc_c(&mut self) -> i32
    where T: Clone + Default + Serialize
    {
        self.reset_and_infect();
        self.total_sim_counter += 1;
        
        'scope: {
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
        }

        let c = self.dual_graph
            .graph_2()
            .contained_iter()
            .filter(|node| !node.is_susceptible())
            .count() as i32;

        if c == 0 {
            if self.neg_bins == 0 {
                return 0;
            }
            let mut max_gamma = -10000.0_f64;
            for (container, adj) in self.dual_graph.graph_1().container_iter().zip(self.dual_graph.adj_1().iter())
            {
                if adj.is_something() && container.contained().was_ever_infected()
                {
                    let gamma = container.contained().get_gamma();
                    max_gamma = max_gamma.max(gamma);
                }
            }
            let bins = self.neg_bins.abs() as f64;

            let a = 20.0 * bins/(21.0-20.0*self.reset_gamma);
            let b = -a*self.reset_gamma;


            let which_bin = a.mul_add(max_gamma, b).clamp(0.0, bins) as i32;
            let which_bin = which_bin + self.neg_bins;
            return which_bin;
        }
        //let dogs = self.dual_graph.graph_1()
        //    .contained_iter()
        //    .filter(|node| !node.is_susceptible())
        //    .count();
        //println!("markov c_human {c} vs dogs {dogs}");
        c
    }

    pub fn scan_calc_energy(&mut self) -> i32
    where T: Clone + Default + Serialize
    {
        self.reset_and_infect();
        self.total_sim_counter += 1;
        
        'scope: {
            for i in 0..self.max_time_steps.get()
            {
                self.offset_set_time(i);
                self.iterate_once();

                if !self.infected_list_humans.is_empty()
                {
                    i_am_cold();
                    return 1;
                }

                if self.infections_empty()
                {
                    self.last_extinction = i + 1;
                    break 'scope;
                }
            }
            self.last_extinction = self.max_time_steps.get() + 1;
            self.unfinished_sim_counter += 1;
        }

        let mut max_gamma = -10000.0_f64;
        for (container, adj) in self.dual_graph.graph_1().container_iter().zip(self.dual_graph.adj_1().iter())
        {
            if adj.is_something() && container.contained().was_ever_infected()
            {
                let gamma = container.contained().get_gamma();
                max_gamma = max_gamma.max(gamma);
            }
        }
        let bins = self.neg_bins.abs() as f64;
        let a = 20.0 * bins/(21.0-20.0*self.reset_gamma);
        let b = -a*self.reset_gamma;
        let which_bin = a.mul_add(max_gamma, b).clamp(0.0, bins) as i32;
        which_bin + self.neg_bins
    }

}

#[cold]
fn i_am_cold(){}


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
    pub fn get_slice_mut<'b, T>(&self, slice: &'b mut [T]) -> &'b mut [T]
    {
        let start = self.lookup_index(0);
        let n = self.n;
        &mut slice[start..start+n]
    }
}


#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum InfectedBy
{
    InitialInfected,
    By(u16),
    NotInfected
}

/*
    TODO

    Change assertions to Debugassertions

    Include "Vorfahre" in InfoNode <--- assert that it works
    -> this will greatly speed up the process.
    Maybe I can even get rid of the is_used vector that way
    -> include "vorfahre" in reset -> either as option or just set it to u32::MAX to ensure a out of bounds panic
    in case I messed up

*/

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct  InfoNode{
    pub time_step: Option<NonZeroU16>,
    pub layer: Option<NonZeroU16>,
    pub infected_by: InfectedBy,
    pub prev_dogs: u16,
    pub gamma_trans: Option<GammaTrans>,
    pub recovery_time: Option<NonZeroU16>
}


impl InfoNode
{
    #[inline]
    pub fn was_infected(&self) -> bool
    {  
        !matches!(self.infected_by, InfectedBy::NotInfected)
    }


    pub fn get_lambda_human(&self) -> f64
    {
        self.gamma_trans.unwrap().trans_human
    }

    pub fn get_lambda_dog(&self) -> f64
    {
        self.gamma_trans.unwrap().trans_animal
    }

    pub fn get_time(&self) -> f64
    {
        self.time_step.unwrap().get() as f64
    }

    pub fn get_gamma(&self) -> f64
    {
        self.gamma_trans.unwrap().gamma
    }

    pub fn get_recovery_time(&self) -> f64
    {
        self.recovery_time.unwrap_or(NonZeroU16::new(10000).unwrap()).get() as f64
    }

    pub fn get_time_difference(&self) -> f64
    {
        (self.recovery_time.unwrap().get() - self.time_step.unwrap().get()) as f64
    }
}

impl Node for InfoNode
{
    fn new_from_index(_: usize) -> Self {
        Self { 
            time_step: None, 
            layer: None,
            infected_by: InfectedBy::NotInfected,
            prev_dogs: 0,
            gamma_trans: None,
            recovery_time: None
        }
    } 
}

#[derive(Serialize, Deserialize)]
pub struct GammaHelper
{
    pub current_gamma: f64,
    pub gammas: Vec<f64>,
    pub next_idx: usize
}

impl GammaHelper
{
    pub fn new(gamma: f64, next_idx: usize) -> Self
    {
        Self{
            gammas: Vec::new(),
            next_idx,
            current_gamma: gamma
        }
    }

    pub fn clear_with(&mut self, gamma: f64, next_idx: usize)
    {
        self.gammas.clear();
        self.current_gamma = gamma;
        self.next_idx = next_idx;
    }

    #[must_use]
    pub fn add_gamma(&mut self, gamma: f64, gamma_threshold: f64) -> u32
    {
        self.gammas.push(self.current_gamma);
        self.current_gamma = gamma;
        for i in (0..self.gammas.len()).rev()
        {
            let dist = (self.gammas[i] - gamma).abs();
            if dist > gamma_threshold
            {
                self.gammas.swap_remove(i);
            }
        }
        self.gammas.len() as u32
    }
}

const HEAD: &str = "digraph T{
    ratio=\"compress\";
    bgcolor=\"transparent\";
";

pub enum HumanOrDog
{
    Dog,
    Human
}

pub fn write_dot<W: Write, F, F2>(
    info: &mut InfoGraph,
    mut writer: W,
    mut color_fun: F,
    mut label_fun: F2
)
where F: FnMut (&InfoGraph, HumanOrDog, usize) -> u32,
    F2: FnMut (&InfoGraph, HumanOrDog, usize) -> String
{
    eprintln!("WARNING: info.calc() was removed! So gamma thresholds are not working! -> info[x].disease_children_count not working. Ret *should* be unaffected");
    let dog_count = info.dog_count;
    let initial_infected = info.initial_infection[0];
    let graph = &info.info;
    let _ = write!(writer, "{HEAD}");
    let human_str = "node [shape=ellipse, penwidth=1, fontname=\"Courier\", pin=true, style=filled ];";
    let (dogs, humans) = graph.get_vertices().split_at(dog_count);
    if humans.iter().any(|human| human.contained().layer.is_some())
    {
        let _ = writeln!(writer, "{human_str}");
        for (index, human) in humans.iter().enumerate()
        {
            if human.contained().layer.is_some()
            {
                let id = index + dog_count;
                let color = color_fun(info, HumanOrDog::Human, id);
                let label = label_fun(info, HumanOrDog::Human, id);
                let _ = write!(writer, " {id} [label=\"{label}\", fillcolor=\"#{color:06x}\"]");
            }
        }
        let _ = writeln!(writer, ";");
    }
    
    let dog_str = "node [shape=box, penwidth=1, fontname=\"Courier\", pin=true, style=filled ];";
    let _ = writeln!(writer, "{dog_str}");
    for (id, dog) in dogs.iter().enumerate()
    {
        if dog.contained().layer.is_some(){
            let color = color_fun(info, HumanOrDog::Dog, id);
            let label = label_fun(info, HumanOrDog::Dog, id);
            let _ = write!(writer, " {id} [label=\"{label}\", fillcolor=\"#{color:06x}\"]");
        }
    }
    let _ = writeln!(writer, ";");

    let bfs = graph.bfs_index_depth(initial_infected);
    for (index, node, _depth) in bfs 
    {
        let adj = graph.container(index).edges();
        let vorfahre = &node.infected_by;
        match vorfahre
        {
            InfectedBy::InitialInfected => {
                let _ = write!(writer, "{index} -> {{");
                for other in adj 
                {
                    let _ = write!(writer, " {other}");
                }
                let _ = writeln!(writer, "}}");
            },
            InfectedBy::NotInfected => continue,
            InfectedBy::By(by) => {
                let _ = write!(writer, "{index} -> {{");
                for other in adj 
                {
                    if *by != *other as u16{
                        let _ = write!(writer, " {other}");
                    }
                }
                let _ = writeln!(writer, "}}");
            }
        }

    }

    let _ = write!(writer, "}}");

}

#[derive(Serialize, Deserialize)]
pub struct CondensedInfo
{
    pub dogs: u16,
    pub total: usize,
    pub initial_infected: usize,
    pub indices: Vec<u32>,
    pub nodes: Vec<InfoNode>
}

impl CondensedInfo {
    pub fn to_info_graph(&self) -> InfoGraph
    {
        let mut graph = Graph::<InfoNode>::new(self.total);

        self.indices.iter()
            .zip(self.nodes.iter())
            .for_each(
                |(&index, node)|
                {
                    let idx = index as usize;
                    *graph.at_mut(idx) = node.clone();
                    if let InfectedBy::By(some) = node.infected_by
                    {
                        let _ = graph.add_edge(idx, some as usize);
                    }
                }
            );


        let waiting_helper_count = vec![0; graph.vertex_count()];
        let disease_children_count = vec![0; graph.vertex_count()];

        InfoGraph { 
            info: graph, 
            disease_children_count, 
            dog_count: self.dogs as usize, 
            initial_infection: vec![self.initial_infected], 
            waiting_helper_count, 
            gamma_helper_in_use: Vec::new(), 
            unused_gamma_helper: Vec::new() 
        }
    }

}


pub struct LayerHelper
{
    pub graph: InfoGraph,
    pub layer_dogs: Vec<Option<NonZeroUsize>>,
    pub layer_humans: Vec<Option<NonZeroUsize>>,
    pub humans_infected_by: Vec<Option<DualIndex>>,
    pub animals_infected_by: Vec<Option<DualIndex>>,
    pub dogs_infected_by_humans: usize,
    pub humans_infected_by_dogs: usize,
    pub index_of_first_infected_human: Option<u32>
}

#[derive(PartialEq, Debug)]
pub struct LayerRes
{
    pub max_index: usize,
    pub max_count: u32,
    pub layer: NonZeroU16,
    pub gamma: f64
}

impl LayerHelper
{
    pub fn reset(&mut self, initial_patients: &[usize])
    {
        self.graph.reset();

        for initial in initial_patients
        {
            self.graph.initial_infection(*initial);
        }

        self.layer_dogs
            .iter_mut()
            .for_each(|val| *val = None);
        self.layer_humans
            .iter_mut()
            .for_each(|val| *val = None);

        self.humans_infected_by
            .iter_mut()
            .for_each(|val| *val = None);

        self.animals_infected_by
            .iter_mut()
            .for_each(|val| *val = None);

        self.dogs_infected_by_humans = 0;
        self.humans_infected_by_dogs = 0;

        for &index in initial_patients
        {
            self.layer_dogs[index] = NonZeroUsize::new(1)
        }

        self.index_of_first_infected_human = None;
    }

    pub fn calc_layer_res<T>(&self, threshold: f64, graph: &DefaultSDG<SirFun<T>, SirFun<T>>) -> (Option<LayerRes>, Option<LayerRes>)
    where T: TransFun
    {
        let mut count_humans = vec![0_u32; graph.graph_2().vertex_count()];
        let mut count_dogs = vec![0_u32; graph.graph_1().vertex_count()];

        self.animals_infected_by.iter()
            .enumerate()
            .for_each(
                |(index, &by)|
                {
                    if by.is_some(){
                        let gamma = graph.graph_1().at(index).get_gamma();
                        let mut node_tracker = by;
                        while let Some(next_node) = node_tracker{
                            match next_node{
                                WhichGraph::Graph1(node_index) => {
                                    let node_gamma = graph.graph_1().at(node_index).get_gamma();
                                    if (node_gamma - gamma).abs() < threshold {
                                        count_dogs[node_index] += 1;
                                        node_tracker = self.animals_infected_by[node_index];
                                    } else {
                                        break;
                                    }
                                },
                                WhichGraph::Graph2(node_index) => {
                                    let node_gamma = graph.graph_2().at(node_index).get_gamma();
                                    if (node_gamma - gamma).abs() < threshold {
                                        count_humans[node_index] += 1;
                                        node_tracker = self.humans_infected_by[node_index];
                                    } else {
                                        break;
                                    }
                                }
                            } 
                        }
                    }
                }
            );

        self.humans_infected_by.iter()
            .enumerate()
            .for_each(
                |(index, &by)|
                {
                    if by.is_some(){
                        let gamma = graph.graph_2().at(index).get_gamma();
                        let mut node_tracker = by;
                        while let Some(next_node) = node_tracker{
                            match next_node{
                                WhichGraph::Graph1(node_index) => {
                                    let node_gamma = graph.graph_1().at(node_index).get_gamma();
                                    if (node_gamma - gamma).abs() < threshold {
                                        count_dogs[node_index] += 1;
                                        node_tracker = self.animals_infected_by[node_index];
                                    } else {
                                        break;
                                    }
                                },
                                WhichGraph::Graph2(node_index) => {
                                    let node_gamma = graph.graph_2().at(node_index).get_gamma();
                                    if (node_gamma - gamma).abs() < threshold {
                                        count_humans[node_index] += 1;
                                        node_tracker = self.humans_infected_by[node_index];
                                    } else {
                                        break;
                                    }
                                }
                            } 
                        }
                    }
                }
            );

        let mut max_dog = 0;
        let mut index_of_max_dog = 0;

        count_dogs
            .iter()
            .enumerate()
            .for_each(
                |(index, &val)|
                {
                    if val > max_dog {
                        max_dog = val;
                        index_of_max_dog = index;
                    }
                }
            );

        let dog_res = if max_dog > 0 {
            let layer = self.layer_dogs[index_of_max_dog].unwrap().get() as u16;
            let layer = NonZeroU16::new(layer).unwrap();
            let gamma = graph.graph_1().at(index_of_max_dog).get_gamma();
            Some(LayerRes{
                layer,
                gamma,
                max_index: index_of_max_dog,
                max_count: max_dog
            })
        } else {
            None
        };

        let mut max_human = 0;
        let mut index_of_max_human = 0;


        count_humans
            .iter()
            .enumerate()
            .for_each(
                |(index, &val)|
                {
                    if val > max_human {
                        max_human = val;
                        index_of_max_human = index;
                    }
                }
            );

        let human_res = if max_human > 0 {
            let layer = self.layer_humans[index_of_max_human].unwrap().get() as u16;
            let layer = NonZeroU16::new(layer).unwrap();
            let gamma = graph.graph_2().at(index_of_max_human).get_gamma();
            Some(LayerRes{
                layer,
                gamma,
                max_index: index_of_max_human,
                max_count: max_human
            })
        } else {
            None
        };

        (dog_res, human_res)
    }

    #[inline]
    pub fn dogs_infected_by_humans_p1(&mut self, _index_dog: usize)
    {
        //println!("dog inf by: {index_dog}");
        self.dogs_infected_by_humans += 1;
    }

    #[inline]
    pub fn humans_infected_by_dogs_p1(&mut self, index_human: usize)
    {
        if self.index_of_first_infected_human.is_none()
        {
            self.index_of_first_infected_human = Some(index_human as u32);
        }
        //println!("human inf by: {index_human}");
        self.humans_infected_by_dogs += 1;
    }

    pub fn new(size_humans: usize, size_dogs: usize) -> Self
    {
        Self{
            graph: InfoGraph::new(size_dogs, size_humans),
            layer_dogs: vec![None; size_dogs],
            layer_humans: vec![None; size_humans],
            dogs_infected_by_humans: 0,
            humans_infected_by_dogs: 0,
            humans_infected_by: vec![None; size_humans],
            animals_infected_by: vec![None; size_dogs],
            index_of_first_infected_human: None
        }
    }
}


pub fn color(lambda: f64) -> u32
{
    let s = 1.0;
    let r = 1.5;
    let hue = 0.5;
    let gamma = 0.6;
    
    let lg = lambda.powf(gamma);

    let a = hue * lg * (1.0-lg) * 0.5;

    let phi = (s/3.0 + r * lambda) * std::f64::consts::TAU;

    let (sin, cos) = phi.sin_cos();

    let red = a * (-0.14861*cos + 1.78277 * sin) + lg;
    let green = lg + a* (-0.29227*cos -0.90649*sin);
    let blue = lg + a * cos * 1.97294;

    let red = (red * 255.0) as u32;
    let blue = (blue * 255.0) as u32;
    let green = (green * 255.0) as u32;

    blue + 256*green + 256*256 * red
}

pub fn color2(val: f64) -> u32
{

    let red = val;
    let red = 1.0 - red.clamp(0.0, 1.0);
    let red = (red * 255.0) as u32;
    let blue = 100;
    let green = 10;

    blue + 256*green + 256*256 * red
}