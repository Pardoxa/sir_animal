use crate::{sir_nodes::*, simple_sample::{BaseModel, PATIENTS_USIZE}};
use net_ensembles::{dual_graph::*, rand::{SeedableRng, seq::SliceRandom, Rng}, MarkovChain};
use rand_distr::{Uniform, StandardNormal, Distribution, Binomial};
use rand_pcg::Pcg64;
use serde::{Serialize, Deserialize};
use std::num::*;
use net_ensembles::{AdjList, AdjContainer};

const ROTATE: f64 = 0.01;
const PATIENT_MOVE: f64 = 0.03;
const P0_RAND: f64 = 0.04;
const MUTATION_MOVE: f64 = 0.05;
const BY_WHOM: f64 = 0.06; 

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
    pub mutation_vec_dogs: Vec<f64>,
    pub mutation_vec_humans: Vec<f64>,
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
    pub initial_patients: [usize;PATIENTS_USIZE]
}

pub enum Direction
{
    Left,
    Right
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
pub union StepEntry
{
    exchange: ExchangeInfo,
    patient: PatientMove
}

pub struct MarkovStep
{
    pub which: WhichMove,
    pub list_animals_trans: Vec<StepEntry>,
    pub list_humans_trans: Vec<ExchangeInfo>,
    pub list_animals_rec: Vec<ExchangeInfo>,
    pub list_humans_rec: Vec<ExchangeInfo>
}

impl Default for MarkovStep
{
    fn default() -> Self {
        Self { 
            which: WhichMove::ByWhom, 
            list_animals_trans: Vec::new(), 
            list_humans_trans: Vec::new(),
            list_animals_rec: Vec::new(),
            list_humans_rec: Vec::new()
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
                list_humans_rec: Vec::new()
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
pub enum WhichMove
{
    RotateLeftBoth,
    RotateRightBoth,
    RotateLeftAnimal,
    RotateRightAnimal,
    RotateLeftHuman,
    RotateRightHuman,
    PatientMove,
    MutationChange,
    ByWhom,
    TransRec
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

    fn steps_accepted(&mut self, _steps: &[MarkovStep]) {
        
    }

    fn steps_rejected(&mut self, _steps: &[MarkovStep]) {
        
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
                        step.which = WhichMove::RotateLeftAnimal;
                    },
                    Direction::Right => {
                        self.offset_dogs.minus_1();
                        step.which = WhichMove::RotateRightAnimal;
                    }
                }
            } else if which_rotation < 2.0 / 3.0 
            {
                // only human
                match direction
                {
                    Direction::Left => {
                        self.offset_humans.plus_1();
                        step.which = WhichMove::RotateLeftHuman;
                    },
                    Direction::Right => {
                        self.offset_humans.minus_1();
                        step.which = WhichMove::RotateRightHuman;
                    }
                }
            } else {
                // both
                match direction
                {
                    Direction::Left => {
                        self.offset_dogs.plus_1();
                        self.offset_humans.plus_1();
                        step.which = WhichMove::RotateLeftBoth;
                    }, 
                    Direction::Right => {
                        self.offset_dogs.minus_1();
                        self.offset_humans.minus_1();
                        step.which = WhichMove::RotateRightBoth;
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
                        ExchangeInfo { index: rand_index, old_val: old_rec_val }
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
                        ExchangeInfo { index: rand_index, old_val: old_trans_val }
                    );
                    step.list_humans_rec.push(
                        ExchangeInfo { index: rand_index, old_val: old_rec_val }
                    );
                }
            }
        } else if which < MUTATION_MOVE
        {
            step.which = WhichMove::MutationChange;
            let humans = self.dual_graph.graph_2().vertex_count();
            let animals = self.dual_graph.graph_1().vertex_count();
            let index = self.markov_rng.gen_range(0..humans+animals);
            let val: f64 = StandardNormal.sample(&mut self.markov_rng);
            let mut mutation = val * self.sigma;
            if index < humans
            {
                std::mem::swap(&mut mutation, &mut self.mutation_vec_humans[index]);
                step.list_humans_rec.push(
                    ExchangeInfo { index, old_val: mutation }
                );
            } else {
                let index = index - humans;
                std::mem::swap(&mut mutation, &mut self.mutation_vec_dogs[index]);
                step.list_animals_rec.push(
                    ExchangeInfo { index, old_val: mutation }
                );
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
                    ExchangeInfo{
                        index,
                        old_val: by_whom
                    }
                );
            } else {
                let index = index - humans;
                std::mem::swap(&mut by_whom, &mut self.infected_by_whom_dogs[index]);
                step.list_animals_rec.push(
                    ExchangeInfo { index, old_val: by_whom }
                );
            }
        } else {
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

                                ExchangeInfo { index, old_val: old_rec }
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

                                ExchangeInfo{index, old_val: old_trans}
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

                                ExchangeInfo{index, old_val: old_trans}
                            }
                        )   
                );
            }
        }

    }

    fn undo_step_quiet(&mut self, step: &MarkovStep) {
        match step.which
        {
            WhichMove::RotateLeftBoth => {
                self.offset_dogs.minus_1();
                self.offset_humans.minus_1()
            },
            WhichMove::RotateRightBoth => {
                self.offset_dogs.plus_1();
                self.offset_humans.plus_1();
            },
            WhichMove::RotateLeftAnimal => {
                self.offset_dogs.minus_1();
            },
            WhichMove::RotateRightAnimal => {
                self.offset_dogs.plus_1();
            },
            WhichMove::RotateLeftHuman => {
                self.offset_humans.minus_1();
            },
            WhichMove::RotateRightHuman => {
                self.offset_humans.plus_1();
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
                            self.mutation_vec_dogs[entry.index] = entry.old_val;
                        }
                    );
                step.list_humans_rec
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            self.mutation_vec_humans[entry.index] = entry.old_val;
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
                            self.infected_by_whom_dogs[entry.index] = entry.old_val;
                        }
                    );
                step.list_humans_rec
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
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
                            self.recovery_rand_vec_dogs[entry.index] = entry.old_val
                        }
                    );
                step.list_humans_trans
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            self.trans_rand_vec_humans[entry.index] = entry.old_val
                        }
                    );
                step.list_humans_rec
                    .iter()
                    .rev()
                    .for_each(
                        |entry|
                        {
                            self.recovery_rand_vec_humans[entry.index] = entry.old_val;
                        }
                    );
            },
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

        s_normal(&mut self.mutation_vec_dogs);
        s_normal(&mut self.mutation_vec_humans);

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
        let mutation_vec_humans = s_normal(n_humans);

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
            mutation_vec_humans
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
                            let new_gamma = gamma + self.mutation_vec_dogs[index];
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
                            let new_gamma = gamma + self.mutation_vec_dogs[index];
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
                    let mut iter = self.dual_graph.graph_1_contained_iter(index);
                    for node in &mut iter
                    {
                        if node.is_infected()
                        {
                            sum += node.get_gamma_trans().trans_animal;
                            if which <= sum {
                                let gamma = node.get_gamma();
                                let new_gamma = gamma + self.mutation_vec_dogs[index];
                                drop(iter);
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
                            let new_gamma = gamma + self.mutation_vec_humans[index];
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
                            let new_gamma = gamma + self.mutation_vec_humans[index];
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
                    let mut iter = self.dual_graph.graph_2_contained_iter(index);
                    for node in &mut iter
                    {
                        if node.is_infected()
                        {
                            sum += node.get_gamma_trans().trans_human;
                            if which <= sum {
                                let gamma = node.get_gamma();
                                let new_gamma = gamma + self.mutation_vec_humans[index];
                                drop(iter);
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
            .filter(|node| node.is_susceptible())
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
                    break 'scope;
                }
            }
            self.unfinished_sim_counter += 1;
            break 'scope;
        }

        self.dual_graph
            .graph_2()
            .contained_iter()
            .filter(|node| !node.is_susceptible())
            .count()
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
    time_with_offset: usize,
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
}