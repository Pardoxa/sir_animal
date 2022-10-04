use crate::{sir_nodes::*, simple_sample::{BaseModel, PATIENTS_USIZE}};
use net_ensembles::{dual_graph::*, rand::{SeedableRng, seq::SliceRandom}};
use rand_distr::{Uniform, StandardNormal, Distribution, Standard};
use rand_pcg::Pcg64;
use serde::{Serialize, Deserialize};
use std::num::*;
use crate::simple_sample::PATIENTS;
use net_ensembles::{AdjList, AdjContainer};

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
    pub max_time_steps: usize,
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

impl LdModel
{
    pub fn new(mut base: BaseModel, markov_seed: u64, max_sir_steps: usize) -> Self
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


        let trans_rand_vec_humans = collector(n_humans * max_sir_steps);
        let recovery_rand_vec_humans = collector(trans_rand_vec_humans.len());

        let trans_rand_vec_dogs = collector(n_dogs * max_sir_steps);
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
            offset_humans: Offset::new(max_sir_steps, n_humans), 
            offset_dogs: Offset::new(max_sir_steps, n_dogs), 
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
                            let node = self.dual_graph.graph_1_mut().at_mut(index);
                            node.set_gt_and_transition(new_gamma, self.max_lambda);
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
                            let node = self.dual_graph.graph_1_mut().at_mut(index);
                            node.set_gt_and_transition(new_gamma, self.max_lambda);
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
                let mut sum = 0.0;
                
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
                            let node = self.dual_graph.graph_1_mut().at_mut(index);
                            node.set_gt_and_transition(new_gamma, self.max_lambda);
                            break;
                        }
                    }
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
                            let node = self.dual_graph.graph_2_mut().at_mut(index);
                            node.set_gt_and_transition(new_gamma, self.max_lambda);
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
                            let node = self.dual_graph.graph_2_mut().at_mut(index);
                            node.set_gt_and_transition(new_gamma, self.max_lambda);
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
                        sum += node.get_gamma_trans().trans_animal;
                    }
                }
            
                let which = self.infected_by_whom_humans[index] * sum;
                let mut sum = 0.0;
                
                let mut iter = self.dual_graph.graph_2_contained_iter(index);
                for node in &mut iter
                {
                    if node.is_infected()
                    {
                        sum += node.get_gamma_trans().trans_animal;
                        if which <= sum {
                            let gamma = node.get_gamma();
                            let new_gamma = gamma + self.mutation_vec_humans[index];
                            drop(iter);
                            let node = self.dual_graph.graph_2_mut().at_mut(index);
                            node.set_gt_and_transition(new_gamma, self.max_lambda);
                            break;
                        }
                    }
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

    #[allow(clippy::never_loop)]
    pub fn calc_c(&mut self) -> usize
    {
        self.reset_and_infect();
        self.total_sim_counter += 1;
        
        'scope: loop{
            for i in 0..self.max_time_steps
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