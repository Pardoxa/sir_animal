use net_ensembles::{rand::seq::SliceRandom};

use {
    net_ensembles::{
        dual_graph::*,
        rand::Rng
    },
    crate::sir_nodes::*,
    rand_pcg::Pcg64,
    rand_distr::*
};


pub const PATIENTS: u32 = 1;
pub const PATIENTS_USIZE: usize = PATIENTS as usize;

#[derive(Clone)]
pub struct BaseModel
{
    pub dual_graph: DefaultSDG<SirFun, SirFun>,
    pub reset_gamma: f64,
    pub sir_rng: Pcg64,
    pub recovery_prob: f64,
    pub max_lambda: f64,
    pub sigma: f64,
    pub initial_gt: GammaTrans,
    pub infected_list: Vec<WhichGraph<usize>>,
    pub new_infected_list: Vec<WhichGraph<usize>>,
    pub possible_patients: Vec<usize>
}

impl BaseModel{

    pub fn gamma_iter_animals(&'_ self) -> impl Iterator<Item=f64> + '_
    {
        self.dual_graph.graph_1()
            .contained_iter()
            .filter_map(
                |val|
                {
                    if !val.is_susceptible()
                    {
                        Some(val.get_gamma())
                    } else {
                        None
                    }
                }
            )
    }

    pub fn gamma_iter_humans(&'_ self) -> impl Iterator<Item=f64> + '_
    {
        self.dual_graph.graph_2()
            .contained_iter()
            .filter_map(
                |val|
                {
                    if !val.is_susceptible()
                    {
                        Some(val.get_gamma())
                    } else {
                        None
                    }
                }
            )
    }

    pub fn reset_and_infect_simple(&mut self)
    {
        self.dual_graph
            .graph_1_mut()
            .contained_iter_mut()
            .for_each(
                |val|
                {
                    val.set_s()
                }
            );

        self.dual_graph
            .graph_2_mut()
            .contained_iter_mut()
            .for_each(
                |state|
                {
                    state.set_s()
                }
            );
        assert!(self.infected_list.is_empty());
        let initial_infections = sample_inplace(&mut self.possible_patients, PATIENTS, &mut self.sir_rng);
        self.infected_list.extend(
            initial_infections.iter()
                .map(
                    |&idx|
                    {
                        WhichGraph::Graph1(idx)
                    }
                )
        );
        for &i in initial_infections{
            self.dual_graph.graph_1_mut()
                .at_mut(i)
                .progress_to_i_with_gt(self.initial_gt);
        }
    }

    pub fn iterate_once(&mut self)
    {
        let dist = Uniform::new(0.0, 1.0);
        let gaussian = rand_distr::StandardNormal;
        

        #[inline]
        fn is_sus(which_graph: &WhichGraph<(usize, &mut SirFun)>) -> bool {
            which_graph.1.is_susceptible()
        }

        // iterate over currently infected nodes in random order
        self.infected_list.shuffle(&mut self.sir_rng);
        for which_graph in self.infected_list.iter()
        {
            match which_graph {
                WhichGraph::Graph1(i) => {
                    let gt = self.dual_graph.graph_1().at(*i).get_gamma_trans();
                    let iter =  self
                        .dual_graph
                        .graph_1_contained_iter_mut_which_graph_with_index(*i)
                        .filter(is_sus);
                    for which_graph in iter {
                        let prob = dist.sample(&mut self.sir_rng);
                        match which_graph
                        {
                            WhichGraph::Graph1(node) => {
                                if prob < gt.trans_animal
                                {
                                    let mut new_gamma = gaussian.sample(&mut self.sir_rng);
                                    new_gamma = new_gamma*self.sigma + gt.gamma;
                                    node.1.progress_to_i(new_gamma, self.max_lambda);
                                    self.new_infected_list.push(WhichGraph::Graph1(node.0));
                                }
                            },
                            WhichGraph::Graph2(node) => {
                                if prob < gt.trans_human
                                {
                                    println!("human infected by dog: {}", node.0);
                                    let mut new_gamma = gaussian.sample(&mut self.sir_rng);
                                    new_gamma = new_gamma*self.sigma + gt.gamma;
                                    node.1.progress_to_i(new_gamma, self.max_lambda);
                                    self.new_infected_list.push(WhichGraph::Graph2(node.0));
                                }
                            }
                        }
                    }
                },
                WhichGraph::Graph2(i) => {
                    let gt = self.dual_graph.graph_2().at(*i).get_gamma_trans();
                    let iter =  self
                        .dual_graph
                        .graph_2_contained_iter_mut_which_graph_with_index(*i)
                        .filter(is_sus);
                    for which_graph in iter {
                        let prob = dist.sample(&mut self.sir_rng);
                        match which_graph
                        {
                            WhichGraph::Graph1(node) => {
                                if prob < gt.trans_animal
                                {
                                    let mut new_gamma = gaussian.sample(&mut self.sir_rng);
                                    new_gamma = new_gamma*self.sigma + gt.gamma;
                                    node.1.progress_to_i(new_gamma, self.max_lambda);
                                    self.new_infected_list.push(WhichGraph::Graph1(node.0));
                                }
                            },
                            WhichGraph::Graph2(node) => {
                                if prob < gt.trans_human
                                {
                                    let mut new_gamma = gaussian.sample(&mut self.sir_rng);
                                    new_gamma = new_gamma*self.sigma + gt.gamma;
                                    node.1.progress_to_i(new_gamma, self.max_lambda);
                                    self.new_infected_list.push(WhichGraph::Graph2(node.0));
                                }
                            }
                        }
                    }
                }
            }
        }

        // old infected nodes get a chance to recover
        for (i, prob) in (0..self.infected_list.len()).rev()
            .zip(dist.sample_iter(&mut self.sir_rng))
        {
            if prob < self.recovery_prob
            {
                let which = self.infected_list.swap_remove(i);
                match which{
                    WhichGraph::Graph1(index) => {
                        self.dual_graph.graph_1_mut().at_mut(index).progress_to_r();
                    },
                    WhichGraph::Graph2(index) => {
                        self.dual_graph.graph_2_mut().at_mut(index).progress_to_r();
                    }
                }
            }
        }

        self.infected_list.append(&mut self.new_infected_list);

    }

    pub fn iterate_until_extinction(&mut self)
    {
        self.reset_and_infect_simple();

        loop{
            self.iterate_once();
            if self.infected_list.is_empty(){
                break;
            }
        }
    }

    pub fn count_c_humans(&self) -> usize
    {
        self.dual_graph
            .graph_2()
            .contained_iter()
            .filter(|sir| !sir.is_susceptible())
            .count()
    }

    pub fn count_c_dogs(&self) -> usize
    {
        self.dual_graph
            .graph_1()
            .contained_iter()
            .filter(|sir| !sir.is_susceptible())
            .count()
    }

}

/// Randomly sample exactly `amount` indices from `0..length`, using an inplace
/// partial Fisher-Yates method.
/// Sample an amount of indices using an inplace partial fisher yates method.
///
/// This  randomizes only the first `amount`.
/// It returns the corresponding slice
///
/// This method is not appropriate for large `length` and potentially uses a lot
/// of memory; because of this we only implement for `u32` index (which improves
/// performance in all cases).
///
/// shuffling is `O(amount)` time.
pub(crate) fn sample_inplace<'a, 'b, R>(idxs: &'a mut[usize], amount: u32, rng: &'b mut R) -> &'a [usize]
where R: Rng + ?Sized {
    
    let len = idxs.len() as u32;
    for i in 0..amount {
        let j: u32 = rng.gen_range(i..len);
        idxs.swap(i as usize, j as usize);
    }
    
    &idxs[0..amount as usize]
}