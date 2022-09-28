use net_ensembles::{WithGraph, spacial::DogEnsemble, dual_graph::{DualGraph, SingleDualGraph, self}, rand::seq::SliceRandom};

use{
    std::num::*,
    serde::{Serialize, Deserialize},
    rand_pcg::Pcg64,
    net_ensembles::
    {
        rand::SeedableRng,
        watts_strogatz::SmallWorldWS,
        EmptyNode
    },
    crate::sir_nodes::*,
    structopt::StructOpt
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseOpts
{
    pub dog_graph_seed: u64,
    pub human_graph_seed: u64,
    pub system_size_humans: NonZeroUsize,
    pub system_size_dogs: NonZeroUsize,
    pub dual_seed: u64,
    pub sir_seed: u64,
    pub rewire_prob: f64,
    pub human_distance: NonZeroUsize,
    pub sigma: f64,
    pub max_lambda: f64,
    pub initial_gamma: f64
}

/// Do a simple sampling simulation and get P(C) and Var<C> 
pub struct SimpleSample{
    pub base_opts: BaseOpts,
    pub samples: NonZeroUsize,
}

impl Default for SimpleSample{
    fn default() -> Self {
        Self { base_opts: BaseOpts::default(), samples: NonZeroUsize::new(10000).unwrap() }
    }
}

impl BaseOpts
{
    pub fn construct(&self)
    {
        let human_rng = Pcg64::seed_from_u64(self.human_graph_seed);
        let ensemble = SmallWorldWS::<EmptyNode, _>::new(
            self.system_size_humans.get() as u32, 
            self.human_distance, 
            self.rewire_prob, 
            human_rng
        ).unwrap();

        let human_graph = ensemble.graph().clone_topology(|_| SirFun::default());
        drop(ensemble);

        let factor = ((self.system_size_dogs.get() as f64) / 66.0).sqrt();

        let dog_rng = Pcg64::seed_from_u64(self.dog_graph_seed);

        let dog: DogEnsemble<EmptyNode, _> = 
            DogEnsemble::new(self.system_size_dogs.get(), dog_rng, factor * 10.0, 0.7, 7);
        let dog_graph = dog.graph().clone_topology(|_| SirFun::default());

        let mut dual = SingleDualGraph::new(dog_graph, human_graph);

        let mut dual_rng = Pcg64::seed_from_u64(self.dual_seed);
        let mut nodes_dogs: Vec<_> = (0..self.system_size_dogs.get()).collect();
        let mut nodes_humans: Vec<_> = (0..self.system_size_humans.get()).collect();
        nodes_humans.shuffle(&mut dual_rng);
        nodes_dogs.shuffle(&mut dual_rng);
        let interconnections = (self.system_size_dogs.get() as f64 * 0.85).round() as usize;

        let dogs_with_owners = &nodes_dogs[0..interconnections];
        let owners = &nodes_humans[0..interconnections];

        for &dog in dogs_with_owners{
            for &human in owners{
                dual.add_edge(dog, human).unwrap();
            }
        }

    }
}

impl Default for BaseOpts{
    fn default() -> Self {
        Self { 
            dog_graph_seed: 2389478734, 
            human_graph_seed: 234875023, 
            system_size_humans: NonZeroUsize::new(1320).unwrap(), 
            system_size_dogs: NonZeroUsize::new(66).unwrap(),
            sir_seed: 284798374, 
            sigma: 0.01, 
            max_lambda: 0.2, 
            initial_gamma: 0.0,
            dual_seed: 8932469261,
            rewire_prob: 0.1,
            human_distance: NonZeroUsize::new(8).unwrap()
        } 
    }
}

#[derive(Debug, StructOpt, Clone)]
pub struct DefaultOpts
{
    /// Specify the json file with the options
    /// If not given, an example json will be printed
    #[structopt(long)]
    pub json: Option<String>,

    /// Number of threads to use
    #[structopt(long)]
    pub num_threads: Option<NonZeroUsize>
}
