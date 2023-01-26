use net_ensembles::{WithGraph, spacial::DogEnsemble, dual_graph::{SingleDualGraph}, rand::seq::SliceRandom, HasRng};
use super::BaseModel;

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
    pub recovery_prob: f64,
    pub human_distance: NonZeroUsize,
    pub sigma: f64,
    pub max_lambda: f64,
    pub initial_gamma: f64,
    pub fun: FunChooser
}

impl BaseOpts
{
    pub fn quick_name(&self) -> String
    {
        format!(
            "v{}{}A{}H{}D{}r{}l{}g{}s{}D{}-{}-{}-{}",
            crate::misc::VERSION,
            self.fun.get_str(),
            self.system_size_dogs,
            self.system_size_humans,
            self.human_distance,
            self.recovery_prob,
            self.max_lambda,
            self.initial_gamma,
            self.sigma,
            self.dog_graph_seed,
            self.human_graph_seed,
            self.dual_seed,
            self.sir_seed
        )
    }

    pub fn construct<T>(&self) -> BaseModel<T>
    where T: Default + 'static + TransFun
    {
        let tid = self.fun.get_type_id();
        if std::any::TypeId::of::<T>() != tid {
            panic!("Type ids do not match!")
        }

        let human_rng = Pcg64::seed_from_u64(self.human_graph_seed);
        let mut ensemble = SmallWorldWS::<EmptyNode, _>::new(
            self.system_size_humans.get() as u32, 
            self.human_distance, 
            self.rewire_prob, 
            human_rng
        ).unwrap();

        while !ensemble.graph().is_connected().unwrap(){
            println!("Initial network was not connected! New try");
            let mut rng = Pcg64::new(0, 0);
            ensemble.swap_rng(&mut rng);
            ensemble = SmallWorldWS::<EmptyNode, _>::new(
                self.system_size_humans.get() as u32, 
                self.human_distance, 
                self.rewire_prob, 
                rng
            ).unwrap();
        }

        let human_graph = ensemble.graph().clone_topology(|_| SirFun::<T>::default());
        drop(ensemble);

        let factor = ((self.system_size_dogs.get() as f64) / 66.0).sqrt();

        let dog_rng = Pcg64::seed_from_u64(self.dog_graph_seed);

        let dog: DogEnsemble<EmptyNode, _> = 
            DogEnsemble::new(self.system_size_dogs.get(), dog_rng, factor * 10.0, 0.7, 7);
        let dog_graph = dog.graph().clone_topology(|_| SirFun::<T>::default());

        let mut dual = SingleDualGraph::new(dog_graph, human_graph);

        let mut dual_rng = Pcg64::seed_from_u64(self.dual_seed);
        let mut nodes_dogs: Vec<_> = (0..self.system_size_dogs.get()).collect();
        let mut nodes_humans: Vec<_> = (0..self.system_size_humans.get()).collect();
        nodes_humans.shuffle(&mut dual_rng);
        nodes_dogs.shuffle(&mut dual_rng);
        let interconnections = (self.system_size_dogs.get() as f64 * 0.85).round() as usize;

        let dogs_with_owners = &nodes_dogs[0..interconnections];
        let owners = &nodes_humans[0..interconnections];

        for (&dog, &human) in dogs_with_owners.iter().zip(owners)
        {
            dual.add_edge(dog, human).unwrap();
        }

        let sir_rng = Pcg64::seed_from_u64(self.sir_seed);

        let possible_patients = (0..dual.graph_1().vertex_count()).collect();

        BaseModel{
            dual_graph: dual,
            reset_gamma: self.initial_gamma,
            sir_rng,
            max_lambda: self.max_lambda,
            sigma: self.sigma,
            initial_gt: T::trans_fun(self.initial_gamma, self.max_lambda),
            recovery_prob: self.recovery_prob,
            infected_list: Vec::new(),
            new_infected_list: Vec::new(),
            possible_patients
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
            recovery_prob: 0.14,
            human_distance: NonZeroUsize::new(8).unwrap(),
            fun: FunChooser::default()
        } 
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SimpleSampleSpecific
{
    pub opts: SimpleSample,
    pub sigma_list: Vec<f64>
}

impl Default for SimpleSampleSpecific
{
    fn default() -> Self {
        Self { opts: SimpleSample::default(), sigma_list: vec![1.0] }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SimpleSampleScan
{
    pub opts: SimpleSample,
    pub sigma_end: f64,
    pub gamma_end: f64,
    pub mut_samples: NonZeroUsize
}

impl Default for SimpleSampleScan
{
    fn default() -> Self {
        SimpleSampleScan {sigma_end: 10.0, mut_samples: NonZeroUsize::new(100).unwrap(), opts: Default::default(), gamma_end: 0.5 }
    }
}


/// Do a simple sampling simulation and get P(C) and Var<C> 
#[derive(Serialize, Deserialize, Clone)]
pub struct SimpleSample{
    pub base_opts: BaseOpts,
    pub samples: NonZeroUsize,
    pub bins: Option<NonZeroUsize>
}

impl SimpleSample
{
    pub fn quick_name(&self) -> String
    {
        let bins = if let Some(b) = self.bins {
            format!("_B{b}")
        } else {
            "".to_owned()
        };
        format!(
            "{}_S{}{bins}",
            self.base_opts.quick_name(),
            self.samples
        )
    }
}

impl Default for SimpleSample{
    fn default() -> Self {
        Self { base_opts: BaseOpts::default(), samples: NonZeroUsize::new(10000).unwrap(), bins: None }
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
