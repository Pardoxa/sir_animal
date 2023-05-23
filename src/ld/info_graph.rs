use{
    super::{
        InfoNode, 
        InfectedBy, 
        GammaHelper, 
        CondensedInfo,
        TopologyGraph
    },
    crate::sir_nodes::*,
    serde::{Serialize, Deserialize},
    std::{
        num::*
    },
    net_ensembles::{
        traits::*, 
        Graph,
        dual_graph::*,
        EmptyNode,
        graph::NodeContainer
    },
    itertools::Itertools
};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum InfectionHelper
{
    InitialInfected,
    By(u16),
    NotInfected,
    Removed
}

impl From<InfectedBy> for InfectionHelper
{
    fn from(value: InfectedBy) -> Self {
        match value{
            InfectedBy::By(by) => InfectionHelper::By(by),
            InfectedBy::NotInfected => InfectionHelper::NotInfected,
            InfectedBy::InitialInfected => InfectionHelper::InitialInfected
        }
    }
}


#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct HelperNode{
    pub infected_by: InfectionHelper,
    pub gamma_trans: Option<GammaTrans>
}


impl Node for HelperNode
{
    fn new_from_index(_: usize) -> Self {
        unreachable!()
    }
}


#[derive(Serialize, Deserialize)]
pub struct InfoGraph
{
    pub info: Graph<InfoNode>,
    pub disease_children_count: Vec<u32>,
    pub dog_count: usize,
    pub initial_infection: Vec<usize>,
    pub waiting_helper_count: Vec<u32>,
    pub gamma_helper_in_use: Vec<GammaHelper>,
    pub unused_gamma_helper: Vec<GammaHelper>
}

pub struct MutationInfo
{
    pub number_of_jumps: u32,
    pub dogs_prior_to_jump: Option<u16>,
    pub max_mutation: f64,
    pub average_mutation_on_first_infected_path: f64
}

// Functions for analyzing later
impl InfoGraph
{   
    pub fn total_mutation_iter(&'_ self) -> impl Iterator<Item=f64> + '_ 
    {
        self.info.contained_iter()
            .filter_map(
                |node|
                {
                    if let InfectedBy::By(by) = node.infected_by
                    {
                        let gamma_self = node.get_gamma();
                        let gamma_old = self.info.at(by as usize).get_gamma();
                        Some(gamma_self - gamma_old)
                    }else {
                        None
                    }
                }
            )
    }

    pub fn leaf_node_iter(&'_ self) -> impl Iterator<Item=&'_ InfoNode>
    {
        self.info.container_iter()
            .filter_map(
                |container|
                {
                    if container.degree() == 1 {
                        Some(container.contained())
                    } else {
                        None
                    }
                }
            )
    }

    pub fn including_non_infected_nodes_with_descendent_count_iter(&'_ self) -> impl Iterator<Item=(u16, &'_ InfoNode)>
    {
        let mut descendant_count = vec![0_u16; self.info.vertex_count()];

        let mut counted = vec![false; descendant_count.len()];

        for (index, degree) in self.info.degree_iter().enumerate()
        {
            if degree == 1 {
                let mut current_node = self.info.at(index);
                let mut counter = 1;
                counted[index] = true;
                while let InfectedBy::By(by) = current_node.infected_by {
                    let by = by as usize;
                    descendant_count[by] += counter;
                    if !counted[by]{
                        counted[by] = true;
                        counter += 1;
                    }
                    
                    current_node = self.info.at(by);
                }
            }
        }

        descendant_count.into_iter()
            .zip(
                self.info.contained_iter()
            )
        }

    // iterates over nodes and gives child count. Only nodes that were infected will be counted
    pub fn nodes_with_descendent_count_iter(&'_ self) -> impl Iterator<Item=(u16, &'_ InfoNode)>
    {
        self.including_non_infected_nodes_with_descendent_count_iter()
            .filter(|(_, node)| node.was_infected())

    }

    pub fn nodes_with_at_least_n_descendants(&'_ self, n: u16) -> impl Iterator<Item=&'_ InfoNode>
    {
        self.nodes_with_descendent_count_iter()
            .filter_map(
                move |(count, node)|
                {
                    if count >= n {
                        Some(node)
                    } else {
                        None
                    }
                }
            )

    }

    pub fn animal_mutation_iter(&'_ self) -> impl Iterator<Item=f64> + '_ 
    {
        self.info.contained_iter()
            .take(self.dog_count)
            .filter_map(
                |node|
                {
                    if let InfectedBy::By(by) = node.infected_by
                    {
                        let gamma_self = node.get_gamma();
                        let gamma_old = self.info.at(by as usize).get_gamma();
                        Some(gamma_self - gamma_old)
                    }else {
                        None
                    }
                }
            )
    }

    pub fn human_mutation_iter(&'_ self) -> impl Iterator<Item=f64> + '_ 
    {
        self.info.contained_iter()
            .skip(self.dog_count)
            .filter_map(
                |node|
                {
                    if let InfectedBy::By(by) = node.infected_by
                    {
                        let gamma_self = node.get_gamma();
                        let gamma_old = self.info.at(by as usize).get_gamma();
                        Some(gamma_self - gamma_old)
                    } else {
                        None
                    }
                }
            )
    }

    pub fn animal_gamma_iter(&'_ self) -> impl Iterator<Item=f64> + '_
    {
        self.info.contained_iter()
            .take(self.dog_count)
            .filter_map(
                |node|
                {
                    if node.was_infected(){
                        Some(
                            node.get_gamma()
                        )
                    } else {
                        None
                    }
                }
            )
    }

    pub fn animals_infecting_humans_node_iter(&'_ self) -> impl Iterator<Item=&InfoNode> + '_
    {
        self.info
            .container_iter()
            .take(self.dog_count)
            .filter_map(
                |node|
                {
                    if node.contained().was_infected(){
                        // did it infect at least one human?
                        let mut at_least_one = false;
                        for other_node_index in node.neighbors().filter(|&other| *other > self.dog_count)
                        {
                            let other_node = self.info.at(*other_node_index);
                            assert!(other_node.was_infected());
                            at_least_one = true;
                        }
                        if at_least_one
                        {
                            return Some(node.contained());
                        }
                    } 
                    None
                    
                }
            )
    }

    pub fn first_animal_infecting_a_human(&self) -> Option<&InfoNode>
    {
        let mut infection_time = u16::MAX;
        let mut first_animal = None;
        self.info
            .container_iter()
            .take(self.dog_count)
            .for_each(
                |node|
                {
                    if node.contained().was_infected(){
                        for other_node_index in node.neighbors().filter(|&other| *other > self.dog_count)
                        {
                            let other_node = self.info.at(*other_node_index);
                            if let Some(time) = other_node.time_step{
                                if time.get() < infection_time {
                                    infection_time = time.get();
                                    first_animal = Some(node.contained());
                                }
                            }
                        }
                    } 
                    
                }
            );
        first_animal
    }

    pub fn path_from_first_animal_infecting_human_to_root(&'_ self) -> Option<impl Iterator<Item=&'_ InfoNode>>
    {
        let first = self.first_animal_infecting_a_human()?;
               
        let iter = std::iter::successors(
            Some(first), 
            |prev| 
            {
                if let InfectedBy::By(by) = prev.infected_by
                {
                    let node = self.info.at(by as usize);
                    Some(node)
                } else {
                    None
                }
            }
        );
        Some(iter)
    }

    pub fn human_with_most_descendants(&'_ self) -> Option<usize>
    {
        let mut max_human_count = 0;
        let mut max_human_index = None;
        for (index, (count, _)) in self.including_non_infected_nodes_with_descendent_count_iter().enumerate().skip(self.dog_count)
        {
            if count > max_human_count
            {
                max_human_count = count;
                max_human_index = Some(index);
            }
        }
        max_human_index
    }

    pub fn path_from_human_with_most_descendants_to_root(&'_ self) -> Option<impl Iterator<Item=(usize, &InfoNode)> + '_>
    {
        let human_index = self.human_with_most_descendants()?;

        let iter = std::iter::successors(
            Some(human_index),
            |idx| {
                let node = self.info.at(*idx);
                if let InfectedBy::By(by) = node.infected_by
                {
                    Some(by as usize)
                } else {
                    None
                }
            }
        ).map(
            |idx|
            {
                let node = self.info.at(idx);
                (idx, node)
            }
        );
        Some(iter)
    } 

    pub fn iter_gamma_change_from_animal_that_infects_human_with_most_descendants_to_root(&'_ self) -> Option<impl Iterator<Item=f64> + '_>
    {
        let iter = self.path_from_human_with_most_descendants_to_root()?
            .skip(1)
            .tuple_windows::<(_,_)>()
            .map(
                |(child, parent)|
                {
                    let child_gamma = child.1.get_gamma();
                    let parent_gamma = parent.1.get_gamma();
                    child_gamma - parent_gamma
                }
            );
        Some(iter)
    }

    pub fn iter_lambda_change_from_animal_that_infects_human_with_most_descendants_to_root(&'_ self) -> Option<impl Iterator<Item=f64> + '_>
    {
        let iter = self.path_from_human_with_most_descendants_to_root()?
            .skip(1)
            .tuple_windows::<(_,_)>()
            .map(
                |(child, parent)|
                {
                    let child_lambda = child.1.get_lambda_human();
                    let parent_lambda = parent.1.get_lambda_human();
                    child_lambda - parent_lambda
                }
            );
        Some(iter)
    }

    pub fn lambda_changes_human_human_transmission(&'_ self) -> impl Iterator<Item=f64> + '_
    {
        let root = self.initial_infection[0];

        self.info.bfs_index_depth(root)
            .filter_map(
                |(index, node, _)|
                {
                    if index < self.dog_count
                    {
                        None
                    } else if let InfectedBy::By(by) = node.infected_by {
                        let by = by as usize;
                        if by < self.dog_count {
                            None
                        } else {
                            let parent = self.info.at(by);
                            Some(
                                node.get_lambda_human() - parent.get_lambda_human()
                            )
                        }
                    } else {
                        None
                    }
                }
            )
    }

    pub fn path_from_first_animal_infecting_human_to_root_mutation_iter(&'_ self) -> impl Iterator<Item=f64> + '_
    {
        let first = self.first_animal_infecting_a_human();
        let mut iter = std::iter::successors(
            first, 
            |prev| 
            {
                if let InfectedBy::By(by) = prev.infected_by
                {
                    let node = self.info.at(by as usize);
                    Some(node)
                } else {
                    None
                }
            }
        );
        let mut newer_gamma = iter.next().map(|node| node.get_gamma()).unwrap_or(f64::NAN);
        iter.map(
            move |node|
            {
                let older_gamma = node.get_gamma();
                let mutation = newer_gamma - older_gamma;
                newer_gamma = older_gamma;
                mutation
            }
        )
    }

    pub fn humans_infected_by_animals(&'_ self) -> impl Iterator<Item=&InfoNode> + '_
    {
        self.info
            .contained_iter()
            .skip(self.dog_count)
            .filter(
                |node|
                {
                    if let InfectedBy::By(by) = node.infected_by
                    {
                        (by as usize) < self.dog_count
                    } else {
                        false
                    }
                    
                }
            )
    }

    pub fn humans_infected_by_animals_info_node_and_global_node<'a>(&'a self, global: &'a TopologyGraph) -> impl Iterator<Item=(&InfoNode, &NodeContainer<EmptyNode>)> + 'a
    {
        self.info
            .contained_iter()
            .skip(self.dog_count)
            .zip(0..)
            .filter_map(
                |(node, index)|
                {
                    if let InfectedBy::By(by) = node.infected_by
                    {
                        if (by as usize) < self.dog_count
                        {
                            let global_node = global.container(index);
                            Some((node, global_node))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                    
                }
            )
    }

    pub fn animal_gamma_trans_iter(&'_ self) -> impl Iterator<Item=GammaTrans> + '_
    {
        self.info.contained_iter()
            .take(self.dog_count)
            .filter_map(
                |node|
                {
                    if node.was_infected(){
                        Some(
                            node.gamma_trans.unwrap()
                        )
                    } else {
                        None
                    }
                }
            )
    }


    pub fn human_gamma_trans_iter(&'_ self) -> impl Iterator<Item=GammaTrans> + '_
    {
        self.info.contained_iter()
            .skip(self.dog_count)
            .filter_map(
                |node|
                {
                    if node.was_infected(){
                        Some(
                            node.gamma_trans.unwrap()
                        )
                    } else {
                        None
                    }
                }
            )
    }

    pub fn human_gamma_iter(&'_ self) -> impl Iterator<Item=f64> + '_
    {
        self.info.contained_iter()
            .skip(self.dog_count)
            .filter_map(
                |node|
                {
                    if node.was_infected(){
                        Some(
                            node.get_gamma()
                        )
                    } else {
                        None
                    }
                }
            )
    }

    pub fn human_node_iter(&'_ self) -> impl Iterator<Item=&InfoNode> + '_
    {
        self.info.contained_iter()
            .skip(self.dog_count)
    }




    pub fn iter_nodes_and_mutation_child_count_unfiltered(&'_ self, max_mutation_distance: f64) -> impl Iterator<Item=(usize, &InfoNode)> + '_
    {
        let mut child_count = vec![0; self.info.vertex_count()];
        let mut gamma_list = Vec::new();
        let mut already_counted = vec![false; self.info.vertex_count()];

        struct CountHelper{
            gamma: f64,
            already_counted: bool
        }

        let mut initial = 0;
        let mut any = false;


        for (index, degree) in self.info.degree_iter().enumerate()
        {
            let mut current_node = self.info.at(index);
            if matches!(current_node.infected_by, InfectedBy::InitialInfected)
            {
                initial = index;
                continue;
            }
            if degree == 1 {
                gamma_list.clear();
                any = true;
                
                let mut current_index = index;
                loop {
                   
                    gamma_list.push(
                        CountHelper{
                            gamma: current_node.get_gamma(),
                            already_counted: already_counted[current_index]
                        }
                    );
                    already_counted[current_index] = true;
                    
                    let current_gamma = current_node.get_gamma();
                    // I want to remove all gamma that where earlier than the first one that violates the condition
                    let right_most_pos = gamma_list.iter().rposition(|this| (this.gamma - current_gamma).abs() > max_mutation_distance);
                    if let Some(pos) = right_most_pos
                    {
                        gamma_list.drain(..=pos);
                    }
                    
                    child_count[current_index] += gamma_list.iter().filter(|helper| !helper.already_counted).count();
                    if let InfectedBy::By(by) = current_node.infected_by {
                        current_node = self.info.at(by as usize);
                        current_index = by as usize;
                    } else {
                        break;
                    }
                }
            }
        }
        if !any {
            child_count[initial] += 1;
        }

        child_count.into_iter()
            .zip(self.info.contained_iter())
    }

    pub fn get_mutation_component(&self, max_mutation_distance: f64, start: usize) -> Vec<usize> 
    {
        // contains min and max encountered through path as well as id of current node
        #[derive(Clone)]
        struct IdMinMax{
            min: f64,
            max: f64,
            id: usize
        }

        let mut used = vec![false; self.info.vertex_count()];
        let mut stack = Vec::new();
        let mut removed_component = Vec::new();
        let container = self.info.container(start);
        if let InfectedBy::By(by) = container.contained().infected_by
        {
            used[by as usize] = true;
        }
        let root_gamma = container.contained().get_gamma();
        let initial = IdMinMax{
            max: root_gamma,
            min: root_gamma,
            id: start
        };
        stack.push(initial);
        
        while let Some(top) = stack.pop() {
            used[top.id] = true;
            removed_component.push(top.id);
            let container = self.info.container(top.id);
            for &e in container.edges()
            {
                if used[e] {
                    continue;
                }
                let neighbor_gamma = self.info.at(e).get_gamma();
                let distance1 = (neighbor_gamma - top.max).abs();
                let distance2 = (neighbor_gamma - top.min).abs();
                let distance = distance1.max(distance2);
                let mut for_stack = top.clone();
                if neighbor_gamma > for_stack.max {
                    for_stack.max = neighbor_gamma;
                } else if neighbor_gamma < for_stack.min 
                {
                    for_stack.min = neighbor_gamma;
                }
                if distance <= max_mutation_distance {
                    for_stack.id = e;
                    stack.push(for_stack);
                }
            }
        }
        removed_component
    }

    pub fn largest_and_second_largest_given_mutation(self, max_mutation_distance: f64, only_humans: bool) -> (usize, usize)
    {

        let mut largest_component_id = 0;
        let mut largest_component_size = 0;
        let skip = if only_humans{
            self.dog_count
        } else {
            0
        };
        self.iter_nodes_and_mutation_child_count_unfiltered(max_mutation_distance)
            .enumerate()
            .skip(skip)
            .for_each(
                |(id, (size, _))|
                {
                    if size > largest_component_size {
                        largest_component_size = size;
                        largest_component_id = id;
                    }
                }
            );

        let removed_component = if only_humans && largest_component_size == 0{
            Vec::new()
        } else {
            self.get_mutation_component(max_mutation_distance, largest_component_id)
        };

        assert_eq!(removed_component.len(), largest_component_size);

        // helper topology
        let mut topology = self.info.clone_topology(
            |old: &InfoNode| 
            {
                let inf = old.infected_by.clone().into();
                HelperNode{
                    infected_by: inf,
                    gamma_trans: old.gamma_trans
                }
            }
        );

        // set all nodes that are in largest component to removed
        for &node in removed_component.iter()
        {
            topology.at_mut(node).infected_by = InfectionHelper::Removed;
        }

        

        let mut child_count = vec![0; self.info.vertex_count()];
        let mut gamma_list = Vec::new();
        let mut already_counted = vec![false; self.info.vertex_count()];

        struct CountHelper{
            gamma: f64,
            already_counted: bool
        }

        let mut leaf_count = 0;

        for (index, container) in topology.container_iter().enumerate()
        {
            // check if this is a leaf in the new graph:
            if matches!(container.contained().infected_by, InfectionHelper::NotInfected | InfectionHelper::Removed){
                continue;
            }
            let is_leaf = if let InfectionHelper::By(this_parent) = container.contained().infected_by
            {
                // has parent
                container
                    .edges()
                    .iter()
                    .all(
                        |neighbor|
                        {
                            this_parent == *neighbor as u16 || matches!(topology.at(*neighbor).infected_by, InfectionHelper::Removed)
                        }
                    )
            } else {
                // does not have parent
                container
                    .edges()
                    .iter()
                    .all(
                        |neighbor|
                        {
                            matches!(topology.at(*neighbor).infected_by, InfectionHelper::Removed)
                        }
                    )
            };
            
            if is_leaf {
                leaf_count += 1;
                let mut current_node = topology.at(index);
                gamma_list.clear();
                
                let mut current_index = index;
                loop {
                    //let InfectedBy::By(by) = current_node.infected_by
                    gamma_list.push(
                        CountHelper{
                            gamma: current_node.gamma_trans.unwrap().gamma,
                            already_counted: already_counted[current_index]
                        }
                    );
                    already_counted[current_index] = true;
                    
                    let current_gamma = current_node.gamma_trans.unwrap().gamma;
                    // I want to remove all gamma that where earlier than the first one that violates the condition
                    let right_most_pos = gamma_list.iter().rposition(|this| (this.gamma - current_gamma).abs() > max_mutation_distance);
                    if let Some(pos) = right_most_pos
                    {
                        gamma_list.drain(..=pos);
                    }
                    
                    child_count[current_index] += gamma_list.iter().filter(|helper| !helper.already_counted).count();
                    if let InfectionHelper::By(by) = current_node.infected_by {
                        current_node = topology.at(by as usize);
                        current_index = by as usize;
                    } else {
                        break;
                    }
                }
            }
        }
        if leaf_count == 0 {
            let total = self.info.contained_iter().filter(|node| node.was_infected()).count();
            if largest_component_size != total {
                println!("largest: {largest_component_size} of {total}");
                panic!("unreasonable");
            }
        }
        let mut second_largest = 0;
        for &count in child_count.iter().skip(skip){
            if count > second_largest 
            {
                second_largest = count;
            }
        }
        assert!(largest_component_size >= second_largest);
        (largest_component_size, second_largest)
    }

    pub fn iter_nodes_and_mutation_child_count(&'_ self, max_mutation_distance: f64) -> impl Iterator<Item=(usize, &InfoNode)> + '_
    {
        self.iter_nodes_and_mutation_child_count_unfiltered(max_mutation_distance)
            .filter(|(_, node)| node.was_infected())
    }

    pub fn iter_human_nodes_and_child_count(&'_ self) -> impl Iterator<Item=(usize, &InfoNode)> + '_
    {
        let mut child_count = vec![0; self.info.vertex_count()];
        let mut gamma_list = Vec::new();
        let mut already_counted = vec![false; self.info.vertex_count()];

        struct CountHelper{
            already_counted: bool
        }

        for (index, degree) in self.info.degree_iter().enumerate()
        {
            if degree == 1 {
                gamma_list.clear();
                let mut current_node = self.info.at(index);
                let mut current_index = index;
                while let InfectedBy::By(by) = current_node.infected_by {
                    gamma_list.push(
                        CountHelper{
                            already_counted: already_counted[current_index]
                        }
                    );
                    if !already_counted[current_index]
                    {
                        already_counted[current_index] = true;

                    }
                    let previous_node = self.info.at(by as usize);
                    
                    child_count[by as usize] += gamma_list.iter().filter(|helper| !helper.already_counted).count();
                    current_node = previous_node;
                    current_index = by as usize;
                }
            }
        }

        child_count.into_iter()
            .zip(self.info.contained_iter())
            .skip(self.dog_count)
            .filter(|(_, node)| node.was_infected())
    }

    // no other human is allowed in path of nodes! Animals are not counted!
    pub fn iter_human_nodes_and_child_count_of_first_infected_humans(&'_ self) -> impl Iterator<Item=(usize, &InfoNode)> + '_
    {
        let mut child_count = vec![0; self.info.vertex_count()];
        let mut gamma_list = Vec::new();
        let mut already_counted = vec![false; self.info.vertex_count()];

        struct CountHelper{
            already_counted: bool
        }
        let dog_count = self.dog_count;
        let mut to_check_list = Vec::new();

        //already_counted.iter_mut().take(dog_count).for_each(|val| *val = true);

        for (index, degree) in self.info.degree_iter().enumerate()
        {
            let mut current_node = self.info.at(index);
            if index > dog_count
            {
                if let InfectedBy::By(by) = current_node.infected_by
                {
                    if (by as usize) < dog_count
                    {
                        to_check_list.push(index);
                    }
                }
            }
            if degree == 1 {
                gamma_list.clear();
                
                let mut current_index = index;
                

                while let InfectedBy::By(by) = current_node.infected_by {
                    gamma_list.push(
                        CountHelper{
                            already_counted: already_counted[current_index]
                        }
                    );
                    if !already_counted[current_index]
                    {
                        already_counted[current_index] = true;

                    }
                    let by = by as usize;
                    let previous_node = self.info.at(by);
                    
                    child_count[by] += gamma_list.iter().filter(|helper| !helper.already_counted).count();
                    current_node = previous_node;
                    current_index = by;
                }
            }
        }
        for i in (0..to_check_list.len()).rev()
        {
            let index = to_check_list[i];
            let mut node = self.info.at(index);
            let mut is_fine = true;
            while let InfectedBy::By(by) = node.infected_by
            {
                let by = by as usize;
                if by > dog_count
                {
                    is_fine = false;
                    break;
                }
                node = self.info.at(by);
            }
            if !is_fine
            {
                to_check_list.swap_remove(i);
            }
        }
        //dbg!(&to_check_list);
        to_check_list
            .into_iter()
            .map(move |i| (child_count[i], self.info.at(i)))
    }

    pub fn dog_mutations(&self) -> MutationInfo
    {
        assert_eq!(self.initial_infection.len(), 1);
        let mut number_of_jumps = 0;
        let mut dogs_prior_to_jump= None;
        let dog_count_u16 = self.dog_count as u16;
        
        let humans = &self.info.get_vertices()[self.dog_count..];

        let mut idx_first_human = None;

        for (idx, human) in humans.iter().enumerate() {
            let contained = human.contained();
            match &contained.infected_by
            {
                InfectedBy::NotInfected => continue,
                InfectedBy::InitialInfected => {
                    unreachable!()
                },
                InfectedBy::By(by) => {
                    if *by < dog_count_u16 {
                        number_of_jumps += 1;
                        let prev = contained.prev_dogs;
                        if let Some(old) = dogs_prior_to_jump
                        {
                            if old < prev
                            {
                                dogs_prior_to_jump = Some(prev);
                                idx_first_human = Some(idx);
                            }
                        } else {
                            dogs_prior_to_jump = Some(prev);
                            idx_first_human = Some(idx);
                        }
                        
                    }
                }
            }
        }

        let mut max_mutation = f64::NAN;
        let mut average_mutation = f64::NAN;

        if let Some(idx) = idx_first_human
        {
            let mut idx = idx + self.dog_count;
            max_mutation = f64::NEG_INFINITY;
            average_mutation = 0.0;
            let mut counter = 0_u32;
            loop{
                let node = self.info.at(idx);
                let gamma = node.get_gamma();
                match node.infected_by{
                    InfectedBy::NotInfected => {
                        unreachable!()
                    },
                    InfectedBy::InitialInfected => {
                        break;
                    },
                    InfectedBy::By(by) => {
                        idx = by as usize;
                        let old_node = self.info.at(idx);
                        let old_gamma = old_node.get_gamma();
                        let mutation = gamma - old_gamma;
                        average_mutation += mutation;
                        max_mutation = max_mutation.max(mutation);
                        counter += 1;
                    }
                }
            }
            average_mutation /= counter as f64;
        }
        MutationInfo { 
            number_of_jumps, 
            dogs_prior_to_jump,
            max_mutation, 
            average_mutation_on_first_infected_path: average_mutation 
        }
    }
}

impl InfoGraph
{
    pub fn create_condensed_info(&self) -> CondensedInfo
    {
        let (nodes, indices) = self.info.get_vertices().iter().zip(0..)
            .filter_map(
                |(node, index)|
                {
                    let contained = node.contained();
                    if matches!(contained.infected_by, InfectedBy::NotInfected)
                    {
                        None
                    } else {
                        Some((contained.clone(), index))
                    }
                }
            ).unzip();
        let initial_infected = self.initial_infection[0];
        CondensedInfo { indices, nodes, dogs: self.dog_count as u16, total: self.info.vertex_count(), initial_infected }
    }

    pub fn set_gamma_trans<T>(&mut self, other: &DefaultSDG<SirFun<T>, SirFun<T>>)
    where SirFun<T>: Node
    {
        self.info.contained_iter_mut()
            .zip(other.graph_1().contained_iter().chain(other.graph_2().contained_iter()))
            .for_each(
                |(this, other)|
                {
                    if this.layer.is_some()
                    {
                        this.gamma_trans = Some(unsafe{other.fun_state.gamma});
                    }
                }
            )
    }

    

    pub fn new(dogs: usize, humans: usize) -> Self
    {
        let info = Graph::new(dogs + humans);
        Self{
            info, 
            dog_count: dogs,
            disease_children_count: vec![0; dogs + humans],
            initial_infection: Vec::new(),
            waiting_helper_count: vec![0; dogs + humans],
            gamma_helper_in_use: Vec::new(),
            unused_gamma_helper: Vec::new()
        }
    }

    pub fn reset(&mut self)
    {
        self.unused_gamma_helper.append(&mut self.gamma_helper_in_use);
        self.initial_infection.clear();

        self.disease_children_count
            .iter_mut()
            .for_each(
                |val|
                {
                    *val = 0
                }
            );
        self.waiting_helper_count
            .iter_mut()
            .for_each(|val| *val = 0);
        self.info.clear_edges();
        self.info.contained_iter_mut()
            .for_each(
                |node|
                {
                    node.time_step = None;
                    node.layer = None;
                    node.infected_by = InfectedBy::NotInfected;
                    node.gamma_trans = None;
                    node.recovery_time = None;
                }
            );
    }

    pub fn get_human_index(&self, human: usize) -> usize
    {
        self.dog_count + human
    }

    pub fn recover_dog(&mut self, dog_id: usize, time: NonZeroU16)
    {
        self.info.at_mut(dog_id).recovery_time = Some(time);
    }

    pub fn recover_human(&mut self, human_index: usize, time: NonZeroU16)
    {
        let id = human_index + self.dog_count;
        self.info.at_mut(id).recovery_time = Some(time);
    }

    pub fn a_infects_b(&mut self, a: usize, b: usize, time_step: NonZeroU16)
    {
        
        let res = self.info.add_edge(a, b);
        assert!(res.is_ok());
        let node_a_clone = self.info.at(a).clone();
        assert!(!matches!(node_a_clone.infected_by, InfectedBy::NotInfected));
        
        let layer_p1 = node_a_clone.layer.unwrap().saturating_add(1);
        let node_b = self.info.at_mut(b);
        node_b.layer = Some(layer_p1);
        node_b.infected_by = InfectedBy::By(a as u16);
        node_b.time_step = Some(time_step);
        let mut prev_dogs = node_a_clone.prev_dogs;
        if a < self.dog_count
        {
            prev_dogs += 1;
        }
        node_b.prev_dogs = prev_dogs;

    }

    pub fn initial_infection(&mut self, index: usize)
    {
        self.initial_infection.push(index);
        let init = self.info.at_mut(index);
        init.time_step= NonZeroU16::new(1);
        init.layer = NonZeroU16::new(1);
        init.infected_by = InfectedBy::InitialInfected;
        init.prev_dogs = 0;
    }
}