use super::*;

pub(crate) fn c_and_average_human_gamma(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut average = 0.0;
    let mut count = 0_u32;
    for gamma in item.1.human_gamma_iter()
    {
        count += 1;
        average += gamma;
    }
    (item.0, average / count as f64)
}


pub(crate) fn c_and_max_human_gamma(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_gamma = f64::NEG_INFINITY;
    for gamma in item.1.human_gamma_iter()
    {
        if max_gamma < gamma {
            max_gamma = gamma;
        }
    }
    if max_gamma == f64::NEG_INFINITY {
        max_gamma = f64::NAN;
    }
    (item.0, max_gamma)
}

pub(crate) fn c_and_average_positive_mutation_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for mutation in item.1.human_mutation_iter()
    {
        if mutation > 0.0 {
            sum += mutation;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

pub(crate) fn c_and_average_negative_mutation_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for mutation in item.1.human_mutation_iter()
    {
        if mutation < 0.0 {
            sum += mutation;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

pub(crate) fn c_and_average_mutation_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for mutation in item.1.human_mutation_iter()
    {
        
        sum += mutation;
        count += 1;
        
    }
    (item.0, sum / count as f64)
}

pub(crate) fn c_and_all_mutations_humans(item: &'_ (usize, InfoGraph)) -> (usize, Box<dyn Iterator<Item=f64> + '_>)
{
    let iter = item.1.human_mutation_iter();
    (item.0, Box::new(iter))
}

pub(crate) fn c_and_max_human_human_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_lambda = f64::NEG_INFINITY;
    for gt in item.1.human_gamma_trans_iter()
    {
        let lambda = gt.trans_human;
        if max_lambda < lambda {
            max_lambda = lambda;
        }
    }
    if max_lambda == f64::NEG_INFINITY {
        max_lambda = f64::NAN;
    }
    (item.0, max_lambda)
}

// maybe create a integer heatmap for this one if the results look interesting
pub(crate) fn c_and_previous_dogs_of_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut prev = 0;
    for human_node in item.1.human_node_iter()
    {
        if prev < human_node.prev_dogs{
            prev = human_node.prev_dogs;
        }
    }
    (item.0, prev as f64)
}

pub(crate) fn c_and_median_human_human_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut median_list: Vec<_> = item.1.human_gamma_trans_iter()
        .map(|val| val.trans_human)
        .collect();
    median_list.sort_unstable_by(f64::total_cmp);
    let median = if median_list.is_empty()
    {
        f64::NAN
    } else {
        let len = median_list.len();
        median_list[len / 2]
    };
    (item.0, median)
}

pub(crate) fn c_and_average_prev_dogs_of_humans_infected_by_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0;
    for human_node in item.1.humans_infected_by_animals()
    {
        let prev = human_node.prev_dogs;
        sum += prev as f64;
        count += 1;
    }
    (item.0, sum / count as f64)
}

pub(crate) fn c_and_frac_max_children_of_humans_infected_by_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_child_count = 0;
    for (count, node) in item.1.iter_human_nodes_and_child_count()
    {
        if let InfectedBy::By(by) = node.infected_by
        {
            if (by as usize) < item.1.dog_count && count > max_child_count
            {
                max_child_count = count;
            }
        }
    }
    (item.0, max_child_count as f64/ item.0 as f64)
}

pub(crate) fn c_and_average_human_human_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut average = 0.0;
    let mut count = 0_u32;
    for gt in item.1.human_gamma_trans_iter()
    {
        let lambda = gt.trans_human;
        count += 1;
        average += lambda;
    }
    (item.0, average / count as f64)
}

pub(crate) fn c_and_average_human_animal_lambda(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut average = 0.0;
    let mut count = 0_u32;
    for gt in item.1.human_gamma_trans_iter()
    {
        let lambda = gt.trans_animal;
        count += 1;
        average += lambda;
    }
    (item.0, average / count as f64)
}

pub(crate) fn c_and_max_children_of_humans_infected_by_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut max_child_count = 0;
    for (count, node) in item.1.iter_human_nodes_and_child_count()
    {
        if let InfectedBy::By(by) = node.infected_by
        {
            if (by as usize) < item.1.dog_count && count > max_child_count
            {
                max_child_count = count;
            }
        }
    }
    (item.0, max_child_count as f64)
}

pub(crate) fn c_and_second_largest_children_of_humans_infected_by_animals(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut child_count = Vec::new();
    for (count, node) in item.1.iter_human_nodes_and_child_count()
    {
        if let InfectedBy::By(by) = node.infected_by
        {
            if (by as usize) < item.1.dog_count
            {
                child_count.push(count);
            }
        }
    }

    let count = if child_count.len() >= 2 {
        child_count.sort_unstable_by_key(|item| std::cmp::Reverse(*item));
        child_count[1]
    } else {
        0
    };

    (item.0, count as f64)
}

pub(crate) fn c_and_second_largest_children_of_humans_infected_by_animals_strict(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut child_count = Vec::new();
    for (count, node) in item.1.iter_human_nodes_and_child_count_of_first_infected_humans()
    {
        if let InfectedBy::By(by) = node.infected_by
        {
            assert!((by as usize) < item.1.dog_count);
            child_count.push(count);
        }
    }

    let count = if child_count.len() >= 2 {
        child_count.sort_unstable_by_key(|item| std::cmp::Reverse(*item));
        //dbg!(&child_count);
        child_count[1]
    } else {
        0
    };

    (item.0, count as f64)
}

pub(crate) fn max_tree_width_div_total_humans_only(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut current_width = 0_u32;
    let mut current_depth = 0;
    let mut max_width = 0;
    let root = item.1.initial_infection[0];
    let mut total_number_of_nodes = 0_u32;
    for (index, _, depth) in item.1.info.bfs_index_depth(root)
    {
        if index < item.1.dog_count{
            continue;
        }
        if depth == current_depth{
            current_width += 1;
        } else {
            if max_width < current_width {
                max_width = current_width;
            }
            current_depth = depth;
            current_width = 1;
        }
        total_number_of_nodes += 1;
    }
    if current_width > max_width {
        max_width = current_width;
    }
    (item.0, max_width as f64 / total_number_of_nodes as f64)
}

pub(crate) fn av_positive_lambda_change_human_human_trans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum_positives = 0.0;
    let mut count = 0_u32;

    for lambda in item.1.lambda_changes_human_human_transmission()
    {
        if lambda > 0.0 {
            count += 1;
            sum_positives += lambda;
        }
    }
    (item.0, sum_positives / count as f64)
}

pub(crate) fn av_negative_lambda_change_human_human_trans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum_negatives = 0.0;
    let mut count = 0_u32;

    for lambda in item.1.lambda_changes_human_human_transmission()
    {
        if lambda < 0.0 {
            count += 1;
            sum_negatives += lambda;
        }
    }
    (item.0, sum_negatives / count as f64)
}

pub(crate) fn frac_negative_lambda_change_human_human_trans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut count_negatives = 0_u32;
    let mut count = 0_u32;

    for lambda in item.1.lambda_changes_human_human_transmission()
    {
        count += 1;
        if lambda < 0.0 {
            count_negatives += 1;
        }
    }
    (item.0, count_negatives as f64 / count as f64)
}

pub(crate) fn frac_negative_gamma_change_human_human_trans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut count_negatives = 0_u32;
    let mut count = 0_u32;

    for gamma in item.1.gamma_changes_human_human_transmission()
    {
        count += 1;
        if gamma < 0.0 {
            count_negatives += 1;
        }
    }
    (item.0, count_negatives as f64 / count as f64)
}

pub(crate) fn frac_human_gamma_larger_first_min(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut count_above_min = 0_u32;
    let mut count = 0_u32;
    
    
    for gamma in item.1.human_gamma_iter()
    {
        let other_g = gamma - 2.4214957;
        if other_g > -0.663
        {
            count_above_min +=1;
        }
        count +=1;
    }
    (item.0, count_above_min as f64 / count as f64)
}
pub(crate) fn frac_human_gamma_larger_first_max(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut count_above_min = 0_u32;
    let mut count = 0_u32;
    
    
    for gamma in item.1.human_gamma_iter()
    {
        let other_g = gamma - 2.4214957;
        if other_g > -1.00728
        {
            count_above_min +=1;
        }
        count +=1;
    }
    (item.0, count_above_min as f64 / count as f64)
}

pub(crate) fn frac_human_gamma_close_to_first_max(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut close_count = 0_u32;
    let mut count = 0_u32;
    
    
    for gamma in item.1.human_gamma_iter()
    {
        let other_g = gamma - 2.4214957;
        let dist = (other_g- -1.00728).abs();
        if dist < 0.02
        {
            close_count +=1;
        }
        count +=1;
    }
    (item.0, close_count as f64 / count as f64)
}

pub(crate) fn median_human_gamma(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut list = Vec::new();
    for gamma in item.1.human_gamma_iter()
    {
        let other_g = gamma - 2.4214957;
        list.push(other_g);
        
    }

    list.sort_unstable_by(f64::total_cmp);

    let median = if list.is_empty(){
        f64::NAN
    } else {
        list[list.len()/2]
    };
    

    (item.0, median)
}

pub(crate) fn av_lambda_change_human_human_trans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for lambda in item.1.lambda_changes_human_human_transmission()
    {
        sum += lambda;
        count += 1;
    }
    (item.0, sum / count as f64)
}

pub(crate) fn c_and_average_recovery_time_humans(item: (usize, InfoGraph)) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;
    for node in item.1.info.contained_iter().skip(item.1.dog_count)
    {
        if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
        {
            sum += (recovery.get() - infection.get()) as f64;
            count += 1;
        }
    }
    (item.0, sum / count as f64)
}

pub(crate) fn c_and_human_average_recovery_time_exact_descendants(item: (usize, InfoGraph), desired_descendants: u16) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    for (descendants, node) in item.1.including_non_infected_nodes_with_descendent_count_iter().skip(item.1.dog_count)
    {
        if node.was_infected() && descendants == desired_descendants{
            if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
            {
                sum += (recovery.get() - infection.get()) as f64;
                count += 1;
            }
        }

    }
    let res = sum / count as f64;
    (item.0, res)
}


pub(crate) fn c_and_human_average_recovery_time_exact_children(item: (usize, InfoGraph), desired_descendants: u16) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    let desired_descendants = desired_descendants as usize;

    for container in item.1.info.container_iter().skip(item.1.dog_count)
    {
        let node = container.contained();
        let des_count = container.edges().len().saturating_sub(1);
        if node.was_infected() && des_count == desired_descendants{
            if let (Some(recovery), Some(infection)) = (node.recovery_time, node.time_step)
            {
                sum += (recovery.get() - infection.get()) as f64;
                count += 1;
            }
        }

    }
    let res = sum / count as f64;
    (item.0, res)
}

pub(crate) fn c_and_human_average_human_lambda_exact_children(item: (usize, InfoGraph), desired_descendants: u16) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    let desired_descendants = desired_descendants as usize;

    for container in item.1.info.container_iter().skip(item.1.dog_count)
    {
        let node = container.contained();
        let des_count = container.edges().len().saturating_sub(1);
        if node.was_infected() && des_count == desired_descendants{
            sum += node.get_lambda_human();
            count += 1;
        }

    }
    let res = sum / count as f64;
    (item.0, res)
}

pub(crate) fn c_and_human_average_animal_lambda_exact_children(item: (usize, InfoGraph), desired_descendants: u16) -> (usize, f64)
{
    let mut sum = 0.0;
    let mut count = 0_u32;

    let desired_descendants = desired_descendants as usize;

    for container in item.1.info.container_iter().skip(item.1.dog_count)
    {
        let node = container.contained();
        let des_count = container.edges().len().saturating_sub(1);
        if node.was_infected() && des_count == desired_descendants{
            sum += node.get_lambda_dog();
            count += 1;
        }

    }
    let res = sum / count as f64;
    (item.0, res)
}


pub(crate) fn c_and_human_max_children_give_mutation(item: (usize, InfoGraph), _: u16, mutation: f64) -> (usize, f64)
{
    let mut max_count = 0;
    for (count, _) in item.1.iter_nodes_and_mutation_child_count_unfiltered(mutation).skip(item.1.dog_count)
    {
        if count > max_count{
            max_count = count;
        }
    }
    (item.0, max_count as f64)
}

pub fn c_and_human_second_largest_children_given_mutation(item: (usize, InfoGraph), _: u16, mutation: f64) -> (usize, f64)
{

    let (_, second_largest) = item.1.largest_and_second_largest_given_mutation(mutation, true);
    (item.0, second_largest as f64)
}

pub fn c_and_human_largest_children_given_mutation(item: (usize, InfoGraph), _: u16, mutation: f64) -> (usize, f64)
{

    let mut largest_component_size = 0;
    item.1.iter_nodes_and_mutation_child_count_unfiltered(mutation)
        .skip(item.1.dog_count)
        .for_each(
            |(size, _)|
            {
                if size > largest_component_size {
                    largest_component_size = size;
                }
            }
        );
    (item.0, largest_component_size as f64)
}

