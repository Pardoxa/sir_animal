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