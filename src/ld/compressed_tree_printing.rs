use std::io::{Write, BufWriter};
use std::fs::File;
use std::num::NonZeroUsize;
use structopt::*;
use super::{TopAnalyzer, InfoGraph, InfectedBy};

const BRANCH_THICKNESS: usize = 1;

#[derive(StructOpt, Debug, Clone)]
pub struct CompressedTreeOptions
{
    /// Globbing to the bh files
    #[structopt(long, short)]
    pub glob: String,

    /// Choose to print a tree with a specific human c value
    #[structopt(long, short)]
    pub human_c: Option<usize>,

    /// Print the print_idxTH tree that fulfills the other criteria
    #[structopt(long, short)]
    pub print_idx: Option<usize>,

    #[structopt(long, short)]
    pub output_name: String,

    #[structopt(long)]
    pub pdf: bool,

    #[structopt(long, short)]
    pub unsorted_children: bool,

    #[structopt(long)]
    pub print_multiple: Option<NonZeroUsize>
}



pub fn compressed_tree_printer(opts: CompressedTreeOptions)
{
    println!("You will now examine");

    let print_idx = opts.print_idx.unwrap_or_default();
    let mut current_idx = 0;

    let print_amount = opts.print_multiple.map(NonZeroUsize::get).unwrap_or(1);
    let mut printed = 0;

    for file in glob::glob(&opts.glob).unwrap()
    {
        let file = file.unwrap();
        println!("file {file:?}");
        let mut analyzer = TopAnalyzer::new(file);
        let iter = analyzer.iter_all_info();
        
        for (c, info_graph) in iter {
            let valid_c = if let Some(human_c) = opts.human_c {
                human_c == c
            } else {
                true
            };
            if valid_c{
                if current_idx >= print_idx {
                    let already_printed = if opts.print_multiple.is_none(){
                        None
                    } else {
                        Some(printed)
                    };
                    create_gp(info_graph, &opts, already_printed);
                    printed += 1;
                    if printed == print_amount{
                        return;
                    }
                }
                current_idx += 1;
            }
        }
    }
}

fn create_gp(info_graph: InfoGraph, opts: &CompressedTreeOptions, print_index: Option<usize>)
{
    

    let mut descendant_count = vec![0_u16; info_graph.info.vertex_count()];

    for (index, degree) in info_graph.info.degree_iter().enumerate()
    {
        if degree == 1 {
            let mut current_node = info_graph.info.at(index);
            let mut counter = 1;
            while let InfectedBy::By(by) = current_node.infected_by {
                let increment = descendant_count[by as usize] == 0;
                descendant_count[by as usize] += counter;
                if increment{
                    counter += 1;
                }
                current_node = info_graph.info.at(by as usize);
            }
        }
    }

    let initial_infected = info_graph.initial_infection[0];

    let mut husks = vec![None; info_graph.info.vertex_count()];
    
    let mut max_depth = 0;

    for (index, node, depth) in  info_graph.info.bfs_index_depth(initial_infected)
    {
        if depth > max_depth 
        {
            max_depth = depth;
        }
        
        let species = if index < info_graph.dog_count {
            HumanOrDog::Dog
        } else {
            HumanOrDog::Human
        };

        assert!(husks[index].is_none());

        husks[index] = Some(
            SegmentVerticalHusk{
                depth,
                species,
                children: Vec::new()
            }
        );
        
        if let InfectedBy::By(by) = node.infected_by
        {
            let parent = husks[by as usize].as_mut().unwrap();
            parent.children.push(index as u16);
        }
    }

    let mut husks_specifying_space_req: Vec<_> = 
        husks.iter()
            .map(
                |husk|
                {
                    husk.clone().map(
                        |husk|
                        {
                            SegmentVerticalHuskSpaceReq{
                                species: husk.species,
                                depth: husk.depth,
                                children: husk.children,
                                space_req: 1
                            }
                        }
                    )
                }
            ).collect();
    

    let mut depth_map = vec![Vec::new(); max_depth + 1];
    let mut max_depth_children = vec![0; husks.len()];

    for (index, _, depth) in info_graph.info.bfs_index_depth(initial_infected)
    {
        depth_map[depth].push(index);
        max_depth_children[index] = depth;
    }

    while let Some(depth_list) = depth_map.pop()
    {
        for node_idx in depth_list
        {
            let this = husks_specifying_space_req[node_idx].as_mut().unwrap();
            this.space_req += this.children.len().saturating_sub(1) * BRANCH_THICKNESS;

            let space_req = this.space_req;

            if let InfectedBy::By(by) = info_graph.info.at(node_idx).infected_by
            {
                if max_depth_children[by as usize] < max_depth_children[node_idx]
                {
                    max_depth_children[by as usize] = max_depth_children[node_idx];
                }
                let parent = husks_specifying_space_req[by as usize].as_mut().unwrap();
                parent.space_req += space_req;
            }
        }
    }

    if !opts.unsorted_children{
        // now I need to know the number of children…
        husks_specifying_space_req.iter_mut()
            .for_each(
                |husk_opt|
                {
                    if let Some(husk) = husk_opt
                    {
                        husk.children.sort_by_key(
                            |child|
                            {
                                max_depth_children[*child as usize]
                            }
                        );
                    }
                }
            );
    }

    let mut husks_y =vec![None; husks_specifying_space_req.len()];

    let initial = husks_specifying_space_req[initial_infected].as_ref().unwrap();

    husks_y[initial_infected] = Some(
        initial.with_y(0)
    );


    let mut stack = vec![initial_infected];

    while let Some(idx) = stack.pop()
    {
        let parent = husks_y[idx].as_ref().unwrap();
        let last_child_id = parent.children.len().saturating_sub(1);

        let mut current_y = parent.y_min;
        

        // how I need to assign all children husks.
        for (index, &child) in husks_specifying_space_req[idx].as_ref().unwrap().children.iter().enumerate()
        {
            if index != 0 {
                current_y += BRANCH_THICKNESS;
            }

            if index == last_child_id {
                current_y += 1;
            }

            let child_husk = husks_specifying_space_req[child as usize].as_ref().unwrap();
            let child_y_min = current_y;
            
            current_y += child_husk.space_req; 

            husks_y[child as usize] = Some(
                child_husk.with_y(child_y_min)
            );

            stack.push(child as usize);
        }
    }

    stack.push(initial_infected);

    let name = match print_index{
        Some(idx) => {
            format!("{}_idx{idx}.gp", opts.output_name)
        }
        None => {
            format!("{}.gp", opts.output_name)
        }
    };
    println!("creating {name}");

    let file = File::create(name).unwrap();
    let mut buf = BufWriter::new(file);
    let mut counter = 0;
    let mut colors = Vec::new();

    if opts.pdf{
        writeln!(buf, "set t pdf").unwrap();
        writeln!(buf, "set output \"{}.pdf\"", opts.output_name).unwrap();
    }


    while let Some(idx) = stack.pop()
    {
        draw(
            idx, 
            &mut buf, 
            &mut counter, 
            &mut colors, 
            &husks_y,
            &info_graph,
        );
        for (_, &child) in husks[idx].as_ref().unwrap().children.iter().enumerate()
        {
            stack.push(child as usize);
        }
    }

    write!(buf, "p ").unwrap();
    for (i, color) in (0..counter).zip(colors.iter()) {
        writeln!(buf, "$data{i} u 2:1 w l lc {} t \"\",\\", color.line_color()).unwrap();
    }
    writeln!(buf).unwrap();
    if opts.pdf{
        writeln!(buf, "set output").unwrap();
    }
}


#[derive(Clone, Copy, PartialEq, Eq)]
enum HumanOrDog
{
    Human,
    Dog
}

#[derive(Clone)]
struct SegmentVerticalHusk
{
    depth: usize,
    species: HumanOrDog,
    children: Vec<u16>
} 



#[derive(Clone)]
struct SegmentVerticalHuskSpaceReq
{
    depth: usize,
    species: HumanOrDog,
    children: Vec<u16>,
    space_req: usize
}

impl SegmentVerticalHuskSpaceReq{
    pub fn with_y(&self, y_min: usize) -> SegmentVerticalHuskY
    {
        SegmentVerticalHuskY{
            depth: self.depth,
            species: self.species,
            children: self.children.clone(),
            y_min,
            y_max: y_min + self.space_req
        }
    }
}

#[derive(Clone)]
struct SegmentVerticalHuskY
{
    depth: usize,
    species: HumanOrDog,
    children: Vec<u16>,
    y_min: usize,
    y_max: usize
}

pub enum Color{
    Red,
    Blue,
    Cyan,
    Black,
    Gray,
    Magenta
}

impl Color {
    fn line_color(&self) -> &'static str 
    {
        match self {
            Self::Red => "rgb \"#FF0000\"",
            Self::Blue => "rgb \"#0000FF\"",
            Self::Black => "rgb \"#000000\"",
            Self::Magenta=> "rgb \"#FF00FF\"",
            Self::Cyan => "rgb \"#00aaaa\"",
            Self::Gray => "rgb \"#999999\""
        }
    }
}

fn draw<W: Write>(
    index_self: usize, 
    mut writer: W, 
    eof_counter: &mut usize, 
    color_vec: &mut Vec<Color>,
    husks: &[Option<SegmentVerticalHuskY>],
    info_graph: &InfoGraph
)
{
    let this_parent = husks[index_self].as_ref().unwrap();

    if this_parent.children.len() > 1 {
        writeln!(writer, "$data{eof_counter}<<EOF").unwrap();
        writeln!(writer, "{} {}", this_parent.depth, this_parent.y_min).unwrap();
        writeln!(writer, "{} {}", this_parent.depth as f64 + 0.00000000001, this_parent.y_max).unwrap();
        writeln!(writer, "EOF").unwrap();
        *eof_counter += 1;
        let color = match this_parent.species{
            HumanOrDog::Human => {
                Color::Blue
            },
            HumanOrDog::Dog => {
                Color::Black
            }
        };
        color_vec.push(color);
    }

    let parent_depth = this_parent.depth;
    let parent_species = this_parent.species;




    let mut child_iter = this_parent.children.iter();
    let mut first = None;
    let mut last = None;

    let get_color = |child_husk: &SegmentVerticalHuskY|
    {
        let is_leaf = child_husk.children.is_empty();
        match (parent_species, child_husk.species)
        {
            (HumanOrDog::Human, HumanOrDog::Human) => {
                if is_leaf {
                    Color::Cyan
                } else {
                    Color::Blue
                }
            },
            (HumanOrDog::Dog, HumanOrDog::Dog) => {
                if is_leaf {
                    Color::Gray
                } else {
                    Color::Black
                }
            },
            (HumanOrDog::Dog, HumanOrDog::Human) => Color::Red,
            (HumanOrDog::Human, HumanOrDog::Dog) => Color::Magenta
        }
    };

    let mut draw_edge = |y_pos, color| {
        writeln!(writer, "$data{eof_counter}<<EOF").unwrap();
        writeln!(writer, "{} {}", parent_depth, y_pos).unwrap();
        writeln!(writer, "{} {}", parent_depth + 1, y_pos).unwrap();
        writeln!(writer, "EOF").unwrap();
        *eof_counter += 1;
        color_vec.push(color);
    };

    #[allow(clippy::comparison_chain)]
    if child_iter.len() > 1 {
        first = child_iter.next().copied();
        last = child_iter.next_back().copied();
    } else if child_iter.len() == 1 {

        let child = *child_iter.next().unwrap();
        let child_husk = husks[child as usize].as_ref().unwrap();

        // recursively check if parent was drawn up or down
        let mut parent_id = index_self;

        let pos = loop{
            let parents_parent = if let InfectedBy::By(by) = info_graph.info.at(parent_id).infected_by
            {
                by as usize
            } else {
                // this parent is initial infected…
                
                let parent_husk = husks[parent_id].as_ref().unwrap();

                break parent_husk.y_min as f64 + ((parent_husk.y_max  - parent_husk.y_min) as f64).abs() / 2.0;
            };

            let parents_parent_husk = husks[parents_parent].as_ref().unwrap();
            if parents_parent_husk.children.len() == 1 {
                parent_id = parents_parent;
                continue;
            }
            let position_parent = parents_parent_husk.children.iter().position(|&idx| idx as usize == parent_id).unwrap();
            if position_parent == 0 {
                break parents_parent_husk.y_min as f64;
            } else if position_parent == parents_parent_husk.children.len() - 1 
            {
                break parents_parent_husk.y_max as f64;
            } else {
                // bug in here... find first descendant with more than one child or last descendant
                let mut child_id = child as usize;
                loop{
                    let husk = husks[child_id].as_ref().unwrap();
                    if husk.children.len() != 1 {
                        break;
                    }
                    child_id = husk.children[0] as usize;
                };
                let child_husk = husks[child_id].as_ref().unwrap();

                break child_husk.y_min as f64 + ((child_husk.y_max  - child_husk.y_min) as f64).abs() / 2.0;
            }

        };
        let color = get_color(child_husk);
        draw_edge(pos, color);
    }


 

    for &middle_child in child_iter{
        let child_husk = husks[middle_child as usize].as_ref().unwrap();

        let pos = if child_husk.children.len() == 1 {
            let mut child_id = middle_child as usize;
                loop{
                    let husk = husks[child_id].as_ref().unwrap();
                    if husk.children.len() != 1 {
                        break;
                    }
                    child_id = husk.children[0] as usize;
                };
                let child_husk = husks[child_id].as_ref().unwrap();

                child_husk.y_min as f64 + ((child_husk.y_max  - child_husk.y_min) as f64).abs() / 2.0
        } else {
            child_husk.y_min as f64 + ((child_husk.y_max  - child_husk.y_min) as f64).abs() / 2.0
        };

        let color = get_color(child_husk);
        draw_edge(pos, color);
    }



    if let Some(child_id) = first {
        let child_husk = husks[child_id as usize].as_ref().unwrap();
        let color = get_color(child_husk);

        draw_edge(child_husk.y_min as f64, color);
    }

    if let Some(child_id) = last {
        let child_husk = husks[child_id as usize].as_ref().unwrap();
        let color = get_color(child_husk);

        draw_edge((child_husk.y_max) as f64, color);
    }

    
}