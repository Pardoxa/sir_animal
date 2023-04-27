use std::io::{Write, BufWriter};
use std::fs::File;
use structopt::*;
use super::{TopAnalyzer, InfoGraph, InfectedBy};

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
    pub print_idx: Option<usize>
}



pub fn compressed_tree_printer(opts: CompressedTreeOptions)
{
    println!("You will now examine");

    let print_idx = opts.print_idx.unwrap_or_default();
    let mut current_idx = 0;

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
                if current_idx == print_idx {
                    create_gp(info_graph);
                    return;
                }
                current_idx += 1;
            }
        }
    }
}

fn create_gp(info_graph: InfoGraph)
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
    let mut number_of_nodes: usize = 0;

    for (index, node, depth) in  info_graph.info.bfs_index_depth(initial_infected)
    {
        number_of_nodes += 1;
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
                num_descendants: descendant_count[index],
                children: Vec::new()
            }
        );
        
        if let InfectedBy::By(by) = node.infected_by
        {
            let parent = husks[by as usize].as_mut().unwrap();
            parent.children.push(index as u16);
        }
    }

    let mut husks_y = vec![None; husks.len()];
    
    
    let initial = husks[initial_infected].as_ref().unwrap();
    
    husks_y[initial_infected] = Some(
        initial.with_y(0, number_of_nodes)
    );


    let mut stack = vec![initial_infected];

    while let Some(idx) = stack.pop()
    {
        let parent = husks_y[idx].as_ref().unwrap();
        let y_min = parent.y_min;
        let y_max = parent.y_max;

        let mut current_y = y_min;

        let total_children = parent.children.len();


        // how I need to assign all children husks.
        for (child_count, &child) in husks[idx].as_ref().unwrap().children.iter().enumerate()
        {
            let child_husk = husks[child as usize].as_ref().unwrap();
            let child_y_min = current_y;
            // if last child
            let child_y_max = if child_count == total_children - 1
            {
                dbg!("last child");
                current_y += 1 + child_husk.num_descendants as usize;
                current_y
            } else {
                current_y += 1 + child_husk.num_descendants as usize;
                current_y
            };

            current_y += 1;

            husks_y[child as usize] = Some(
                child_husk.with_y(child_y_min, child_y_max)
            );

            stack.push(child as usize);
        }
    }

    stack.push(initial_infected);

    let file = File::create("test_compressed.gp").unwrap();
    let mut buf = BufWriter::new(file);
    let mut counter = 0;
    let mut colors = Vec::new();

    while let Some(idx) = stack.pop()
    {
        let parent = husks_y[idx].as_ref().unwrap();
        parent.draw(&mut buf, &mut counter, &mut colors, &husks);
        for (_, &child) in husks[idx].as_ref().unwrap().children.iter().enumerate()
        {
            stack.push(child as usize);
        }
    }

    write!(buf, "p ").unwrap();
    for (i, color) in (0..counter).zip(colors.iter()) {
        writeln!(buf, "$data{i} w l lc {} t \"\",\\", color.line_color()).unwrap();
    }
    writeln!(buf).unwrap();

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
    num_descendants: u16,
    children: Vec<u16>
} 

impl SegmentVerticalHusk{
    pub fn with_y(&self, y_min: usize, y_max: usize) -> SegmentVerticalHuskY
    {
        SegmentVerticalHuskY{
            depth: self.depth,
            species: self.species,
            children: self.children.clone(),
            num_descendants: self.num_descendants,
            y_min,
            y_max
        }
    }
}


#[derive(Clone)]
struct SegmentVerticalHuskY
{
    depth: usize,
    species: HumanOrDog,
    num_descendants: u16,
    children: Vec<u16>,
    y_min: usize,
    y_max: usize
}

pub enum Color{
    Red,
    Blue,
    Black,
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
        }
    }
}

impl SegmentVerticalHuskY{
    fn draw<W: Write>(
        &self, 
        mut writer: W, 
        eof_counter: &mut usize, 
        color_vec: &mut Vec<Color>,
        husks: &[Option<SegmentVerticalHusk>]
    )
    {
        if self.children.len() > 1 {
            writeln!(writer, "$data{eof_counter}<<EOF").unwrap();
            writeln!(writer, "{} {}", self.depth, self.y_min).unwrap();
            writeln!(writer, "{} {}", self.depth as f64 + 0.00000000001, self.y_max).unwrap();
            writeln!(writer, "EOF").unwrap();
            *eof_counter += 1;
            let color = match self.species{
                HumanOrDog::Human => {
                    Color::Blue
                },
                HumanOrDog::Dog => {
                    Color::Black
                }
            };
            color_vec.push(color);
        }

        let mut draw_edge = |y_pos, color| {
            writeln!(writer, "$data{eof_counter}<<EOF").unwrap();
            writeln!(writer, "{} {}", self.depth, y_pos).unwrap();
            writeln!(writer, "{} {}", self.depth + 1, y_pos).unwrap();
            writeln!(writer, "EOF").unwrap();
            *eof_counter += 1;
            color_vec.push(color);
        };

        if self.children.len() == 1 {
            let mid = self.y_min as f64 + ((self.y_max  - self.y_min) as f64).abs() / 2.0;
            let child_species = husks[self.children[0] as usize].as_ref().unwrap().species;
            let color = match (child_species, self.species)
            {
                (HumanOrDog::Human, HumanOrDog::Human) => Color::Blue,
                (HumanOrDog::Dog, HumanOrDog::Dog) => Color::Black,
                (HumanOrDog::Dog, HumanOrDog::Human) => Color::Red,
                (HumanOrDog::Human, HumanOrDog::Dog) => Color::Magenta
            };
            draw_edge(mid, color);
        } else {
            
        }

        
    }
}