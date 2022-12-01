use{
    std::time::*,
    structopt::StructOpt
};


mod sir_nodes;
mod misc;
mod simple_sample;
mod ld;

fn main() {
    let start_time = Instant::now();
    let cmd = CmdOptions::from_args();

    match cmd
    {
        CmdOptions::SimpleSample(def) => {
            simple_sample::execute_simple_sample(def);
        },
        CmdOptions::Wl(def) => {
            ld::execute_wl(def, start_time)
        },
        CmdOptions::WlContinue(def) => {
            ld::execute_wl_continue(def, start_time)
        },
        CmdOptions::MarkovSimpleSample(def) => {
            ld::execute_markov_ss(def)
        },
        CmdOptions::Entropic(def) => {
            ld::exec_entropic_beginning(def, start_time)
        },
        CmdOptions::REWl(def) => {
            ld::execute_rewl(def, start_time)
        },
        CmdOptions::Rees(def) => {
            ld::exec_rees_beginning(def, start_time)
        },
        CmdOptions::PrintDot(o) => {
            o.execute()
        }
    }
    println!(
        "Execution took {}",
        humantime::format_duration(start_time.elapsed())
    );
}

#[derive(Debug, StructOpt, Clone)]
#[structopt(about = "Simulations for the SIR Model")]
pub enum CmdOptions
{
    SimpleSample(simple_sample::DefaultOpts),
    /// Wang Landau Simulation
    Wl(simple_sample::DefaultOpts),
    /// Replica Exchange Wang Landau Simulation
    REWl(simple_sample::DefaultOpts),
    /// Continue a WL simulation
    WlContinue(simple_sample::DefaultOpts),
    /// Test simple sampling via Markov chain
    MarkovSimpleSample(simple_sample::DefaultOpts),
    /// Start entropic sampling
    Entropic(simple_sample::DefaultOpts),
    /// Start a REES simulation
    Rees(simple_sample::DefaultOpts),
    /// Read file, print dot options
    PrintDot(ld::PrintOpts)
    
}