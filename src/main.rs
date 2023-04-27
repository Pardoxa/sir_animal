use ld::{ExamineOptions, CompressedTreeOptions};

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
        CmdOptions::SimpleSampleScan(def) => {
            simple_sample::execute_simple_sample_scan(def)
        },
        CmdOptions::SimpleSampleSpecific(def) => {
            simple_sample::execute_simple_sample_specific(def)
        },
        CmdOptions::SimpleSampleCheck(def) => {
            simple_sample::execute_ss_check(def)
        }
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
        },
        CmdOptions::ScanJump(o) => {
            ld::execute_mutation_scan(o, start_time)
        },
        CmdOptions::ContinueScanJump(o) => {
            ld::execute_mutation_scan_continue(o, start_time)
        },
        CmdOptions::BhAnalysis(opts) => {
            ld::examine(opts)
        },
        CmdOptions::CompressedTreePrint(opts) => {
            ld::compressed_tree_printer(opts)
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
    /// Scan mutation range with simple sample
    SimpleSampleScan(simple_sample::DefaultOpts),
    /// Sample specific sigma to get jump prob
    SimpleSampleSpecific(simple_sample::DefaultOpts),
    /// For checking our idea why starting in mid can have higher jump probability
    SimpleSampleCheck(simple_sample::DefaultOpts),
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
    PrintDot(ld::PrintOpts),
    /// Scan probability of jumping
    ScanJump(simple_sample::DefaultOpts),
    /// Continue the scan of jumps
    ContinueScanJump(simple_sample::DefaultOpts),
    /// Examine the BH file
    BhAnalysis(ExamineOptions),
    /// Print Tree as abstract gnuplot representation
    CompressedTreePrint(CompressedTreeOptions)
    
}