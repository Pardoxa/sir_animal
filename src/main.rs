use{
    std::time::*,
    structopt::StructOpt
};


mod sir_nodes;
mod misc;
mod simple_sample;

fn main() {
    let start_time = Instant::now();
    let cmd = CmdOptions::from_args();

    match cmd
    {
        CmdOptions::SimpleSample(def) => {
            simple_sample::execute_simple_sample(def);
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
    SimpleSample(simple_sample::DefaultOpts)
}