use{
    indicatif::{ProgressBar, ProgressStyle},
    std::{
        io::{Write, BufReader},
        process::exit,
        fs::File,
        path::Path,
        num::*
    },
    serde_json::Value,
    serde::{Serialize, de::DeserializeOwned, Deserialize},
    net_ensembles::sampling::histogram::*
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn write_json<W: Write>(mut writer: W, json: &Value)
{
    write!(writer, "#").unwrap();
    serde_json::to_writer(&mut writer, json).unwrap();
    writeln!(writer).unwrap();
}

pub fn indication_bar(len: u64) -> ProgressBar
{
        // for indication on when it is finished
        let bar = ProgressBar::new(len);
        bar.set_style(ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise} - {eta_precise}] {wide_bar}")
            .unwrap()
        );
        bar
}


pub fn write_commands<W: Write>(mut w: W) -> std::io::Result<()>
{
    write!(w, "#")?;
    for arg in std::env::args()
    {
        write!(w, " {arg}")?;
    }
    writeln!(w)
}

pub fn parse<P, T>(file: Option<P>) -> (T, Value)
where P: AsRef<Path>,
    T: Default + Serialize + DeserializeOwned
{
    match file
    {
        None => {
            let example = T::default();
            serde_json::to_writer_pretty(
                std::io::stdout(),
                &example
            ).expect("Unable to reach stdout");
            exit(0)
        }, 
        Some(file) => {
            let f = File::open(file)
                .expect("Unable to open file");
            let buf = BufReader::new(f);

            let json_val: Value = match serde_json::from_reader(buf)
            {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("json parsing error!");
                    dbg!(e);
                    exit(1);
                }
            };

            let opt: T = match serde_json::from_value(json_val.clone())
            {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("json parsing error!");
                    dbg!(e);
                    exit(1);
                }
            };

            (opt, json_val)    
        }
    }
}



#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct RequestedTime
{
    pub seconds: Option<NonZeroU64>,
    pub minutes: Option<NonZeroU64>,
    pub hours: Option<NonZeroU64>,
    pub days: Option<NonZeroU64>
}

impl RequestedTime
{
    #[allow(dead_code)]
    pub fn in_seconds(&self) -> u64
    {
        let mut time = self.seconds.map_or(0, NonZeroU64::get);
        if let Some(min) = self.minutes
        {
            time += min.get() * 60;
        }

        if let Some(h) = self.hours
        {
            time += h.get() * (60*60);
        }

        if let Some(d) = self.days
        {
            time += d.get() * (24*60*60);
        }

        if time == 0 {
            eprintln!("Time is zero! That is invalid! Fatal Error");
            panic!("No Time")
        }

        time
    }
}

#[derive(Clone, Copy, Deserialize, Serialize, Default)]
pub struct Interval
{
    pub start: i32,
    pub end_inclusive: i32
}

impl Interval{
    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool
    {
        self.start < self.end_inclusive
    }

    pub fn get_hist(&self, neg_start: i32) -> HistI32Fast
    {
        let start = if self.start > 0 
        {
            self.start
        } else if neg_start > 0{
            -neg_start
        } else {
            neg_start
        };
        HistI32Fast::new_inclusive(start, self.end_inclusive)
            .expect("unable to create hist")
    }
}

pub trait QuickName
{
    fn quick_name(&self) -> String;
    fn quick_name_with_ending(&self, ending: &str) -> String
    {
        format!("{}{ending}", self.quick_name())
    }
}