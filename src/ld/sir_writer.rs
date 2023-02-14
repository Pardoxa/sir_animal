use std::fmt::Display;

use net_ensembles::Node;
use serde_json::Value;
use std::mem::ManuallyDrop;
use std::ops::DerefMut;
use std::process::Command;

use{
    std::{
        fs::File,
        io::{Write, BufWriter}
    },
    net_ensembles::{GenericGraph, AdjContainer},
    crate::sir_nodes::*,
    rayon::prelude::*
};

#[allow(dead_code)]
pub struct ZippingWriter
{
    pub writer: ManuallyDrop<BufWriter<File>>,
    pub path: String
}

#[allow(dead_code)]
impl ZippingWriter {
    pub fn new(path: String) -> Self 
    {
        let file = File::create(&path)
            .expect("unable to create file");
        let buf = BufWriter::with_capacity(1024*64, file);
        Self { writer: ManuallyDrop::new(buf), path }
    }
}

impl Write for ZippingWriter
{
    #[inline]
    fn write_fmt(&mut self, fmt: std::fmt::Arguments<'_>) -> std::io::Result<()> {
        self.writer.write_fmt(fmt)
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.writer.write_all(buf)
    }

    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.writer.write(buf)
    }

    #[inline]
    fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
        self.writer.write_vectored(bufs)
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }

}

impl Drop for ZippingWriter
{
    fn drop(&mut self) {
        unsafe{ManuallyDrop::drop(&mut self.writer)};
        let out = Command::new("gzip")
            .arg(&self.path)
            .output();
        match out {
            Ok(_) => println!("Success! Zipped {}", self.path),
            Err(e) => println!("Error! Failed to zip {} due to {:?}", self.path, e)
        }
    }
}

pub type CurveWriter = BufWriter<File>;
pub struct SirWriter
{
    pub writer_s: ManuallyDrop<CurveWriter>,
    pub writer_r: ManuallyDrop<CurveWriter>,
    pub writer_i: ManuallyDrop<CurveWriter>,
    pub writer_ever: ManuallyDrop<CurveWriter>,
    pub paths: [String; 4]
}

impl Drop for SirWriter
{
    fn drop(&mut self)
    {
        // first drop all the Writers!
        unsafe{
            ManuallyDrop::drop(&mut self.writer_s);
            ManuallyDrop::drop(&mut self.writer_r);
            ManuallyDrop::drop(&mut self.writer_i);
            ManuallyDrop::drop(&mut self.writer_ever);
        };

        // next: Zipping time!
        self.paths.par_iter()
            .for_each(
                |path|
                {
                    let out = Command::new("gzip")
                        .arg(path)
                        .output();
                    match out {
                        Ok(_) => println!("Success! Zipped {path}"),
                        Err(e) => println!("Error! Failed to zip {path} due to {e:?}")
                    }
                }
            );
    }
}

#[allow(dead_code)]
impl SirWriter
{
    #[inline]
    pub fn writer_iter(&mut self) -> impl Iterator<Item=&mut CurveWriter>
    {
        let slice = [
            self.writer_s.deref_mut(), 
            self.writer_r.deref_mut(), 
            self.writer_i.deref_mut(), 
            self.writer_ever.deref_mut()
        ];
        slice.into_iter()
    }

    pub fn new(name: &str, number: usize) -> Self
    {
        let names: [String; 4] = [
            format!("{name}_{number}_s.curves"),
            format!("{name}_{number}_r.curves"),
            format!("{name}_{number}_i.curves"),
            format!("{name}_{number}_e.curves")
        ];

        let mut files = names.clone().map(
            |name| 
            {
                BufWriter::with_capacity(1024*64,
                    File::create(name)
                        .expect("unable to create file S")
                )
            }
        ).into_iter();


        Self{
            writer_s: ManuallyDrop::new(files.next().unwrap()),
            writer_r: ManuallyDrop::new(files.next().unwrap()),
            writer_i: ManuallyDrop::new(files.next().unwrap()),
            writer_ever: ManuallyDrop::new(files.next().unwrap()),
            paths: names
        }
    }

    pub fn write_energy<E, I>(&mut self, energy: E, extinction_index: I) -> std::io::Result<()>
    where E: Display,
        I: Display
    {
        for w in self.writer_iter()
        {
            write!(w, "{energy} {extinction_index} ")?;
        }
        
        Ok(())
    }

    pub fn write_current<A, T>(&mut self, graph: &GenericGraph<SirFun<T>, A>) -> std::io::Result<()>
    where A: AdjContainer<SirFun<T>>,
        SirFun<T>: Node
    {
        let mut i = 0_u32;
        let mut r = 0_u32;
        let mut s = 0_u32;
        
        graph.contained_iter()
            .for_each(
                |contained|
                match contained.sir{
                    SirState::I => i +=1,
                    SirState::R => r +=1,
                    SirState::S => s += 1,
                    _ => ()
                }
            );

        write!(self.writer_i, "{i} ")?;
        write!(self.writer_r, "{r} ")?;
        write!(self.writer_s, "{s} ")?;
        let e = i + r;
        write!(self.writer_ever, "{e} ")
    }

    pub fn write_line(&mut self) -> std::io::Result<()>
    {
        for w in self.writer_iter()
        {
            writeln!(w)?;
        }
        Ok(())
    }

    pub fn write_header(&mut self, jsons: &[Value]) -> std::io::Result<()>
    {
        for w in self.writer_iter()
        {
            write_header(&mut *w)?;
            for json in jsons{
                crate::misc::write_json(&mut *w, json);
            }
        }

        Ok(())
    }
}

fn write_header(writer: &mut CurveWriter) -> std::io::Result<()>
{
    writeln!(writer, "#Energy Extinction_index Curve[0] Curve[1] …")?;
    Ok(())
}