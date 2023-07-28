use bio::io::fastq;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

fn main() {
    let binding = std::env::args().nth(1).unwrap();
    let target: &str = binding.as_str();

    let mut params = std::collections::HashMap::new();
    params.insert("file", target);
    params.insert("edge", "40");
    params.insert("init_wave", "100");
    params.insert("from", "60");
    params.insert("to", "600");
    params.insert("num", "50");
    params.insert("tolerance", "4");
    params.insert("min_read_length", "1000");

    let path = std::path::Path::new(params.get("file").unwrap());
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    // check whether the file is fasta or fastq
    let one_writer = fastq::Writer::to_file("./result/filtered.fastq").unwrap();
    let writer = Arc::new(Mutex::new(one_writer));

    // Parallelize the loop using Rayon's par_iter.
    fastq::Reader::new(reader)
        .records()
        .par_iter()  // Use par_iter() to parallelize the iterator.
        .enumerate()
        .for_each(|(cnt, record)| {
            let record = record.unwrap();
            if record.seq().len() < params["min_read_length"].parse::<usize>().unwrap() {
                return;
            }
            let seq = record.seq().to_vec();
            let sig_seq = clean_wrf::tosignal::convert_to_signal(&seq);
            let cwt_seq = clean_wrf::cwt::cwt(sig_seq, &params);

            if clean_wrf::filter::linear_argmax(&cwt_seq, &params) {
                // Lock the writer before writing to it.
                let mut writer_lock = writer.lock().unwrap();
                writer_lock.write_record(&record).unwrap();
            }

            if cnt % 10000 == 0 {
                println!("{} reads processed", cnt);
            }
        });
}