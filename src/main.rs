use bio::io::fastq;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use bio::io::fastq::Record;

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
    let one_writer = fastq::Writer::to_file("./result/filtered.fastq").unwrap();
    let mut writer = Arc::new(Mutex::new(one_writer));
    let fastq_reader = fastq::Reader::new(reader).records().collect::<Vec<Record>>();

    // parallelize for loop above with rayon
    fastq_reader
        .par_iter()
        .for_each(|record| {
        if record.seq().len() < params["min_read_length"].parse::<usize>().unwrap() {
            return;
        }
        let seq = record.seq().to_vec();
        let sig_seq = clean_wrf::tosignal::convert_to_signal(&seq);
        let cwt_seq = clean_wrf::cwt::cwt(sig_seq, &params);
        if clean_wrf::filter::linear_argmax(&cwt_seq, &params) {
            let mut writer = writer.lock().unwrap();
            writer.write_record(&record).unwrap();
        }
    });

}