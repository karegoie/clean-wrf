use bio::io::fastq;
fn main() {
    let binding = std::env::args().nth(1).unwrap();
    let target: &str = binding.as_str();
    let mut params = std::collections::HashMap::new();
    params.insert("file", target);
    params.insert("edge", "40");
    params.insert("init_wave", "100");
    params.insert("from", "60");
    params.insert("to", "600");
    params.insert("num", "25");
    params.insert("tolerance", "4");
    params.insert("min_read_length", "10000");
    let path = std::path::Path::new(params.get("file").unwrap());
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    // check whether the file is fasta or fastq
    let mut writer = fastq::Writer::to_file("./result/filtered.fastq").unwrap();
    let mut cnt = 0;
    for record in fastq::Reader::new(reader).records() {
        cnt += 1;
        if cnt % 100 == 0 {
            println!("{} reads processed", cnt);
        }
        let record = record.unwrap();
        if record.seq().len() < params["min_read_length"].parse::<usize>().unwrap() {
            continue;
        }
        let seq = record.seq().to_vec();
        let sig_seq = clean_wrf::tosignal::convert_to_signal(&seq);
        let cwt_seq = clean_wrf::cwt::cwt(sig_seq, &params);
        if clean_wrf::filter::linear_argmax(&cwt_seq, &params) {
            writer.write_record(&record).unwrap();
        }
    }
}