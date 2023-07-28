use bio::io::fastq;

fn main() {
    let target: &str = std::env::args().nth(1).unwrap().as_str();
    let mut params = std::collections::HashMap::new();
    params.insert("file", target);
    params.insert("edge", "20");
    params.insert("init_wave", "100");
    params.insert("from", "10");
    params.insert("to", "1200");
    params.insert("num", "50");
    params.insert("tolerance", "4");


    let path = std::path::Path::new(params.get("file").unwrap());
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);
    // check whether the file is fasta or fastq
    let mut writer = fastq::Writer::to_file("./result/filtered.fastq").unwrap();
    for record in fastq::Reader::new(reader).records() {
        let record = record.unwrap();

        let sig_seq = clean_wrf::tosignal::convert_to_signal(&seq);
        let cwt_seq = clean_wrf::cwt::cwt(sig_seq, &params);

        if clean_wrf::filter::linear_argmax(&cwt_seq, &params) {
            writer.write_record(&record).unwrap();
        }

    }
}