use ndarray::Array1;

fn a_convert(value: u8) -> f64 {
    if value == b'A' || value == b'a'{
        1.0
    } else {
        0.0
    }
}
// Returns 1 if value is "C"
fn c_convert(value: u8) -> f64 {
    if value == b'C' || value == b'c'{
        1.0
    } else {
        0.0
    }
}
// Returns 1 if value is "G"
fn g_convert(value: u8) -> f64 {
    if value == b'G' || value == b'g'{
        1.0
    } else {
        0.0
    }
}
// Returns 1 if value is "T"
fn t_convert(value: u8) -> f64 {
    if value == b'T' || value == b't'{
        1.0
    } else {
        0.0
    }
}

pub fn convert_to_signal(sequence: &Vec<u8>) -> Vec<Array1<f64>> {
    let a_seq: Vec<f64> = sequence.iter().map(|x| a_convert(*x)).collect();
    let c_seq: Vec<f64> = sequence.iter().map(|x| c_convert(*x)).collect();
    let g_seq: Vec<f64> = sequence.iter().map(|x| g_convert(*x)).collect();
    let t_seq: Vec<f64> = sequence.iter().map(|x| t_convert(*x)).collect();
    let mut sig_seqs: Vec<Array1<f64>> = Vec::new();
    sig_seqs.push(Array1::from(a_seq));
    sig_seqs.push(Array1::from(c_seq));
    sig_seqs.push(Array1::from(g_seq));
    sig_seqs.push(Array1::from(t_seq));
    sig_seqs
}