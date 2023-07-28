use ndarray::{Array2, Axis};
use std::collections::HashMap;

fn position_max(slice: &[f64]) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
}


pub fn linear_argmax(array: &Array2<f64>, para: &HashMap<&str, &str>) -> bool {
    let idx = para["edge"].parse::<usize>().unwrap();
    let max1 = position_max(&array.index_axis(Axis(1), idx).to_vec()).unwrap();
    let max2 = position_max(&array.index_axis(Axis(1), array.len_of(Axis(1)) - idx -1).to_vec()).unwrap();
    if ((max1 - max2) as f64).abs() < para["tolerance"].parse::<f64>().unwrap() {
        true
    } else {
        false
    }
}