use std::f64::consts::PI;

use rayon::prelude::*;
use std::sync::{Arc, Mutex, MutexGuard};
use ndarray::{Array, Array1, s, Array2};
use rustfft::{FftPlanner, num_complex::Complex};
use rustfft::num_complex::ComplexFloat;
use std::collections::HashMap;

fn linspace(start: f64, stop: f64, num: usize) -> Array1<f64> {
    let delta = (stop - start) / ((num - 1) as f64);
    Array::from_shape_fn(num, |i| start + (i as f64) * delta)
}

fn linspace_ceil(start: f64, stop: f64, num: usize) -> Array1<usize> {
    let delta = (stop - start) as usize / (num - 1);
    Array::from_shape_fn(num, |i| (start as usize) + i * delta)
}

fn psi(T: usize, f0: f64) -> Array1<f64> {
    let x = linspace(-2.0 * PI, 2.0 * PI, T);

    let a = 3.0 * f0 * f0;
    let A = 2.0 / (f64::sqrt(a * PI) * f64::sqrt(f64::sqrt(3.0) * PI));
    let wsq = f0 * f0;

    let vec = Array1::linspace(0.0, (x.len() - 1) as f64, x.len());
    let tsq = vec.mapv(|v| v * v);
    let modulus = 1.0 - tsq.clone() / wsq;
    let gauss = (-tsq / (2.0 * wsq)).mapv(|v| v.exp());
    let total = A * modulus * gauss;

    total
}

fn wavelet_convolution(tup: (&Array1<f64>, usize), init_wave: f64) -> Array1<f64> {
    let f = tup.0;
    let T = tup.1;
    let f_len = f.len();

    let mut f_hat = Array1::zeros(f_len + T);
    f_hat.slice_mut(s![..f_len]).assign(f);
    let h = psi(T, init_wave);
    let mut h_hat = Array1::zeros(f_len + T);
    h_hat.slice_mut(s![..h.len()]).assign(&h);

    let mut planner = FftPlanner::new();
    let fft_len = f_len + T;
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    let mut f_hat_complex: Vec<Complex<f64>> = f_hat.iter().map(|&val| Complex::new(val, 0.0)).collect();
    let mut h_hat_complex: Vec<Complex<f64>> = h_hat.iter().map(|&val| Complex::new(val, 0.0)).collect();

    fft.process(&mut f_hat_complex);
    fft.process(&mut h_hat_complex);

    let mut result_complex: Vec<Complex<f64>> = f_hat_complex.iter().zip(h_hat_complex.iter()).map(|(&a, &b)| a * b).collect();

    ifft.process(&mut result_complex);

    let result_real: Vec<f64> = result_complex.iter().map(|&val| val.re).collect();
    //let result_view = ArrayView1::from(result_real);
    let result_view = Array::from_shape_vec(f_len + T, result_real).unwrap();
    let start_index = T / 2;
    let end_index = start_index + f_len;

    result_view.slice(s![start_index..end_index]).to_owned()
}


fn cwt_perform(f: &Array1<f64>, para: &HashMap<&str, &str>) -> Array2<f64> {
    let f_len = f.len();

    let num = para["num"].parse::<usize>().unwrap();
    let from = para["from"].parse::<usize>().unwrap();
    let to = para["to"].parse::<usize>().unwrap();

    let t_values: Vec<usize> = linspace_ceil(from as f64, to as f64, num).to_vec(); // TODO: Check whether true

    //put result of wavelet convolution into vector with rayon
    let result: Vec<Array1<f64>> = t_values.par_iter().map(|&t| wavelet_convolution(
        (&f, t), para["init_wave"].parse::<f64>().unwrap())).collect();

    //convert result to 2d array
    let mut result_2d = Array2::zeros((result.len(), f_len));
    for (i, row) in result.iter().enumerate() {
        result_2d.slice_mut(s![i, ..]).assign(row);
    }

    result_2d
}

pub fn cwt(sig_seqs: Vec<Array1<f64>>, para: &HashMap<&str, &str>) -> Array2<f64> {
    let mut cwt_result : MutexGuard<Vec<Array2<f64>>> = Arc::new(Mutex::new(Vec::new())).lock().unwrap();
    sig_seqs.par_iter().for_each(|sig_seq| {
        let result = cwt_perform(sig_seq, para);
        let mut cwt_result = cwt_result.lock().unwrap();
        cwt_result.push(result);
    });
    let mut mean_result = Array2::zeros(cwt_result[0].dim());
    for result in cwt_result.iter() {
        mean_result = mean_result + result;
    }
    mean_result = mean_result / (cwt_result.len() as f64);
    mean_result
}