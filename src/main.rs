use rustograd::nn::{Module, MLP};
use rustograd::ValueWrapper;

use rand::seq::SliceRandom;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

fn read_dataset(filename: &str) -> (Vec<Vec<f64>>, Vec<f64>) {
    let file = File::open(filename).expect("Unable to open file");
    let reader = BufReader::new(file);
    let mut x = Vec::new();
    let mut y = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue;
        } // Skip header
        let line = line.expect("Unable to read line");
        let values: Vec<f64> = line
            .split(',')
            .map(|s| s.parse().expect("Parse error"))
            .collect();
        x.push(vec![values[0], values[1]]);
        y.push(values[2]);
    }

    (x, y)
}

fn main() {
    let a = ValueWrapper::new(-4.0);
    let b = ValueWrapper::new(2.0);

    let mut c = a.clone() + b.clone();
    let mut d = a.clone() * b.clone() + b.clone().pow(3.0);

    c += c.clone() + 1.0;
    c += 1.0 + c.clone() + (-a.clone());
    d += d.clone() * 2.0 + (b.clone() + a.clone()).relu();
    d += 3.0 * d.clone() + (b.clone() - a.clone()).relu();

    let e = c - d;
    let f = e.pow(2.0);
    let mut g = f.clone() / 2.0;
    g += 10.0 / f;

    println!("{:.4}", g.0.borrow().data);

    g.backward();

    println!("{:.4}", a.0.borrow().grad);
    println!("{:.4}", b.0.borrow().grad);

    // Read the dataset
    let (x, y) = read_dataset("moon_dataset.csv");
    let n_samples = x.len();

    // Create the MLP
    let mut model = MLP::new(2, &[16, 16, 1]);
    println!("Model: {:?}", model);
    println!("Number of parameters: {}", model.parameters().len());

    // Training loop
    let n_epochs = 2000;
    let mut rng = rand::thread_rng();
    for epoch in 0..n_epochs {
        let mut total_loss = ValueWrapper::new(0.0);
        let mut correct = 0;

        // Shuffle indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        for &i in &indices {
            // Convert input to ValueWrapper
            let input: Vec<ValueWrapper> = x[i].iter().map(|&x| ValueWrapper::new(x)).collect();
            let target = y[i];

            // Forward pass
            let output = model.call(input)[0].clone();

            // Compute loss (hinge loss)
            let mut loss = (ValueWrapper::new(1.0) + (-target * output.clone())).relu();
            total_loss += loss.clone();

            // Compute accuracy
            if (output.0.borrow().data.signum() as i32) == (target as i32) {
                correct += 1;
            }

            // Backward pass
            model.zero_grad();
            loss.backward();

            // Update weights (SGD)
            let learning_rate = 0.01;

            for p in model.parameters() {
                let mut p_mut = p.0.borrow_mut();
                p_mut.data -= learning_rate * p_mut.grad;
            }
        }

        // Print progress
        if epoch % 50 == 0 {
            println!(
                "Epoch {}: Loss = {:.4}, Accuracy = {}/{}",
                epoch,
                total_loss.0.borrow().data,
                correct,
                n_samples
            );
        }
    }

    // Generate points for visualization

    let output_file_name = "decision_boundary_data2.csv";

    let mut file = File::create(output_file_name).unwrap();
    writeln!(file, "x,y,z").unwrap();

    let step = 0.1;
    for j in -30..=30 {
        for i in -30..=30 {
            let x = i as f64 * step;
            let y = j as f64 * step;
            let input = vec![ValueWrapper::new(x), ValueWrapper::new(y)];
            let output = model.call(input)[0].clone();
            writeln!(file, "{},{},{}", x, y, output.0.borrow().data).unwrap();
        }
    }

    println!("Decision boundary data saved to '{}'", output_file_name);
    println!("Use the Python visualization script to see the results.");
}
