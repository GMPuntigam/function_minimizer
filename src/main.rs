mod sub;

use favannat::{MatrixRecurrentFabricator, StatefulFabricator, StatefulEvaluator, MatrixRecurrentEvaluator};
use rand::{
    distributions::{Uniform},
    seq::SliceRandom,
    thread_rng, Rng,
};
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::{cmp::Ordering, fs};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use set_genome::{Genome, Parameters, Structure, Mutations, activations::Activation};

const STEPS: usize = 20;
const N_TRYS: usize = 10;
const N_TESTFUNCTIONS: usize =2;
const SAMPLEPOINTS: usize = 6;
const POPULATION_SIZE: usize = 1000;
const GENERATIONS: usize = 500;
const MAXDEV: f32 = 100.0;

#[derive(Debug)]
pub struct Prediction{
    fitness: f32,
    x_guess: f32,
    x_minus: f32,
    x_plus: f32
}
#[derive(Debug, Copy, Clone)]
pub struct FitnessEval{
    fitness: f32,
    x_worst: f32,
    x_minus: f32,
    x_plus: f32
}



#[derive(Debug)]
pub struct TestfunctionEval{
    fitnessEval: FitnessEval,
    functionName: String,
}

#[derive(Debug)]
pub struct OverallFitness{
    fitness: f32,
    fitnessvec: Vec<TestfunctionEval>
}


fn main() {
    // env::set_var("RUST_BACKTRACE", "1");
    let parameters = Parameters {
        structure: Structure::basic(2*SAMPLEPOINTS, 3),
        mutations: vec![
            Mutations::ChangeWeights {
                chance: 0.8,
                percent_perturbed: 0.5,
                standard_deviation: 0.2,
            },
            Mutations::AddNode {
                chance: 0.05,
                activation_pool: vec![
                    Activation::Sigmoid,
                    Activation::Tanh,
                    Activation::Gaussian,
                    Activation::Step,
                    // Activation::Sine,
                    // Activation::Cosine,
                    Activation::Inverse,
                    Activation::Absolute,
                    Activation::Relu,
                ],
            },
            Mutations::AddConnection { chance: 0.2 },
            Mutations::AddConnection { chance: 0.02 },
            Mutations::AddRecurrentConnection { chance: 0.04 },
            Mutations::RemoveConnection { chance: 0.06 },
            Mutations::RemoveNode { chance: 0.04 }
        ],
    };

    let mut current_population = Vec::with_capacity(POPULATION_SIZE);

    for _ in 0..POPULATION_SIZE {
        current_population.push(Genome::initialized(&parameters))
    }

    let mut champion = current_population[0].clone();
    // let mut summed_diff = 0.0;
    for gen_iter in 0..GENERATIONS {
        // ## Evaluate current nets
        print!("Generation {} of {}\n",gen_iter, GENERATIONS);
        let mut population_fitnesses = current_population
            .par_iter()
            .map(evaluate_net_fitness)
            .enumerate()
            .collect::<Vec<_>>();
        population_fitnesses.sort_unstable_by(|a, b| match (a.1.fitness.is_nan(), b.1.fitness.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => a.1.fitness.partial_cmp(&b.1.fitness).unwrap(),
        });

        // ## Reproduce best nets

        let mut new_population = Vec::with_capacity(POPULATION_SIZE);

        for &(index, _) in population_fitnesses.iter().take(POPULATION_SIZE/10) {
            new_population.push(current_population[index].clone());
        }

        let mut rng = thread_rng();
        while new_population.len() < POPULATION_SIZE {
            let mut child = new_population.choose(&mut rng).unwrap().clone();
            child.mutate(&parameters);
            new_population.push(child);
        }

        champion = current_population[population_fitnesses[0].0].clone();
        // let mut fitness_delta = population_fitnesses.iter().as_slice().sum() ;
        current_population = new_population;
        
        dbg!(population_fitnesses.iter().take(1).collect::<Vec<_>>());
        // dbg!(&population_fitnesses[0].0);
        // print!("Best fitness: {}", &population_fitnesses[0].0)
    }
    print!("{}", Genome::dot(&champion));
    evaluate_champion(&champion, sub::func1, "data/powerfour.txt");
    evaluate_champion(&champion, sub::func2, "data/quadratic_sinus.txt");
    
}

fn evaluate_champion(champion: &Genome, f: fn(f32) -> f32, filepath: &str) {
    let contents = String::from(format!("XVal,x-plus,x-minus\n"));
    fs::write(filepath, contents).expect("Unable to write file");
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(filepath)
        .unwrap();
    let mut x_guess: f32 = 0.0;
    let mut x_minus: f32 = MAXDEV;
    let mut x_plus: f32 = MAXDEV;
    let mut between: Uniform<f32> = Uniform::from(x_guess - x_minus.abs()..x_guess +x_plus.abs());
    let mut x_vals: Vec<f32> = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut f_vals: Vec<f32> = x_vals.iter().enumerate().map(|(_, x)| sub::func2(*x).clone()).collect::<Vec<_>>();
    let mut x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
    let mut x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
    // for _ in 0..N_TRYS {
    let mut evaluator = MatrixRecurrentFabricator::fabricate(champion).expect("didnt work");
    for step in 0..STEPS {
        let input_values: Vec<f32> = [x_vals, f_vals].concat();
        let prediction: Vec<f64> = evaluator.evaluate(input_values.clone().iter().map(|x | *x as f64).collect() );
        x_guess = prediction[0] as f32*(x_max-x_min) + x_min;
        x_minus = x_minus+ x_minus*(prediction[1]as f32-0.5);
        x_plus = x_plus+ x_plus*(prediction[2]as f32-0.5);
        let prediction =Prediction {
            fitness: f(x_guess) + x_minus.abs() + x_plus.abs(),
            x_guess: x_guess,
            x_plus: x_plus.abs(),
            x_minus: x_minus.abs()};
        print!("Champion Step {}: {:?}\n",step, prediction);
        between = Uniform::from(x_guess - x_minus.abs()..x_guess +x_plus.abs());
        x_vals = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
        x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        f_vals = x_vals.iter().enumerate().map(|(_, x)| sub::func2(*x).clone()).collect::<Vec<_>>();
        x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
        x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
        // let contents = String::from(format!());
        if let Err(e) = writeln!(file, "{x_val1},{x_plus},{x_minus}\n",x_val1 = x_guess, x_plus = x_plus.abs(), x_minus = x_minus.abs()) {
            eprintln!("Couldn't write to file: {}", e);
        }
    }
    
    // fs::write("data/out.txt", contents).expect("Unable to write file");
}


fn evaluate_on_testfunction(f: fn(f32) -> f32, mut evaluator:  MatrixRecurrentEvaluator) -> FitnessEval {
    let mut fitness_vec= Vec::with_capacity(N_TRYS);
    let mut x_guess: f32;
    let mut x_worst: f32 = 0.0;
    let mut x_minus: f32 = MAXDEV;
    let mut x_plus: f32 = MAXDEV;
    for _ in 0..N_TRYS {
        x_guess = 0.0;
        x_minus = MAXDEV;
        x_plus = MAXDEV;
        let mut lower_range_limit = x_guess - x_minus.abs();
        let mut upper_range_limit = x_guess + x_plus.abs();
        let mut between = Uniform::from(lower_range_limit..upper_range_limit);
        let mut x_vals: Vec<f32>  = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
        x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x).clone()).collect::<Vec<_>>();
        // let f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
        // f_vals = f_vals.iter().enumerate().map(|(_, x)| *x/f_max).collect::<Vec<_>>();
        let mut x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
        let mut x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
        for _ in 0..STEPS {
            let input_values: Vec<f32> = [x_vals, f_vals].concat();
            let prediction: Vec<f64> = evaluator.evaluate(input_values.clone().iter().map(|x | *x as f64).collect() );
            x_guess = prediction[0] as f32*(x_max-x_min) + x_min;
            x_minus = x_minus+ x_minus*((prediction[1]as f32)-0.5);
            x_plus = x_plus+ x_plus*((prediction[2]as f32)-0.5);
            if x_guess.is_nan() || x_minus.is_nan()|| x_plus.is_nan(){
                return FitnessEval {
                    fitness: f32::INFINITY,
                    x_worst: f32::INFINITY,
                    x_plus: f32::INFINITY,
                    x_minus: f32::INFINITY};
            }
            

            // dbg!([x_guess, x_minus, x_plus]);
            lower_range_limit = x_guess - x_minus.abs();
            upper_range_limit = x_guess + x_plus.abs();
            if upper_range_limit - lower_range_limit == f32::INFINITY {
                // dbg!([x_guess, x_minus, x_plus]);
                return FitnessEval {
                    fitness: f32::INFINITY,
                    x_worst: f32::INFINITY,
                    x_plus: f32::INFINITY,
                    x_minus: f32::INFINITY};
            }
            if lower_range_limit >= upper_range_limit{
                // dbg!([lower_range_limit,x_guess, upper_range_limit]);
                x_vals = vec![x_guess; SAMPLEPOINTS];
                f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x).clone()).collect::<Vec<_>>();
            } else {
                between = Uniform::from(lower_range_limit..upper_range_limit);
                x_vals = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
                x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x).clone()).collect::<Vec<_>>();
            }
            
            
            x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
            x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
        }
        if f(x_guess) > f(x_worst)
        {
            x_worst = x_guess;
        }
        
        fitness_vec.push(f(x_guess) + (x_minus.abs() + x_plus.abs())*0.25);
        
    }
    let fitness = fitness_vec.iter().copied().fold(f32::NAN, f32::max);
    FitnessEval{
        fitness: fitness,
        x_worst: x_worst,
        x_plus: x_plus.abs(),
        x_minus: x_minus.abs()
    }
}

fn evaluate_net_fitness(net: &Genome) -> OverallFitness {
    let mut fitness_vec: Vec<FitnessEval>= Vec::with_capacity(N_TESTFUNCTIONS);
    // let mut x_guess_vec= Vec::with_capacity(N_TRYS);
    
    let mut evaluator = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    let mut fitness = evaluate_on_testfunction(sub::func1, evaluator);
    fitness_vec.push(fitness);
    evaluator = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    fitness = evaluate_on_testfunction(sub::func2, evaluator);
    fitness_vec.push(fitness);
    // dbg!(&delta_f_values);
    fitness_vec.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    OverallFitness{
        fitness: fitness_vec[1].fitness,
        fitnessvec: vec![TestfunctionEval {
            fitnessEval: fitness_vec[0],
            functionName: String::from("Power four")
        } ,
        TestfunctionEval {
            fitnessEval:fitness_vec[1],
            functionName: String::from("Quadratic Sinus")
        }]}        
}

// fn net_as_dot(net: &Genome) -> String {
//     let mut dot = "digraph {\n".to_owned();

//     for node in net.nodes() {
//         dot.push_str(&format!("{} [label={:?}];\n", node.id.0, node.activation));
//     }

//     for connection in net.connections() {
//         dot.push_str(&format!(
//             "{} -> {} [label={:?}];\n",
//             connection.input.0, connection.output.0, connection.weight
//         ));
//     }

//     dot.push_str("}\n");
//     dot
// }
