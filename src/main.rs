mod sub;
mod data_structs;
use data_structs::{FitnessEval, OverallFitness, TestfunctionEval, AllXVals};
use favannat::{MatrixRecurrentFabricator, StatefulFabricator, StatefulEvaluator, MatrixRecurrentEvaluator};
use rand::{
    distributions::{Uniform},
    seq::SliceRandom,
    thread_rng, Rng,
};
use std::{fs::OpenOptions};
use std::io::prelude::*;
use std::{cmp::Ordering, fs};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use set_genome::{Genome, Parameters, Structure, Mutations, activations::Activation};

use crate::data_structs::Prediction;

const STEPS: usize = 25;
const N_TRYS: usize = 10;
const N_TESTFUNCTIONS: usize =4;
const SAMPLEPOINTS: usize = 6;
const POPULATION_SIZE: usize = 600;
const GENERATIONS: usize = 150;
const MAXDEV: f32 = 100.0;
const FUNCTION_HANDLE_UNTRAINED: fn(f32) -> f32 = |x|  (x-3.0).abs()+ 1.5*x.sin();





fn main() {
    // env::set_var("RUST_BACKTRACE", "1");
    let parameters = Parameters {
        structure: Structure { number_of_inputs: (2*SAMPLEPOINTS + 1), number_of_outputs: (3), percent_of_connected_inputs: (0.5), outputs_activation: (Activation::Sigmoid), seed: (42) },
        // structure: Structure::basic(2*SAMPLEPOINTS + 2, 3),
        mutations: vec![
            Mutations::ChangeWeights {
                chance: 0.8,
                percent_perturbed: 0.5,
                standard_deviation: 0.2,
            },
            Mutations::AddNode {
                chance: 0.1,
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
            Mutations::AddRecurrentConnection { chance: 0.1 },
            Mutations::RemoveConnection { chance: 0.05 },
            Mutations::RemoveConnection { chance: 0.01 },
            Mutations::RemoveNode { chance: 0.05 }
        ],
    };

    let mut network_population = Vec::with_capacity(POPULATION_SIZE);

    for _ in 0..POPULATION_SIZE {
        network_population.push(Genome::initialized(&parameters))
    }

    let mut champion = network_population[0].clone();
    // let mut summed_diff = 0.0;
    for gen_iter in 0..GENERATIONS {
        // ## Evaluate current nets
        let mut rng = rand::thread_rng();
        let y =rng.gen_range(0.0..10.0);
        let y2 =rng.gen_range(0.0..10.0);
        
        print!("Generation {} of {}\n",gen_iter, GENERATIONS);
        let mut population_fitnesses = network_population
            .par_iter()
            .map(|a|evaluate_net_fitness(a, y, y2)).enumerate()
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
            new_population.push(network_population[index].clone());
        }

        let mut rng = thread_rng();
        while new_population.len() < POPULATION_SIZE {
            let mut child = new_population.choose(&mut rng).unwrap().clone();
            if let Err(_) = child.mutate(&parameters) {
            };
            new_population.push(child);
        }

        champion = network_population[population_fitnesses[0].0].clone();
        // let mut fitness_delta = population_fitnesses.iter().as_slice().sum() ;
        network_population = new_population;
        
        dbg!(population_fitnesses.iter().take(1).collect::<Vec<_>>());
        // dbg!(&population_fitnesses[0].0);
        // print!("Best fitness: {}", &population_fitnesses[0].0)
    }
    print!("Global net\n {}", Genome::dot(&champion));
    print!("Local net\n {}", Genome::dot(&champion));
    evaluate_champion(&champion, sub::func1, "data/powerfour.txt");
    evaluate_champion(&champion, sub::func2, "data/quadratic_sinus.txt");
    evaluate_champion(&champion, sub::func3, "data/abs.txt");
    evaluate_champion(&champion, sub::func4, "data/x-squared.txt");
    evaluate_champion(&champion, FUNCTION_HANDLE_UNTRAINED, "data/x-abs+sin.txt");
    
}
fn generate_sampleheader()-> String {
    let mut helpvec = Vec::with_capacity(SAMPLEPOINTS);
    for i in 0..SAMPLEPOINTS {
        helpvec.push(format!("x{i}", i=i));
    }
    for i in 0..SAMPLEPOINTS {
        helpvec.push(format!("f(x{i})", i=i));
    }
    let joined = helpvec.join(",").to_owned();
    joined
}

fn evaluate_values(evaluator: &mut MatrixRecurrentEvaluator, input_values_normalized: Vec<f32>, all_xvals: &AllXVals, inneronly:bool) ->(f32, f32, f32) {
    // let x_guess_old = all_xvals.x_guess.clone();
    let prediction: Vec<f64> = evaluator.evaluate(input_values_normalized.clone().iter().map(|x | *x as f64).collect() );
    let pred_x_guess = prediction[0] as f32;
    let pred_x_minus = prediction[1] as f32;
    let pred_x_plus = prediction[2] as f32;
    let x_minus;
    let x_plus;
    // if pred_x_minus < 0.0{
    //     pred_x_minus = 0.0;
    // }   else if pred_x_minus > 1.0 {
    //     pred_x_minus = 1.0;
    // }
    // if pred_x_plus < 0.0{
    //     pred_x_plus = 0.0;
    // }   else if pred_x_plus > 1.0 {
    //     pred_x_plus = 1.0;
    // }
    // if pred_x_guess < 0.0{
    //     pred_x_guess = 0.0;
    // }   else if pred_x_guess > 1.0 {
    //     pred_x_guess = 1.0;
    // }
    // dbg!(&pred_x_guess, x_max.clone()-x_min.clone(), x_min, x_minus);
    let x_guess = (pred_x_guess)*(all_xvals.x_max-all_xvals.x_min)+all_xvals.x_min -all_xvals.x_minus + all_xvals.x_guess;
    // let delta_x_guess = x_guess - x_guess_old; 
    let x_prediction_in_intervall = (pred_x_guess)*(all_xvals.x_max-all_xvals.x_min) + all_xvals.x_min;
    if inneronly{
        x_minus = x_prediction_in_intervall*(pred_x_minus);
        x_plus = (all_xvals.x_max-x_prediction_in_intervall)*(pred_x_plus);
    }   else {
        x_minus = all_xvals.x_minus*(0.5+pred_x_minus);
        x_plus = all_xvals.x_plus*(0.5+pred_x_plus);
    }
    (x_guess, x_minus, x_plus)
}

fn evaluate_champion(net: &Genome, f: fn(f32) -> f32, filepath: &str) {
    let mut all_xvals = AllXVals{
        x_guess : 0.0,
        x_max: 0.0,
        x_min: 0.0,
        x_minus: 0.0,
        x_plus: 0.0
    };
    let contents = String::from(format!(",x-best,Step,XVal,x-plus,x-minus\n"));
    let mut sampleheader = generate_sampleheader();
    sampleheader.push_str(&contents);
    fs::write(filepath, sampleheader).expect("Unable to write file");
    // let mut rng = rand::thread_rng();
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(filepath)
        .unwrap();
    let mut x_vals_normalized;
    let x_guess : f32 = 0.0;
    let x_minus: f32 = MAXDEV;
    let x_plus: f32 = MAXDEV;
    // let x_best: f32 = rng.gen_range(0.0..x_minus + x_plus);
    let mut between: Uniform<f32> = Uniform::from(0.0..x_minus + x_plus);
    let mut x_vals: Vec<f32> = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mut f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x-x_minus).clone()).collect::<Vec<_>>();
    let mut f_vals_normalized : Vec<f32> = normalise_vector(&f_vals);
    // norm f_vals
    // let mut f_max;
    // f_vals = f_vals.iter().enumerate().map(|(_, x)| *x/f_max).collect::<Vec<_>>();
    // get max and min of x_samples for interpretation of prediction
    let x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
    let x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
    // norm x_vals
    x_vals_normalized = normalise_vector(&x_vals);
    // x_vals should be between 0 and 1 now, the first always0, the last always 1.
    // for _ in 0..N_TRYS {
    let mut evaluator = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    // let mut evaluator_local = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    // for step in 0..STEPS {
    all_xvals.x_guess = x_guess;
    all_xvals.x_max = x_max;
    all_xvals.x_min = x_min;
    all_xvals.x_minus = x_minus;
    all_xvals.x_plus = x_plus;
    let mut step = 0;
        // while ((x_minus + x_plus) > 1.0e-4_f32) & (step < STEPS) & !extremum_found {
        while step < STEPS {
            // dbg!(&f_vals);
            step+=1;
            let input_values: Vec<f32> = [x_vals.clone().iter().enumerate().map(|(_, x)| *x-all_xvals.x_minus+all_xvals.x_guess).collect::<Vec<_>>(), f_vals.clone(), vec![(STEPS -step) as f32/STEPS as f32]].concat();    
            let input_values_normalized: Vec<f32> = [x_vals_normalized, f_vals_normalized, vec![(STEPS -step) as f32/STEPS as f32]].concat();
            

            network_step(input_values_normalized, &mut evaluator, &mut all_xvals, false);
            let prediction =Prediction {
                fitness: f(all_xvals.x_guess) + (all_xvals.x_minus + all_xvals.x_plus),
                x_guess: all_xvals.x_guess,
                x_plus: all_xvals.x_plus,
                x_minus: all_xvals.x_minus};
            print!("champion_global Step {}: {:?}\n",step, prediction); 
            let samples_string: String = input_values.iter().map( |&id| id.to_string() + ",").collect();
            if let Err(e) = writeln!(file, "{samples}{step},{x_val1},{x_plus},{x_minus}",samples =samples_string, step=step, x_val1 = all_xvals.x_guess, x_plus = all_xvals.x_plus, x_minus = all_xvals.x_minus) {
                eprintln!("Couldn't write to file: {}", e);
            }
            between = Uniform::from(0.0..all_xvals.x_minus + all_xvals.x_plus);
            x_vals = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
            x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x+all_xvals.x_guess-all_xvals.x_minus).clone()).collect::<Vec<_>>();
            // f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
            f_vals_normalized = normalise_vector(&f_vals);
            all_xvals.x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
            all_xvals.x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
            x_vals_normalized = normalise_vector(&x_vals);
        } 
        // let mut  extremum_found = false;
        // while (step < STEPS*2) & !extremum_found {
        //     extremum_found = check_extremum(f_vals.clone(), x_max, x_min);
        //     // dbg!(&f_vals);
        //     step+=1;    
        //     let input_values: Vec<f32> = [x_vals.clone().iter().enumerate().map(|(_, x)| *x-all_xvals.x_minus+all_xvals.x_guess).collect::<Vec<_>>(), f_vals.clone(), vec![(STEPS*2 -step) as f32/STEPS as f32]].concat();    
        //     let input_values_normalized: Vec<f32> = [x_vals_normalized, f_vals_normalized, vec![(STEPS*2 -step) as f32/STEPS as f32]].concat();
        //     network_step(input_values_normalized, &mut evaluator_local, &mut all_xvals, true);
        //     let prediction =Prediction {
        //         fitness: f(all_xvals.x_guess) + (all_xvals.x_minus + all_xvals.x_plus),
        //         x_guess: all_xvals.x_guess,
        //         x_plus: all_xvals.x_plus,
        //         x_minus: all_xvals.x_minus};
        //     print!("champion_local Step {}: {:?}\n",step, prediction); 
        //     let samples_string: String = input_values.iter().map( |&id| id.to_string() + ",").collect();
        //     if let Err(e) = writeln!(file, "{samples}{step},{x_val1},{x_plus},{x_minus}",samples =samples_string, step=step, x_val1 = all_xvals.x_guess, x_plus = all_xvals.x_plus, x_minus = all_xvals.x_minus) {
        //         eprintln!("Couldn't write to file: {}", e);
        //     }
        //     between = Uniform::from(0.0..all_xvals.x_minus + all_xvals.x_plus);
        //     x_vals = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
        //     x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        //     f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x+all_xvals.x_guess-all_xvals.x_minus).clone()).collect::<Vec<_>>();
        //     // f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
        //     f_vals_normalized = normalise_vector(&f_vals);
        //     all_xvals.x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
        //     all_xvals.x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
        //     x_vals_normalized = normalise_vector(&x_vals);

        // }    
}

// fn check_extremum(f_vals:Vec<f32>, x_max: f32, x_min: f32) -> bool {
//     let precision:f32= 1.0e-2_f32;
//     let mut extremum_found = false;
//     let f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
//     let f_min = f_vals.iter().copied().fold(f32::NAN, f32::min);
//     let dif_first_to_last = (f_vals[0] -f_vals[SAMPLEPOINTS-1]).abs();
//     if f_max - f_min == 0.0 {
//         extremum_found = false;
//     } else if ((f_max - f_min)/(x_max-x_min) <= 0.1) & (dif_first_to_last < precision) {
//         extremum_found = true;
//         // dbg!(f_max, f_min, x_max, x_min);
//         // dbg!((f_max - f_min)/(x_max-x_min));
//         // dbg!(dif_first_to_last);
//     }
//     extremum_found
// }

fn network_step(input_values: Vec<f32>, evaluator: &mut MatrixRecurrentEvaluator,  all_xvals: &mut AllXVals, inneronly:bool) -> () {
    
            // let prediction: Vec<f64> = evaluator.evaluate(input_values.clone().iter().map(|x | *x as f64).collect() );
            // dbg!(&prediction);
            
            // x_guess = (prediction[0] as f32)*(x_max) + (1.0-prediction[0] as f32)*x_min;
            (all_xvals.x_guess, all_xvals.x_minus, all_xvals.x_plus) = evaluate_values(evaluator, input_values, all_xvals, inneronly);
            // dbg!(&x_minus, &x_guess, &x_plus);
            // if simulation runs out of bounds, return inf values

            

            // dbg!([x_guess, x_minus, x_plus]);
            

            // x_vals should be between 0 and 1 now, the first always0, the last always 1.
            // dbg!("new xvals",&x_vals);

            // if f(x_guess) + g(x_guess, y, y2) - (f(x_best) + g(x_best, y, y2)) < 0.0 {
            //     counter = 0
            // } else {
            //     counter+=1;
            // }
            // if f(x_guess) + g(x_guess, y, y2) < (f(x_best) + g(x_best, y, y2))
            // {
            //     x_best = x_guess;
            // }
}

fn normalise_vector(vec: &Vec<f32>) -> Vec<f32> {
    let x_max = vec.iter().copied().fold(f32::NAN, f32::max);
    let x_min = vec.iter().copied().fold(f32::NAN, f32::min);
    let vec_normalized = vec.iter().enumerate().map(|(_, x)| (*x-x_min)/(x_max-x_min)).collect::<Vec<_>>();
    vec_normalized
}

fn evaluate_on_testfunction(f: fn(f32) -> f32, y: f32, z:f32, evaluator:  &mut MatrixRecurrentEvaluator) -> FitnessEval {
    let mut fitness_vec= Vec::with_capacity(N_TRYS);
    let mut x_guess: f32;
    let mut x_worst: f32 = 0.0;
    let mut x_minus: f32;
    let mut x_plus: f32;
    let mut step_total: usize = 0;
    let mut all_xvals = AllXVals{
        x_guess : 0.0,
        x_max: 0.0,
        x_min: 0.0,
        x_minus: 0.0,
        x_plus: 0.0
    };
    let g: fn(f32, f32, f32) -> f32 = |x,y,z|  y*(x*z).sin();
    for try_enum in 0..N_TRYS {
        // let mut counter: usize = 0;        
        // let mut x_best: f32 = rng.gen_range(0.0..x_minus + x_plus);
        x_guess = 0.0;
        x_minus = MAXDEV;
        x_plus = MAXDEV;
        // set upper range of randomness generation
        let mut lower_range_limit;
        let mut upper_range_limit;
        // generate positive random values within the span of xminus + x plus
        let mut between = Uniform::from(0.0..x_minus + x_plus);
        // generate some random values
        let mut x_vals: Vec<f32>  = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
        // sort the values
        x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // calculate the function at the xvalues
        let mut f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x-x_minus).clone() +g(*x-x_minus, y, z)).collect::<Vec<_>>();
        // norm f_vals
        let f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
        // let f_min_start = f_vals.iter().copied().fold(f32::NAN, f32::min);
        // let f_max_start = f_max.clone();
        let mut f_vals_normalized = f_vals.iter().enumerate().map(|(_, x)| *x/f_max).collect::<Vec<_>>();
        

        // get max and min of x_samples for interpretation of prediction
        let x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
        let x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
        all_xvals.x_guess = x_guess;
        all_xvals.x_max = x_max;
        all_xvals.x_min = x_min;
        all_xvals.x_minus = x_minus;
        all_xvals.x_plus = x_plus;
        // norm x_vals
        let mut x_vals_normalized = x_vals.iter().enumerate().map(|(_, x)| (*x-x_min)/(x_max-x_min)).collect::<Vec<_>>();
        // x_vals should be between 0 and 1 now, the first always0, the last always 1.
        // dbg!(&x_vals);
        // for step in 0..STEPS {
        
        let mut step = 0;
        // while ((x_minus + x_plus) > 1.0e-4_f32) & (step < STEPS) & !extremum_found {
        while step < STEPS {
            
            // dbg!(&f_vals);
            step+=1;    
            let input_values: Vec<f32> = [x_vals_normalized, f_vals_normalized, vec![(STEPS -step) as f32/STEPS as f32]].concat();
            
            // dbg!("before",&all_xvals);
            network_step(input_values, evaluator, &mut all_xvals, false);
            // dbg!("after",&all_xvals);
            if all_xvals.x_guess.is_nan() || all_xvals.x_minus.is_nan()|| all_xvals.x_plus.is_nan(){
                return FitnessEval {
                    fitness: f32::INFINITY,
                    x_worst: f32::INFINITY,
                    x_plus: f32::INFINITY,
                    x_minus: f32::INFINITY,
                    step: STEPS};
            }
            lower_range_limit = all_xvals.x_guess - all_xvals.x_minus;
            upper_range_limit = all_xvals.x_guess + all_xvals.x_plus;
            if upper_range_limit - lower_range_limit == f32::INFINITY {
                // dbg!([x_guess, x_minus, x_plus]);
                return FitnessEval {
                    fitness: f32::INFINITY,
                    x_worst: f32::INFINITY,
                    x_plus: f32::INFINITY,
                    x_minus: f32::INFINITY,
                    step: STEPS
                } 
            };
            if lower_range_limit >= upper_range_limit{
                // if lower range limit is buggy, just return multiple x guesses
                // x_vals = vec![x_guess; SAMPLEPOINTS];
                // f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x).clone() +g(*x, y, y2)).collect::<Vec<_>>();
                return FitnessEval {
                    fitness: f32::INFINITY,
                    x_worst: f32::INFINITY,
                    x_plus: f32::INFINITY,
                    x_minus: f32::INFINITY,
                    step: STEPS
                } 
            }else {
                between = Uniform::from(0.0..all_xvals.x_minus + all_xvals.x_plus);
                x_vals = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
                x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x+all_xvals.x_guess-all_xvals.x_minus).clone() +g(*x+all_xvals.x_guess-all_xvals.x_minus, y, z)).collect::<Vec<_>>();
                // f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
                f_vals_normalized = normalise_vector(&f_vals);
            };
            all_xvals.x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
            all_xvals.x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
            x_vals_normalized = normalise_vector(&x_vals);
        } 
        // print new lower range, guessed x and upper range
        // dbg!([lower_range_limit,x_guess, upper_range_limit]);
        // if extremum_found {
        // fitness_vec.push(step as f32);
        // }
        if try_enum == 0 {
            x_worst = all_xvals.x_guess;
        }
        if f(all_xvals.x_guess) + g(all_xvals.x_guess, y, z) > f(x_worst) + g(x_worst, y, z)
        {
            x_worst = all_xvals.x_guess;
        }
        
        step_total = step_total + step;
        // fitness_vec.push(f(x_guess) + g(x_guess, y, y2) + (x_minus + x_plus)*(1.0+step as f32));
        // fitness_vec.push((f(x_guess) + g(x_guess, y, y2) - f_min_start)/f_max_start + (1.0/STEPS as f32)* step as f32);
        // fitness_vec.push(f(x_guess) + g(x_guess, y, y2) + (x_minus + x_plus)*(1.0+step as f32) + counter as f32);
        let fitness = f(all_xvals.x_guess) + g(all_xvals.x_guess, y, z);
        fitness_vec.push(FitnessEval{
            fitness: fitness,
            x_worst: x_worst,
            x_plus: all_xvals.x_plus,
            x_minus: all_xvals.x_minus,
            step: step_total
        });
        
    }
    fitness_vec.sort_unstable_by(|a, b| match (a.fitness.is_nan(), b.fitness.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.fitness.partial_cmp(&b.fitness).unwrap(),
    });
    // let fitness = fitness_vec.iter().copied().fold(f32::NAN, f32::max);
    // let fitness = fitness_vec.iter().sum::<f32>() as f32;
    fitness_vec[0]
    
}

fn evaluate_net_fitness(net: &Genome, y: f32, z:f32) -> OverallFitness {
    let mut fitness_vec: Vec<f32>= Vec::with_capacity(N_TESTFUNCTIONS);
    let fitness_max: f32;
    let mut eval_vec: Vec<FitnessEval>= Vec::with_capacity(N_TESTFUNCTIONS);
    // let mut x_guess_vec= Vec::with_capacity(N_TRYS);
    
    let mut evaluator_global = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    // let mut evaluator_local = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    let mut fitness = evaluate_on_testfunction(sub::func1,y, z, &mut evaluator_global);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    evaluator_global.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func2,y, z, &mut evaluator_global);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    evaluator_global.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func3,y, z, &mut evaluator_global);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    evaluator_global.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func4,y, z, &mut evaluator_global);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    // dbg!(&delta_f_values);
    fitness_max = fitness_vec.iter().copied().fold(f32::NAN, f32::max);
    // fitness_max= fitness_vec.iter().sum::<f32>() as f32/4.0;
    OverallFitness{
        fitness: fitness_max,
        fitnessvec: vec![TestfunctionEval {
            fitness_eval: eval_vec[0],
            function_name: String::from("Power four")
        } ,
        TestfunctionEval {
            fitness_eval:eval_vec[1],
            function_name: String::from("Quadratic Sinus")
        },
        TestfunctionEval {
            fitness_eval:eval_vec[2],
            function_name: String::from("Abs")
        },
        TestfunctionEval {
            fitness_eval:eval_vec[3],
            function_name: String::from("X-Squared")
        }]}        
}