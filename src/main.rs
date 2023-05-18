mod sub;
mod data_structs;
mod util;
use chrono::{Utc, Datelike};
use data_structs::{FitnessEval, OverallFitness, TestfunctionEval, AllXVals};
use favannat::{MatrixRecurrentFabricator, StatefulFabricator, StatefulEvaluator, MatrixRecurrentEvaluator};
use rand::{
    seq::SliceRandom,
    thread_rng, Rng,
};
use std::{fs::{OpenOptions}};
use std::io::prelude::*;
use std::{cmp::Ordering, fs};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use set_genome::{Genome, Parameters, Structure, Mutations, activations::Activation};
use std::time::{Instant};
use crate::{data_structs::Prediction};

const DIMENSIONS: usize = 2;
const STEPS: usize = 200;
const N_TRYS: usize = 3;
const N_TESTFUNCTIONS: usize =4;
const SAMPLEPOINTS: usize = 6;
const POPULATION_SIZE: usize = 1000;
const GENERATIONS: usize = 100;
const TRAINFROMRESULTS: bool=false;
const FUNCTION_HANDLE_UNTRAINED: fn(&Vec<f32>) -> f32 = |x|  (x[0]-3.14).powi(2) + (x[1]-2.72).powi(2)+ (3.0*x[0] +1.41).sin() + (4.0*x[1] -1.73).sin();





fn main() {
    let max_dev: Vec<f32> = vec![100.0, 50.0];
    // let n_inputs = (DIMENSIONS+1)*(SAMPLEPOINTS).pow(DIMENSIONS.try_into().unwrap()) + 1;
    // let n_inputs = 2*(SAMPLEPOINTS).pow(DIMENSIONS.try_into().unwrap()) + 1;
    let n_inputs = 2*(SAMPLEPOINTS) + 1;
    let parameters = Parameters {
        // structure: Structure { number_of_inputs: n_inputs, number_of_outputs: (3*DIMENSIONS), percent_of_connected_inputs: (1.0), outputs_activation: (Activation::Sigmoid), seed: (42) },
        structure: Structure { number_of_inputs: n_inputs, number_of_outputs: 3, percent_of_connected_inputs: (1.0), outputs_activation: (Activation::Sigmoid), seed: (42) },
        // structure: Structure::basic(2*SAMPLEPOINTS + 2, 3),
        mutations: vec![
            Mutations::ChangeWeights {
                chance: 0.8,
                percent_perturbed: 0.3,
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
                    // Activation::Absolute,
                    Activation::Relu,
                ],
            },
            Mutations::AddConnection { chance: 0.2 },
            Mutations::AddConnection { chance: 0.02 },
            Mutations::AddRecurrentConnection { chance: 0.1 },
            Mutations::AddRecurrentConnection { chance: 0.01 },
            Mutations::RemoveConnection { chance: 0.01 },
            Mutations::RemoveConnection { chance: 0.001 },
            Mutations::RemoveNode { chance: 0.1 }
        ],
    };
    let mut network_population = Vec::with_capacity(POPULATION_SIZE);
    let gen: Genome;
    if TRAINFROMRESULTS{
        let champion_bytes = &fs::read_to_string(format!("example_runs/2023-5-17_somewhat_convergent/winner.json")).expect("Unable to read champion");
        gen = serde_json::from_str(champion_bytes).unwrap();
    }else {
        gen = Genome::initialized(&parameters)
    }

    for _ in 0..POPULATION_SIZE {  
        network_population.push(gen.clone())
    }
    
    let mut champion = network_population[0].clone();
    // let mut summed_diff = 0.0;
    for gen_iter in 0..GENERATIONS {
        // ## Evaluate current nets
        let mut rng = rand::thread_rng();
        let y =rng.gen_range(-5.0..5.0);
        let y2 =rng.gen_range(-5.0..5.0);
        
        print!("Generation {} of {}\n",gen_iter, GENERATIONS);
        let mut population_fitnesses = network_population
            .par_iter()
            .map(|a|evaluate_net_fitness(a, y, y2, &max_dev)).enumerate()
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
    // save the champion net
    print!("Global net\n {}", Genome::dot(&champion));
    let now = Utc::now();
    let time_stamp = format!("{year}-{month}-{day}", year = now.year(), month = now.month(), day = now.day());
    // fs::try_exists(format!("example_runs/{}/", time_stamp)).is_err();
    fs::create_dir_all(format!("example_runs/{}/", time_stamp)).expect("Unable to create dir");
    fs::write(
        format!("example_runs/{}/winner.json", time_stamp),
        serde_json::to_string(&champion).unwrap(),
    )
    .expect("Unable to write file");
    evaluate_champion(&champion, sub::func1, format!("example_runs/{}/powerfour.txt", time_stamp), false, &max_dev);
    evaluate_champion(&champion, sub::func2, format!("example_runs/{}/quadratic-sinus.txt", time_stamp), false, &max_dev);
    let start = Instant::now();
    evaluate_champion(&champion, sub::func2, format!("example_runs/{}/quadratic-sinus.txt", time_stamp), true, &max_dev);
    let duration = start.elapsed();
    println!("Champion took {:?} to find the minimum", duration);
    evaluate_champion(&champion, sub::func3, format!("example_runs/{}/abs.txt", time_stamp), false, &max_dev);
    evaluate_champion(&champion, sub::func4, format!("example_runs/{}/x-squared.txt", time_stamp), false, &max_dev);
    evaluate_champion(&champion, FUNCTION_HANDLE_UNTRAINED, format!("example_runs/{}/x-abs+sin.txt", time_stamp), false, &max_dev);
    
}




fn evaluate_champion(net: &Genome, f: fn(&Vec<f32>) -> f32, filepath: String, timeonly: bool, max_dev: &Vec<f32>) {
    
    if !timeonly {
        let contents = String::from(format!(";Step;XVal;x-plus;x-minus\n"));
        let mut sampleheader = util::generate_sampleheader(SAMPLEPOINTS,DIMENSIONS);
        sampleheader.push_str(&contents);
        fs::write(filepath.clone(), sampleheader).expect("Unable to write file");

    }
    // let mut x_vals_normalized;
    let x_guess: Vec<f32> = vec![0.0; DIMENSIONS];
    let mut function_domain = util::FunctionDomain{
        dimensions:DIMENSIONS,
        samplepoints: SAMPLEPOINTS,
        upper_limits: (0..DIMENSIONS).map(|i| x_guess[i] + max_dev[i]).collect::<Vec<_>>(),
        lower_limits: (0..DIMENSIONS).map(|i| x_guess[i] - max_dev[i]).collect::<Vec<_>>()
    };
    // let mut set_of_sample_points = util::generate_samplepoint_matrix_single_dimension_view(function_domain);
    let mut all_xvals = AllXVals{
        x_guess : vec![0.0; DIMENSIONS],
        x_max: max_dev.clone(),
        x_min: max_dev.clone(),
        x_minus: max_dev.clone(),
        x_plus: max_dev.clone()
    };
    let mut f_vals: Vec<f32>;
    let mut f_vals_normalized: Vec<f32>;
    let mut evaluator = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    let mut step = 0;
        while step < STEPS {
            for dim in 0..DIMENSIONS{
                step+=1;
                let set_of_sample_points = util::generate_1d_samplepoints(function_domain, dim);
                all_xvals.x_max= set_of_sample_points.max;
                all_xvals.x_min= set_of_sample_points.min;
                f_vals = set_of_sample_points.coordinates.iter().enumerate().map(|(_, x)| f(x)).collect::<Vec<_>>();
                // norm f_vals
                let f_max: f32 = f_vals.iter().copied().fold(f32::NAN, f32::max);
                f_vals_normalized = f_vals.iter().enumerate().map(|(_, x)| *x/f_max).collect::<Vec<_>>();
                let input_values: Vec<f32> = [set_of_sample_points.coordinates_by_dimension_normalised[dim].clone(), f_vals_normalized.clone(), vec![(STEPS -step) as f32/STEPS as f32]].concat();
                network_step(input_values, &mut evaluator, &mut all_xvals, dim, false);
                let prediction =Prediction {
                    fitness: f(&all_xvals.x_guess),
                    x_guess: all_xvals.x_guess.clone(),
                    x_plus: all_xvals.x_plus.clone(),
                    x_minus: all_xvals.x_minus.clone()};
                if !timeonly {
                    let mut file_write = OpenOptions::new()
                    .write(true)
                    .append(true)
                    .open(filepath.clone())
                    .unwrap();
                    print!("champion_global Step {}: {:?}\n",step, prediction); 
                    let coordinates_string: String = set_of_sample_points.coordinates.iter().map(|x|  x.iter().map( |&id| id.to_string()).collect::<Vec<String>>().join(",")).collect::<Vec<String>>().join(";");
                    let f_vals_string: String = f_vals.iter().map( |&id| id.to_string()).collect::<Vec<String>>().join(";");
                    let x_guess_string: String = all_xvals.x_guess.clone().into_iter().map(|i| i.to_string()).collect::<Vec<String>>().join(",");
                    let x_plus_string: String = all_xvals.x_plus.clone().into_iter().map(|i| i.to_string()).collect::<Vec<String>>().join(",");
                    let x_minus_string: String = all_xvals.x_minus.clone().into_iter().map(|i| i.to_string()).collect::<Vec<String>>().join(",");
                    if let Err(e) = writeln!(file_write, "{coords};{fvals};{step};{x_val1};{x_plus};{x_minus}",coords =coordinates_string,fvals=f_vals_string, step=step, x_val1 =x_guess_string , x_plus = x_plus_string, x_minus = x_minus_string) {
                        eprintln!("Couldn't write to file: {}", e);
                    }
                }
                function_domain = util::FunctionDomain{
                    dimensions:DIMENSIONS,
                    samplepoints: SAMPLEPOINTS,
                    upper_limits: (0..DIMENSIONS).map(|i| all_xvals.x_guess[i] + all_xvals.x_plus[i]).collect::<Vec<_>>(),
                    lower_limits: (0..DIMENSIONS).map(|i| all_xvals.x_guess[i] - all_xvals.x_minus[i]).collect::<Vec<_>>()
            };
        }
    } 
}


// fn evaluate_values(evaluator: &mut MatrixRecurrentEvaluator, input_values_normalized: Vec<f32>, all_xvals: &AllXVals, dim:usize, inneronly:bool) ->(Vec<f32>, Vec<f32>, Vec<f32>) {
fn evaluate_values(evaluator: &mut MatrixRecurrentEvaluator, input_values_normalized: Vec<f32>, all_xvals: &AllXVals, dim:usize, inneronly:bool) ->(f32, f32, f32) {
    let prediction: Vec<f64> = evaluator.evaluate(input_values_normalized.clone().iter().map(|x | *x as f64).collect() );
    let pred_x_guess: f32 = prediction[0] as f32;
    let pred_x_minus: f32 = prediction[1] as f32;
    let pred_x_plus: f32 = prediction[2] as f32;
    let x_minus: f32;
    let x_plus: f32;
    let x_guess: f32 = pred_x_guess*(all_xvals.x_max[dim]-all_xvals.x_min[dim])+all_xvals.x_min[dim];
    x_minus = all_xvals.x_minus[dim]*(0.5+pred_x_minus);
    x_plus = all_xvals.x_plus[dim]*(0.5+pred_x_plus);
    (x_guess, x_minus, x_plus)
}

fn network_step(input_values: Vec<f32>, evaluator: &mut MatrixRecurrentEvaluator,  all_xvals: &mut AllXVals, dim:usize, inneronly:bool) -> () {
            (all_xvals.x_guess[dim], all_xvals.x_minus[dim], all_xvals.x_plus[dim]) = evaluate_values(evaluator, input_values, all_xvals, dim, inneronly);
}

fn normalise_vector(vec: &Vec<f32>) -> Vec<f32> {
    let x_max = vec.iter().copied().fold(f32::NAN, f32::max);
    let x_min = vec.iter().copied().fold(f32::NAN, f32::min);
    let vec_normalized = vec.iter().enumerate().map(|(_, x)| (*x-x_min)/(x_max-x_min)).collect::<Vec<_>>();
    vec_normalized
}

fn evaluate_on_testfunction(f: fn(&Vec<f32>) -> f32, y: f32, z:f32, evaluator:  &mut MatrixRecurrentEvaluator, max_dev: &Vec<f32>) -> f32 {
    let mut fitness_vec= Vec::with_capacity(N_TRYS);
    let mut x_guess: Vec<f32>;
    let mut scores_vec= Vec::with_capacity(N_TRYS);
    
    let g: fn(&Vec<f32>, f32, f32) -> f32 = |x,y,z|  (0..DIMENSIONS).map(|i| y*(x[i]*z).sin()).sum();
    for _try_enum in 0..N_TRYS {
        x_guess = vec![0.0;DIMENSIONS];
        // x_minus = max_dev;
        // x_plus = max_dev;
        let mut function_domain = util::FunctionDomain{
            dimensions:DIMENSIONS,
            samplepoints: SAMPLEPOINTS,
            upper_limits: (0..DIMENSIONS).map(|i| x_guess[i] + max_dev[i]).collect::<Vec<_>>(),
            lower_limits: (0..DIMENSIONS).map(|i| x_guess[i] - max_dev[i]).collect::<Vec<_>>()
        };
        let mut all_xvals = AllXVals{
            x_guess : vec![0.0; DIMENSIONS],
            x_max: max_dev.clone(),
            x_min: max_dev.clone(),
            x_minus: max_dev.clone(),
            x_plus: max_dev.clone()
        };
        let mut f_vals: Vec<f32>;
        let mut f_vals_normalized: Vec<f32>;
        let mut step = 0;
        while step < STEPS {
            for dim in 0..DIMENSIONS{
                step+=1;   
                let set_of_sample_points = util::generate_1d_samplepoints(function_domain, dim);
                all_xvals.x_max= set_of_sample_points.max;
                all_xvals.x_min= set_of_sample_points.min;
                f_vals = set_of_sample_points.coordinates.iter().enumerate().map(|(_, x)| f(x).clone() +g(x, y, z)).collect::<Vec<_>>();
                // norm f_vals
                let f_max: f32 = f_vals.iter().copied().fold(f32::NAN, f32::max);
                f_vals_normalized = f_vals.iter().enumerate().map(|(_, x)| *x/f_max).collect::<Vec<_>>();
                let input_values: Vec<f32> = [set_of_sample_points.coordinates_by_dimension_normalised[dim].clone(), f_vals_normalized.clone(), vec![(STEPS -step) as f32/STEPS as f32]].concat();
                network_step(input_values, evaluator, &mut all_xvals, dim, false);
                if all_xvals.x_guess.iter().any(|i| i.is_nan()) || all_xvals.x_minus.iter().any(|i| i.is_nan())|| all_xvals.x_plus.iter().any(|i| i.is_nan()){
                    return f32::INFINITY;
                }
                let lower_range_limit = all_xvals.x_guess.iter().enumerate().map(|(i, val)| val - all_xvals.x_minus[i]).collect::<Vec<f32>>();
                let upper_range_limit = all_xvals.x_guess.iter().enumerate().map(|(i, val)| val + all_xvals.x_plus[i]).collect::<Vec<f32>>();
                if upper_range_limit.iter().enumerate().any(|(i, val)| val - lower_range_limit[i] == f32::INFINITY)  {
                    return f32::INFINITY;
                };
                if upper_range_limit.iter().enumerate().any(|(i, val)| *val <= lower_range_limit[i]){
                    return f32::INFINITY;
                }
                function_domain = util::FunctionDomain{
                    dimensions:DIMENSIONS,
                    samplepoints: SAMPLEPOINTS,
                    upper_limits: (0..DIMENSIONS).map(|i| all_xvals.x_guess[i] + all_xvals.x_plus[i]).collect::<Vec<_>>(),
                    lower_limits: (0..DIMENSIONS).map(|i| all_xvals.x_guess[i] - all_xvals.x_minus[i]).collect::<Vec<_>>()
                };
            }
        } 
        let fitness = f(&all_xvals.x_guess) + g(&all_xvals.x_guess, y, z);
        let fitness_eval = FitnessEval{
            fitness: fitness.clone(),
            x_guess: all_xvals.x_guess.clone(),
            x_plus: all_xvals.x_plus.clone(),
            x_minus: all_xvals.x_minus.clone()
        };
        fitness_vec.push(fitness_eval.clone());
        scores_vec.push(fitness);
        
    }
    scores_vec.iter().sum::<f32>()/(N_TRYS as f32)
    
}

fn evaluate_net_fitness(net: &Genome, y: f32, z:f32, max_dev: &Vec<f32>) -> OverallFitness {
    let mut fitness_vec: Vec<f32>= Vec::with_capacity(N_TESTFUNCTIONS);
    // let fitness_max: f32;
    // let mut eval_vec: Vec<FitnessEval>= Vec::with_capacity(N_TESTFUNCTIONS);
    
    let mut evaluator_global = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    let mut fitness = evaluate_on_testfunction(sub::func1,y, z, &mut evaluator_global, max_dev);
    // fitness_vec.push(fitness.fitness);
    fitness_vec.push(fitness);
    evaluator_global.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func2,y, z, &mut evaluator_global, max_dev);
    // fitness_vec.push(fitness.fitness);
    fitness_vec.push(fitness);
    evaluator_global.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func3,y, z, &mut evaluator_global, max_dev);
    // fitness_vec.push(fitness.fitness);
    fitness_vec.push(fitness);
    evaluator_global.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func4,y, z, &mut evaluator_global, max_dev);
    // fitness_vec.push(fitness.fitness);
    fitness_vec.push(fitness);
    // fitness_max = fitness_vec.iter().copied().fold(f32::NAN, f32::max);
    let fitness_average = fitness_vec.iter().sum::<f32>()/4.0;
    OverallFitness{
        fitness: fitness_average,
        fitnessvec: vec![TestfunctionEval {
            fitness_eval: fitness_vec[0].clone(),
            function_name: String::from("Power four")
        } ,
        TestfunctionEval {
            fitness_eval:fitness_vec[1].clone(),
            function_name: String::from("Quadratic Sinus")
        },
        TestfunctionEval {
            fitness_eval:fitness_vec[2].clone(),
            function_name: String::from("Abs")
        },
        TestfunctionEval {
            fitness_eval:fitness_vec[3].clone(),
            function_name: String::from("X-Squared")
        }]}        
}