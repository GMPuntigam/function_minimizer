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
const STEPS: usize = 50;
const N_TRYS: usize = 10;
const N_TESTFUNCTIONS: usize =4;
const SAMPLEPOINTS: usize = 6;
const POPULATION_SIZE: usize = 1000;
const GENERATIONS: usize = 100;

const FUNCTION_HANDLE_UNTRAINED: fn(&Vec<f32>) -> f32 = |x|  (x[0]-3.0).abs()+ 1.5*x[0].sin()+ x[1].powi(2);





fn main() {
    let max_dev: Vec<f32> = vec![100.0, 50.0];
    // let function_domain = util::FunctionDomain{
    //     dimensions:DIMENSIONS,
    //     samplepoints: SAMPLEPOINTS,
    //     upper_limits: vec![100.0, 50.0],
    //     lower_limits: vec![-100.0, -50.0]
    // };
    // let matrix = util::generate_samplepoint_matrix(function_domain);
    // dbg!(matrix);
    let n_inputs = (DIMENSIONS+1)*(SAMPLEPOINTS).pow(DIMENSIONS.try_into().unwrap()) + 1;
    // dbg!(&n_inputs);
    let parameters = Parameters {
        structure: Structure { number_of_inputs: n_inputs, number_of_outputs: (3*DIMENSIONS), percent_of_connected_inputs: (0.5), outputs_activation: (Activation::Sigmoid), seed: (42) },
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



fn evaluate_champion(net: &Genome, f: fn(&Vec<f32>) -> f32, filepath: String, timeonly: bool, max_dev: &Vec<f32>) {
    let mut all_xvals = AllXVals{
        x_guess : vec![0.0; DIMENSIONS],
        x_max: vec![0.0; DIMENSIONS],
        x_min: vec![0.0; DIMENSIONS],
        x_minus: vec![0.0; DIMENSIONS],
        x_plus: vec![0.0; DIMENSIONS]
    };
    if !timeonly {
        let contents = String::from(format!(",x-best,Step,XVal,x-plus,x-minus\n"));
        let mut sampleheader = generate_sampleheader();
        sampleheader.push_str(&contents);
        fs::write(filepath.clone(), sampleheader).expect("Unable to write file");

    }
    // let mut x_vals_normalized;
    let x_guess: Vec<f32> = vec![0.0; DIMENSIONS];
    // let x_minus: f32 = max_dev;
    // let x_plus: f32 = max_dev;
    // let mut between: Uniform<f32> = Uniform::from(0.0..x_minus + x_plus);
    // let mut x_vals: Vec<f32> = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
    // x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let function_domain = util::FunctionDomain{
        dimensions:DIMENSIONS,
        samplepoints: SAMPLEPOINTS,
        upper_limits: (0..DIMENSIONS).map(|i| x_guess[i] + max_dev[i]).collect::<Vec<_>>(),
        lower_limits: (0..DIMENSIONS).map(|i| x_guess[i] - max_dev[i]).collect::<Vec<_>>()
    };
    let set_of_sample_points = util::generate_samplepoint_matrix(function_domain);


    let mut f_vals = set_of_sample_points.coordinates.iter().enumerate().map(|(_, x)| f(x)).collect::<Vec<_>>();
    let mut f_vals_normalized : Vec<f32> = normalise_vector(&f_vals);
    // let x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
    // let x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
    // norm x_vals
    // x_vals_normalized = normalise_vector(&x_vals);
    let mut evaluator = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    // all_xvals.x_guess = x_guess;
    // all_xvals.x_max = x_max;
    // all_xvals.x_min = x_min;
    // all_xvals.x_minus = x_minus;
    // all_xvals.x_plus = x_plus;
    let mut step = 0;
        while step < STEPS {
            step+=1;
            let input_values: Vec<f32> = [set_of_sample_points.coordinates.concat(), f_vals.clone(), vec![(STEPS -step) as f32/STEPS as f32]].concat();    
            let input_values_normalized: Vec<f32> = [set_of_sample_points.coordinates_normalised.concat(), f_vals_normalized, vec![(STEPS -step) as f32/STEPS as f32]].concat();
            

            network_step(input_values_normalized, &mut evaluator, &mut all_xvals, false);
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
                let samples_string: String = input_values.iter().map( |&id| id.to_string() + ",").collect();
                let x_guess_string: String = all_xvals.x_guess.clone().into_iter().map(|i| i.to_string()).collect::<Vec<String>>().join(",");
                let x_plus_string: String = all_xvals.x_plus.clone().into_iter().map(|i| i.to_string()).collect::<Vec<String>>().join(",");
                let x_minus_string: String = all_xvals.x_minus.clone().into_iter().map(|i| i.to_string()).collect::<Vec<String>>().join(",");
                if let Err(e) = writeln!(file_write, "{samples}{step},{x_val1},{x_plus},{x_minus}",samples =samples_string, step=step, x_val1 =x_guess_string , x_plus = x_plus_string, x_minus = x_minus_string) {
                eprintln!("Couldn't write to file: {}", e);
            }
            }
            let function_domain = util::FunctionDomain{
                dimensions:DIMENSIONS,
                samplepoints: SAMPLEPOINTS,
                upper_limits: (0..DIMENSIONS).map(|i| x_guess[i] + all_xvals.x_plus[i]).collect::<Vec<_>>(),
                lower_limits: (0..DIMENSIONS).map(|i| x_guess[i] - all_xvals.x_minus[i]).collect::<Vec<_>>()
            };
            let set_of_sample_points = util::generate_samplepoint_matrix(function_domain);
            f_vals = set_of_sample_points.coordinates.iter().enumerate().map(|(_, x)| f(x)).collect::<Vec<_>>();
            f_vals_normalized = normalise_vector(&f_vals);
            // between = Uniform::from(0.0..all_xvals.x_minus + all_xvals.x_plus);
            // x_vals = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
            // x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x+all_xvals.x_guess-all_xvals.x_minus).clone()).collect::<Vec<_>>();
            // f_vals_normalized = normalise_vector(&f_vals);
            // all_xvals.x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
            // all_xvals.x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
            // x_vals_normalized = normalise_vector(&x_vals);
        } 
}


fn evaluate_values(evaluator: &mut MatrixRecurrentEvaluator, input_values_normalized: Vec<f32>, all_xvals: &AllXVals, inneronly:bool) ->(Vec<f32>, Vec<f32>, Vec<f32>) {
    let prediction: Vec<f64> = evaluator.evaluate(input_values_normalized.clone().iter().map(|x | *x as f64).collect() );
    let pred_x_guess: Vec<f32> = prediction[0..DIMENSIONS].iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let pred_x_minus: Vec<f32> = prediction[DIMENSIONS..DIMENSIONS*2].iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let pred_x_plus: Vec<f32> = prediction[DIMENSIONS*2..DIMENSIONS*3].iter().map(|x| *x as f32).collect::<Vec<f32>>();
    let x_minus: Vec<f32>;
    let x_plus: Vec<f32>;
    let x_guess: Vec<f32> = pred_x_guess.iter().enumerate().map(|(i,val)| val*(all_xvals.x_max[i]-all_xvals.x_min[i])+all_xvals.x_min[i] -all_xvals.x_minus[i] + all_xvals.x_guess[i]).collect::<Vec<f32>>();
    // let x_prediction_in_intervall = (pred_x_guess)*(all_xvals.x_max-all_xvals.x_min) + all_xvals.x_min;
    x_minus = all_xvals.x_minus.iter().enumerate().map(|(i,val)| val*(0.5+pred_x_minus[i])).collect::<Vec<f32>>();
    x_plus = all_xvals.x_plus.iter().enumerate().map(|(i,val)| val*(0.5+pred_x_plus[i])).collect::<Vec<f32>>();
    // if inneronly{
    //     x_minus = x_prediction_in_intervall*(pred_x_minus);
    //     x_plus = (all_xvals.x_max-x_prediction_in_intervall)*(pred_x_plus);
    // }   else {
    //     x_minus = all_xvals.x_minus*(0.5+pred_x_minus);
    //     x_plus = all_xvals.x_plus*(0.5+pred_x_plus);
    // }
    (x_guess, x_minus, x_plus)
}

fn network_step(input_values: Vec<f32>, evaluator: &mut MatrixRecurrentEvaluator,  all_xvals: &mut AllXVals, inneronly:bool) -> () {
            (all_xvals.x_guess, all_xvals.x_minus, all_xvals.x_plus) = evaluate_values(evaluator, input_values, all_xvals, inneronly);
}

fn normalise_vector(vec: &Vec<f32>) -> Vec<f32> {
    let x_max = vec.iter().copied().fold(f32::NAN, f32::max);
    let x_min = vec.iter().copied().fold(f32::NAN, f32::min);
    let vec_normalized = vec.iter().enumerate().map(|(_, x)| (*x-x_min)/(x_max-x_min)).collect::<Vec<_>>();
    vec_normalized
}

fn evaluate_on_testfunction(f: fn(&Vec<f32>) -> f32, y: f32, z:f32, evaluator:  &mut MatrixRecurrentEvaluator, max_dev: &Vec<f32>) -> FitnessEval {
    let mut fitness_vec= Vec::with_capacity(N_TRYS);
    let mut x_guess: Vec<f32>;
    let mut x_worst: Vec<f32> = vec![0.0; DIMENSIONS];
    // let mut x_minus: f32;
    // let mut x_plus: f32;
    
    let g: fn(&Vec<f32>, f32, f32) -> f32 = |x,y,z|  (0..DIMENSIONS-1).map(|i| y*(x[i]*z).sin()).sum();
    for try_enum in 0..N_TRYS {
        x_guess = vec![0.0;DIMENSIONS];
        // x_minus = max_dev;
        // x_plus = max_dev;
        let function_domain = util::FunctionDomain{
            dimensions:DIMENSIONS,
            samplepoints: SAMPLEPOINTS,
            upper_limits: (0..DIMENSIONS).map(|i| x_guess[i] + max_dev[i]).collect::<Vec<_>>(),
            lower_limits: (0..DIMENSIONS).map(|i| x_guess[i] - max_dev[i]).collect::<Vec<_>>()
        };
        let set_of_sample_points = util::generate_samplepoint_matrix(function_domain);
        let mut all_xvals = AllXVals{
            x_guess : vec![0.0; DIMENSIONS],
            x_max: set_of_sample_points.max,
            x_min: set_of_sample_points.min,
            x_minus: max_dev.clone(),
            x_plus: max_dev.clone()
        };
        // set upper range of randomness generation
        // let mut lower_range_limit;
        // let mut upper_range_limit;
        // // generate positive random values within the span of xminus + x plus
        // let mut between = Uniform::from(0.0..x_minus + x_plus);
        // // generate some random values
        // let mut x_vals: Vec<f32>  = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
        // // sort the values
        // x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // // calculate the function at the xvalues

        let mut f_vals = set_of_sample_points.coordinates.iter().enumerate().map(|(_, x)| f(x).clone() +g(x, y, z)).collect::<Vec<_>>();
        // norm f_vals
        let f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
        let mut f_vals_normalized = f_vals.iter().enumerate().map(|(_, x)| *x/f_max).collect::<Vec<_>>();
        // dbg!(&set_of_sample_points);
        // dbg!(&f_vals);

        // get max and min of x_samples for interpretation of prediction
        // let x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
        // let x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
        // all_xvals.x_guess = x_guess;
        // all_xvals.x_max = x_max;
        // all_xvals.x_min = x_min;
        // all_xvals.x_minus = x_minus;
        // all_xvals.x_plus = x_plus;
        // norm x_vals
        // let mut x_vals_normalized = x_vals.iter().enumerate().map(|(_, x)| (*x-x_min)/(x_max-x_min)).collect::<Vec<_>>();
        // x_vals should be between 0 and 1 now, the first always0, the last always 1.
        
        let mut step = 0;
        while step < STEPS {
            
            step+=1;   
            // dbg!(&all_xvals);
            let input_values: Vec<f32> = [set_of_sample_points.coordinates_normalised.concat(), f_vals_normalized, vec![(STEPS -step) as f32/STEPS as f32]].concat();
            network_step(input_values, evaluator, &mut all_xvals, false);
            // dbg!(&all_xvals);
            if all_xvals.x_guess.iter().any(|i| i.is_nan()) || all_xvals.x_minus.iter().any(|i| i.is_nan())|| all_xvals.x_plus.iter().any(|i| i.is_nan()){
                return FitnessEval {
                    fitness: f32::INFINITY,
                    x_worst: vec![f32::INFINITY; DIMENSIONS],
                    x_plus: vec![f32::INFINITY; DIMENSIONS],
                    x_minus: vec![f32::INFINITY; DIMENSIONS]};
            }
            let lower_range_limit = all_xvals.x_guess.iter().enumerate().map(|(i, val)| val - all_xvals.x_minus[i]).collect::<Vec<f32>>();
            let upper_range_limit = all_xvals.x_guess.iter().enumerate().map(|(i, val)| val + all_xvals.x_plus[i]).collect::<Vec<f32>>();
            // upper_range_limit = all_xvals.x_guess + all_xvals.upper_range_limit;
            if upper_range_limit.iter().enumerate().any(|(i, val)| val - lower_range_limit[i] == f32::INFINITY)  {
                return FitnessEval {
                    fitness: f32::INFINITY,
                    x_worst: vec![f32::INFINITY; DIMENSIONS],
                    x_plus: vec![f32::INFINITY; DIMENSIONS],
                    x_minus: vec![f32::INFINITY; DIMENSIONS]
                } 
            };
            if upper_range_limit.iter().enumerate().any(|(i, val)| *val <= lower_range_limit[i]){
                return FitnessEval {
                    fitness: f32::INFINITY,
                    x_worst: vec![f32::INFINITY; DIMENSIONS],
                    x_plus: vec![f32::INFINITY; DIMENSIONS],
                    x_minus: vec![f32::INFINITY; DIMENSIONS]
                } 
            }
            let function_domain = util::FunctionDomain{
                dimensions:DIMENSIONS,
                samplepoints: SAMPLEPOINTS,
                upper_limits: (0..DIMENSIONS).map(|i| x_guess[i] + all_xvals.x_plus[i]).collect::<Vec<_>>(),
                lower_limits: (0..DIMENSIONS).map(|i| x_guess[i] - all_xvals.x_minus[i]).collect::<Vec<_>>()
            };
            let set_of_sample_points = util::generate_samplepoint_matrix(function_domain);
            all_xvals.x_max= set_of_sample_points.max;
            all_xvals.x_min= set_of_sample_points.min;
            // between = Uniform::from(0.0..all_xvals.x_minus + all_xvals.x_plus);
            // x_vals = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
            // x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            f_vals = set_of_sample_points.coordinates.iter().enumerate().map(|(_, x)| f(&x) +g(&x, y, z)).collect::<Vec<_>>();
            f_vals_normalized = normalise_vector(&f_vals);
            // all_xvals.x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
            // all_xvals.x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
            // x_vals_normalized = normalise_vector(&x_vals);
        } 
        if try_enum == 0 {
            x_worst = all_xvals.x_guess.clone();
        }
        if f(&all_xvals.x_guess) + g(&all_xvals.x_guess, y, z) > f(&x_worst) + g(&x_worst, y, z)
        {
            x_worst = all_xvals.x_guess.clone();
        }
        let fitness = f(&all_xvals.x_guess) + g(&all_xvals.x_guess, y, z);
        let fitness_eval = FitnessEval{
            fitness: fitness.clone(),
            x_worst: x_worst.clone(),
            x_plus: all_xvals.x_plus.clone(),
            x_minus: all_xvals.x_minus.clone()
        };
        fitness_vec.push(fitness_eval.clone());
        
    }
    fitness_vec.sort_unstable_by(|a, b| match (a.fitness.is_nan(), b.fitness.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.fitness.partial_cmp(&b.fitness).unwrap(),
    });
    fitness_vec[0].clone()
    
}

fn evaluate_net_fitness(net: &Genome, y: f32, z:f32, max_dev: &Vec<f32>) -> OverallFitness {
    let mut fitness_vec: Vec<f32>= Vec::with_capacity(N_TESTFUNCTIONS);
    let fitness_max: f32;
    let mut eval_vec: Vec<FitnessEval>= Vec::with_capacity(N_TESTFUNCTIONS);
    
    let mut evaluator_global = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    let mut fitness = evaluate_on_testfunction(sub::func1,y, z, &mut evaluator_global, max_dev);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    evaluator_global.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func2,y, z, &mut evaluator_global, max_dev);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    evaluator_global.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func3,y, z, &mut evaluator_global, max_dev);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    evaluator_global.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func4,y, z, &mut evaluator_global, max_dev);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    fitness_max = fitness_vec.iter().copied().fold(f32::NAN, f32::max);
    OverallFitness{
        fitness: fitness_max,
        fitnessvec: vec![TestfunctionEval {
            fitness_eval: eval_vec[0].clone(),
            function_name: String::from("Power four")
        } ,
        TestfunctionEval {
            fitness_eval:eval_vec[1].clone(),
            function_name: String::from("Quadratic Sinus")
        },
        TestfunctionEval {
            fitness_eval:eval_vec[2].clone(),
            function_name: String::from("Abs")
        },
        TestfunctionEval {
            fitness_eval:eval_vec[3].clone(),
            function_name: String::from("X-Squared")
        }]}        
}