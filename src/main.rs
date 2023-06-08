mod sub;
mod data_structs;
mod util;
use chrono::{Utc, Datelike};
use data_structs::{OverallFitness, TestfunctionEval, AllXValsCircle, FitnessEvalCircle, PredictionCircle};
use favannat::{MatrixRecurrentFabricator, StatefulFabricator, StatefulEvaluator, MatrixRecurrentEvaluator};
use rand::{
    seq::SliceRandom,
    thread_rng, Rng,
};
use libm::tanhf;
use std::{fs::{OpenOptions}, f32::consts::PI};
use std::io::prelude::*;
use std::{cmp::Ordering, fs};
use rayon::{prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator}};
use set_genome::{Genome, Parameters, Structure, Mutations, activations::Activation};
use std::time::{Instant};
// use crate::{data_structs::};

const DIMENSIONS: usize = 2;
const STEPS: usize = 40;
const N_TRYS: usize = 3;
const N_DRILL: usize = 2;
const N_OVERDRILL: usize = 0;
const N_TESTFUNCTIONS: usize =6;
const SAMPLEPOINTS: usize = 6;
const POPULATION_SIZE: usize = 1000;
const GENERATIONS: usize = 10;
const TRAINFROMRESULTS: bool=false;
const FUNCTION_HANDLE_UNTRAINED1D: fn(&Vec<f32>) -> f32 = |x|  x[0].abs() + 1.9* x[0].sin();
const FUNCTION_HANDLE_UNTRAINED: fn(&Vec<f32>) -> f32 = |x|  (x[0]-3.14).powi(2) + (x[1]-2.72).powi(2)+ (3.0*x[0] +1.41).sin() + (4.0*x[1] -1.73).sin();
const FUNCTION_HANDLE_HIMMELBLAU: fn(&Vec<f32>) -> f32 = |x|  ((x[0]).powi(2) + x[1] - 11.0).powi(2)+ ((x[0]) + x[1].powi(2) - 7.0).powi(2);
const FUNCTION_HANDLE_ROSENBROCK: fn(&Vec<f32>) -> f32 = |x|  100.0*(x[1] - x[0].powi(2)).powi(2) + (1.0 - (x[0])).powi(2);
const FUNCTION_HANDLE_RASTRIGIN: fn(&Vec<f32>) -> f32 = |x|  20.0 + (x[0]).powi(2) - 10.0*(2.0*PI*x[0]).cos() + (x[1]).powi(2) - 10.0*(2.0*PI*x[1]).cos();



fn main() {
    let n_inputs = 2*(SAMPLEPOINTS) + 4;
    let parameters = Parameters {
        structure: Structure { number_of_inputs: n_inputs, number_of_outputs: 2, percent_of_connected_inputs: (1.0), outputs_activation: (Activation::Sigmoid), seed: (42) },
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
                    Activation::Inverse,
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
    let n_inputs = 2*(SAMPLEPOINTS) + 4;
    let parameters_straightlines = Parameters {
        structure: Structure { number_of_inputs: n_inputs, number_of_outputs: 3, percent_of_connected_inputs: (1.0), outputs_activation: (Activation::Sigmoid), seed: (42) },
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
                    Activation::Inverse,
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
    let gen2: Genome;
    if TRAINFROMRESULTS{
        let champion_bytes = &fs::read_to_string(format!("example_runs/2023-6-8/winner-circle.json")).expect("Unable to read champion");
        let champion_bytes2 = &fs::read_to_string(format!("example_runs/2023-6-8/winner-line.json")).expect("Unable to read champion");
        gen = serde_json::from_str(champion_bytes).unwrap();
        gen2 = serde_json::from_str(champion_bytes2).unwrap();
    }else {
        gen = Genome::initialized(&parameters);
        gen2 = Genome::initialized(&parameters_straightlines);
    }

    for _ in 0..POPULATION_SIZE {  
        network_population.push(vec![gen.clone(), gen2.clone()])
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
            .map(|a|evaluate_net_fitness(&a[0], &a[1], y, y2)).enumerate()
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
            if let Err(_) = child[0].mutate(&parameters) {
            };
            if let Err(_) = child[1].mutate(&parameters) {
            };
            new_population.push(child);
        }

        champion = network_population[population_fitnesses[0].0].clone();
        network_population = new_population;
        
        dbg!(population_fitnesses.iter().take(1).collect::<Vec<_>>());
    }
    // save the champion net
    print!("Circle net\n {}", Genome::dot(&champion[0]));
    print!("Line net\n {}", Genome::dot(&champion[1]));
    let now = Utc::now();
    let time_stamp = format!("{year}-{month}-{day}", year = now.year(), month = now.month(), day = now.day());
    // fs::try_exists(format!("example_runs/{}/", time_stamp)).is_err();
    fs::create_dir_all(format!("example_runs/{}/", time_stamp)).expect("Unable to create dir");
    fs::write(
        format!("example_runs/{}/winner-circle.json", time_stamp),
        serde_json::to_string(&champion[0]).unwrap(),
    )
    .expect("Unable to write file");
    fs::write(
        format!("example_runs/{}/winner-line.json", time_stamp),
        serde_json::to_string(&champion[1]).unwrap(),
    )
    .expect("Unable to write file");
    evaluate_champion(&champion, sub::func1, format!("example_runs/{}/powerfour.txt", time_stamp), false, 50.0, &vec![-30.0,0.0]);
    time_evaluation(&champion, sub::func1, 70.0, &vec![-30.0,0.0]);
    evaluate_champion(&champion, sub::func2, format!("example_runs/{}/quadratic-sinus.txt", time_stamp), false, 70.0, &vec![0.0,0.0]);
    time_evaluation(&champion, sub::func2, 70.0, &vec![0.0,0.0]);
    evaluate_champion(&champion, sub::func3, format!("example_runs/{}/abs.txt", time_stamp), false, 40.0, &vec![0.0,0.0]);
    time_evaluation(&champion, sub::func3, 40.0, &vec![0.0,0.0]);
    evaluate_champion(&champion, sub::func4, format!("example_runs/{}/x-squared.txt", time_stamp), false, 40.0, &vec![-30.0,0.0]);
    time_evaluation(&champion, sub::func4, 40.0, &vec![-30.0,0.0]);
    // evaluate_champion(&champion, FUNCTION_HANDLE_UNTRAINED1D, format!("example_runs1d/{}/abs_sin.txt", time_stamp), false, &max_dev);
    // time_evaluation(&champion, FUNCTION_HANDLE_UNTRAINED1D, &max_dev);    
    evaluate_champion(&champion, FUNCTION_HANDLE_UNTRAINED, format!("example_runs/{}/x-abs+sin.txt", time_stamp), false, 20.0, &vec![0.0,0.0]);
    time_evaluation(&champion, FUNCTION_HANDLE_UNTRAINED, 20.0, &vec![0.0,0.0]);
    // evaluate_champion(&champion, FUNCTION_HANDLE_HIMMELBLAU, format!("example_runs/{}/himmelblau.txt", time_stamp), false, &max_dev);
    // time_evaluation(&champion, FUNCTION_HANDLE_HIMMELBLAU, &max_dev);
    evaluate_champion(&champion, FUNCTION_HANDLE_ROSENBROCK, format!("example_runs/{}/rosenbrock.txt", time_stamp), false, 20.0, &vec![0.0,0.0]);
    time_evaluation(&champion, FUNCTION_HANDLE_ROSENBROCK, 20.0, &vec![0.0,0.0]);
    evaluate_champion(&champion, FUNCTION_HANDLE_RASTRIGIN, format!("example_runs/{}/rastringin.txt", time_stamp), false, 10.0, &vec![0.0,0.0]);
    time_evaluation(&champion, FUNCTION_HANDLE_RASTRIGIN, 10.0, &vec![0.0,0.0]);
    
}

fn time_evaluation(champion: &Vec<Genome>,f: fn(&Vec<f32>) -> f32, radius: f32, center: &Vec<f32>){
    let evaluations= 1000;
    let mut f_vals: Vec<f32> = Vec::with_capacity(evaluations);
    let mut step_vec: Vec<usize>= Vec::with_capacity(evaluations);
    let mut x_vals: Vec<Vec<f32>>= Vec::with_capacity(evaluations);
    let mut durations: Vec<f32>= Vec::with_capacity(evaluations);
    let mut cumulative_duration: f32 = 0.0;
    let mut f_best_approx: f32 = f32::INFINITY;
    let mut approx_to_best: Vec<Vec<f32>> =  Vec::with_capacity(evaluations);
    for _ in 0..evaluations{
        let start = Instant::now();
        let f_val: f32;
        let x_min: Vec<f32>;
        let steps: usize;
        (f_val, x_min, steps)= evaluate_champion(&champion, f, "".to_string(), true,  radius, center );
        // dbg!(&f_val);
        let duration = start.elapsed().as_secs_f32();
        f_vals.push(f_val);
        x_vals.push(x_min.clone());
        step_vec.push(steps);
        durations.push(duration);
        cumulative_duration = cumulative_duration + duration;
        if f_val < f_best_approx {
            f_best_approx = f_val;
            approx_to_best.push(vec![cumulative_duration,f_best_approx])
        }
        

    }
    let f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
    let f_min = f_vals.iter().copied().fold(f32::NAN, f32::min);
    let average_time = durations.iter().sum::<f32>()/(evaluations as f32);
    let max_time = durations.iter().copied().fold(f32::NAN, f32::max);
    let mut x_min: Vec<f32> = Vec::with_capacity(DIMENSIONS);
    let mut x_min_worst: Vec<f32> = Vec::with_capacity(DIMENSIONS);
    for (i, f_val) in f_vals.iter().enumerate(){
        if f_val == &f_min {
            x_min = x_vals[i].clone();
        }
        if f_val == &f_max {
            x_min_worst = x_vals[i].clone();
        }
    }
    let average_steps = (step_vec.iter().sum::<usize>()) as f32/(evaluations as f32);
    let median = median(&mut f_vals);
    println!("Ran 500 iterations to find the minimum");
    dbg!(approx_to_best);
    println!("Champion took on average {:?} to find the minimum, worst time {}", average_time, max_time);
    colour::green_ln!("Minimum has value {} ", f_min);
    colour::yellow_ln!("Median of Minimum {} ", median);
    colour::red_ln!("Wort minimum approximation {} ", f_max);
    colour::cyan_ln!("Minimum location {:?} ", x_min);
    colour::red_ln!("Worst minimum approximation location {:?} ", x_min_worst);
    colour::grey_ln!("Took {} steps on average", average_steps)
}

fn median(numbers: &mut [f32]) -> f32 {
    numbers.sort_unstable_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap(),
    });
    let mid = numbers.len() / 2;
    numbers[mid]
}


fn evaluate_champion(net: &Vec<Genome>, f: fn(&Vec<f32>) -> f32, filepath: String, timeonly: bool, radius: f32, center: &Vec<f32>) -> (f32, Vec<f32>, usize) {
    
    if !timeonly {
        let contents = String::from(format!(";Step;XVal;radius\n"));
        let mut sampleheader = util::generate_sampleheader(SAMPLEPOINTS,DIMENSIONS);
        sampleheader.push_str(&contents);
        fs::write(filepath.clone(), sampleheader).expect("Unable to write file");

    }

    let mut function_domain = util::CircleGeneratorInput{
        dimensions:DIMENSIONS,
        samplepoints: SAMPLEPOINTS,
        radius: radius,
        center: center.to_vec()
    };
    let mut all_xvals = AllXValsCircle{
        x_guess : center.to_vec(),
        radius: radius
    };
    // let mut set_of_sample_points = util::generate_1d_samplepoints(&function_domain, 0);
    let mut set_of_samples_line =  util::generate_samplepoint_vector(&vec![0.0,0.0], &vec![1.0,1.0], SAMPLEPOINTS);
    let mut set_of_sample_points: util::SetOfSamplesCircle;
    let mut f_vals: Vec<f32>;
    let mut f_vals_normalized: Vec<f32>;
    let mut evaluator_circle = MatrixRecurrentFabricator::fabricate(&net[0]).expect("didnt work");
    let mut evaluator_line = MatrixRecurrentFabricator::fabricate(&net[1]).expect("didnt work");
    let mut step = 0;
    let mut delta_fitness = f32::INFINITY;
    let mut last_fitness = f32::INFINITY;
    let mut skiprest: bool = false;
    let mut fitness_change_limited: f32;
    let mut vector_start: Vec<f32>;
    let mut vector_end: Vec<f32>;
    for n_drill in 0..N_DRILL+N_OVERDRILL {
        step = 0;
        fitness_change_limited = 0.0;
        while (step < STEPS) & (delta_fitness.abs() > 1e-10_f32) {
            // if  step%3 == 0{
                set_of_sample_points = util::generate_samplepoint_2d_random_slice(&function_domain);
                f_vals = set_of_sample_points.coordinates.iter().enumerate().map(|(_, x)| f(x)).collect::<Vec<_>>();
                f_vals_normalized = normalise_vector(&f_vals);
                let input_values: Vec<f32> = [set_of_sample_points.coordinates_normalised.clone(), f_vals_normalized.clone(), vec![(n_drill as f32), (STEPS -step) as f32/STEPS as f32, 1.0, fitness_change_limited]].concat();
                network_step(input_values, &mut evaluator_circle, &mut all_xvals, &function_domain, step);
                vector_start  = all_xvals.x_guess.clone();
                vector_end = function_domain.center.clone();
                if vector_start.iter().enumerate().all(|(i,x)| (vector_end[i]-x).abs() < 1e-8_f32) {
                    skiprest = true;
                    delta_fitness = 0.0;
                    all_xvals.x_guess = vector_start.clone();
                // }
            }
            // if skiprest {
            //     dbg!(&n_drill, &step, &skiprest);
            // }
            if !skiprest{
            let vector_start_resize = vector_start.iter().enumerate().map(|(i,x)| x -((STEPS -step) as f32/STEPS as f32)*2.0*( vector_end[i] - x)).collect::<Vec<f32>>();
            let vector_end_resize = vector_end.iter().enumerate().map(|(i,x)| x +0.0*( x - vector_start[i])).collect::<Vec<f32>>();
            set_of_samples_line = util::generate_samplepoint_vector(&vector_start_resize, &vector_end_resize, SAMPLEPOINTS);
            f_vals = set_of_samples_line.coordinates.iter().enumerate().map(|(_, x)| f(x).clone()).collect::<Vec<_>>();
            f_vals_normalized = normalise_vector(&f_vals);
            let input_values: Vec<f32> = [set_of_samples_line.coordinates_normalised.clone(), f_vals_normalized.clone(), vec![n_drill as f32, (STEPS -step) as f32/STEPS as f32, 1.0, fitness_change_limited]].concat();
            network_step_line(input_values, &mut evaluator_line, &mut all_xvals, &vector_start_resize, &vector_end_resize);
            function_domain.center = all_xvals.x_guess.clone();
            function_domain.radius = all_xvals.radius.clone();
            delta_fitness = last_fitness - f(&all_xvals.x_guess);
            fitness_change_limited = tanhf(delta_fitness);
            last_fitness = f(&all_xvals.x_guess);
            }
            if !timeonly {
                let prediction =PredictionCircle {
                    fitness: f(&all_xvals.x_guess),
                    x_guess: all_xvals.x_guess.clone(),
                    radius: all_xvals.radius.clone()};
                let mut file_write = OpenOptions::new()
                .write(true)
                .append(true)
                .open(filepath.clone())
                .unwrap();
                // print!("champion_global Step {}: {:?}\n",step, prediction); 
                let coordinates_string: String = set_of_samples_line.coordinates.iter().map(|x|  x.iter().map( |&id| id.to_string()).collect::<Vec<String>>().join(",")).collect::<Vec<String>>().join(";");
                let f_vals_string: String = f_vals.iter().map( |&id| id.to_string()).collect::<Vec<String>>().join(";");
                let x_guess_string: String = all_xvals.x_guess.clone().into_iter().map(|i| i.to_string()).collect::<Vec<String>>().join(",");
                let radius_string: String = all_xvals.radius.to_string();
                if let Err(e) = writeln!(file_write, "{coords};{fvals};{step};{x_val1};{radius}",coords =coordinates_string,fvals=f_vals_string, step=step, x_val1 =x_guess_string , radius = radius_string) {
                    eprintln!("Couldn't write to file: {}", e);
                }
            } 
            step+=1;
        }
    }
    (f(&all_xvals.x_guess), all_xvals.x_guess, step)
}

fn evaluate_values_line(evaluator: &mut MatrixRecurrentEvaluator, input_values_normalized: Vec<f32>, vector_start: &Vec<f32>, vector_end: &Vec<f32>, all_xvals: &AllXValsCircle) ->(Vec<f32>, f32) {
    let prediction: Vec<f64> = evaluator.evaluate(input_values_normalized.clone().iter().map(|x | *x as f64).collect() );
    let pred_x_guess: f32 = prediction[0] as f32;
    let pred_plus: f32 = prediction[1] as f32;
    let pred_minus: f32 = prediction[2] as f32;
    let x_plus: f32;
    let x_minus: f32;
    let mut radius: f32;
    let mut x_guess: Vec<f32> = vec![0.0, 0.0];

    x_plus = all_xvals.radius*(0.5+pred_plus);
    x_minus = all_xvals.radius*(0.5+pred_minus);
    radius = (x_plus + x_minus)/2.0;
    for dim in 0..2{
        x_guess[dim] = pred_x_guess*(vector_end[dim]-vector_start[dim])+vector_start[dim];
        if x_guess[dim].is_nan() {
            x_guess[dim] = vector_start[dim];    
        }
    }
    let len_x_guess :f32 =  x_guess.iter().map(|x| x.powi(2)).collect::<Vec<f32>>().iter().sum::<f32>().sqrt();
    let x_guess_norm :Vec<f32> =  x_guess.iter().map(|x| x/len_x_guess).collect::<Vec<f32>>();
    x_guess = x_guess.iter().enumerate().map(|(i,x)| ((x - x_guess_norm[i]*x_minus) + (x + x_guess_norm[i]*x_plus))/2.0).collect::<Vec<f32>>();
    if radius.is_nan() {
        radius = all_xvals.radius;
    }
    (x_guess, radius)
}

fn evaluate_values(evaluator: &mut MatrixRecurrentEvaluator, input_values_normalized: Vec<f32>, all_xvals: &AllXValsCircle, function_domain: & util::CircleGeneratorInput, step:usize) ->(Vec<f32>,f32) {
    let prediction: Vec<f64> = evaluator.evaluate(input_values_normalized.clone().iter().map(|x | *x as f64).collect() );
    let pred_x_guess: f32 = prediction[0] as f32;
    
    let mut x_guess: Vec<f32> = vec![0.0, 0.0];
    x_guess[0] = function_domain.center[0] + (pred_x_guess*2.0*PI).cos()*all_xvals.radius;
    x_guess[1] = function_domain.center[1] + (pred_x_guess*2.0*PI).sin()*all_xvals.radius;
    for dim in 0..2{
        if x_guess[dim].is_nan() {
            x_guess[dim] = function_domain.center[dim];
        }
    }

    let pred_radius: f32 = prediction[1] as f32;
    let mut radius: f32;
    // circle step can't change radius anymore after half the steps. For better convergence.
    if step < STEPS - 2 {
        radius = all_xvals.radius*(0.5+pred_radius);
    } else {
        radius = all_xvals.radius
    }
    
    if radius.is_nan() {
        radius = all_xvals.radius;
    }

    (x_guess, radius)
}

fn network_step(input_values: Vec<f32>, evaluator: &mut MatrixRecurrentEvaluator,  all_xvals: &mut AllXValsCircle, functiondomain: & util::CircleGeneratorInput, step: usize) -> () {
    (all_xvals.x_guess, all_xvals.radius) = evaluate_values(evaluator, input_values, all_xvals, functiondomain, step);
}

fn network_step_line(input_values: Vec<f32>, evaluator: &mut MatrixRecurrentEvaluator,  all_xvals: &mut AllXValsCircle, vector_start: &Vec<f32>, vector_end: &Vec<f32>) -> () {
    (all_xvals.x_guess, all_xvals.radius) = evaluate_values_line(evaluator, input_values, vector_start, vector_end, all_xvals);
}

fn normalise_vector(vec: &Vec<f32>) -> Vec<f32> {
    let x_max = vec.iter().copied().fold(f32::NAN, f32::max);
    let x_min = vec.iter().copied().fold(f32::NAN, f32::min);
    let vec_normalized: Vec<f32>;
    if x_max-x_min < 1e-22_f32 {
        vec_normalized = vec![1.0; vec.len()]
    } else {
        vec_normalized = vec.iter().enumerate().map(|(_, x)| (*x-x_min)/(x_max-x_min)).collect::<Vec<_>>();
    }
    vec_normalized
}

fn evaluate_on_testfunction(f: fn(&Vec<f32>) -> f32, y: f32, z:f32, evaluator_circle:  &mut MatrixRecurrentEvaluator, evaluator_line:  &mut MatrixRecurrentEvaluator, radius: f32, center: &Vec<f32>) -> f32 {
    let mut fitness_vec= Vec::with_capacity(N_TRYS);
    // let mut x_guess: Vec<f32>;
    let mut scores_vec= Vec::with_capacity(N_TRYS);
    
    let g: fn(&Vec<f32>, f32, f32) -> f32 = |x,y,z|  (0..DIMENSIONS).map(|i| y*(x[i]*z).sin()).sum();
    for _try_enum in 0..N_TRYS {
        let mut f_vals: Vec<f32>;
        let mut f_vals_normalized: Vec<f32>;
        let mut step = 0;
        let mut delta_fitness = f32::INFINITY;
        let mut last_fitness = f32::INFINITY;
        
        let mut function_domain = util::CircleGeneratorInput{
            dimensions:DIMENSIONS,
            samplepoints: SAMPLEPOINTS,
            radius: radius,
            center: center.to_vec()
        };
        let mut all_xvals = AllXValsCircle{
            x_guess : center.to_vec(),
            radius: radius
        };
        let mut vector_start: Vec<f32>;
        let mut vector_end: Vec<f32>;
        for n_drill in 0..N_DRILL {
            step = 0;
            let mut fitness_change_limited = 0.0;
            let mut skiprest: bool = false;
            
            while (step < STEPS) & (delta_fitness.abs() > 1e-10_f32) {
                // if step%3 == 0{   
                    let set_of_sample_points = util::generate_samplepoint_2d_random_slice(&function_domain);
                    f_vals = set_of_sample_points.coordinates.iter().enumerate().map(|(_, x)| f(x).clone() +g(x, y, z)).collect::<Vec<_>>();
                    f_vals_normalized = normalise_vector(&f_vals);
                    let input_values: Vec<f32> = [set_of_sample_points.coordinates_normalised.clone(), f_vals_normalized.clone(), vec![n_drill as f32, (STEPS -step) as f32/STEPS as f32, 1.0,fitness_change_limited]].concat();
                    network_step(input_values, evaluator_circle, &mut all_xvals, &function_domain, step);
                    if all_xvals.x_guess.iter().any(|i| i.is_nan()) || all_xvals.radius.is_nan() || f_vals.iter().any(|i| i.is_nan()) {
                        return f32::INFINITY;
                    }
                    vector_start = all_xvals.x_guess.clone();
                    vector_end  = function_domain.center.clone();
                    if vector_start.iter().enumerate().all(|(i,x)| (vector_end[i]-x).abs() < 1e-8_f32) {
                        skiprest = true;
                        delta_fitness = 0.0;
                    }
                // }
                if !skiprest{
                    let vector_start_resize = vector_start.iter().enumerate().map(|(i,x)| x -((STEPS -step) as f32/STEPS as f32)*2.0*( vector_end[i] - x)).collect::<Vec<f32>>();
                    let vector_end_resize = vector_end.iter().enumerate().map(|(i,x)| x +0.0*( x - vector_start[i])).collect::<Vec<f32>>();
                    let set_of_samples_line = util::generate_samplepoint_vector(&vector_start_resize, &vector_end_resize, SAMPLEPOINTS);
                    f_vals = set_of_samples_line.coordinates.iter().enumerate().map(|(_, x)| f(x).clone() +g(x, y, z)).collect::<Vec<_>>();
                    f_vals_normalized = normalise_vector(&f_vals);
                    // if f_vals_normalized.iter().any(|i| i.is_nan()){
                    //     dbg!(&f_vals_normalized);
                    //     dbg!(&f_vals);
                    //     dbg!(&set_of_samples_line);
                    //     dbg!(&vector_end_resize);
                    //     dbg!(&vector_start_resize);
                    // }
                    let input_values: Vec<f32> = [set_of_samples_line.coordinates_normalised.clone(), f_vals_normalized.clone(), vec![n_drill as f32, (STEPS -step) as f32/STEPS as f32, 1.0,fitness_change_limited]].concat();
                    network_step_line(input_values, evaluator_line, &mut all_xvals, &vector_start_resize, &vector_end_resize);
                    function_domain.center = all_xvals.x_guess.clone();
                    function_domain.radius = all_xvals.radius.clone();
                    delta_fitness = last_fitness - (f(&all_xvals.x_guess) + g(&all_xvals.x_guess, y, z));
                    fitness_change_limited = tanhf(delta_fitness);
                    last_fitness = f(&all_xvals.x_guess) + g(&all_xvals.x_guess, y, z);
                }
                step+=1;
            } 
        }
        let fitness = f(&all_xvals.x_guess) + g(&all_xvals.x_guess, y, z);
        let fitness_eval = FitnessEvalCircle{
            fitness: fitness.clone(),
            x_guess: all_xvals.x_guess.clone(),
            radius: all_xvals.radius.clone()
        };
        fitness_vec.push(fitness_eval.clone());
        scores_vec.push(fitness);
    }
    scores_vec.iter().sum::<f32>()/(N_TRYS as f32)
    // scores_vec.iter().copied().fold(f32::NAN, f32::min)
    
}

fn evaluate_net_fitness(net: &Genome, net2: &Genome, y: f32, z:f32) -> OverallFitness {
    let mut fitness_vec: Vec<f32>= Vec::with_capacity(N_TESTFUNCTIONS);
    let mut rng = rand::thread_rng();
    let c1 =rng.gen_range(-5.0..5.0);
    let c2 =rng.gen_range(-5.0..5.0);
    let mut evaluator_circle = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    let mut evaluator_line = MatrixRecurrentFabricator::fabricate(net2).expect("didnt work");
    let mut fitness = evaluate_on_testfunction(sub::func1,y, z, &mut evaluator_circle, &mut evaluator_line, 50.0, &vec![-30.0+c1,0.0+c2]);
    fitness_vec.push(fitness);
    evaluator_circle.reset_internal_state();
    evaluator_line.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func2,y, z, &mut evaluator_circle,&mut evaluator_line, 70.0, &vec![0.0+c1,0.0+c2]);
    fitness_vec.push(fitness);
    evaluator_circle.reset_internal_state();
    evaluator_line.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func3,y, z, &mut evaluator_circle,&mut evaluator_line, 40.0, &vec![0.0+c1,0.0+c2]);
    fitness_vec.push(fitness);

    evaluator_circle.reset_internal_state();
    evaluator_line.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func4,y, z, &mut evaluator_circle,&mut evaluator_line, 40.0, &vec![-30.0+c1,0.0+c2]);
    fitness_vec.push(fitness);
    evaluator_circle.reset_internal_state();
    evaluator_line.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func5,y, z, &mut evaluator_circle,&mut evaluator_line, 20.0, &vec![0.0+c1,0.0+c2]);
    fitness_vec.push(fitness);
    evaluator_circle.reset_internal_state();
    evaluator_line.reset_internal_state();
    fitness = evaluate_on_testfunction(sub::func6,y, z, &mut evaluator_circle,&mut evaluator_line, 10.0, &vec![0.0+c1,0.0+c2]);
    fitness_vec.push(fitness);
    // let fitness_max = fitness_vec.iter().copied().fold(f32::NAN, f32::max);
    let fitness_average = fitness_vec.iter().sum::<f32>()/6.0;
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
        },
        TestfunctionEval {
            fitness_eval:fitness_vec[4].clone(),
            function_name: String::from("Rosenbrock")
        },
        TestfunctionEval {
            fitness_eval:fitness_vec[5].clone(),
            function_name: String::from("Rastringin")
        }]}        
}