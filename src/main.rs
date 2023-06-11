mod sub;
mod data_structs;
mod util;
use chrono::{Utc, Datelike};
use data_structs::{OverallFitness, TestfunctionEval, AllXValsCircle, FitnessEvalCircle, PredictionCircle, WallTimeEval};
use favannat::{MatrixRecurrentFabricator, StatefulFabricator, StatefulEvaluator, MatrixRecurrentEvaluator};
use rand::{
    seq::SliceRandom,
    thread_rng, Rng,
};
use libm::tanhf;
use std::{fs::{OpenOptions}, f32::consts::PI};
use std::io::prelude::*;
use std::{cmp::Ordering, fs};
use rayon::{prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator, IntoParallelIterator}};
use set_genome::{Genome, Parameters, Structure, Mutations, activations::Activation};
use std::time::{Instant};

use crate::util::Evaluation;
// use crate::{data_structs::};

const DIMENSIONS: usize = 2;
const STEPS: usize = 30;
const N_TRYS: usize = 3;
const N_DRILL: usize = 1;
const N_PARTICLES: usize = 8;
const N_OVERDRILL: usize = 0;
const N_TESTFUNCTIONS: usize =6;
const SAMPLEPOINTS: usize = 6;
const POPULATION_SIZE: usize = 100;
const GENERATIONS: usize = 0;
const TRAINFROMRESULTS: bool=true;
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
        let champion_bytes = &fs::read_to_string(format!("example_runs/2023-6-11/winner-circle.json")).expect("Unable to read champion");
        let champion_bytes2 = &fs::read_to_string(format!("example_runs/2023-6-11/winner-line.json")).expect("Unable to read champion");
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
    println!("Powerfour");
    evaluate_champion(&champion, sub::func1, format!("example_runs/{}/powerfour.txt", time_stamp), false, vec![70.0; N_PARTICLES], vec![vec![0.0,0.0]; N_PARTICLES]);
    time_evaluation(&champion, sub::func1, 70.0, &vec![0.0,0.0]);
    println!("Quadratic Sinus");
    evaluate_champion(&champion, sub::func2, format!("example_runs/{}/quadratic-sinus.txt", time_stamp), false, vec![70.0; N_PARTICLES], vec![vec![0.0,0.0]; N_PARTICLES]);
    time_evaluation(&champion, sub::func2, 70.0, &vec![0.0,0.0]);
    println!("Abs");
    evaluate_champion(&champion, sub::func3, format!("example_runs/{}/abs.txt", time_stamp), false, vec![70.0; N_PARTICLES], vec![vec![0.0,0.0]; N_PARTICLES]);
    time_evaluation(&champion, sub::func3, 70.0, &vec![0.0,0.0]);
    println!("Offset Sphere");
    evaluate_champion(&champion, sub::func4, format!("example_runs/{}/x-squared.txt", time_stamp), false, vec![70.0; N_PARTICLES], vec![vec![0.0,0.0]; N_PARTICLES]);
    time_evaluation(&champion, sub::func4, 70.0, &vec![0.0,0.0]);
    // evaluate_champion(&champion, FUNCTION_HANDLE_UNTRAINED1D, format!("example_runs1d/{}/abs_sin.txt", time_stamp), false, &max_dev);
    // time_evaluation(&champion, FUNCTION_HANDLE_UNTRAINED1D, &max_dev);    
    println!("Abs + Sinus");
    evaluate_champion(&champion, FUNCTION_HANDLE_UNTRAINED, format!("example_runs/{}/x-abs+sin.txt", time_stamp), false, vec![70.0; N_PARTICLES], vec![vec![0.0,0.0]; N_PARTICLES]);
    time_evaluation(&champion, FUNCTION_HANDLE_UNTRAINED, 70.0, &vec![0.0,0.0]);
    // evaluate_champion(&champion, FUNCTION_HANDLE_HIMMELBLAU, format!("example_runs/{}/himmelblau.txt", time_stamp), false, &max_dev);
    // time_evaluation(&champion, FUNCTION_HANDLE_HIMMELBLAU, &max_dev);
    println!("Rosenbrock");
    evaluate_champion(&champion, FUNCTION_HANDLE_ROSENBROCK, format!("example_runs/{}/rosenbrock.txt", time_stamp), false, vec![70.0; N_PARTICLES], vec![vec![0.0,0.0]; N_PARTICLES]);
    time_evaluation(&champion, FUNCTION_HANDLE_ROSENBROCK, 70.0, &vec![0.0,0.0]);
    println!("Rastrigin");
    evaluate_champion(&champion, FUNCTION_HANDLE_RASTRIGIN, format!("example_runs/{}/rastringin.txt", time_stamp), false, vec![70.0; N_PARTICLES], vec![vec![0.0,0.0]; N_PARTICLES]);
    time_evaluation(&champion, FUNCTION_HANDLE_RASTRIGIN, 70.0, &vec![0.0,0.0]);
    
}

fn evaluate_wall_time(champion: &Vec<Genome>,f: fn(&Vec<f32>) -> f32, radius: f32, center: &Vec<f32>) -> WallTimeEval{
    let evaluations= 10;
    let mut f_vals: Vec<f32>;
    let mut x_radius: Vec<f32>;
    let n_time_eval: usize = 30;
    let mut step_vec= Vec::<usize>::with_capacity(n_time_eval);
    let mut x_vals: Vec<Vec<f32>>;
    // let durations: Vec<f32>;
    let mut f_best_approx: f32 = f32::INFINITY;
    let mut approx_to_best: Vec<Vec<f32>> =  Vec::with_capacity(evaluations);
    let mut eval_results: Vec<Evaluation>;
    let mut x_min: Vec<f32> = Vec::with_capacity(DIMENSIONS);
    let mut x_min_worst: Vec<f32> = Vec::with_capacity(DIMENSIONS);
    let mut f_val_results: Vec<f32> =  Vec::with_capacity(evaluations);
    
    let start = Instant::now();
    for _ in 0..n_time_eval {
        let par_iter = (0..evaluations).into_par_iter().map(|_| evaluate_champion(&champion, f, "".to_string(), true,  vec![radius; N_PARTICLES], vec![center.clone() ; N_PARTICLES]));
        eval_results = par_iter.collect::<Vec<Evaluation>>();
        // f_vals = eval_results.iter().map(|a| a.fval.clone()).collect();
        for i in 0..2 {
            x_vals = eval_results.iter().map(|a| a.x_min.clone()).collect();
            x_radius = eval_results.iter().map(|a| a.radius.clone() + 0.1_f32.powi(i)).collect();
            let par_iter = (0..evaluations).into_par_iter().map(|_| evaluate_champion(&champion, f, "".to_string(), true,  x_radius.clone(), x_vals.clone()));
            eval_results = par_iter.collect::<Vec<Evaluation>>();
            }
        f_vals = eval_results.iter().map(|a| a.fval.clone()).collect();
        x_vals = eval_results.iter().map(|a| a.x_min.clone()).collect();
        step_vec.push(eval_results.iter().map(|a| a.steps).collect::<Vec<usize>>().iter().sum());
        
        let f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
        let f_min = f_vals.iter().copied().fold(f32::NAN, f32::min);
        f_val_results.push(f_min);
        // let average_time = durations.iter().sum::<f32>()/(evaluations as f32);
        // let max_time = durations.iter().copied().fold(f32::NAN, f32::max);
        
        for (i, f_val) in f_vals.iter().enumerate(){
            if f_val == &f_min {
                x_min = x_vals[i].clone();
            }
            if f_val == &f_max {
                x_min_worst = x_vals[i].clone();
            }
        }
        
        
    // dbg!(&f_vals);
    }
    let average_wall_time = start.elapsed().as_secs_f32()/32.0;
    let f_min = f_val_results.iter().copied().fold(f32::NAN, f32::min);
    let f_max = f_val_results.iter().copied().fold(f32::NAN, f32::max);
    let median = median(&mut f_val_results);
    let average_steps = (step_vec.iter().sum::<usize>()) as f32/(n_time_eval as f32);
    
    WallTimeEval { f_min, f_average: median, f_worst: f_max, x_best: x_min, x_worst: x_min_worst, walltime: average_wall_time, average_steps}
}


fn time_evaluation(champion: &Vec<Genome>,f: fn(&Vec<f32>) -> f32, radius: f32, center: &Vec<f32>){
    let eval = evaluate_wall_time(champion, f , radius, center);
    println!("Ran 30 iterations to find the minimum");
    println!("Average total wall time to find the minimum {:.3} ms", eval.walltime*1000.0);
    if eval.f_min.abs() < 0.1 {
        colour::green_ln!("Minimum has value {:+.6e} ", eval.f_min);
        colour::yellow_ln!("Median of Minimum {:+.6e} ", eval.f_average);
        colour::red_ln!("Wort minimum approximation {:+.6e} ", eval.f_worst);
    } else {
        colour::green_ln!("Minimum has value {:.6} ", eval.f_min);
        colour::yellow_ln!("Median of Minimum {:.6} ", eval.f_average);
        colour::red_ln!("Wort minimum approximation {:.6} ", eval.f_worst);
    }
    
    colour::cyan_ln!("Minimum location {:?} ", eval.x_best);
    colour::red_ln!("Worst minimum approximation location {:?} ", eval.x_worst);
    colour::grey_ln!("Took {} steps on average", eval.average_steps)
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


fn evaluate_champion(net: &Vec<Genome>, f: fn(&Vec<f32>) -> f32, filepath: String, timeonly: bool, radius: Vec<f32>, center: Vec<Vec<f32>>) -> Evaluation {
    let start = Instant::now();
    if !timeonly {
        let contents = String::from(format!(";Step;XVal;radius\n"));
        let mut sampleheader = util::generate_sampleheader(N_PARTICLES,DIMENSIONS);
        sampleheader.push_str(&contents);
        fs::write(filepath.clone(), sampleheader).expect("Unable to write file");

    }

    let mut particle_pos = (0..N_PARTICLES).enumerate().map(|(_,i)| util::CircleGeneratorInput{
        dimensions:DIMENSIONS,
        samplepoints: SAMPLEPOINTS,
        radius: radius[i],
        center: center[i].to_vec()
    }).collect::<Vec<util::CircleGeneratorInput>>();
    let mut all_xvals = (0..N_PARTICLES).enumerate().map(|(_,i)| AllXValsCircle{
        x_guess : center[i].to_vec(),
        radius: radius[i],
        f_val: f32::INFINITY,
        velocity: vec![0.0; DIMENSIONS],
        delta_fitness: f32::INFINITY,
        fitness_change_limited : 0.0,
        last_fitness: f32::INFINITY,
        currentbest: false
    }).collect::<Vec<AllXValsCircle>>();
    // let mut set_of_sample_points = util::generate_1d_samplepoints(&function_domain, 0);
    let mut set_of_samples_line =  util::generate_samplepoint_vector(&vec![0.0,0.0], &vec![1.0,1.0], SAMPLEPOINTS);
    let mut set_of_sample_points: util::SetOfSamplesCircle;
    let mut f_vals: Vec<f32>;
    let mut f_vals_normalized: Vec<f32>;
    let mut x_velocity_point: Vec<f32>;
    let mut evaluator_circle = MatrixRecurrentFabricator::fabricate(&net[0]).expect("didnt work");
    let mut evaluator_line = MatrixRecurrentFabricator::fabricate(&net[1]).expect("didnt work");
    let mut step = 0;
    let mut steptotal = 0;
    let mut delta_fitness = f32::INFINITY;
    let mut radius_guess: f32 = 0.0;
    // let mut last_fitness = f32::INFINITY;
    let mut skiprest: bool;
    // let mut fitness_change_limited: f32;
    let mut vector_start: Vec<f32> = vec![0.0; DIMENSIONS];
    let mut vector_end: Vec<f32> = vec![0.0; DIMENSIONS];
    let mut x_guess_best: Vec<f32> = vec![0.0; DIMENSIONS];
    for n_drill in 0..N_DRILL+N_OVERDRILL {
        step = 0;
        skiprest = false;
        for n_particle in 0..N_PARTICLES {
            particle_pos[n_particle].radius = particle_pos[n_particle].radius+0.005;
            all_xvals[n_particle].fitness_change_limited = 0.0;
            
        }
        while (step < STEPS) & (delta_fitness.abs() > 1e-10_f32) {
            for n_particle in 0..N_PARTICLES {
            // if  step%3 == 0{
                // if all_xvals[n_particle].fitness_change_limited  >= - 0.0 {
                    set_of_sample_points = util::generate_samplepoint_2d_random_slice(&particle_pos[n_particle]);
                    f_vals = set_of_sample_points.coordinates.iter().enumerate().map(|(_, x)| f(x)).collect::<Vec<_>>();
                    let mut i_min: usize = 0;
                        for i in 0..SAMPLEPOINTS {
                            if f_vals[i] < f_vals[i_min] {
                                i_min = i;
                            };
                        }
                        if step == 0 {
                            for dim in 0..DIMENSIONS {
                                all_xvals[n_particle].velocity[dim] = set_of_sample_points.coordinates[i_min][dim].clone() - all_xvals[n_particle].x_guess[dim];
                            }
                        }
                        x_velocity_point = all_xvals[n_particle].x_guess.iter().enumerate().map(|(dim, x)| x + all_xvals[n_particle].velocity[dim]).collect::<Vec<f32>>();
                        if step > 0 {
                            if f(&set_of_sample_points.coordinates[i_min])< f(&x_velocity_point) {
                                for dim in 0..DIMENSIONS {
                                    all_xvals[n_particle].velocity[dim] = set_of_sample_points.coordinates[i_min][dim].clone() - all_xvals[n_particle].x_guess[dim];
                                }
                                x_velocity_point= all_xvals[n_particle].x_guess.iter().enumerate().map(|(dim, x)| x + all_xvals[n_particle].velocity[dim]).collect::<Vec<f32>>();
                            }
                        }
                        vector_start = x_velocity_point;
                    // f_vals_normalized = normalise_vector(&f_vals);
                    // let input_values: Vec<f32> = [set_of_sample_points.coordinates_normalised.clone(), f_vals_normalized.clone(), vec![(n_drill as f32)/(N_DRILL+N_OVERDRILL) as f32, (STEPS -step) as f32/STEPS as f32, 1.0, all_xvals[n_particle].fitness_change_limited]].concat();
                    // network_step(input_values, &mut evaluator_circle, &mut all_xvals[n_particle], &particle_pos[n_particle], step);
                    // vector_start  = all_xvals[n_particle].x_guess.clone();
                    vector_end = particle_pos[n_particle].center.clone();
                    if vector_start.iter().enumerate().all(|(i,x)| (vector_end[i]-x).abs() < 1e-8_f32) {
                        skiprest = true;
                        delta_fitness = 0.0;
                        all_xvals[n_particle].x_guess = vector_start.clone();
                    // }
                    }
                // }
                // if skiprest {
                //     dbg!(&n_drill, &step, &skiprest);
                // }
                if !skiprest{
                    let vector_start_resize = vector_start.iter().enumerate().map(|(i,x)| x -((STEPS -step) as f32/STEPS as f32)*2.0*( vector_end[i] - x)).collect::<Vec<f32>>();
                    let vector_end_resize = vector_end.iter().enumerate().map(|(i,x)| x +0.0*( x - vector_start[i])).collect::<Vec<f32>>();
                    set_of_samples_line = util::generate_samplepoint_vector(&vector_start_resize, &vector_end_resize, SAMPLEPOINTS);
                    f_vals = set_of_samples_line.coordinates.iter().enumerate().map(|(_, x)| f(x).clone()).collect::<Vec<_>>();
                    f_vals_normalized = normalise_vector(&f_vals);
                    let input_values: Vec<f32> = [set_of_samples_line.coordinates_normalised.clone(), f_vals_normalized.clone(), vec![n_drill as f32/(N_DRILL+N_OVERDRILL) as f32, (STEPS -step) as f32/STEPS as f32, 1.0, all_xvals[n_particle].fitness_change_limited]].concat();
                    network_step_line(input_values, &mut evaluator_line, &mut all_xvals[n_particle], &vector_start_resize, &vector_end_resize);
                    particle_pos[n_particle].center = all_xvals[n_particle].x_guess.clone();
                    particle_pos[n_particle].radius = all_xvals[n_particle].radius.clone();
                    all_xvals[n_particle].f_val= f(&all_xvals[n_particle].x_guess);
                    all_xvals[n_particle].delta_fitness = all_xvals[n_particle].last_fitness - all_xvals[n_particle].f_val;
                    all_xvals[n_particle].fitness_change_limited = tanhf(all_xvals[n_particle].delta_fitness);
                    all_xvals[n_particle].last_fitness = all_xvals[n_particle].f_val;
                }
            }
            let fitness_min = all_xvals.iter().map(|x| x.f_val).collect::<Vec<f32>>().iter().copied().fold(f32::NAN, f32::min);
            
            for n_particle in 0..N_PARTICLES {
                if all_xvals[n_particle].f_val == fitness_min{
                    all_xvals[n_particle].currentbest = true;
                    x_guess_best = all_xvals[n_particle].x_guess.clone();
                } else {
                    all_xvals[n_particle].currentbest = false;
                }
            }
            if step > STEPS/2 {
                for n_particle in 0..N_PARTICLES {
                    if (all_xvals[n_particle].currentbest == false) & (all_xvals[n_particle].fitness_change_limited > -0.5){
                        vector_start = all_xvals[n_particle].x_guess.clone();
                        vector_end  = x_guess_best.clone();
                        let set_of_samples_line = util::generate_samplepoint_vector(&vector_start, &vector_end, SAMPLEPOINTS);
                        f_vals = set_of_samples_line.coordinates.iter().enumerate().map(|(_, x)| f(x).clone()).collect::<Vec<_>>();
                        f_vals_normalized = normalise_vector(&f_vals);
                        let input_values: Vec<f32> = [set_of_samples_line.coordinates_normalised.clone(), f_vals_normalized.clone(), vec![n_drill as f32/N_DRILL as f32, (STEPS -step) as f32/STEPS as f32, 1.0,all_xvals[n_particle].fitness_change_limited]].concat();
                        network_step_line(input_values, &mut evaluator_line, &mut all_xvals[n_particle], &vector_start, &vector_end);
                    } 
                }
            }
            
            for n_particle in 0..N_PARTICLES {
                if all_xvals[n_particle].f_val == fitness_min{
                    radius_guess = particle_pos[n_particle].radius;
                } 
            }
            if !timeonly {
                // let prediction =PredictionCircle {
                //     fitness: fitness_min,
                //     x_guess: x_guess_best.clone(),
                //     radius: radius_guess};
                let mut file_write = OpenOptions::new()
                .write(true)
                .append(true)
                .open(filepath.clone())
                .unwrap();
                // print!("champion_global Step {}: {:?}\n",step, prediction); 
                let coordinates_string: String = all_xvals.iter().map(|x|  x.x_guess.iter().map( |&id| id.to_string()).collect::<Vec<String>>().join(",")).collect::<Vec<String>>().join(";");
                let f_vals_string: String = all_xvals.iter().map( |i| i.f_val.to_string()).collect::<Vec<String>>().join(";");
                let x_guess_string: String = x_guess_best.clone().into_iter().map(|i| i.to_string()).collect::<Vec<String>>().join(",");
                let radius_string: String = radius_guess.to_string();
                if let Err(e) = writeln!(file_write, "{coords};{fvals};{step};{x_val1};{radius}",coords =coordinates_string,fvals=f_vals_string, step=steptotal, x_val1 =x_guess_string , radius = radius_string) {
                    eprintln!("Couldn't write to file: {}", e);
                }
            } 
            step+=1;
            steptotal+=1;
        }
    }
    let duration = start.elapsed().as_secs_f32();
    let eval =  Evaluation { fval: f(&x_guess_best), x_min: x_guess_best, steps: steptotal, duration, radius:radius_guess };
    eval
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
        let mut step: usize;
        let mut x_velocity_point: Vec<f32>;
        let mut delta_fitness = f32::INFINITY;
        let mut x_velocity_point: Vec<f32>;
        let mut particle_pos = vec![util::CircleGeneratorInput{
            dimensions:DIMENSIONS,
            samplepoints: SAMPLEPOINTS,
            radius: radius,
            center: center.to_vec()
        }; N_PARTICLES];
        let mut all_xvals = vec![AllXValsCircle{
            x_guess : center.to_vec(),
            velocity: vec![0.0; DIMENSIONS],
            radius: radius,
            f_val: f32::INFINITY,
            delta_fitness: f32::INFINITY,
            fitness_change_limited : 0.0,
            last_fitness: f32::INFINITY,
            currentbest: false
        }; N_PARTICLES];
        let mut vector_start: Vec<f32>  =vec![0.0; DIMENSIONS];
        let mut vector_end: Vec<f32> = vec![0.0; DIMENSIONS];
        for n_drill in 0..N_DRILL {
            step = 0;
            let mut skiprest: bool = false;
            for n_particle in 0..N_PARTICLES {
                particle_pos[n_particle].radius = particle_pos[n_particle].radius+0.005;
                all_xvals[n_particle].fitness_change_limited = 0.0;
                
            }
                
            while (step < STEPS) & (delta_fitness.abs() > 1e-10_f32) {
                for n_particle in 0..N_PARTICLES {
                // if step%3 == 0{
                    // if all_xvals[n_particle].fitness_change_limited >= - 0.0 {   
                        let set_of_sample_points = util::generate_samplepoint_2d_random_slice(&particle_pos[n_particle]);
                        f_vals = set_of_sample_points.coordinates.iter().enumerate().map(|(_, x)| f(x).clone() +g(x, y, z)).collect::<Vec<_>>();
                        let mut i_min: usize = 0;
                        for i in 0..SAMPLEPOINTS {
                            if f_vals[i] < f_vals[i_min] {
                                i_min = i;
                            };
                        }
                        if step == 0 {
                            for dim in 0..DIMENSIONS {
                                all_xvals[n_particle].velocity[dim] = set_of_sample_points.coordinates[i_min][dim].clone() - all_xvals[n_particle].x_guess[dim];
                            }
                        }
                        x_velocity_point = all_xvals[n_particle].x_guess.iter().enumerate().map(|(dim, x)| x + all_xvals[n_particle].velocity[dim]).collect::<Vec<f32>>();
                        if step > 0 {
                            if f(&set_of_sample_points.coordinates[i_min]) +g(&set_of_sample_points.coordinates[i_min], y, z) < f(&x_velocity_point) +g(&x_velocity_point, y, z) {
                                for dim in 0..DIMENSIONS {
                                    all_xvals[n_particle].velocity[dim] = set_of_sample_points.coordinates[i_min][dim].clone() - all_xvals[n_particle].x_guess[dim];
                                }
                                x_velocity_point= all_xvals[n_particle].x_guess.iter().enumerate().map(|(dim, x)| x + all_xvals[n_particle].velocity[dim]).collect::<Vec<f32>>();
                            }
                        }
                        vector_start = x_velocity_point;
                        // f_vals_normalized = normalise_vector(&f_vals);
                        // let input_values: Vec<f32> = [set_of_sample_points.coordinates_normalised.clone(), f_vals_normalized.clone(), vec![n_drill as f32/N_DRILL as f32, (STEPS -step) as f32/STEPS as f32, 1.0,all_xvals[n_particle].fitness_change_limited]].concat();
                        // network_step(input_values, evaluator_circle, &mut all_xvals[n_particle], &particle_pos[n_particle], step);
                        // if all_xvals[n_particle].x_guess.iter().any(|i| i.is_nan()) || all_xvals[n_particle].radius.is_nan() || f_vals.iter().any(|i| i.is_nan()) {
                        //     return f32::INFINITY;
                        // }
                        // vector_start = all_xvals[n_particle].x_guess.clone();
                        vector_end  = particle_pos[n_particle].center.clone();
                        if vector_start.iter().enumerate().all(|(i,x)| (vector_end[i]-x).abs() < 1e-8_f32) {
                            skiprest = true;
                            delta_fitness = 0.0;
                        }
                    // }
                    // }
                    if !skiprest{
                        let vector_start_resize = vector_start.iter().enumerate().map(|(i,x)| x -((STEPS -step) as f32/STEPS as f32)*2.0*( vector_end[i] - x)).collect::<Vec<f32>>();
                        let vector_end_resize = vector_end.iter().enumerate().map(|(i,x)| x +0.0*( x - vector_start[i])).collect::<Vec<f32>>();
                        let set_of_samples_line = util::generate_samplepoint_vector(&vector_start_resize, &vector_end_resize, SAMPLEPOINTS);
                        f_vals = set_of_samples_line.coordinates.iter().enumerate().map(|(_, x)| f(x).clone() +g(x, y, z)).collect::<Vec<_>>();
                        f_vals_normalized = normalise_vector(&f_vals);
                        let input_values: Vec<f32> = [set_of_samples_line.coordinates_normalised.clone(), f_vals_normalized.clone(), vec![n_drill as f32/N_DRILL as f32, (STEPS -step) as f32/STEPS as f32, 1.0,all_xvals[n_particle].fitness_change_limited]].concat();
                        network_step_line(input_values, evaluator_line, &mut all_xvals[n_particle], &vector_start_resize, &vector_end_resize);
                        particle_pos[n_particle].center = all_xvals[n_particle].x_guess.clone();
                        particle_pos[n_particle].radius = all_xvals[n_particle].radius.clone();
                        all_xvals[n_particle].f_val= f(&all_xvals[n_particle].x_guess) + g(&all_xvals[n_particle].x_guess, y, z);
                        all_xvals[n_particle].delta_fitness = all_xvals[n_particle].last_fitness - all_xvals[n_particle].f_val;
                        all_xvals[n_particle].fitness_change_limited = tanhf(all_xvals[n_particle].delta_fitness);
                        all_xvals[n_particle].last_fitness = all_xvals[n_particle].f_val;
                    }
                }
                let fitness_min = all_xvals.iter().map(|x| x.f_val).collect::<Vec<f32>>().iter().copied().fold(f32::NAN, f32::min);
                let mut x_guess_best: Vec<f32> = vec![0.0; DIMENSIONS];
                for n_particle in 0..N_PARTICLES {
                    if all_xvals[n_particle].f_val == fitness_min{
                        all_xvals[n_particle].currentbest = true;
                        x_guess_best = all_xvals[n_particle].x_guess.clone();
                    } else {
                        all_xvals[n_particle].currentbest = false;
                    }
                }
                if step > STEPS/2 {
                    for n_particle in 0..N_PARTICLES {
                        if (all_xvals[n_particle].currentbest == false) & (all_xvals[n_particle].fitness_change_limited > -0.5){ 
                            vector_start = all_xvals[n_particle].x_guess.clone();
                            vector_end  = x_guess_best.clone();
                            let set_of_samples_line = util::generate_samplepoint_vector(&vector_start, &vector_end, SAMPLEPOINTS);
                            f_vals = set_of_samples_line.coordinates.iter().enumerate().map(|(_, x)| f(x).clone() +g(x, y, z)).collect::<Vec<_>>();
                            f_vals_normalized = normalise_vector(&f_vals);
                            let input_values: Vec<f32> = [set_of_samples_line.coordinates_normalised.clone(), f_vals_normalized.clone(), vec![n_drill as f32/N_DRILL as f32, (STEPS -step) as f32/STEPS as f32, 1.0,all_xvals[n_particle].fitness_change_limited]].concat();
                            network_step_line(input_values, evaluator_line, &mut all_xvals[n_particle], &vector_start, &vector_end);
                        } 
                    }
                }

                step+=1;
            } 
        }
        let mut x_guess: Vec<f32> = vec![0.0; DIMENSIONS];
        let mut radius: f32 = 0.0;
        let fitness_min = all_xvals.iter().map(|x| x.f_val).collect::<Vec<f32>>().iter().copied().fold(f32::NAN, f32::min);
        for n_particle in 0..N_PARTICLES {
            if all_xvals[n_particle].f_val == fitness_min{
                x_guess = particle_pos[n_particle].center.clone();
                radius = particle_pos[n_particle].radius;
            } 
        }
        let fitness_eval = FitnessEvalCircle{
            fitness: fitness_min,
            x_guess: x_guess,
            radius: radius
        };
        fitness_vec.push(fitness_eval.clone());
        scores_vec.push(fitness_min);
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