mod sub;

use favannat::{MatrixRecurrentFabricator, StatefulFabricator, StatefulEvaluator, MatrixRecurrentEvaluator};
use rand::{
    distributions::{Uniform},
    seq::SliceRandom,
    thread_rng, Rng,
};
use std::{fs::OpenOptions, cmp::max};
use std::io::prelude::*;
use std::{cmp::Ordering, fs};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use set_genome::{Genome, Parameters, Structure, Mutations, activations::Activation};

const STEPS: usize = 100;
const N_TRYS: usize = 10;
const N_TESTFUNCTIONS: usize =4;
const SAMPLEPOINTS: usize = 6;
const POPULATION_SIZE: usize = 400;
const GENERATIONS: usize = 300;
const MAXDEV: f32 = 100.0;
const FUNCTION_HANDLE_UNTRAINED: fn(f32) -> f32 = |x|  (x-3.0).abs()+ 1.5*x.sin();


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
    x_plus: f32,
    step: usize
}



#[derive(Debug)]
pub struct TestfunctionEval{
    fitness_eval: FitnessEval,
    function_name: String,
}

#[derive(Debug)]
pub struct OverallFitness{
    fitness: f32,
    fitnessvec: Vec<TestfunctionEval>
}


fn main() {
    // env::set_var("RUST_BACKTRACE", "1");
    let parameters = Parameters {
        structure: Structure { number_of_inputs: (2*SAMPLEPOINTS + 1), number_of_outputs: (3), percent_of_connected_inputs: (1.0), outputs_activation: (Activation::Sigmoid), seed: (42) },
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
            if let Err(_) = child.mutate(&parameters) {
            };
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

fn evaluate_champion(champion: &Genome, f: fn(f32) -> f32, filepath: &str) {
    let contents = String::from(format!(",x-best,Step,XVal,x-plus,x-minus\n"));
    let mut sampleheader = generate_sampleheader();
    sampleheader.push_str(&contents);
    fs::write(filepath, sampleheader).expect("Unable to write file");
    let mut rng = rand::thread_rng();
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open(filepath)
        .unwrap();
    let mut x_vals_normalized;
    let mut x_guess : f32 = 0.0;
    let mut x_minus: f32 = MAXDEV;
    let mut x_plus: f32 = MAXDEV;
    let mut x_best: f32 = rng.gen_range(0.0..x_minus + x_plus);
    let mut between: Uniform<f32> = Uniform::from(0.0..x_minus + x_plus);
    let mut x_vals: Vec<f32> = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut f_vals_normalized : Vec<f32>;
    let mut f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x-x_minus).clone()).collect::<Vec<_>>();
    // norm f_vals
    let mut f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
    // f_vals = f_vals.iter().enumerate().map(|(_, x)| *x/f_max).collect::<Vec<_>>();
    // get max and min of x_samples for interpretation of prediction
    let mut x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
    let mut x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
    // norm x_vals
    x_vals_normalized = x_vals.iter().enumerate().map(|(_, x)| (*x-x_min)/(x_max-x_min)).collect::<Vec<_>>();
    // x_vals should be between 0 and 1 now, the first always0, the last always 1.
    // for _ in 0..N_TRYS {
    let mut evaluator = MatrixRecurrentFabricator::fabricate(champion).expect("didnt work");
    // for step in 0..STEPS {
    let mut step = 0;
        while ((x_minus + x_plus) > 1.0e-4_f32) & (step < STEPS){
            step+=1; 
            f_vals = f_vals.clone();
            f_vals_normalized = f_vals.iter().enumerate().map(|(_, x)| *x/f_max).collect::<Vec<_>>();
            let input_values: Vec<f32> = [x_vals.clone().iter().enumerate().map(|(_, x)| *x-x_minus+x_guess).collect::<Vec<_>>(), f_vals.clone(), vec![(STEPS -step) as f32/STEPS as f32]].concat();
            let input_values_normalized: Vec<f32> = [x_vals_normalized, f_vals_normalized, vec![(STEPS -step) as f32/STEPS as f32]].concat();
            let prediction: Vec<f64> = evaluator.evaluate(input_values_normalized.clone().iter().map(|x | *x as f64).collect() );
            let mut pred_x_guess = prediction[0] as f32;
            let mut pred_x_minus = prediction[1]as f32;
            let mut pred_x_plus = prediction[2]as f32;
            if pred_x_minus < 0.0{
                pred_x_minus = 0.0;
            }   else if pred_x_minus > 1.0 {
                pred_x_minus = 1.0;
            }
            if pred_x_plus < 0.0{
                pred_x_plus = 0.0;
            }   else if pred_x_plus > 1.0 {
                pred_x_plus = 1.0;
            }
            if pred_x_guess < 0.0{
                pred_x_guess = 0.0;
            }   else if pred_x_guess > 1.0 {
                pred_x_guess = 1.0;
            }
            // dbg!(&pred_x_guess, x_max.clone()-x_min.clone(), x_min, x_minus);
            x_guess = (pred_x_guess)*(x_max-x_min)+x_min -x_minus + x_guess;
            x_minus = x_minus*(0.5+pred_x_minus);
            x_plus = x_plus*(0.5+pred_x_plus);

            let prediction =Prediction {
                fitness: f(x_guess) + (x_minus + x_plus),
                x_guess: x_guess,
                x_plus: x_plus,
                x_minus: x_minus};
            print!("Champion Step {}: {:?}\n",step, prediction); 
            let samples_string: String = input_values.iter().map( |&id| id.to_string() + ",").collect();
            if let Err(e) = writeln!(file, "{samples}{step},{x_val1},{x_plus},{x_minus}",samples =samples_string, step=step, x_val1 = x_guess, x_plus = x_plus, x_minus = x_minus) {
                eprintln!("Couldn't write to file: {}", e);
            }
            between = Uniform::from(0.0..x_minus + x_plus);
            x_vals = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
            x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x-x_minus+x_guess).clone()).collect::<Vec<_>>();
            f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
            x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
            x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
            x_vals_normalized = x_vals.iter().enumerate().map(|(_, x)| (*x-x_min)/(x_max-x_min)).collect::<Vec<_>>();
            if f(x_guess) < f(x_best) {
                x_best = x_guess;
            }
            // let contents = String::from(format!());
            
    }
    
    // fs::write("data/out.txt", contents).expect("Unable to write file");
}

fn evaluate_on_testfunction(f: fn(f32) -> f32, mut evaluator:  MatrixRecurrentEvaluator) -> FitnessEval {
    let mut fitness_vec= Vec::with_capacity(N_TRYS);
    let mut x_guess: f32;
    let mut x_worst: f32 = 0.0;
    let mut x_minus: f32 = MAXDEV;
    let mut x_plus: f32 = MAXDEV;
    let mut step_worst: usize = 0;
    for try_enum in 0..N_TRYS {
        let mut counter: usize = 0;
        let mut rng = rand::thread_rng();
        let y =rng.gen_range(0.0..10.0);
        let y2 =rng.gen_range(0.0..10.0);
        let g: fn(f32, f32, f32) -> f32 = |x, y, z| y*x.sin() + z*x.powi(2);
        let mut x_best: f32 = rng.gen_range(0.0..x_minus + x_plus);
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
        let mut f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x-x_minus).clone() +g(*x-x_minus, y, y2)).collect::<Vec<_>>();
        // norm f_vals
        let mut f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
        f_vals = f_vals.iter().enumerate().map(|(_, x)| *x/f_max).collect::<Vec<_>>();
        

        // get max and min of x_samples for interpretation of prediction
        let mut x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
        let mut x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
        // norm x_vals
        x_vals = x_vals.iter().enumerate().map(|(_, x)| (*x-x_min)/(x_max-x_min)).collect::<Vec<_>>();
        // x_vals should be between 0 and 1 now, the first always0, the last always 1.
        // dbg!(&x_vals);
        // for step in 0..STEPS {
        let mut step = 0;
        while ((x_minus + x_plus) > 1.0e-4_f32) & (step < STEPS) {
            // dbg!(&f_vals);
            step+=1;    
            let input_values: Vec<f32> = [x_vals, f_vals, vec![(STEPS -step) as f32/STEPS as f32]].concat();
            let prediction: Vec<f64> = evaluator.evaluate(input_values.clone().iter().map(|x | *x as f64).collect() );
            // dbg!(&prediction);
            
            // x_guess = (prediction[0] as f32)*(x_max) + (1.0-prediction[0] as f32)*x_min;
            let mut pred_x_guess = prediction[0] as f32;
            let mut pred_x_minus = prediction[1]as f32;
            let mut pred_x_plus = prediction[2]as f32;
            if pred_x_minus < 0.0{
                pred_x_minus = 0.0;
            }   else if pred_x_minus > 1.0 {
                pred_x_minus = 1.0;
            }
            if pred_x_plus < 0.0{
                pred_x_plus = 0.0;
            }   else if pred_x_plus > 1.0 {
                pred_x_plus = 1.0;
            }
            if pred_x_guess < 0.0{
                pred_x_guess = 0.0;
            }   else if pred_x_guess > 1.0 {
                pred_x_guess = 1.0;
            }
            x_guess = (pred_x_guess)*(x_max-x_min)+x_min -x_minus + x_guess;
            
            // dbg!(&x_guess);
            x_minus = x_minus*(0.5+pred_x_minus);
            x_plus = x_plus*(0.5+pred_x_plus);
            // dbg!(&x_minus, &x_guess, &x_plus);
            // if simulation runs out of bounds, return inf values
            if x_guess.is_nan() || x_minus.is_nan()|| x_plus.is_nan(){
                return FitnessEval {
                    fitness: f32::INFINITY,
                    x_worst: f32::INFINITY,
                    x_plus: f32::INFINITY,
                    x_minus: f32::INFINITY,
                    step: STEPS};
            }
            

            // dbg!([x_guess, x_minus, x_plus]);
            lower_range_limit = x_guess - x_minus;
            upper_range_limit = x_guess + x_plus;
            if upper_range_limit - lower_range_limit == f32::INFINITY {
                // dbg!([x_guess, x_minus, x_plus]);
                return FitnessEval {
                    fitness: f32::INFINITY,
                    x_worst: f32::INFINITY,
                    x_plus: f32::INFINITY,
                    x_minus: f32::INFINITY,
                    step: STEPS
                };
            }
            // print new lower range, guessed x and upper range
            // dbg!([lower_range_limit,x_guess, upper_range_limit]);
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
                };
            } else {
                between = Uniform::from(0.0..x_minus + x_plus);
                x_vals = rand::thread_rng().sample_iter(&between).take(SAMPLEPOINTS).collect();
                x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                f_vals = x_vals.iter().enumerate().map(|(_, x)| f(*x+x_guess-x_minus).clone() +g(*x+x_guess-x_minus, y, y2)).collect::<Vec<_>>();
                f_max = f_vals.iter().copied().fold(f32::NAN, f32::max);
                f_vals = f_vals.iter().enumerate().map(|(_, x)| *x/f_max).collect::<Vec<_>>();
            }
            x_max = x_vals.iter().copied().fold(f32::NAN, f32::max);
            x_min = x_vals.iter().copied().fold(f32::NAN, f32::min);
            // norm x_vals
            x_vals = x_vals.iter().enumerate().map(|(_, x)| (*x-x_min)/(x_max-x_min)).collect::<Vec<_>>();
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
        if try_enum == 0 {
            x_worst = x_guess;
        }
        if f(x_guess) + g(x_guess, y, y2) > f(x_worst) + g(x_worst, y, y2)
        {
            x_worst = x_guess;
        }
        
        step_worst = max(step_worst, step);
        fitness_vec.push(f(x_guess) + g(x_guess, y, y2) + (x_minus + x_plus)*(1.0+step as f32));
        // fitness_vec.push(f(x_guess) + g(x_guess, y, y2) + (x_minus + x_plus)*(1.0+step as f32) + counter as f32);
        // fitness_vec.push(f(x_guess) + g(x_guess, y, y2));
        
    }
    let fitness = fitness_vec.iter().copied().fold(f32::NAN, f32::max);
    FitnessEval{
        fitness: fitness,
        x_worst: x_worst,
        x_plus: x_plus,
        x_minus: x_minus,
        step: step_worst
    }
}

fn evaluate_net_fitness(net: &Genome) -> OverallFitness {
    let mut fitness_vec: Vec<f32>= Vec::with_capacity(N_TESTFUNCTIONS);
    let fitness_max: f32;
    let mut eval_vec: Vec<FitnessEval>= Vec::with_capacity(N_TESTFUNCTIONS);
    // let mut x_guess_vec= Vec::with_capacity(N_TRYS);
    
    let mut evaluator = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    let mut fitness = evaluate_on_testfunction(sub::func1, evaluator);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    evaluator = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    fitness = evaluate_on_testfunction(sub::func2, evaluator);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    evaluator = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    fitness = evaluate_on_testfunction(sub::func3, evaluator);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    evaluator = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
    fitness = evaluate_on_testfunction(sub::func4, evaluator);
    fitness_vec.push(fitness.fitness);
    eval_vec.push(fitness);
    // dbg!(&delta_f_values);
    fitness_max = fitness_vec.iter().copied().fold(f32::NAN, f32::max);
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
