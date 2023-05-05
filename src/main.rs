mod sub;

use favannat::{Evaluator, Fabricator, MatrixRecurrentFabricator, StatefulFabricator, StatefulEvaluator};
use rand::{
    distributions::{Distribution, Uniform},
    seq::SliceRandom,
    thread_rng,
};
use std::cmp::Ordering;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use set_genome::{activations::Activation, Genome, Mutations, Parameters, Structure};
use std::fs;

const STEPS: usize = 10;
const POPULATION_SIZE: usize = 100;
const GENERATIONS: usize = 300;
fn main() {
    let parameters = Parameters {
        structure: Structure::basic(4, 1),
        mutations: vec![
            Mutations::ChangeWeights {
                chance: 1.0,
                percent_perturbed: 0.5,
                standard_deviation: 0.1,
            },
            Mutations::AddNode {
                chance: 0.1,
                activation_pool: vec![
                    Activation::Sigmoid,
                    Activation::Tanh,
                    Activation::Gaussian,
                    Activation::Step,
                    Activation::Sine,
                    Activation::Cosine,
                    Activation::Inverse,
                    Activation::Absolute,
                    Activation::Relu,
                ],
            },
            Mutations::AddConnection { chance: 0.2 },
            Mutations::AddRecurrentConnection { chance: 0.01 },
            Mutations::RemoveConnection { chance: 0.15 },
            Mutations::RemoveNode { chance: 0.05 }
        ],
    };

    let mut current_population = Vec::with_capacity(POPULATION_SIZE);

    for _ in 0..POPULATION_SIZE {
        current_population.push(Genome::initialized(&parameters))
    }

    let mut champion = current_population[0].clone();
    // let mut summed_diff = 0.0;
    for _ in 0..GENERATIONS {
        // ## Evaluate current nets
        
        let mut population_fitnesses = current_population
            .par_iter()
            .map(evaluate_net_fitness)
            .enumerate()
            .collect::<Vec<_>>();
        population_fitnesses.sort_unstable_by(|a, b| match (a.1 .1.is_nan(), b.1 .1.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => a.1 .1.partial_cmp(&b.1 .1).unwrap(),
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
        
        dbg!(population_fitnesses.iter().take(10).collect::<Vec<_>>());
    }
    print!("{}", net_as_dot(&champion));
    let x1= 0.0;
    let x12= 0.5;
    let x2= -2.0;
    let x22= -2.5;
    let input_values_right = vec![x1, sub::func1(x1), x12, sub::func1(x12)];
    let input_values_left = vec![x2, sub::func1(x2), x22, sub::func1(x22)];
    let mut evaluator = MatrixRecurrentFabricator::fabricate(&champion).expect("didnt work");
    let mut x_eval = x1;
    for _ in 0..STEPS {
        let input_values = vec![x_eval, sub::func1(x_eval).clone(), x12, sub::func1(x12)];
            let delta = evaluator.evaluate(input_values)[0];
            x_eval = x_eval + delta;
            print!("Champion Walk x from right side: {}\n", x_eval);
    }
    let mut x_eval = x2;
    for _ in 0..STEPS {
        let input_values = vec![x_eval, sub::func1(x_eval).clone(), x22, sub::func1(x22)];
            let delta = evaluator.evaluate(input_values)[0];
            x_eval = x_eval + delta;
            print!("Champion Walk x from left side: {}\n", x_eval);
    }
    let delta1=evaluator.evaluate(input_values_left)[0];
    let delta2=evaluator.evaluate(input_values_right)[0];
    let contents = String::from(format!("XVals,Delta\n{x_val1},{delta1}\n{x_val2},{delta2}",x_val1 = x1, x_val2 = x2, delta1 = delta1, delta2 = delta2));
    fs::write("data/out.txt", contents).expect("Unable to write file");
    print!("Champion Delta x from right side: {}\n", delta1);
    
    print!("Champion Delta x from left side: {}\n", delta2);
}

fn median(numbers: &mut Vec<f64>) -> f64 {
    numbers.sort_unstable_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap(),
    });
    let mid = numbers.len() / 2;
    numbers[mid]
}


fn evaluate_net_fitness(net: &Genome) -> (f64, f64) {
    let between = Uniform::from(-100.0..100.0);
    let mut rng = thread_rng();

    let mut x_values = Vec::with_capacity(STEPS);
    let mut fitness_values = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let x_start = vec![between.sample(&mut rng), between.sample(&mut rng)];
        let mut evaluator = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");
        let mut x1 = x_start[0].clone();
        let mut x2 = x_start[1].clone();
        for _ in 0..STEPS {
            let input_values = vec![x1, sub::func1(x1).clone(), x2, sub::func1(x2).clone()];
            let delta = evaluator.evaluate(input_values)[0];
            x1 = x1 + delta;
        }
        let delta_f = (sub::func1(x_start[0]) - sub::func1(x1))/sub::func1(x_start[0]);
        x_values.push(x1);
        fitness_values.push(sub::func1(x1) + delta_f.abs().exp());
    }
    // dbg!(&delta_f_values);
    (
        x_values.iter().sum::<f64>() / STEPS as f64,
        median(&mut fitness_values) as f64,
      
    )
}

fn net_as_dot(net: &Genome) -> String {
    let mut dot = "digraph {\n".to_owned();

    for node in net.nodes() {
        dot.push_str(&format!("{} [label={:?}];\n", node.id.0, node.activation));
    }

    for connection in net.connections() {
        dot.push_str(&format!(
            "{} -> {} [label={:?}];\n",
            connection.input.0, connection.output.0, connection.weight
        ));
    }

    dot.push_str("}\n");
    dot
}
