use favannat::{Evaluator, Fabricator, MatrixRecurrentFabricator, StatefulFabricator, StatefulEvaluator};
use rand::{
    distributions::{Distribution, Uniform},
    seq::SliceRandom,
    thread_rng,
};
use std::cmp::Ordering;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use set_genome::{activations::Activation, Genome, Mutations, Parameters, Structure};

const STEPS: usize = 10;
const POPULATION_SIZE: usize = 100;
const GENERATIONS: usize = 100;
const X_SQUARE: fn(f64) -> f64 = |x| 0.5 * x.powi(4) - 0.7 * x.powi(2) + 0.1 * x;

fn main() {
    let parameters = Parameters {
        structure: Structure::basic(2, 1),
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
            Mutations::AddConnection { chance: 0.3 },
            Mutations::AddRecurrentConnection { chance: 0.01 },
        ],
    };

    let mut current_population = Vec::with_capacity(POPULATION_SIZE);

    for _ in 0..POPULATION_SIZE {
        current_population.push(Genome::initialized(&parameters))
    }

    let mut champion = current_population[0].clone();

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
        current_population = new_population;

        dbg!(population_fitnesses.iter().take(10).collect::<Vec<_>>());
    }

    dbg!(&champion);

    print!("{}", net_as_dot(&champion));
    let mut evaluator = MatrixRecurrentFabricator::fabricate(&champion).expect("didnt work");
    let mut input_values = vec![0.0, X_SQUARE(0.0)];
    print!("Champion Delta x from right side: {}\n", &evaluator.evaluate(input_values)[0]);
    input_values = vec![-3.0, X_SQUARE(-3.3)];
    print!("Champion Delta x from left side: {}\n", &evaluator.evaluate(input_values)[0]);
}

fn evaluate_net_fitness(net: &Genome) -> (f64, f64) {
    let between = Uniform::from(-100.0..100.0);
    let mut rng = thread_rng();

    let mut x_values = Vec::with_capacity(STEPS);
    let mut fitness_values = Vec::with_capacity(STEPS);

    for _ in 0..STEPS {
        let mut x = vec![between.sample(&mut rng)];
        let mut evaluator = MatrixRecurrentFabricator::fabricate(net).expect("didnt work");

        for _ in 0..STEPS {
            let mut input_values = x.clone();
            input_values.push(X_SQUARE(x[0]).clone());
            x[0] = x[0] + evaluator.evaluate(input_values)[0];
        }

        x_values.push(x[0]);
        fitness_values.push(X_SQUARE(x[0]));
    }

    (
        x_values.iter().sum::<f64>() / STEPS as f64,
        fitness_values.iter().sum::<f64>() / STEPS as f64,
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
