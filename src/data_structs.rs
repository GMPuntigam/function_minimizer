use set_genome::{Genome};

#[derive(Debug)]
pub struct Prediction{
    pub fitness: f64,
    pub x_guess: Vec<f64>,
    pub x_minus: Vec<f64>,
    pub x_plus: Vec<f64>
}
#[derive(Debug)]
pub struct PredictionCircle{
    pub fitness: f64,
    pub x_guess: Vec<f64>,
    pub radius: f64
}

#[derive(Debug, Clone)]
pub struct FitnessEval{
    pub fitness: f64,
    pub x_guess: Vec<f64>,
    pub x_minus: Vec<f64>,
    pub x_plus: Vec<f64>
}

#[derive(Debug, Clone)]
pub struct FitnessEvalCircle{
    pub fitness: f64,
    pub x_guess: Vec<f64>,
    pub radius: f64
}

#[derive(Debug, Clone)]
pub struct Networkpair {
    pub global_optimizer :Genome,
    pub local_optimizer :Genome
}


#[derive(Debug, Clone)]
pub struct AllXVals {
    pub x_max: Vec<f64>, 
    pub x_min: Vec<f64>, 
    pub x_minus: Vec<f64>, 
    pub x_plus: Vec<f64>, 
    pub x_guess: Vec<f64>

}


#[derive(Debug, Clone)]
pub struct AllXValsCircle {
    pub radius: f64,
    pub x_guess: Vec<f64>,
    pub velocity: Vec<f64>, 
    pub f_val: f64,
    pub delta_fitness: f64,
    pub fitness_change_limited: f64,
    pub last_fitness: f64,
    pub currentbest: bool

}

#[derive(Debug, Clone)]
pub struct TestfunctionEval{
    pub fitness_eval: f64,
    pub function_name: String,
}

#[derive(Debug)]
pub struct OverallFitness{
    pub fitness: f64,
    pub fitnessvec: Vec<TestfunctionEval>
}

#[derive(Debug)]
pub struct WallTimeEval{
    pub f_min: f64,
    pub f_average: f64,
    pub f_worst: f64,
    pub x_best: Vec<f64>,
    pub x_worst: Vec<f64>,
    pub walltime: f64,
    pub average_steps: f64,
}

#[derive(Debug, Clone)]
pub struct FProgress{
    pub f_min: f64,
    pub n_evals: usize,
    pub min_point: Vec<f64>
}

#[derive(Debug)]
pub struct Evaluation {
    pub fval: f64,
    pub x_min: Vec<f64>,
    pub steps: usize,
    pub duration: f64,
    pub radius: f64
}