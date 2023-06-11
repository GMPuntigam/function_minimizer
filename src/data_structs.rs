use set_genome::{Genome};

#[derive(Debug)]
pub struct Prediction{
    pub fitness: f32,
    pub x_guess: Vec<f32>,
    pub x_minus: Vec<f32>,
    pub x_plus: Vec<f32>
}
#[derive(Debug)]
pub struct PredictionCircle{
    pub fitness: f32,
    pub x_guess: Vec<f32>,
    pub radius: f32
}

#[derive(Debug, Clone)]
pub struct FitnessEval{
    pub fitness: f32,
    pub x_guess: Vec<f32>,
    pub x_minus: Vec<f32>,
    pub x_plus: Vec<f32>
}

#[derive(Debug, Clone)]
pub struct FitnessEvalCircle{
    pub fitness: f32,
    pub x_guess: Vec<f32>,
    pub radius: f32
}

#[derive(Debug, Clone)]
pub struct Networkpair {
    pub global_optimizer :Genome,
    pub local_optimizer :Genome
}


#[derive(Debug, Clone)]
pub struct AllXVals {
    pub x_max: Vec<f32>, 
    pub x_min: Vec<f32>, 
    pub x_minus: Vec<f32>, 
    pub x_plus: Vec<f32>, 
    pub x_guess: Vec<f32>

}


#[derive(Debug, Clone)]
pub struct AllXValsCircle {
    pub radius: f32,
    pub x_guess: Vec<f32>,
    pub velocity: Vec<f32>, 
    pub f_val: f32,
    pub delta_fitness: f32,
    pub fitness_change_limited: f32,
    pub last_fitness: f32,
    pub currentbest: bool

}

#[derive(Debug, Clone)]
pub struct TestfunctionEval{
    pub fitness_eval: f32,
    pub function_name: String,
}

#[derive(Debug)]
pub struct OverallFitness{
    pub fitness: f32,
    pub fitnessvec: Vec<TestfunctionEval>
}

#[derive(Debug)]
pub struct WallTimeEval{
    pub f_min: f32,
    pub f_average: f32,
    pub f_worst: f32,
    pub x_best: Vec<f32>,
    pub x_worst: Vec<f32>,
    pub walltime: f32,
    pub average_steps: f32
}