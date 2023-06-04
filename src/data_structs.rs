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
    pub radius: f32,
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
    pub x_guess: Vec<f32>

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