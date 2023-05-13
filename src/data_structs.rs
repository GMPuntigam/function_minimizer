use set_genome::{Genome};

#[derive(Debug)]
pub struct Prediction{
    pub fitness: f32,
    pub x_guess: f32,
    pub x_minus: f32,
    pub x_plus: f32
}
#[derive(Debug, Copy, Clone)]
pub struct FitnessEval{
    pub fitness: f32,
    pub x_worst: f32,
    pub x_minus: f32,
    pub x_plus: f32
}

#[derive(Debug, Clone)]
pub struct Networkpair {
    pub global_optimizer :Genome,
    pub local_optimizer :Genome
}


#[derive(Debug, Clone)]
pub struct AllXVals {
    pub x_max: f32, 
    pub x_min: f32, 
    pub x_minus: f32, 
    pub x_plus: f32, 
    pub x_guess: f32

}
#[derive(Debug)]
pub struct TestfunctionEval{
    pub fitness_eval: FitnessEval,
    pub function_name: String,
}

#[derive(Debug)]
pub struct OverallFitness{
    pub fitness: f32,
    pub fitnessvec: Vec<TestfunctionEval>
}