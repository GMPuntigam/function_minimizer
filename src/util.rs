use rand::{Rng, distributions::Uniform};

#[derive(Debug)]
pub struct FunctionDomain {
    pub dimensions: usize,
    pub samplepoints: usize,
    pub upper_limits: Vec<f32>,
    pub lower_limits: Vec<f32>
}

#[derive(Debug, Clone)]
pub struct SetOfSamples {
    pub dimensions: usize,
    pub n_samplepoints: usize,
    pub max: Vec<f32>,
    pub min: Vec<f32>,
    pub coordinates: Vec<Vec<f32>>,
    pub coordinates_normalised: Vec<Vec<f32>>,

}


pub fn generate_samplepoint_matrix(function_domain: FunctionDomain) -> SetOfSamples {
    let between: Uniform<f32> = Uniform::from(0.0..1.0);
    let mut mins: Vec<f32> = vec![0.0; function_domain.dimensions];
    let mut maxs: Vec<f32> = vec![0.0; function_domain.dimensions];
    let total_samplepoints= function_domain.samplepoints.pow((function_domain.dimensions).try_into().unwrap());
    let mut tensor= Vec::with_capacity(total_samplepoints);
    let mut tensor_normalised= Vec::with_capacity(total_samplepoints);
    tensor_normalised.push(vec![0.0; function_domain.dimensions]);
    for _ in 1..total_samplepoints-1 {
        let point:Vec<f32> = rand::thread_rng().sample_iter(&between).take(function_domain.dimensions).collect();
        tensor_normalised.push(point.clone());
    }
    tensor_normalised.push(vec![1.0; function_domain.dimensions]);
    tensor_normalised.sort_unstable_by(|a: &Vec<f32>, b: &Vec<f32>| a.iter()
                                    .map(|ai| ai.powi(2)).sum::<f32>()
                                    .partial_cmp(&b.iter().map(|bi| bi.powi(2)).sum::<f32>()).unwrap()); 
                             
    for i in 0..total_samplepoints {
        let mut point = tensor_normalised[i].clone();
        for dim in 0..function_domain.dimensions{
            point[dim] = point[dim]*(function_domain.upper_limits[dim]-function_domain.lower_limits[dim])+function_domain.lower_limits[dim];
            if i == 0 {
                mins[dim] = point[dim].clone();
                maxs[dim] = point[dim].clone();
            } else {
                mins[dim] = mins[dim].min(point[dim]);
                maxs[dim] = maxs[dim].max(point[dim]);
            }
        }
        
        tensor.push(point);
    }
    SetOfSamples {
        dimensions: function_domain.dimensions,
        n_samplepoints: total_samplepoints,
        max: maxs,
        min: mins,
        coordinates: tensor,
        coordinates_normalised: tensor_normalised,
    
    }
}