use std::f32::consts::PI;

use rand::{Rng, distributions::Uniform};

#[derive(Debug)]
pub struct FunctionDomain {
    pub dimensions: usize,
    pub samplepoints: usize,
    pub upper_limits: Vec<f32>,
    pub lower_limits: Vec<f32>
}

#[derive(Debug)]
pub struct Evaluation {
    pub fval: f32,
    pub x_min: Vec<f32>,
    pub steps: usize,
    pub duration: f32,
    
}

#[derive(Debug)]
pub struct CircleGeneratorInput {
    pub dimensions: usize,
    pub samplepoints: usize,
    pub radius: f32,
    pub center: Vec<f32>
}

#[derive(Debug, Clone)]
pub struct SetOfSamples {
    pub dimensions: usize,
    pub n_samplepoints: usize,
    pub max: Vec<f32>,
    pub min: Vec<f32>,
    pub coordinates: Vec<Vec<f32>>,
    pub coordinates_normalised: Vec<Vec<f32>>,
    pub coordinates_by_dimension: Vec<Vec<f32>>,
    pub coordinates_by_dimension_normalised: Vec<Vec<f32>>

}

#[derive(Debug, Clone)]
pub struct SetOfSamplesCircle {
    pub dimensions: usize,
    pub n_samplepoints: usize,
    pub coordinates: Vec<Vec<f32>>,
    pub coordinates_normalised: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SetOfSamplesVector {
    pub coordinates: Vec<Vec<f32>>,
    pub coordinates_normalised: Vec<f32>,
}

// pub fn generate_2d_normed_vec_with_rotation(function_domain: FunctionDomain, phi: f32) -> SetOfSamples {
//     let between: Uniform<f32> = Uniform::from(0.0..1.0);
//     let mut mins: Vec<f32> = vec![0.0; function_domain.dimensions];
//     let mut maxs: Vec<f32> = vec![0.0; function_domain.dimensions];
//     let mut tensor= Vec::with_capacity(function_domain.samplepoints);
//     let mut tensor_normalised= Vec::with_capacity(function_domain.samplepoints);
//     // tensor_normalised.push(vec![0.0; function_domain.dimensions]);
//     let mut random_values:Vec<f32> = rand::thread_rng().sample_iter(&between).take(function_domain.samplepoints-2).collect();
//     random_values.sort_unstable_by(|a, b| a.partial_cmp(&b).unwrap());
//     let dim_values: Vec<f32> = [vec![0.0], random_values, vec![1.0]].concat();
//     // let mut rng = rand::thread_rng();
//     for sample_iter in 0..function_domain.samplepoints {
//         let mut point = Vec::with_capacity(function_domain.dimensions);
//         for dim in 0..function_domain.dimensions{
//             if dim == 0{
//                 point.push(dim_values[sample_iter]*phi.cos())
//             }else if dim == 1 {
//                 point.push(dim_values[sample_iter]*phi.sin())
//             }
//         }
//         tensor_normalised.push(point);
//     }

//     let mut coordinates_by_dimension: Vec<Vec<f32>> = vec![Vec::with_capacity(function_domain.samplepoints); function_domain.dimensions];
//     let mut coordinates_by_dimension_normalised: Vec<Vec<f32>> = vec![Vec::with_capacity(function_domain.samplepoints); function_domain.dimensions];

//     for i in 0..function_domain.samplepoints {
//         let mut point = tensor_normalised[i].clone();
//         for dim in 0..function_domain.dimensions{
//             coordinates_by_dimension_normalised[dim].push(point[dim].clone());
//             point[dim] = point[dim]*(function_domain.upper_limits[dim])+ (1.0-point[dim])*function_domain.lower_limits[dim];
//             coordinates_by_dimension[dim].push(point[dim].clone());
//             if i == 0 {
//                 mins[dim] = point[dim].clone();
//                 maxs[dim] = point[dim].clone();
//             } else {
//                 mins[dim] = mins[dim].min(point[dim]);
//                 maxs[dim] = maxs[dim].max(point[dim]);
//             }
//         }
        
//         tensor.push(point);
//     }

//     SetOfSamples {
//         dimensions: function_domain.dimensions,
//         n_samplepoints: function_domain.samplepoints,
//         max: maxs,
//         min: mins,
//         coordinates: tensor,
//         coordinates_normalised: tensor_normalised,
//         coordinates_by_dimension: coordinates_by_dimension,
//         coordinates_by_dimension_normalised: coordinates_by_dimension_normalised
//     }
// }

// fn rotate_point_by_vector(mut point: Vec<f32>, phi_rot:f32) -> Vec<f32> {
//     point = vec![point[0]*phi_rot.cos() - point[1]*phi_rot.sin(), point[0]*phi_rot.sin() + point[1]*phi_rot.cos()];
//     point
// }

pub fn generate_samplepoint_vector(vector_start: &Vec<f32>, vector_end: &Vec<f32>, n_samples: usize) -> SetOfSamplesVector {
    let between: Uniform<f32> = Uniform::from(0.0..1.0);
    let mut tensor: Vec<Vec<f32>>= Vec::with_capacity(n_samples);
    let mut random_values:Vec<f32> = rand::thread_rng().sample_iter(&between).take(n_samples-2).collect();
    random_values.sort_unstable_by(|a, b| a.partial_cmp(&b).unwrap());
    let dim_values: Vec<f32> = [vec![0.0], random_values, vec![1.0]].concat();
    for sample_iter in 0..n_samples {
        let mut point = Vec::with_capacity(2);
        for dim in 0..2{
            point.push(dim_values[sample_iter].clone()*(vector_end[dim]-vector_start[dim])+vector_start[dim])
        }
        tensor.push(point);
    }

    SetOfSamplesVector {
        coordinates: tensor,
        coordinates_normalised: dim_values 
    }
}


pub fn generate_samplepoint_2d_random_slice(circle_generator_input: &CircleGeneratorInput) -> SetOfSamplesCircle {
    let between: Uniform<f32> = Uniform::from(0.0..1.0);
    // let mut mins: Vec<f32> = vec![0.0; 2];
    // let mut maxs: Vec<f32> = vec![0.0; 2];
    let mut tensor= Vec::with_capacity(circle_generator_input.samplepoints);
    let mut tensor_normalised= Vec::with_capacity(circle_generator_input.samplepoints);
    // tensor_normalised.push(vec![0.0; function_domain.dimensions]);
    let mut random_values_angle:Vec<f32> = rand::thread_rng().sample_iter(&between).take(circle_generator_input.samplepoints).collect();
    let random_values_radius:Vec<f32> = vec![1.0; circle_generator_input.samplepoints];
    // let mut random_values_radius:Vec<f32> = rand::thread_rng().sample_iter(&between).take(circle_generator_input.samplepoints).collect();
    random_values_angle.sort_unstable_by(|a, b| a.partial_cmp(&b).unwrap());
    // let dim_values: Vec<f32> = [vec![0.0], random_values, vec![1.0]].concat();
    for sample_iter in 0..circle_generator_input.samplepoints {
        let mut point = Vec::with_capacity(2);
        for dim in 0..2{
            if dim == 0{
                point.push((random_values_radius[sample_iter])*(random_values_angle[sample_iter]*2.0*PI).cos())
            }else if dim == 1 {
                point.push((random_values_radius[sample_iter])*(random_values_angle[sample_iter]*2.0*PI).sin())
            }
        }
        tensor_normalised.push(point);
    }

    let mut coordinates_by_dimension: Vec<Vec<f32>> = vec![Vec::with_capacity(circle_generator_input.samplepoints); circle_generator_input.dimensions];
    let mut coordinates_by_dimension_normalised: Vec<Vec<f32>> = vec![Vec::with_capacity(circle_generator_input.samplepoints); circle_generator_input.dimensions];

    for i in 0..circle_generator_input.samplepoints {
        let mut point = tensor_normalised[i].clone();
        for dim in 0..circle_generator_input.dimensions{
            coordinates_by_dimension_normalised[dim].push(point[dim].clone());
            point[dim] = point[dim]*circle_generator_input.radius + circle_generator_input.center[dim];
            coordinates_by_dimension[dim].push(point[dim].clone());
        }
        tensor.push(point);
    }

    SetOfSamplesCircle {
        dimensions: circle_generator_input.dimensions,
        n_samplepoints: circle_generator_input.samplepoints,
        coordinates: tensor,
        coordinates_normalised: random_values_angle,
    }
}

pub fn generate_1d_samplepoints(function_domain: &FunctionDomain, currentdim: usize) -> SetOfSamples {
    let between: Uniform<f32> = Uniform::from(0.0..1.0);
    let mut mins: Vec<f32> = vec![0.0; function_domain.dimensions];
    let mut maxs: Vec<f32> = vec![0.0; function_domain.dimensions];
    let mut tensor= Vec::with_capacity(function_domain.samplepoints);
    let mut tensor_normalised= Vec::with_capacity(function_domain.samplepoints);
    // tensor_normalised.push(vec![0.0; function_domain.dimensions]);
    let mut random_values:Vec<f32> = rand::thread_rng().sample_iter(&between).take(function_domain.samplepoints-2).collect();
    random_values.sort_unstable_by(|a, b| a.partial_cmp(&b).unwrap());
    let dim_values: Vec<f32> = [vec![0.0], random_values, vec![1.0]].concat();
    // let mut rng = rand::thread_rng();
    for sample_iter in 0..function_domain.samplepoints {
        let mut point = Vec::with_capacity(function_domain.dimensions);
        for dim in 0..function_domain.dimensions{
            if dim == currentdim{
                point.push(dim_values[sample_iter])
            }else {
                point.push(0.5);
            }
        }
        tensor_normalised.push(point);
    }

    let mut coordinates_by_dimension: Vec<Vec<f32>> = vec![Vec::with_capacity(function_domain.samplepoints); function_domain.dimensions];
    let mut coordinates_by_dimension_normalised: Vec<Vec<f32>> = vec![Vec::with_capacity(function_domain.samplepoints); function_domain.dimensions];

    for i in 0..function_domain.samplepoints {
        let mut point = tensor_normalised[i].clone();
        for dim in 0..function_domain.dimensions{
            coordinates_by_dimension_normalised[dim].push(point[dim].clone());
            point[dim] = point[dim]*(function_domain.upper_limits[dim])+ (1.0-point[dim])*function_domain.lower_limits[dim];
            coordinates_by_dimension[dim].push(point[dim].clone());
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
        n_samplepoints: function_domain.samplepoints,
        max: maxs,
        min: mins,
        coordinates: tensor,
        coordinates_normalised: tensor_normalised,
        coordinates_by_dimension: coordinates_by_dimension,
        coordinates_by_dimension_normalised: coordinates_by_dimension_normalised
    }
}

pub fn generate_sampleheader(samplepoints: usize, dimensions: usize)-> String {
    // let total_samplepoints= samplepoints.pow((dimensions).try_into().unwrap());
    let total_samplepoints= samplepoints;
    let mut helpvec_x = Vec::with_capacity(total_samplepoints);
    let mut helpvec_fx = Vec::with_capacity(total_samplepoints);
    let mut dimensioncounter = vec![0; dimensions];
    let mut combination_vec : Vec<usize>= vec![0; dimensions];
    for _ in 0..total_samplepoints {
        for d in 0..dimensions {
            combination_vec[d] = dimensioncounter[d];
        }
        let combination_string = combination_vec.iter().map( |&id| id.to_string()).collect::<Vec<String>>().join(",");
        helpvec_x.push(format!("x{i}", i =combination_string.clone() ));
        helpvec_fx.push(format!("f(x{i})", i =combination_string.clone()));

        dimensioncounter[dimensions-1] = dimensioncounter[dimensions-1] +1;
        for d in 0..dimensions {
            if dimensioncounter[dimensions-1 -d] == samplepoints{
                dimensioncounter[dimensions-1 -d] = 0;
                if dimensions as i32 -1 -d as i32 -1 >=0 {
                    dimensioncounter[dimensions-1 -d -1] = dimensioncounter[dimensions-1 -d -1] + 1;
                }
                
            }
        }
        

        
    }
    let joinedx = helpvec_x.join(";").to_owned();
    let joinedf = helpvec_fx.join(";").to_owned();
    let joinvec = vec![joinedx, joinedf];
    let joined = joinvec.join(";").to_owned();
    joined
}