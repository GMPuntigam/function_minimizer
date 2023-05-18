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
    pub coordinates_by_dimension: Vec<Vec<f32>>,
    pub coordinates_by_dimension_normalised: Vec<Vec<f32>>

}


pub fn generate_1d_samplepoints(function_domain: FunctionDomain, currentdim: usize) -> SetOfSamples {
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

pub fn generate_samplepoint_matrix_single_dimension_view(function_domain: FunctionDomain) -> SetOfSamples {
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

    let mut coordinates_by_dimension: Vec<Vec<f32>> = vec![Vec::with_capacity(total_samplepoints); function_domain.dimensions];
    let mut coordinates_by_dimension_normalised: Vec<Vec<f32>> = vec![Vec::with_capacity(total_samplepoints); function_domain.dimensions];

    for i in 0..total_samplepoints {
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
        n_samplepoints: total_samplepoints,
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
    let mut helpvec = Vec::with_capacity(total_samplepoints);
    let mut dimensioncounter = vec![0; dimensions];
    let mut combination_vec : Vec<usize>= vec![0; dimensions];
    for _ in 0..total_samplepoints {
        for d in 0..dimensions {
            combination_vec[d] = dimensioncounter[d];
        }
        let combination_string = combination_vec.iter().map( |&id| id.to_string()).collect::<Vec<String>>().join(",");
        helpvec.push(format!("x{i}", i =combination_string.clone() ));
        helpvec.push(format!("f(x{i})", i =combination_string.clone()));

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
    let joined = helpvec.join(";").to_owned();
    joined
}