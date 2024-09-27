use alexandria_linalg::dot::dot;
use core::{felt252_div};

type Vector = Array<felt252>;
type Matrix = Array<Vector>;

fn felt_power(mut base: felt252, mut exp: u128) -> felt252 {
    let mut result = 1;

    while exp > 0 {
        if exp % 2 == 1 {
            result *= base;
        }
        base *= base;
        exp /= 2;
    };

    result
}

fn random_matrix(rows: usize, cols: usize) -> Matrix {
    let mut matrix: Matrix = ArrayTrait::new();

    // Generate numbers using a simple deterministic function
    for i in 0
        ..rows {
            let mut row = ArrayTrait::new();
            for j in 0
                ..cols {
                    let value = ((i * cols + j) * 7 + 3) % 11; // Values between 0 and 10
                    row.append(value.into());
                };
            matrix.append(row);
        };

    matrix
}

fn integer_softmax(x: @Vector) -> Vector {
    let scaling_factor = felt_power(10, 12);
    let mut exp_values: Vector = ArrayTrait::new();
    let mut sum: felt252 = 0;

    for i in 0
        ..x
            .len() {
                let val = *x.at(i);
                let exp_value: felt252 = felt_power(val, 5);
                exp_values.append(exp_value);
                sum += exp_value;
            };

    if sum == 0 {
        // Return a vector filled with zeros, same length as input
        let mut zero_output: Vector = ArrayTrait::new();
        for _ in 0..x.len() {
            zero_output.append(0);
        };
        return zero_output; // Return the vector of zeros
    }

    let mut softmax_output: Vector = ArrayTrait::new();
    let non_zero_sum: NonZero<felt252> = sum.try_into().unwrap();
    let inverse_sum = felt252_div(1, non_zero_sum);
    for i in 0
        ..exp_values
            .len() {
                let exp_value = exp_values.at(i);
                let normalized_value = *exp_value * scaling_factor * inverse_sum;
                softmax_output.append(normalized_value);
            };
    softmax_output
}

fn set_vec(vec: Vector, index: usize, value: felt252) -> Vector {
    let mut new_vec: Vector = ArrayTrait::new();

    for i in 0
        ..vec
            .len() {
                if i == index {
                    new_vec.append(value);
                } else {
                    new_vec.append(*vec.at(i));
                }
            };

    new_vec
}

// Basic self-attention mechanism (single-head)
fn self_attention(inputs: Matrix) -> Matrix {
    let sequence_length = inputs.len();
    let embed_size = inputs[0].len();

    let query_matrix = random_matrix(embed_size, embed_size);
    let key_matrix = random_matrix(embed_size, embed_size);

    let mut queries: Matrix = ArrayTrait::new();
    let mut keys: Matrix = ArrayTrait::new();
    let values = inputs.clone();

    // Iterate over the inputs using indexing
    for i in 0
        ..sequence_length {
            let input_vec = values[i];
            let mut query: Vector = ArrayTrait::new();
            let mut key: Vector = ArrayTrait::new();

            for j in 0
                ..embed_size {
                    let query_value = dot(input_vec.span(), query_matrix[j].span());
                    query.append(query_value);

                    let key_value = dot(input_vec.span(), key_matrix[j].span());
                    key.append(key_value);
                };

            queries.append(query);
            keys.append(key);
        };

    // Calculate attention scores
    let mut attention_scores: Matrix = ArrayTrait::new();
    for i in 0
        ..sequence_length {
            let mut scores_row: Vector = ArrayTrait::new();
            for j in 0
                ..sequence_length {
                    let score = dot(queries[i].span(), keys[j].span());
                    scores_row.append(score);
                };
            attention_scores.append(scores_row);
        };

    // Apply softmax to get attention weights
    let mut attention_weights: Matrix = ArrayTrait::new();

    for i in 0
        ..attention_scores
            .len() {
                let weights = integer_softmax(attention_scores.at(i));
                attention_weights.append(weights);
            };

    // Compute weighted sum of values
    let mut attended_output: Matrix = ArrayTrait::new();

    for i in 0
        ..sequence_length {
            let mut weighted_sum: Vector = ArrayTrait::new();
            for _ in 0..embed_size {
                weighted_sum.append(0);
            };

            for j in 0
                ..sequence_length {
                    for k in 0
                        ..embed_size {
                            let attention_weights_row = attention_weights[i];
                            let values_row = values[j];
                            let product = *attention_weights_row.at(j) * *values_row.at(k);
                            let curr_sum = *weighted_sum.at(k) + product;
                            weighted_sum = set_vec(weighted_sum, k, curr_sum);
                        };
                };
            attended_output.append(weighted_sum);
        };

    attended_output
}

// Simple feed-forward network (linear transformation)
fn feed_forward(inputs: Matrix) -> Matrix {
    let sequence_length = inputs.len();
    let embed_size = inputs[0].len();

    let weight_matrix = random_matrix(embed_size, embed_size);

    let mut outputs = ArrayTrait::new();

    for i in 0
        ..sequence_length {
            let input_vec = inputs.at(i);
            let mut output_vec: Vector = ArrayTrait::new();

            // compute the dot product and append the result to `output_vec`
            for j in 0
                ..embed_size {
                    let output_value = dot(input_vec.span(), weight_matrix[j].span());
                    output_vec.append(output_value);
                };

            outputs.append(output_vec);
        };
    outputs
}

// Basic transformer encoder layer
fn transformer_encoder(inputs: Matrix) -> Matrix {
    // Self-attention mechanism
    let attention_output = self_attention(inputs);

    // Simple feed-forward layer
    let output = feed_forward(attention_output);

    output
}

// Simple embedding function (returns "embedded" tokens)
fn embed_tokens(tokens: Array<usize>, vocab_size: usize, embed_size: usize) -> Matrix {
    // Initialize embeddings
    let embeddings = random_matrix(vocab_size, embed_size);

    let mut embedded_tokens: Matrix = ArrayTrait::new();

    for i in 0
        ..tokens
            .len() {
                let token_embeddings = embeddings.at(*tokens.at(i));
                let mut dereferenced_embeddings: Vector = ArrayTrait::new();

                for i in 0
                    ..token_embeddings
                        .len() {
                            dereferenced_embeddings.append(*token_embeddings.at(i));
                        };

                embedded_tokens.append(dereferenced_embeddings);
            };

    embedded_tokens
}

fn zero_matrix(rows: usize, cols: usize) -> Matrix {
    let mut matrix: Matrix = ArrayTrait::new();
    for _ in 0
        ..rows {
            let mut row = ArrayTrait::new();
            for _ in 0..cols {
                row.append(0.into());
            };
            matrix.append(row);
        };
    matrix
}

fn main() {
    let vocab_size = 1_usize;
    let embed_size = 1_usize;
    let sequence_length = 1_usize;

    // Predefined sequence of token indices
    let tokens = array![0_usize];

    let embedded_input = embed_tokens(tokens, vocab_size, embed_size);

    // Check if the output has the expected dimensions
    assert!(
        embedded_input.len() == sequence_length,
        "Expected embedded_input length to be {}, but got {}",
        sequence_length,
        embedded_input.len()
    );

    assert!(
        embedded_input[0].len() == embed_size,
        "Expected embedded_input[0] length to be {}, but got {}",
        embed_size,
        embedded_input[0].len()
    );

    let output = transformer_encoder(embedded_input);

    assert!(
        output.len() == sequence_length,
        "Expected output length to be {}, but got {}",
        sequence_length,
        output.len()
    );

    assert!(
        output[0].len() == embed_size,
        "Expected output[0] length to be {}, but got {}",
        embed_size,
        output[0].len()
    );
}
