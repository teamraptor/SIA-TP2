1;
% entry n * 1
% weights 1 * n
function output = get_output(entry,weights,activation_func)
    output = activation_func(weights * entry);
end

% entry n * 1
% weights 1 * n
function weights = update_weights(learning_factor, output, expected_output, weights, entry)
    weights = weights + learning_factor*(expected_output - output) * entry';
end

function [weights, output] = simple_perceptron_learn(entries, expected_output, activation_func=@sign, learning_factor=.5,
                                                            max_iterations=1000, tolerance=1e-5)
    n = length(entries(:,1));
    % agrego umbral
    entries = [-1*ones(1,2^n);entries];
    weights = rand(1,n+1) .- 0.5;
    
    for iteration = 1:max_iterations
        for index = randperm(2^n);
                entry = entries(:,index);
                output(index) = get_output(entry,weights,activation_func);
                weights = update_weights(learning_factor, output(index), expected_output(index), weights, entry);
        end
        if (sum(abs(expected_output-output)) <= tolerance)
                'solution found'
                return;
        end
    end
    'max iterations reached'
end

function x = id(x)
end

