2;
function output = get_output(entries, weights, neurons_per_layer, activation_func)
    m = 0;
    for i = 2:length(neurons_per_layer)
        m = m + 1;
        layer_entry{m} = [-1, zeros(1, neurons_per_layer(i-1))];
    end
    M = m;
    i = 1;
    for entry = entries
        layer_entry{1}(2:end) = entry;
        for m = 2:M
            layer_entry{m}(2:end) = activation_func(weights{m-1} * layer_entry{m-1}');
        end
        output(i) = activation_func(weights{M} * layer_entry{M}');
        i = i + 1;
    end   
end

function [weights,output,error_per_iteration] = multilayer_perceptron_learn(entries, expected_output, neurons_per_layer, activation_func, activation_der,
                                    learning_factor=.5, max_iterations=1000, tolerance=1e-5)
    %number of entries
    n = length(entries(1,:));
    %number of layers
    m = 0;

    %setup
    for i = 2:length(neurons_per_layer)
        m = m + 1;
        weights{m} = rand(neurons_per_layer(i), neurons_per_layer(i-1)+1) .- 0.5;
        layer_entry{m} = [-1, zeros(1, neurons_per_layer(i-1))];
    end
    %last layer
    M = m;

    for iteration = 1:max_iterations
        %select random entry
        for index = randperm(n);
            %get layers output 
            layer_entry{1}(2:end) = entries(:, index);
            for m = 2:M
                layer_entry{m}(2:end) = activation_func(weights{m-1} * layer_entry{m-1}');
            end
            output(index) = activation_func(weights{M} * layer_entry{M}');
            %get errors
            d{M} = activation_der(weights{M}*layer_entry{M}')*(expected_output(index) - output(index));
            for i = M-1:-1:1
                %[r,c] = size(d{i+1});
                %delta = d{i+1};
                %if(r<c)
                %    delta = d{i+1}';
                %end
                %sum(weights{i+1}(:,2:end) .* delta)
                %d{i+1} * weights{i+1}(:,2,end)
                d{i} = (activation_der(weights{i}*layer_entry{i}')' .* (d{i+1} * weights{i+1}(:,2:end)));
            end
            %correct weights
            d;
            for i = 1:M
                delta_w = learning_factor * d{i}' * layer_entry{i};
                weights{i} = weights{i} + delta_w;
            end
        end
        %get iteration error
        error_per_iteration(iteration) = sum((expected_output - output).^2);
        if error_per_iteration(iteration) <= tolerance
            return
        end
    end

end

