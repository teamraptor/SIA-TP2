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
            layer_entry{1}(2:end) = [entry];
        for m = 2:M
                layer_entry{m}(2:end) = activation_func(weights{m-1} * layer_entry{m-1}');
        end
        output(i) = (weights{M} * layer_entry{M}');
        i = i + 1;
    end   
end

function [weights, output, error_per_iteration, test_error] = multilayer_perceptron_learn(entries, expected_output, train_percentage, neurons_per_layer, activation_func, activation_der,
                                    learning_factor=.5, max_iterations=1000, tolerance=1e-5, alpha=0, adaptative_eta=false, dbug=false)
    
    full_entry = entries;
    full_expected = expected_output;
    n = length(entries(1,:));
    
    train_amount = floor(train_percentage * n);
    counter = 1;
    test_error = [];
    
    if train_percentage ~= 1
      for index = randperm(n)
        if counter < train_amount
          train_entry(:, counter) = full_entry(:, index);
          training_expected_output(counter) = full_expected(index);
        else
          test_entry(:, counter - train_amount + 1) = full_entry(:, index);
          test_expected_output(counter - train_amount + 1) = full_expected(index);
        end
        counter = counter + 1;
      end
      entries = train_entry;
      expected_output = training_expected_output;
    end
    
    n = length(entries(1,:));
    eta = learning_factor;
    a = 0;
    
    %number of layers
    m = 0;
    
    %setup
    for i = 2:length(neurons_per_layer)
        m = m + 1;
        %weights{m} = (2*(rand(neurons_per_layer(i), neurons_per_layer(i-1)+1) .- 0.5))./100;
        weights{m} = (rand(neurons_per_layer(i), neurons_per_layer(i-1)+1) .- 0.5)./(neurons_per_layer(i-1));
        layer_entry{m} = [-1, zeros(1, neurons_per_layer(i-1))];
    	  h{m} = [-1, zeros(1, neurons_per_layer(i-1))];
    end
    
    %last layer
    M = m;
  
    for iteration = 1:max_iterations
    tic; 
	  %select random entry
        for index = randperm(n);
            %get layers output 
            layer_entry{1}(2:end) = entries(:, index);
            
            for m = 2:M
		        h{m-1} = weights{m-1} * layer_entry{m-1}';
                layer_entry{m}(2:end) = activation_func(h{m-1});
            end
            
            if dbug 
                layer_entry
                fflush(1);
	          end
            
            h{M} = weights{M} * layer_entry{M}';
            
            %no linear
            %output(index) = activation_func(h{M});
            %get errors
            %d{M} = activation_der(h{M})*(expected_output(index) - output(index));
            
            %linear
            output(index) = h{M};
            %h{M};
            
            %get errors
            d{M} = (expected_output(index) - output(index));
            %d{M};
            
            for i = M-1:-1:1
                d{i} = (activation_der(h{i})' .* (d{i+1} * weights{i+1}(:, 2:end)));
            end
            
            %correct weights
            d;
            old_weigths = weights;
            if iteration > 1 
                prev_delta_w = delta_w;
            end
            for i = 1:M
              if iteration > 1 && a ~= 0
                delta_w{i} = -learning_factor * d{i}' * layer_entry{i} + a * delta_w{i};
              else 
                delta_w{i} = learning_factor * d{i}' * layer_entry{i};
              end
                weights{i} = weights{i} + delta_w{i};
            end
        end
        
        %get iteration error
        error_per_iteration(iteration) = sum((expected_output - output).^2)/n;
        if adaptative_eta
          if iteration > 1 
            if error_per_iteration(iteration) > error_per_iteration(iteration - 1)
              weights = old_weigths;
              error_per_iteration(iteration) = error_per_iteration(iteration - 1);
              delta_w = prev_delta_w;
              learning_factor = 0.9 * learning_factor;
              a = 0;    
            else 
              a = alpha;
              learning_factor = 0.1 * eta + learning_factor;
            end
          else
            a = alpha;
          end
        end
        if train_percentage ~= 1
          test_error(iteration) = sum((test_expected_output - get_output(test_entry, weights, neurons_per_layer, activation_func)).^2)/length(test_entry(1,:));
          [error_per_iteration(iteration), iteration, toc, learning_factor, test_error(iteration)]
          fflush(1);
        else
          [error_per_iteration(iteration), iteration, toc, learning_factor]
          fflush(1);
        end
        
        if error_per_iteration(iteration) <= tolerance
            return
        end
        
        if(learning_factor < 0.001 * eta)
          learning_factor = 0.1 * eta;
        end
        
    end

end