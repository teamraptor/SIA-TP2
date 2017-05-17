1;
source multilayer_perceptron.m
source configuration.m

terrain = dlmread(data_file);
terrain = terrain(starting_line:end, :);

if activation == 1
  [f,fder] = activation_exp(beta);
else 
  [f,fder] = activation_tanh(beta);
end

[weights, output, mse, test_mse] = multilayer_perceptron_learn(terrain(:,1:2)', terrain(:,3)', train_percentage, net, f, fder, eta, max_iterations, cut_error, alpha, adaptative_eta);

if print_error
  figure(1);
  semilogy(mse, 'LineWidth', 2);
  title("ECM");
  xlabel("Épocas");
  ylabel("ECM en escala logarítmica");
  if length(test_mse) > 0
    hold on;
    semilogy(test_mse, 'LineWidth', 2);
    total_error = mse * train_percentage + test_mse * (1-train_percentage);
    semilogy(total_error, 'LineWidth', 2);
    legend('ECM Entrenamiento','ECM Testeo', 'ECM Total');
    hold off;
  end
end

if print_estimation
  figure(3);
  x = [-3:0.025:3];
  y = x;
  for i = 1:length(x)
          for j = 1:length(y)
              z(i,j) = get_output([x(i);y(j)],weights,net,f); 
          end
  end
  surf(x, y, z);
  mymap = [0, 0, 0.6; 0.2, 0.8, 0; 0.4, 0.8, 0; 0.5, 0.8, 0; 0.7, 0.7, 0; 0.8, 0.8, 0.8; 1, 1, 1];
  colormap(get_map(mymap));
  title("Aproximación de la función");
end
