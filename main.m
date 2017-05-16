1;
source multilayer_perceptron.m
source configuration.m

terrain = dlmread(data_file);
terrain = terrain(starting_line:end,:);

if activation == 'exp'
  [f,fder] = activation_exp(beta);
else 
  [f,fder] = activation_tanh(beta);
end

[weights, output, mse] = multilayer_perceptron_learn(terrain(:,1:2)', terrain(:,3)', net, f, fder, eta, max_iterations, cut_error, alpha, adaptative_eta);

if print_error
  figure(1);
  semilogy(mse, 'LineWidth', 2);
  title("ECM");
  xlabel("Épocas");
  ylabel("ECM en escala logarítmica");
end

if print_estimation
  figure(2);
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
