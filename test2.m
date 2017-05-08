1;
source multilayer_perceptron.m
terrain = dlmread('terrain/terrain03.data');
[f,fder] = activation_tanh(.5);
entries = binary_entry_generator(3);
expected_output = ones(1,8);
expected_output(1) = -1;
expected_output(end) = -1;
[weights,output,mse] = multilayer_perceptron_learn(entries,expected_output,[3,4,1],f,fder,.05,1e3);
