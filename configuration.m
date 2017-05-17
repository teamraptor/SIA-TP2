15;

% Parameters
data_file = 'terrain/terrain03.data';
starting_line = 2;
beta = 15;
activation = 1;
net = [2, 45, 50, 1];
eta = 0.1;
max_iterations = 1e2;
cut_error = 1e-4;
train_percentage = 0.9;

% Optimizations
alpha = 0;
adaptative_eta = true;

% Display
print_error = true;
print_estimation = false;