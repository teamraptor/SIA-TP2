1;
source multilayer_perceptron.m
terrain = dlmread('terrain/terrain03.data');
[f,fder] = activation_tanh(.5);
[weights,output,mse] = multilayer_perceptron_learn(terrain(:,1:2)',terrain(:,3)',[2,20,1],f,fder,.05,1e5);
x = [-2:0.01:2]
y = x;

for i = 1 length(x)
        for j = 1:length(y)
                out = get_output([x(i);y(j)],weights,[2,20,1],f);
        end
end

surf(x,y,out);
