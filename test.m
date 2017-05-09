1;
source multilayer_perceptron.m
terrain = dlmread('terrain/terrain03.data');
[f,fder] = activation_tanh(1);
net = [2,200,50,10,1];
[weights,output,mse] = multilayer_perceptron_learn(terrain(:,1:2)',terrain(:,3)',net ,f,fder,.05,1e4,2e-4);
x = [-3:0.01:3];
y = x;
%dlmwrite('weights.csv',cell2mat(weights));

for i = 1:length(x)
        for j = 1:length(y)
            z(i,j) = get_output([x(i);y(j)],weights,net,f); 
        end
end

plot3(terrain(:,1),terrain(:,2),terrain(:,3),'.','markersize',12)
hold on;
surf(x,y,z);
hold off;
