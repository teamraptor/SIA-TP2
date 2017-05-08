1;
source multilayer_perceptron.m
terrain = dlmread('terrain/terrain03.data');
[f,fder] = activation_exp(1);
net = [2,45,50,1];
[weights,output,mse] = multilayer_perceptron_learn(terrain(:,1:2)',terrain(:,3)',net ,f,fder,1,1e4,5e-4);
x = [-3:0.01:3];
y = x;
%dlmwrite('weights.csv',cell2mat(weights));

for i = 1:length(x)
        for j = 1:length(y)
                out(i,j) = get_output([x(i);y(j)],weights,net,f);
        end
end

plot3(terrain(:,1),terrain(:,2),terrain(:,3),'.')
hold on;
surf(x,y,out);
hold off;
