1;
source multilayer_perceptron.m
terrain = dlmread('terrain/terrain03.data');
terrain = terrain(2:end,:);
[f,fder] = activation_exp(15);
net = [2,45,50,1];
[weights,output,mse] = multilayer_perceptron_learn(terrain(:,1:2)',terrain(:,3)',net ,f,fder,.05,1e4,1e-3);
x = [-3:0.01:3];
y = x;
%dlmwrite('weights.csv',cell2mat(weights));

for i = 1:length(x)
        for j = 1:length(y)
            z(i,j) = get_output([x(i);y(j)],weights,net,f); 
        end
end

plot3(terrain(:,2),terrain(:,1),terrain(:,3),'.','markersize',12)
hold on;
mymap = [0,0,0.6;0.2,0.8,0;0.4,0.8,0;0.5,0.8,0;0.7,0.7,0;0.8,0.8,0.8;1,1,1];
colormap(get_map(mymap));
surf(x,y,z);
hold off;
