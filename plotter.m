source multilayer_perceptron.m
x = [-5:0.01:5];
y = x;
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
