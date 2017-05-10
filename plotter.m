source multilayer_perceptron.m
for i = 1:length(x)
        for j = 1:length(y)
            z(i,j) = get_output([x(i);y(j)],weights,net,f); 
        end
end

plot3(terrain(:,1),terrain(:,2),terrain(:,3),'.','markersize',12)
hold on;
surf(x,y,z);
hold off;
