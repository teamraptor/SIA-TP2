function [f,fder] = activation_exp(b)
   f = @(x) 1 ./ (1+exp(-2*b*x));
   fder = @(x) 2*b*exp(2*b*x)/((exp(2*b*x)+1).^2);
end
