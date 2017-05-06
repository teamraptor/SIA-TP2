function [f,fder] = activation_tanh(b)
   f = @(x) tanh(b*x);
   fder = @(x) b*(1-f(x).^2);
end