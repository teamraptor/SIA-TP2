function entry = binary_entry_generator(n)
    for i = [0:n-1]
        x = 2^i; 
        % 2x*r = 2^n
        % r = 2^n/2x
        % x = 2^i
        % r = 2^(n-1-i)
        entry(i+1,:) = repmat([ones(1,x),-1*ones(1,x)],1,2^(n-i-1));
    end
end
