1;
function res = mac(func, matrix)
   matrix= (matrix.+1)/.2;
   res = matrix(1,:);		
   for i = 2:length(matrix(:,1))
	res = func(res, matrix(i,:));
   end
   res = (res * 2)-1;
end

function res = mac_xor(matrix)
  mac_or = (mac(@or,matrix)+1)/2;
  mac_not_and = -1*(((mac(@and,matrix)+1)/2)-1);
  res = (and(mac_or,mac_not_and)*2)-1;
end
