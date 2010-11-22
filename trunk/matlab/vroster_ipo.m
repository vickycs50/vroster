function [config score] = vroster_ipo(W)

[id ir] = size(W);
W2 = W;
if id < ir
	W2 = [W2; zeros(ir-id, ir)]; 
end

W2 = [W2 ones(size(W2,1), 1)*mean(mean(W))];

[d r] = size(W2);
A = reshape(W2', 1, d*r);
x = intvar(d*r, 1, 'full');

objective = [];

% Every detected face needs one recognizer
for i=1:d
	obj = zeros(1, d*r);
	obj((i-1)*r+1:i*r) = 1;
	objective = [objective obj*x==1];
end
% A recognizer can be used once or never
for i=1:r-1
	obj = zeros(1, d*r);
	obj(i:r:d*r) = 1;
	objective = [objective obj*x==1];
end

constraint = (A*x);

solver_name = 'bnb';
solvesdp(objective, constraint, sdpsettings('verbose', 1, 'showprogress', 1, 'solver', solver_name));

config = reshape(double(x)', r, d)';
config = config(1:id, :);
score = double(A*x);

end