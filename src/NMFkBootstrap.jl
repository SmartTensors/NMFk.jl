import Distributions

"""
Randomize matrix by assigning a random vector to each column based on a multinomial distribution (with the original column vector as the mean)
"""
function bootstrapping(X::Array, scale::Number=0.1, epsilon::Number=sqrt(eps()))
	N = deepcopy(X)
	bootstrapping!(N, scale, epsilon)
	return N
end

function bootstrapping!(X::Array, scale::Number=0.1, epsilon::Number=sqrt(eps()))
	for i in 1:size(X, 2)
		r = round.(Int64, X[:, i])
		n = sum(r)
		p = r ./ n
		v = Distributions.Multinomial(n, p)
		X[:, i] = float(max.(rand(v) .* scale, epsilon))
	end
end
