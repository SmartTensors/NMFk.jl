import Distributions

"""
Randomize matrix by assigning a random vector to each column based on a multinomial distribution (with the original column vector as the mean)
"""
function bootstrapping(X::Array, epsilon::Number=sqrt(eps()))
	N = deepcopy(X)
	bootstrapping!(N, epsilon)
	return N
end

function bootstrapping!(X::Array, epsilon::Number=sqrt(eps()))
	for i in 1:size(X, 2)
		n = sum(X[:, i])
		p = X[:, i] ./ n
		v = Distributions.Multinomial(convert(Int64, round(n)), p)
		X[:, i] = float(max.(rand(v), epsilon))
	end
end
