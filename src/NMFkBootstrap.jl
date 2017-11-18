import Distributions

"""
Randomize matrix by assigning a random vector to each column based on a multinomial distribution (with the original column vector as the mean)
"""
function bootstrapping(X::Array{Float64}, scaling::Number=1.0, epsilon::Number=sqrt(eps()))
	N = deepcopy(X)
	bootstrapping!(N, scaling, epsilon)
	return N
end

function bootstrapping!(X::Array{Float64}, scaling::Number=1.0, epsilon::Number=sqrt(eps()))
	for i in 1:size(X, 2)
		v = convert(Array{Int64}, round.(X[:, i] .* scaling))
		n = sum(v)
		p = v ./ n
		v = Distributions.Multinomial(n, p)
		X[:, i] = float(max.(rand(v) / scaling, epsilon))
	end
end

function bootstrapping(X::Array{Int64})
	N = deepcopy(X)
	bootstrapping!(N)
	return N
end

function bootstrapping!(X::Array{Int64})
	for i in 1:size(X, 2)
		n = sum(X[:, i])
		p = X[:, i] ./ n
		v = Distributions.Multinomial(n, p)
		X[:, i] = rand(v)
	end
end
