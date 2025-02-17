import Distributions
import DocumentFunction

function bootstrapping(X::AbstractMatrix{T}, scaling::Number=1.0, epsilon::Number=sqrt(eps())) where {T <: Number}
	N = deepcopy(X)
	bootstrapping!(N, scaling, epsilon)
	return N
end

function bootstrapping!(X::AbstractMatrix{T}, scaling::Number=1.0, epsilon::Number=sqrt(eps())) where {T <: Number}
	for i in axes(X, 2)
		v = convert(Vector{Int64}, round.(X[:, i] .* scaling))
		n = sum(v)
		p = v ./ n
		v = Distributions.Multinomial(n, p)
		X[:, i] = float(max.(rand(v) / scaling, epsilon))
	end
end

function bootstrapping(X::AbstractMatrix{Int64})
	N = deepcopy(X)
	bootstrapping!(N)
	return N
end

function bootstrapping!(X::AbstractMatrix{Int64})
	for i in axes(X, 2)
		n = sum(X[:, i])
		p = X[:, i] ./ n
		v = Distributions.Multinomial(n, p)
		X[:, i] = rand(v)
	end
end

@doc """
Randomize matrix by assigning a random vector to each column based on a multi-nomial distribution (with the original column vector as the mean)

$(DocumentFunction.documentfunction(bootstrapping))
""" bootstrapping

@doc """
Randomize matrix by assigning a random vector to each column based on a multi-nomial distribution (with the original column vector as the mean)

$(DocumentFunction.documentfunction(bootstrapping!))
""" bootstrapping!
