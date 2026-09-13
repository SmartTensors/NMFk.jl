import Distributions

"""
    bootstrapping(X::AbstractMatrix{Int64})
    bootstrapping(X::AbstractMatrix{T}, scaling::Number=1.0,
                  epsilon::Number=sqrt(eps())) where {T<:Number}

Return an independently resampled copy of `X` by drawing each column from a
multinomial distribution whose probabilities are derived from that column.
The one-argument `Int64` method preserves each column sum exactly, while the
general method converts scaled, rounded values to counts before sampling.

# Arguments

- `X`: Nonnegative matrix to resample; every column must have a positive sum
  after any scaling and rounding.
- `scaling=1.0`: Factor applied before values are rounded to integer counts;
  sampled counts are divided by the same factor.
- `epsilon=sqrt(eps())`: Lower bound applied to every value returned by the
  general numeric method.

# Returns

A resampled matrix with the same shape as `X`.
The input matrix is not modified.

# Examples

```julia
import Random

Random.seed!(2026)
X = Int64[4 1; 1 4]
Y = NMFk.bootstrapping(X)
@assert X == Int64[4 1; 1 4]
@assert sum(Y; dims=1) == sum(X; dims=1)
```
"""
function bootstrapping(X::AbstractMatrix{T}, scaling::Number=1.0, epsilon::Number=sqrt(eps())) where {T <: Number}
	N = deepcopy(X)
	bootstrapping!(N, scaling, epsilon)
	return N
end

"""
    bootstrapping!(X::AbstractMatrix{Int64})
    bootstrapping!(X::AbstractMatrix{T}, scaling::Number=1.0,
                   epsilon::Number=sqrt(eps())) where {T<:Number}

Resample every column of `X` in place from a multinomial distribution whose
probabilities are derived from that column.
The one-argument `Int64` method preserves each column sum exactly, while the
general method converts scaled, rounded values to counts before sampling.

# Arguments

- `X`: Nonnegative matrix to overwrite; every column must have a positive sum
  after any scaling and rounding.
- `scaling=1.0`: Factor applied before values are rounded to integer counts;
  sampled counts are divided by the same factor.
- `epsilon=sqrt(eps())`: Lower bound applied to every value written by the
  general numeric method.

# Returns

`nothing`.

# Examples

```julia
import Random

Random.seed!(2026)
X = Int64[4 1; 1 4]
column_totals = sum(X; dims=1)
NMFk.bootstrapping!(X)
@assert sum(X; dims=1) == column_totals
```
"""
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
