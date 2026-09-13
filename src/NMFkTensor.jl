"""
	tensorfactorization(X::AbstractArray{T,N}, range::Union{AbstractUnitRange{Int},Integer}, dims::Union{AbstractUnitRange{Int},Integer}=1:N, aw...; casefilename::AbstractString="nmfk-tensor", kw...) where {T<:Number,N}
	tensorfactorization(X::AbstractArray{T,N}, range::AbstractVector, dims::Union{AbstractUnitRange{Int},Integer}=1:N, aw...; casefilename::AbstractString="nmfk-tensor", kw...) where {T<:Number,N}

Factorize matrix unfoldings of a numeric tensor along the requested dimensions.

For each selected dimension `d`, this function calls `NMFk.flatten(X, d)` and passes the resulting matrix to `NMFk.execute`.
Each execution receives a case filename formed by appending `-d<d>` to `casefilename`.

# Arguments

- `X`: Numeric tensor to unfold and factorize.
- `range`: Integer rank or unit range applied to every selected dimension, or a vector containing a rank range for each dimension; an empty per-dimension range skips that dimension.
- `dims=1:ndims(X)`: Dimension index or range of dimensions to analyze.
- `aw...`: Additional positional arguments forwarded to `NMFk.execute`, such as the number of factorizations.

# Keywords

- `casefilename="nmfk-tensor"`: Base case filename used for each dimension-specific execution.
- `kw...`: Additional keyword arguments forwarded to `NMFk.execute`.

# Returns

A vector indexed by tensor dimension whose analyzed entries contain the tuples returned by `NMFk.execute`.

# Examples

```julia
import NMFk
import Random

Random.seed!(2026)
X = Random.rand(2, 2, 2)
results = NMFk.tensorfactorization(
	X,
	1,
	1:3,
	1;
	load=false,
	save=false,
	method=:simple,
	quiet=true,
)
length(results)
```
"""
function tensorfactorization(X::AbstractArray{T,N}, range::Union{AbstractUnitRange{Int},Integer}, dims::Union{AbstractUnitRange{Int},Integer}=1:N, aw...; casefilename::AbstractString="nmfk-tensor", kw...) where {T <: Number, N}
	@assert maximum(dims) <= N
	R = Vector{Tuple}(undef, N)
	@info("Analyzed Dimensions: $(dims)")
	for d in dims
		M = NMFk.flatten(X, d)
		@info("Dimension $d: size: $(size(X)) -> $(size(M)) ...")
		R[d] = NMFk.execute(M, range, aw...; casefilename=casefilename * "-d$d", kw...)
	end
	return R
end

function tensorfactorization(X::AbstractArray{T,N}, range::AbstractVector, dims::Union{AbstractUnitRange{Int},Integer}=1:N, aw...; casefilename::AbstractString="nmfk-tensor", kw...) where {T <: Number, N}
	@assert length(range) == length(dims)
	@assert maximum(dims) <= N
	R = Vector{Tuple}(undef, N)
	@info("Analyzed Dimensions: $(dims)")
	for d in dims
		if length(range[d]) > 0
			M = NMFk.flatten(X, d)
			@info("Dimension $d: size: $(size(X)) -> $(size(M)) ...")
			R[d] = NMFk.execute(M, range[d], aw...; casefilename=casefilename * "-d$d", kw...)
		end
	end
	return R
end
