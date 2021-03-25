import DocumentFunction

function tensorfactorization(X::AbstractArray{T,N}, range::Union{AbstractRange{Int},Integer}, dims::Union{AbstractRange{Int},Integer}=1:N, aw...; kw...) where {T,N}
	@assert maximum(dims) <= N
	M = Vector{Tuple}(undef, N)
	for d = dims
		M[d] = NMFk.execute(NMFk.flatten(X, d), range, aw...; kw...)
	end
	return M
end

function tensorfactorization(X::AbstractArray{T,N}, range::AbstractVector, aw...; kw...) where {T,N}
	@assert length(range) == N
	M = Vector{Tuple}(undef, N)
	for d = 1:N
		if length(range[d]) > 0
			M[d] = NMFk.execute(NMFk.flatten(X, d), range[d], aw...; kw...)
		end
	end
	return M
end

@doc """
Tensor Factorization

$(DocumentFunction.documentfunction(tensorfactorization))
""" tensorfactorization