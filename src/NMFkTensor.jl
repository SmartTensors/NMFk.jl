import DocumentFunction

function tensorfactorization(X::AbstractArray{T,N}, range::Union{AbstractRange{Int},Integer}, dims::Union{AbstractRange{Int},Integer}=1:N, aw...; casefilename::AbstractString="nmfk-tensor", kw...) where {T <: Number, N}
	@assert maximum(dims) <= N
	M = Vector{Tuple}(undef, N)
	@info "Dimensions to be analyzed: $dims"
	for d in dims
		@info "Dimension $d ..."
		casefilename *= "-d$d"
		M[d] = NMFk.execute(NMFk.flatten(X, d), range, aw...; casefilename=casefilename, kw...)
	end
	return M
end

function tensorfactorization(X::AbstractArray{T,N}, range::AbstractVector, aw...; casefilename::AbstractString="nmfk-tensor", kw...) where {T <: Number, N}
	@assert length(range) == N
	M = Vector{Tuple}(undef, N)
	@info "Dimensions to be analyzed: 1:$N"
	for d = 1:N
		@info "Dimension $d ..."
		casefilename *= "-d$d"
		if length(range[d]) > 0
			M[d] = NMFk.execute(NMFk.flatten(X, d), range[d], aw...; casefilename=casefilename, kw...)
		end
	end
	return M
end

@doc """
Tensor Factorization

$(DocumentFunction.documentfunction(tensorfactorization))
""" tensorfactorization