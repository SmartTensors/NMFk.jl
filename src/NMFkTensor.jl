import DocumentFunction

function tensorfactorization(X::AbstractArray{T,N}, range::Union{AbstractUnitRange{Int},Integer}, dims::Union{AbstractUnitRange{Int},Integer}=1:N, aw...; casefilename::AbstractString="nmfk-tensor", kw...) where {T <: Number, N}
	@assert maximum(dims) <= N
	R = Vector{Tuple}(undef, N)
	@info("Analyzed Dimensions: $dims")
	for d in dims
		M = NMFk.flatten(X, d)
		@info("Dimension $d: size: $(size(X)) -> $(size(M)) ...")
		R[d] = NMFk.execute(M, range, aw...; casefilename=casefilename * "-d$d", kw...)
	end
	return R
end

function tensorfactorization(X::AbstractArray{T,N}, range::AbstractVector, aw...; casefilename::AbstractString="nmfk-tensor", kw...) where {T <: Number, N}
	@assert length(range) == N
	R = Vector{Tuple}(undef, N)
	@info("Analyzed Dimensions: 1:$N")
	for d = 1:N
		if length(range[d]) > 0
			M = NMFk.flatten(X, d)
			@info("Dimension $d: size: $(size(X)) -> $(size(M)) ...")
			R[d] = NMFk.execute(M, range[d], aw...; casefilename=casefilename * "-d$d", kw...)
		end
	end
	return R
end

@doc """
Tensor Factorization

$(DocumentFunction.documentfunction(tensorfactorization))
""" tensorfactorization