"Normalize"
function normalize!(a::AbstractArray; rev::Bool=false, amax = NMFk.maximumnan(a), amin = NMFk.minimumnan(a))
	dx = amax - amin
	if rev
		a = (amax .- a) ./ dx
		return a, amax, amin
	else
		a = (a .- amin) ./ dx
		return a, amin, amax
	end
end
function normalize!(a; rev::Bool=false, amax = NMFk.maximumnan(a), amin = NMFk.minimumnan(a))
	dx = amax - amin
	if rev
		a = (amax .- a) ./ dx
		return a, amax, amin
	else
		a = (a .- amin) ./ dx
		return a, amin, amax
	end
end

"Denormalize"
function denormalize!(a, amin, amax)
	if all(amax .>= amin)
		a = a .* (amax - amin) .+ amin
	else
		a = a .* (amax - amin) .+ amin
	end
	return a
end

"Normalize matrix (by columns)"
function normalizematrix_col!(a::AbstractMatrix; rev::Bool=false)
	normalizematrix!(a, 2; rev=rev)
end

"Normalize matrix (by rows)"
function normalizematrix_row!(a::AbstractMatrix; rev::Bool=false)
	normalizematrix!(a, 1; rev=rev)
end

function normalizematrix!(a::AbstractMatrix, dim::Integer; rev::Bool=false)
	amin, amax = matrixminmax(a, dim)
	dx = amax - amin
	if length(dx) > 1
		i0 = dx .== 0 # check for zeros
		amin[i0] .= 0
		dx[i0] .= amax[i0]
		i0 = dx .== 0 # check for zeros again
		dx[i0] .= 1
	end
	if rev
		a = (amax .- a) ./ dx
		return a, amax, amin
	else
		a = (a .- amin) ./ dx
		return a, amin, amax
	end
end

function matrixminmax(a::AbstractMatrix, dim::Integer)
	amax = map(i->NMFk.maximumnan(a[ntuple(k->(k == dim ? i : Colon()), ndims(a))...]), 1:size(a, dim))
	amin = map(i->NMFk.minimumnan(a[ntuple(k->(k == dim ? i : Colon()), ndims(a))...]), 1:size(a, dim))
	if dim == 2
		amax = permutedims(amax)
		amin = permutedims(amin)
	end
	return amin, amax
end

"Denormalize matrix"
function denormalizematrix_col!(a::AbstractMatrix, amin::Vector, amax::Vector)
	if all(amax .>= amin)
		a = a .* (amax - amin) + repeat(amin; outer=[size(a, 1), 1])
	else
		a = repeat(amin; outer=[size(a, 1), 1]) + a .* (amax - amin)
	end
	return a
end

"Denormalize matrix"
function denormalizematrix_row!(a::AbstractMatrix, amin::Vector, amax::Vector)
	if all(amax .>= amin)
		a = a .* (amax - amin) + repeat(amin; outer=[1, size(a, 2)])
	else
		a = repeat(amin; outer=[1, size(a, 1)]) + a .* (amax - amin)
	end
	return a
end

"Scale array"
function scalearray!(a::AbstractArray{T,N}; dims=(1,2)) where {T,N}
	amax = vec(maximumnan(a; dims=dims))
	for i = 1:length(amax)
		if amax[i] != 0 && !isnan(amax[i])
			nt = ntuple(k->(k in dims ? Colon() : i), N)
			a[nt...] ./= amax[i]
		end
	end
	return a, amax
end

"Descale array"
function descalearray!(a::AbstractArray{T,N}, amax::Vector; dims=(1,2)) where {T,N}
	for i = 1:length(amax)
		if amax[i] != 0 && !isnan(amax[i])
			nt = ntuple(k->(k in dims ? Colon() : i), N)
			a[nt...] .*= amax[i]
		end
	end
	return a
end

"Scale matrix (by rows)"
function scalematrix_row!(a::AbstractMatrix)
	amax = permutedims(map(i->NMFk.maximumnan(a[:,i]), 1:size(a, 2)))
	a ./= amax
	return a, amax
end

"Scale matrix (by columns)"
function scalematrix_col!(a::AbstractMatrix)
	amax = map(i->NMFk.maximumnan(a[i,:]), 1:size(a, 1))
	a = a ./ amax
	return a, amax
end

"Descale matrix (by rows)"
function descalematrix!(a::AbstractMatrix, amax::AbstractMatrix)
	a .*= amax
	return a
end