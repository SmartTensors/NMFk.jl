function normalizematrix!(a...; kw...)
	normalizematrix_col!(a...; kw...)
end

"Normalize matrix (by columns)"
function normalizematrix_col!(a::AbstractMatrix; rev::Bool=false)
	amax = permutedims(map(i->NMFk.maximumnan(a[:,i]), 1:size(a, 2)))
	amin = permutedims(map(i->NMFk.minimumnan(a[:,i]), 1:size(a, 2)))
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

"Normalize matrix (by rows)"
function normalizematrix_row!(a::AbstractMatrix; rev::Bool=false)
	amax = map(i->NMFk.maximumnan(a[i,:]), 1:size(a, 1))
	amin = map(i->NMFk.minimumnan(a[i,:]), 1:size(a, 1))
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

"Denormalize matrix"
function denormalizematrix!(a...)
	denormalizematrix_col!(a...)
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
		a = repeat(amin; outer=[1, size(a, 1),]) + a .* (amax - amin)
	end
	return a
end

"Scale matrix (by rows)"
function scalematrix!(a::AbstractMatrix)
	amax = permutedims(map(i->NMFk.maximumnan(a[:,i]), 1:size(a, 2)))
	a ./= amax
	return a, amax
end

"Scale array"
function scalearray!(a::Array)
	amax = vec(maximumnan(a; dims=(1,3)))
	for i = 1:length(amax)
		if amax[i] != 0 && !isnan(amax[i])
			a[:, i, :] ./= amax[i]
		end
	end
	return a, amax
end

"Descale matrix (by rows)"
function descalematrix!(a::AbstractMatrix, amax::AbstractMatrix)
	a .*= amax
	return a
end

"Descale matrix (by rows)"
function descalearray!(a::AbstractMatrix, amax::Vector)
	for i = 1:length(amax)
		if amax[i] != 0 && !isnan(amax[i])
			a[:, i, :] .*= amax[i]
		end
	end
	return a
end

"Scale matrix (by columns)"
function scalematrix_col!(a::AbstractMatrix)
	amax = maximum(abs(a); dims=2)
	a = a ./ amax
	return a, amax
end