"Normalize matrix"
function normalizematrix(a::Matrix)
	amin = minimum(a, 1)
	amax = maximum(a, 1)
	dx = amax - amin
	i0 = dx .== 0 # check for zeros
	min[i0] = 0
	dx[i0] = amax[i0]
	i0 = dx .== 0 # check for zeros again
	dx[i0] = 1
	a = (a .- amin) ./ dx
	return a, amin, amax
end

"Denormalize matrix"
function denormalizematrix(a::Matrix, b::Matrix, amin::Matrix, amax::Matrix)
	a = a .* (amax - amin) + pinv(b) * repeat(amin, outer=[size(b, 1), 1])
	return a
end

"Scale matrix (by rows)"
function scalematrix(a::Matrix)
	amax = maximum(abs(a), 1)
	a = a ./ amax
	return a, amax
end

"Descale matrix (by rows)"
function descalematrix(a::Matrix, amax::Matrix)
	a = a .* amax
	return a
end

"Scale matrix (by columns)"
function scalematrix_col(a::Matrix)
	amax = maximum(abs(a), 2)
	a = a ./ amax
	return a, amax
end