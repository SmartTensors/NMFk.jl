"Normalize matrix"
function normalizematrix(a::Matrix)
	min = minimum(a, 1)
	max = maximum(a, 1)
	dx = max - min
	i0 = dx .== 0 # check for zeros
	min[i0] = 0
	dx[i0] = max[i0]
	i0 = dx .== 0 # check for zeros again
	dx[i0] = 1
	a = (a .- min) ./ dx
	return a, min, max
end

"Denormalize matrix"
function denormalizematrix(a::Matrix, b::Matrix, min::Matrix, max::Matrix)
	a = a .* (max - min) + pinv(b) * repeat(min, outer=[size(b, 1), 1])
	return a
end

"Scale matrix (by rows)"
function scalematrix(a::Matrix)
	max = maximum(abs(a), 1)
	a = a ./ max
	return a, max
end

"Descale matrix (by rows)"
function descalematrix(a::Matrix, max::Matrix)
	a = a .* max
	return a
end

"Scale matrix (by columns)"
function scalematrix_col(a::Matrix)
	max = maximum(abs(a), 2)
	a = a ./ max
	return a, max
end