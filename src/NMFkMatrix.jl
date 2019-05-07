"Normalize matrix"
function normalizematrix!(a::Matrix)
	amin = minimum(a; dims=1)
	amax = maximum(a; dims=1)
	dx = amax - amin
	if length(dx) > 1
		i0 = dx .== 0 # check for zeros
		amin[i0] .= 0
		dx[i0] .= amax[i0]
		i0 = dx .== 0 # check for zeros again
		dx[i0] .= 1
	end
	a = (a .- amin) ./ dx
	return a, amin, amax
end

"Denormalize matrix"
function denormalizematrix!(a::Matrix, b::Matrix, amin::Matrix, amax::Matrix)
	a = a .* (amax - amin) + pinv(b) * repeat(amin, outer=[size(b, 1), 1])
	return a
end

"Scale matrix (by rows)"
function scalematrix!(a::Matrix)
	amax = maximum(abs.(a); dims=1)
	a ./= amax
	return a, amax
end

"Scale array"
function scalearray!(a::Array)
	amax = vec(maximum(abs.(a), dims=(1,3)))
	for i = 1:length(amax)
		a[:, i, :] ./= amax[i]
	end
	return a, amax
end

"Descale matrix (by rows)"
function descalematrix!(a::Matrix, amax::Matrix)
	a .*= amax
	return a
end

"Descale matrix (by rows)"
function descalearray!(a::Matrix, amax::Vector)
	a .*= permutedims(amax)
	return a
end

"Scale matrix (by columns)"
function scalematrix_col!(a::Matrix)
	amax = maximum(abs(a); dims=2)
	a = a ./ amax
	return a, amax
end