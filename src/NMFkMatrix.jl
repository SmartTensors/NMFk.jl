"Normalize"
function normalize!(a::AbstractArray; rev::Bool=false, amax = NMFk.maximumnan(a), amin = NMFk.minimumnan(a))
	dx = amax - amin
	if rev
		a .= (amax .- a) ./ dx
		return a, amax, amin
	else
		a .= (a .- amin) ./ dx
		return a, amin, amax
	end
end
function normalize!(a; rev::Bool=false, amax = NMFk.maximumnan(a), amin = NMFk.minimumnan(a))
	dx = amax - amin
	if rev
		a .= (amax .- a) ./ dx
		return a, amax, amin
	else
		a = (a .- amin) ./ dx
		return a, amin, amax
	end
end

"Denormalize"
function denormalize!(a, amin, amax)
	if all(amax .>= amin)
		a .= a .* (amax - amin) .+ amin
	else
		a .= a .* (amax - amin) .+ amin
	end
	return a
end

"Normalize matrix (by columns)"
function normalizematrix_col(a::AbstractMatrix; kw...)
	normalizematrix!(copy(a), 2; kw...)
end
function normalizematrix_col!(a::AbstractMatrix; kw...)
	normalizematrix!(a, 2; kw...)
end

"Normalize matrix (by rows)"
function normalizematrix_row(a::AbstractMatrix; kw...)
	normalizematrix!(copy(a), 1; kw...)
end
function normalizematrix_row!(a::AbstractMatrix; kw...)
	normalizematrix!(a, 1; kw...)
end

function normalizematrix!(a::AbstractMatrix, dim::Integer; rev::Bool=false, log::Bool=false, logv::AbstractVector=fill(log, size(a, dim)), offset::Number=1)
	amin, amax = matrixminmax(a, dim)
	zflag = falses(length(amin))
	for (i, m) in enumerate(amin)
		nt = ntuple(k->(k == dim ? i : Colon()), ndims(a))
		av = view(a, nt...)
		if logv[i]
			iz = av .<= 0
			siz = sum(iz)
			if siz == length(iz)
				av .= abs.(av)
			end
			iz = av .<= 0
			siz = sum(iz)
			siz > 0 && (av[iz] .= NaN)
			av .= log10.(av)
			if siz > 0
				av[iz] .= minimumnan(av) - offset
				zflag[i] = true
			end
		end
	end
	amin, amax = matrixminmax(a, dim)
	dx = amax .- amin
	if length(dx) > 1
		i0 = dx .== 0
		amin[i0] .= 0
		dx[i0] .= amax[i0]
		i0 = dx .== 0 # check for zeros again
		dx[i0] .= 1
	end
	if rev
		a .= (amax .- a) ./ dx
		return a, amax, amin, zflag
	else
		a .= (a .- amin) ./ dx
		return a, amin, amax, zflag
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

function normalizearray!(a::AbstractArray, dim::Integer; rev::Bool=false)
	amin, amax = arrayminmax(a, dim)
	dx = amax .- amin
	if length(dx) > 1
		i0 = dx .== 0
		amin[i0] .= 0
		dx[i0] .= amax[i0]
		i0 = dx .== 0 # check for zeros again
		dx[i0] .= 1
	end
	if rev
		for i = 1:size(a, dim)
			nt = ntuple(k->(k == dim ? (i:i) : Colon()), ndims(a))
			a[nt...] .= (amax[i] .- a[nt...]) ./ dx[i]
		end
		return a, amax, amin
	else
		for i = 1:size(a, dim)
			nt = ntuple(k->(k == dim ? (i:i) : Colon()), ndims(a))
			a[nt...] .= (a[nt...] .- amin[i]) ./ dx[i]
		end
		return a, amin, amax
	end
end

function arrayminmax(a::AbstractArray, dim::Integer)
	amax = map(i->NMFk.maximumnan(a[ntuple(k->(k == dim ? i : Colon()), ndims(a))...]), 1:size(a, dim))
	amin = map(i->NMFk.minimumnan(a[ntuple(k->(k == dim ? i : Colon()), ndims(a))...]), 1:size(a, dim))
	return amin, amax
end

"Denormalize matrix"
function denormalizematrix_col(a::AbstractMatrix, at...; kw...)
	denormalizematrix_col!(copy(a), at...; kw...)
end
function denormalizematrix_col!(a::AbstractMatrix, amin::AbstractMatrix, amax::AbstractMatrix; log::Bool=false, logv::AbstractVector=fill(log, size(a, 2)), zflag::AbstractVector=falses(size(a, 2)))
	if all(amax .>= amin)
		a .= a .* (amax - amin) + repeat(amin; outer=[size(a, 1), 1])
	else
		a .= repeat(amin; outer=[size(a, 1), 1]) + a .* (amax - amin)
	end
	for (i, m) in enumerate(amin)
		av = view(a, :, i)
		if logv[i]
			zflag[i] && (iz = av .== m)
			av .= 10. .^ av
			zflag[i] && (av[iz] .= 0)
		end
	end
	return a
end

"Denormalize matrix"
function denormalizematrix_row(a::AbstractMatrix, at...; kw...)
	denormalizematrix_row!(copy(a), at...; kw...)
end
function denormalizematrix_row!(a::AbstractMatrix, amin::AbstractVector, amax::AbstractVector; log::Bool=false, logv::AbstractVector=fill(log, size(a, 1)), zflag::AbstractVector=falses(size(a, 2)))
	if all(amax .>= amin)
		a .= a .* (amax - amin) + repeat(amin; outer=[1, size(a, 2)])
	else
		a .= repeat(amin; outer=[1, size(a, 1)]) + a .* (amax - amin)
	end
	for (i, m) in enumerate(amin)
		av = view(a, i, :)
		if logv[i]
			iz = av .< m
			av .= 10. .^ av
			av[iz] .= 0
		end
	end
	return a
end

"Normalize array"
function normalizearray!(a::AbstractArray{T,N}; rev::Bool=false, dims=(1,2), amax=vec(maximumnan(a; dims=dims)), amin=vec(minimumnan(a; dims=dims))) where {T,N}
	for i = 1:length(amax)
		dx = amax[i] - amin[i]
		if dx != 0 && !isnan(dx)
			nt = ntuple(k->(k in dims ? Colon() : i), N)
			if rev
				a[nt...] .= (amax[i] .- a[nt...]) ./ dx
			else
				a[nt...] .= (a[nt...] .- amin[i]) ./ dx
			end
		end
	end
	if rev
		return a, amax, amin
	else
		return a, amin, amax
	end
end

"Denormalize array"
function denormalizearray!(a::AbstractArray{T,N}, amin, amax; dims=(1,2)) where {T,N}
	for i = 1:length(amax)
		dx = amax[i] - amin[i]
		if dx != 0 && !isnan(dx)
			nt = ntuple(k->(k in dims ? Colon() : i), N)
			dx = amax[i] - amin[i]
			a[nt...] .= a[nt...] .* dx .+ amin[i]
		end
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
function scalearray!(a::AbstractArray{T,N}, dim::Integer) where {T,N}
	_, amax = matrixminmax(a, dim)
	for i = 1:length(amax)
		if amax[i] != 0 && !isnan(amax[i])
			nt = ntuple(k->(k == dim ? (i:i) : Colon()), N)
			a[nt...] ./= amax[i]
		end
	end
	return a, amax
end

"Descale array"
function descalearray!(a::AbstractArray{T,N}, amax::AbstractVector; dims=(1,2)) where {T,N}
	for i = 1:length(amax)
		if amax[i] != 0 && !isnan(amax[i])
			nt = ntuple(k->(k in dims ? Colon() : i), N)
			a[nt...] .*= amax[i]
		end
	end
	return a
end
function descalearray!(a::AbstractArray{T,N}, amax::AbstractVector, dim::Integer) where {T,N}
	for i = 1:length(amax)
		if amax[i] != 0 && !isnan(amax[i])
			nt = ntuple(k->(k == dim ? (i:i) : Colon()), N)
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