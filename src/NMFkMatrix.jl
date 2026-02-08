"Normalize"
function normalize(a::AbstractArray; kw...)
	normalize!(copy(a); kw...)
end
function normalize!(a::AbstractArray{T, N}; rev::Bool=false, amax=NMFk.maximumnan(a), amin=NMFk.minimumnan(a), logv::AbstractVector=falses(0)) where {T <: Number, N}
	if N == 1 && length(logv) > 0
		@assert length(logv) == length(a)
		@assert length(logv) == length(amin)
		for i = eachindex(logv)
			if logv[i]
				a[i] = log10s(a[i]; min=amin[i])
			end
		end
	end
	dx = amax .- amin
	if dx == 0
		dx = amax
		amin = 0
	end
	if rev
		a .= (amax .- a) ./ dx
		return a, amax, amin
	else
		a .= (a .- amin) ./ dx
		return a, amin, amax
	end
end

"Denormalize"
function denormalize(a::AbstractArray, aw...)
	return denormalize!(copy(a), aw...)
end
function denormalize!(a::AbstractArray, amin, amax)
	if all(amax .>= amin)
		a .= a .* (amax - amin) .+ amin
	else
		a .= a .* (amax - amin) .+ amin
	end
	return a
end

"Normalize matrix (by columns)"
function normalizematrix_col(a::AbstractMatrix; kw...)
	return normalizematrix!(copy(a), 2; kw...)
end
function normalizematrix_col!(a::AbstractMatrix; kw...)
	return normalizematrix!(a, 2; kw...)
end

"Normalize matrix (by rows)"
function normalizematrix_row(a::AbstractMatrix; kw...)
	return normalizematrix!(copy(a), 1; kw...)
end
function normalizematrix_row!(a::AbstractMatrix; kw...)
	return normalizematrix!(a, 1; kw...)
end

"Normalize matrix"
function normalizematrix(a::AbstractMatrix, dim::Integer; kw...)
	return normalizematrix!(copy(a), dim; kw...)
end

function normalizematrix!(a::AbstractMatrix, dim::Integer; amin::AbstractArray=matrixmin(a, dim), amax::AbstractArray=matrixmax(a, dim), rev::Bool=false, log::Bool=false, logv::AbstractVector=fill(log, size(a, dim)), offset::Number=1)
	@assert length(amin) == size(a, dim)
	@assert length(amax) == size(a, dim)
	@assert length(logv) == size(a, dim)
	logtransform_type = Vector{Union{Float64,Symbol}}(undef, length(amin))
	logtransform_type .= :nothing
	lamin = copy(amin)
	lamax = copy(amax)
	for (i, m) in enumerate(lamin)
		nt = ntuple(k->(k == dim ? i : Colon()), ndims(a))
		av = view(a, nt...)
		if logv[i]
			avn = av[.!isnan.(av)]
			inz = avn .<= 0
			if sum(inz) == length(inz) # if all negative or zero
				av .= abs.(av)
				logtransform_type[i] = :absflip
			elseif (sum(avn .< 0) > 0) && (sum(avn .> 0 ) > 0) # if some negative and some positive
				minavn = minimum(avn)
				av .+= abs(minavn) + offset # make all positive by shifting
				logtransform_type[i] = abs(minavn) + offset
			end
			iz = av .== 0
			av[iz] .= NaN # if there are zero values make them NaN
			av .= log10.(av)
			if sum(iz) > 0
				av[iz] .= minimumnan(av) - offset # make the negative and zero values something very small
				logtransform_type[i] = logtransform_type[i] == :absflip ? :absflip_min_zero : :min_zero
			end
			# @show logtransform_type[i]
			lamin[nt...] .= minimumnan(av)
			lamax[nt...] .= maximumnan(av)
		end
	end
	dx = lamax .- lamin
	if length(dx) > 1
		i0 = dx .== 0
		dx[i0] .= 1
	end
	if rev
		a .= (lamax .- a) ./ dx
		return a, lamax, lamin, logtransform_type
	else
		a .= (a .- lamin) ./ dx
		return a, lamin, lamax, logtransform_type
	end
end

function matrixminmax(a::AbstractMatrix, dim::Integer)
	amax = map(i->NMFk.maximumnan(a[ntuple(k->(k == dim ? i : Colon()), ndims(a))...]), axes(a, dim))
	amin = map(i->NMFk.minimumnan(a[ntuple(k->(k == dim ? i : Colon()), ndims(a))...]), axes(a, dim))
	if dim == 2
		amax = permutedims(amax)
		amin = permutedims(amin)
	end
	return amin, amax
end

function matrixmin(a::AbstractMatrix, dim::Integer)
	amin = map(i->NMFk.minimumnan(a[ntuple(k->(k == dim ? i : Colon()), ndims(a))...]), axes(a, dim))
	if dim == 2
		amin = permutedims(amin)
	end
	return amin
end

function matrixmax(a::AbstractMatrix, dim::Integer)
	amax = map(i->NMFk.maximumnan(a[ntuple(k->(k == dim ? i : Colon()), ndims(a))...]), axes(a, dim))
	if dim == 2
		amax = permutedims(amax)
	end
	return amax
end

function normalizearray(a::AbstractArray, dim::Integer; kw...)
	return normalizearray!(copy(a), dim; kw...)
end

function normalizearray!(a::AbstractArray, dim::Integer; rev::Bool=false, log::Bool=false, logv::AbstractVector=[], min_skewness::Number=1, min_log_range::Number=1.5)
	n = size(a, dim)
	if length(logv) == 0
		logv = falses(n)
		if log
			for i = 1:n
				nt = ntuple(k->(k == dim ? (i:i) : Colon()), ndims(a))
				log_range = log10(maximum(a[nt...])) - log10(minimum(a[nt...]))
				s = StatsBase.skewness(vec(a[nt...]))
				if s > min_skewness || log_range > min_log_range
					a[nt...] .= NMFk.log10s(a[nt...])
					logv[i] = true
				end
			end
		end
	else
		@assert length(logv) == n
		for i = 1:n
			nt = ntuple(k->(k == dim ? (i:i) : Colon()), ndims(a))
			if logv[i]
				a[nt...] .= NMFk.log10s(a[nt...])
			end
		end
	end
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
		for i = 1:n
			nt = ntuple(k->(k == dim ? (i:i) : Colon()), ndims(a))
			a[nt...] .= (amax[i] .- a[nt...]) ./ dx[i]
		end
		return a, amax, amin, logv
	else
		for i = 1:n
			nt = ntuple(k->(k == dim ? (i:i) : Colon()), ndims(a))
			a[nt...] .= (a[nt...] .- amin[i]) ./ dx[i]
		end
		return a, amin, amax, logv
	end
end

function arrayminmax(a::AbstractArray, dim::Integer)
	amax = map(i->NMFk.maximumnan(a[ntuple(k->(k == dim ? i : Colon()), ndims(a))...]), axes(a, dim))
	amin = map(i->NMFk.minimumnan(a[ntuple(k->(k == dim ? i : Colon()), ndims(a))...]), axes(a, dim))
	return amin, amax
end

"Denormalize matrix"
function denormalizematrix_row(a::AbstractMatrix, at...; kw...)
	return denormalizematrix_row!(copy(a), at...; kw...)
end
function denormalizematrix_row!(a::AbstractMatrix, amin::Union{AbstractVector,AbstractMatrix}, amax::Union{AbstractVector,AbstractMatrix}; kw...)
	return denormalizematrix!(a, 1, amin, amax; kw...)
end
function denormalizematrix_col(a::AbstractMatrix, at...; kw...)
	return denormalizematrix_col!(copy(a), at...; kw...)
end
function denormalizematrix_col!(a::AbstractMatrix, amin::Union{AbstractVector,AbstractMatrix}, amax::Union{AbstractVector,AbstractMatrix}; kw...)
	return denormalizematrix!(a, 2, amin, amax; kw...)
end
function denormalizematrix(a::AbstractMatrix, at...; kw...)
	return denormalizematrix!(copy(a), at...; kw...)
end
function denormalizematrix!(a::AbstractMatrix, dim::Number, amin::Union{AbstractVector,AbstractMatrix}, amax::Union{AbstractVector,AbstractMatrix}; log::Bool=false, logv::AbstractVector=fill(log, size(a, dim)), logtransform_type::AbstractVector=fill(:nothing, size(a, dim)))
	dx = amax .- amin
	dx[dx .== 0] .= 1
	if dim == 1
		outer1 = [1, size(a, 2)]
		outer2 = [1, size(a, 1)]
	else
		outer1 = [size(a, 1), 1]
		outer2 = [size(a, 1), 1]
	end
	if all(dx .>= 0)
		a .= a .* dx + repeat(amin; outer=outer1)
	else
		a .= repeat(amin; outer=outer2) + a .* dx
	end
	for (i, m) in enumerate(amin)
		nt = ntuple(k->(k == dim ? i : Colon()), ndims(a))
		av = view(a, nt...)
		if logv[i]
			# @show logtransform_type[i]
			if typeof(logtransform_type[i]) <: Number
				av .= 10. .^ av
				av .-= logtransform_type[i]
			else
				if logtransform_type[i] .== :absflip_min_zero || logtransform_type[i] .== :min_zero
					iz = av .== m
					av .= 10. .^ av
					av[iz] .= 0
				else
					av .= 10. .^ av
				end
				if logtransform_type[i] .== :absflip_min_zero || logtransform_type[i] .== :absflip
					av .= -abs.(av)
				end
			end
		end
	end
	return a
end

"Normalize array"
function normalizearray!(a::AbstractArray{T,N}; rev::Bool=false, dims=(1,2), amin=vec(minimumnan(a; dims=dims)), amax=vec(maximumnan(a; dims=dims)), ) where {T <: Number, N}
	for i = eachindex(amax)
		dx = amax[i] - amin[i]
		if dx == 0
			dx = 1
		end
		if !isnan(dx)
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
function denormalizearray(a::AbstractArray{T,N}, aw...; kw...) where {T <: Number, N}
	return denormalizearray!(copy(a), aw...; kw...)
end

function denormalizearray!(a::AbstractArray{T,N}, amin, amax; dims=(1,2)) where {T <: Number, N}
	for i = eachindex(amax)
		dx = amax[i] - amin[i]
		if dx == 0
			dx = 1
		end
		if !isnan(dx)
			nt = ntuple(k->(k in dims ? Colon() : i), N)
			a[nt...] .= a[nt...] .* dx .+ amin[i]
		end
	end
	return a
end

"Scale array"
function scalearray!(a::AbstractArray{T,N}; dims=(1,2)) where {T <: Number, N}
	amax = vec(maximumnan(a; dims=dims))
	for i = eachindex(amax)
		if amax[i] != 0 && !isnan(amax[i])
			nt = ntuple(k->(k in dims ? Colon() : i), N)
			a[nt...] ./= amax[i]
		end
	end
	return a, amax
end
function scalearray!(a::AbstractArray{T,N}, dim::Integer) where {T <: Number, N}
	_, amax = matrixminmax(a, dim)
	for i = eachindex(amax)
		if amax[i] != 0 && !isnan(amax[i])
			nt = ntuple(k->(k == dim ? (i:i) : Colon()), N)
			a[nt...] ./= amax[i]
		end
	end
	return a, amax
end

"Descale array"
function descalearray!(a::AbstractArray{T,N}, amax::AbstractVector; dims=(1,2)) where {T <: Number, N}
	for i = eachindex(amax)
		if amax[i] != 0 && !isnan(amax[i])
			nt = ntuple(k->(k in dims ? Colon() : i), N)
			a[nt...] .*= amax[i]
		end
	end
	return a
end
function descalearray!(a::AbstractArray{T,N}, amax::AbstractVector, dim::Integer) where {T <: Number, N}
	for i = eachindex(amax)
		if amax[i] != 0 && !isnan(amax[i])
			nt = ntuple(k->(k == dim ? (i:i) : Colon()), N)
			a[nt...] .*= amax[i]
		end
	end
	return a
end

"Scale matrix (by rows)"
function scalematrix_row!(a::AbstractMatrix)
	amax = permutedims(map(i->NMFk.maximumnan(a[:,i]), axes(a, 2)))
	a ./= amax
	return a, amax
end

"Scale matrix (by columns)"
function scalematrix_col!(a::AbstractMatrix)
	amax = map(i->NMFk.maximumnan(a[i,:]), axes(a, 1))
	a = a ./ amax
	return a, amax
end

"Descale matrix (by rows)"
function descalematrix!(a::AbstractMatrix, amax::AbstractMatrix)
	a .*= amax
	return a
end