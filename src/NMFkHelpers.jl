import DocumentFunction
import Statistics
import LinearAlgebra

"""
Set image dpi

$(DocumentFunction.documentfunction(setdpi))
"""
function setdpi(dpi::Integer)
	global imagedpi = dpi;
end

toupper(x::String, i=1) = x[1:i-1] * uppercase(x[i:i]) * x[i+1:end]

function r2(x::Vector, y::Vector)
	# rho = Statistics.cov(x, y) / (Statistics.std(x) * Statistics.std(y))
	# r2 = (1 - sum((x .- y).^2) / sum((x .- Statistics.mean(x)).^2))
	(sum((x .- Statistics.mean(x)) .* (y .- Statistics.mean(y)))/sqrt(sum((x .- Statistics.mean(x)).^2 .* sum((y .- Statistics.mean(y)).^2))))^2
end

function maximumnan(X; functionname="isnan", kw...)
	i = Core.eval(NMFk, Meta.parse(functionname)).(X)
	maximum(X[.!i]; kw...)
end

function minimumnan(X; functionname="isnan", kw...)
	i = Core.eval(NMFk, Meta.parse(functionname)).(X)
	minimum(X[.!i]; kw...)
end

function sumnan(X, c=nothing; kw...)
	if c == nothing
		return sum(X[.!isnan.(X)]; kw...)
	else
		count = .*(size(X)[vec(collect(c))]...)
		I = isnan.(X)
		X[I] .= 0
		sX = sum(X; dims=c, kw...)
		X[I] .= NaN
		sI = sum(I; dims=c, kw...)
		sX[sI.==count] .= NaN
		return sX
	end
end

function meannan(X)
	Statistics.mean(X[.!isnan.(X)])
end

function ssqrnan(X)
	sum(X[.!isnan.(X)].^2)
end

function normnan(X)
	LinearAlgebra.norm(X[.!isnan.(X)])
end

function cornan(x, y)
	isn = .!(isnan.(x) .| isnan.(y))
	if length(x) > 0 && length(y) > 0 && sum(isn) > 1
		return Statistics.cov(x[isn], y[isn])
	else
		return NaN
	end
end

function hardencodelength(x::Vector{T}) where {T}
	u = unique(x)
	i = indexin(x, u)
	inan = indexin(true, isnan.(u))[1]
	d = inan != nothing ? length(u) - 1 : d = length(u)
	return d
end

function hardencode(x::Vector{T}) where {T}
	u = unique(x)
	i = indexin(x, u)
	inan = indexin(true, isnan.(u))[1]
	d = inan != nothing ? length(u) - 1 : d = length(u)
	m = zeros(length(x), d)
	for (j, k) in enumerate(i)
		if inan != nothing
			if k == inan
				m[j, :] .= NaN
			elseif k < inan
				m[j, k] = 1
			else
				m[j, k-1] = 1
			end
		else
			m[j, k] = 1
		end
	end
	return m
end


function hardencode(x::Matrix{T}) where {T}
	hcat([hardencode(x[:,i]) for i = 1:size(x, 2)]...)
end

function gettypes(x::Matrix{T}, levels=[0.05,0.35]) where {T}
	nlevels = length(levels)
	nw = size(x, 1)
	ns = size(x, 2)
	s = zeros(T, nw, ns)
	for i = 1:ns
		for l = 1:nlevels
			if l == nlevels
				s[:,i] .+= (x[:,i] .>= levels[l]) .* l
			else
				s[:,i] .+= ((x[:,i] .>= levels[l]) .& (x[:,i] .< levels[l+1])) .* l
			end
		end
	end
	wcode = sum(hcat(map(j->s[:,j].*10^j, 1:ns)...); dims=2)
	uniquewcode = unique(wcode)
	nt = length(uniquewcode)
	@info("Number of unique types is $nt)")
	types = nt < 31 ? collect(range('A'; length=nt)) : ["T$i" for i=1:nt]
	iwcode = indexin(wcode, uniquewcode)
	map(i->types[i], iwcode)
end

function harddecode(x::AbstractMatrix, h::AbstractMatrix{T}) where {T}
	na = size(x, 2)
	d = [hardencodelength(x[:,i]) for i = 1:na]
	ns = size(h, 1)
	s = Matrix{T}(undef, ns, na)
	c = 1
	for i = 1:na
		ce = c + d[i] - 1
		s[:,i] = sum(h[:,c:ce]; dims=2)
		c = ce + 1
	end
	return s
end

function checkcols(x::AbstractMatrix; quiet::Bool=false)
	inans = Vector{Int64}(undef, 0)
	izeros = Vector{Int64}(undef, 0)
	ineg = Vector{Int64}(undef, 0)
	na = size(x, 2)
	for i = 1:na
		isn = .!isnan.(x[:,i])
		if sum(isn) == 0
			!quiet && @info "Column $i has only NaNs!"
			push!(inans, i)
		elseif sum(x[isn, i]) == 0
			!quiet && @info "Column $i has only Zeros!"
			push!(izeros, i)
		elseif any(x[isn, i] .< 0)
			!quiet && @info "Column $i has negative values!"
			push!(ineg, i)
		else

		end
	end
	return inans, izeros, ineg
end

function movingwindow(A::AbstractArray{T, N}, windowsize::Number=1; functionname::String="maximum") where {T, N}
	if windowsize == 0
		return A
	end
	B = similar(A)
	R = CartesianIndices(size(A))
	Istart, Iend = first(R), last(R)
	for I in R
		s = Vector{T}(undef, 0)
		a = max(Istart, I - windowsize * one(I))
		b = min(Iend, I + windowsize * one(I))
		ci = ntuple(i->a[i]:b[i], length(a))
		for J in CartesianIndices(ci)
			push!(s, A[J])
		end
		B[I] = Core.eval(NTFk, Meta.parse(functionname))(s)
	end
	return B
end

function nanmask!(X::Array, mask::Union{Nothing,Number})
	if mask != nothing
		X[X .<= mask] .= NaN
	end
	return nothing
end

function nanmask!(X::Array, mask::BitArray{N}, dim) where {N}
	if length(size(mask)) == length(size(X))
		X[mask] .= NaN
	else
		X[remask(mask, size(X, dim))] .= NaN
	end
	return nothing
end

function nanmask!(X::Array, mask::BitArray{N}) where {N}
	msize = vec(collect(size(mask)))
	xsize = vec(collect(size(X)))
	if length(msize) == length(xsize)
		X[mask] .= NaN
	else
		X[remask(mask, xsize[3:end])] .= NaN
	end
	return nothing
end

function remask(sm, repeats::Integer=1)
	return reshape(repeat(sm, 1, repeats), (size(sm)..., repeats))
end

function remask(sm, repeats::Tuple)
	return reshape(repeat(sm, 1, *(repeats...)), (size(sm)..., repeats...))
end

function remask(sm, repeats::Vector{Int64})
	return reshape(repeat(sm, 1, *(repeats...)), (size(sm)..., repeats...))
end

function bincount(x::Vector; cutoff=0)
	n = unique(sort(x))
	c = map(i->sum(x .== i), n)
	i = sortperm(c; rev=true)
	j = c[i] .> cutoff
	return [n[i][j] c[i][j]]
end