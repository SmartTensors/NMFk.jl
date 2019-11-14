import DocumentFunction
import Statistics
import LinearAlgebra
import Suppressor

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

function findfirst(v::Vector, func::Function=i->i > 0; zerod::Bool=true, funczerod::Function=isnan)
	if zerod
		i = funczerod.(v)
		v[i] .= 0
	end
	vi = Base.findfirst(func, v)
	if zerod
		i = isnan.(v)
		v[i] .= NaN
	end
	return vi
end

function maximumnan(X::AbstractArray; dims=nothing, func::Function=isnan, kw...)
	if dims == nothing
		i = func.(X)
		m = maximum(X[.!i]; kw...)
	else
		i = func.(X)
		X[i] .= 0
		m = maximum(X; dims=dims, kw...)
		X[i] .= NaN
	end
	return m
end

function minimumnan(X::AbstractArray; dims=nothing, func::Function=isnan, kw...)
	if dims == nothing
		i = func.(X)
		m = minimum(X[.!i]; kw...)
	else
		i = func.(X)
		X[i] .= Inf
		m = minimum(X; dims=dims, kw...)
		X[i] .= NaN
	end
	return m
end

function sumnan(X::AbstractArray; dims=nothing, kw...)
	if dims == nothing
		return sum(X[.!isnan.(X)]; kw...)
	else
		ecount = .*(size(X)[vec(collect(dims))]...)
		I = isnan.(X)
		X[I] .= 0
		sX = sum(X; dims=dims, kw...)
		X[I] .= NaN
		sI = sum(I; dims=dims, kw...)
		sX[sI.==ecount] .= NaN
		return sX
	end
end

function meannan(X::AbstractArray)
	Statistics.mean(X[.!isnan.(X)])
end

function ssqrnan(X::AbstractArray)
	sum(X[.!isnan.(X)].^2)
end

function normnan(X::AbstractArray)
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

function movingwindow(A::AbstractArray{T, N}, windowsize::Number=1; func::Function=maximum) where {T, N}
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
		B[I] = func(s)
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

function flip!(X)
	X = -X .+ NMFk.maximumnan(X) .+ NMFk.minimumnan(X)
end

function flip(X)
	-X .+ NMFk.maximumnan(X) .+ NMFk.minimumnan(X)
end

function estimateflip(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, nNNF=10; save=false, method=:ipopt, regularizationweight=1e-8, kw...) where T
	@assert size(X, 2) == size(Y, 2)
	@assert size(A, 2) == size(B, 2)
	@assert size(X, 1) == size(B, 1)
	@assert size(Y, 1) == size(A, 1)
	nparam = size(X, 2)
	nk = size(X, 1)
	vflip = falses(nparam)
	for i = 1:nparam
		local H1
		@Suppressor.suppress W, H1, of, sil, aic = NMFk.execute(permutedims(Y[:,i]), nk, nNNF; Winit=permutedims(X[:,i]), Wfixed=true, save=save, method=method, regularizationweight=regularizationweight, kw...);
		a = Statistics.norm(permutedims(A) .- (permutedims(B) * H1))
		local H2
		@Suppressor.suppress W, H2, of, sil, aic = NMFk.execute(permutedims(NMFk.flip(Y[:,i])), nk, nNNF; Winit=permutedims(NMFk.flip(X[:,i])), Wfixed=true, save=save, method=method, regularizationweight=regularizationweight, kw...);
		b = Statistics.norm(permutedims(A) .- (permutedims(B) * H2))
		vflip[i] = a < b ? false : true
	end
	return vflip
end