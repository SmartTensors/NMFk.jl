import DocumentFunction
import Statistics
import LinearAlgebra
import Suppressor
import Interpolations

"Test NMFk functions"
function test()
	include(joinpath(dir, "test", "runtests.jl"))
end

"""
Set image dpi

$(DocumentFunction.documentfunction(setdpi))
"""
function setdpi(dpi::Integer)
	global imagedpi = dpi;
end

toupper(x::AbstractString, i=1) = x[1:i-1] * uppercase(x[i:i]) * x[i+1:end]

function r2(x::AbstractVector, y::AbstractVector)
	# rho = Statistics.cov(x, y) / (Statistics.std(x) * Statistics.std(y))
	# r2 = (1 - sum((x .- y).^2) / sum((x .- Statistics.mean(x)).^2))
	ix = .!isnan.(x) .* .!isinf.(x)
	iy = .!isnan.(y) .* .!isinf.(y)
	ii = ix .& iy
	mx = x[ii] .- Statistics.mean(x[ii])
	my = y[ii] .- Statistics.mean(y[ii])
	r2 = (sum(mx .* my) / sqrt(sum((mx .^ 2) * sum(my .^ 2))))^2
	return r2
end

function r2(x::AbstractArray, y::AbstractArray)
	return r2(vec(x), vec(y))
end

function findfirst(v::AbstractVector, func::Function=i->i > 0; zerod::Bool=true, funczerod::Function=isnan)
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

function maximumnan(X::AbstractArray; dims=nothing, func::Function=x->isnan(ismissing(isnothing(x))), kw...)
	if isnothing(dims)
		i = func.(X)
		v = X[.!i]
		m = length(v) > 0 ? maximum(v; kw...) : NaN
	else
		i = func.(X)
		X[i] .= -Inf
		m = maximum(X; dims=dims, kw...)
		X[i] .= NaN
		m[isinf.(m)] .= NaN
	end
	return m
end

function minimumnan(X::AbstractArray; dims=nothing, func::Function=x->isnan(ismissing(isnothing(x))), kw...)
	if isnothing(dims)
		i = func.(X)
		v = X[.!i]
		m = length(v) > 0 ? minimum(v; kw...) : NaN
	else
		i = func.(X)
		X[i] .= Inf
		m = minimum(X; dims=dims, kw...)
		X[i] .= NaN
		m[isinf.(m)] .= NaN
	end
	return m
end

function sumnan(X::AbstractArray; dims=nothing, func::Function=x->isnan(ismissing(isnothing(x))), kw...)
	if isnothing(dims)
		ecount = .*(size(X)...)
		I = func.(X)
		if sum(I) == ecount
			return NaN
		else
			return sum(X[.!I]; kw...)
		end
	else
		ecount = .*(size(X)[vec(collect(dims))]...)
		I = func.(X)
		X[I] .= 0
		sX = sum(X; dims=dims, kw...)
		X[I] .= NaN
		sI = sum(I; dims=dims, kw...)
		sX[sI.==ecount] .= NaN
		return sX
	end
end

function cumsumnan(X::AbstractArray; dims=nothing, func::Function=x->isnan(ismissing(isnothing(x))), kw...)
	if isnothing(dims)
		ecount = .*(size(X)...)
		I = func.(X)
		if sum(I) == ecount
			return NaN
		else
			X[I] .= 0
			sX = cumsum(X; kw...)
			X[I] .= NaN
			return sX
		end
	else
		ecount = .*(size(X)[vec(collect(dims))]...)
		I = func.(X)
		X[I] .= 0
		sX = cumsum(X; dims=dims, kw...)
		X[I] .= NaN
		sI = sum(I; dims=dims, kw...)
		sX[sI.==ecount] .= NaN
		return sX
	end
end

function meannan(X::AbstractArray; dims=nothing, func::Function=x->isnan(ismissing(isnothing(x))), kw...)
	if isnothing(dims)
		ecount = .*(size(X)...)
		I = func.(X)
		if sum(I) == ecount
			return NaN
		else
			return sum(X[.!I]; kw...) / (ecount - sum(I))
		end
	else
		ecount = .*(size(X)[vec(collect(dims))]...)
		I = func.(X)
		X[I] .= 0
		sX = sum(X; dims=dims, kw...)
		X[I] .= NaN
		sI = sum(I; dims=dims, kw...)
		sX[sI.==ecount] .= NaN
		sX ./= sum(I .== 0; dims=dims)
		return sX
	end
end

function varnan(X::AbstractArray; dims=nothing, func::Function=x->isnan(ismissing(isnothing(x))), kw...)
	if isnothing(dims)
		ecount = .*(size(X)...)
		I = func.(X)
		if sum(I) == ecount
			return NaN
		else
			return Statistics.var(X[.!I]; kw...)
		end
	else
		ecount = .*(size(X)[vec(collect(dims))]...)
		I = func.(X)
		X[I] .= 0
		sX = sum(X; dims=dims, kw...)
		sX2 = sum(X .^ 2; dims=dims, kw...)
		X[I] .= NaN
		sI = sum(I; dims=dims, kw...)
		sX[sI.==ecount] .= NaN
		n = sum(I .== 0; dims=dims)
		var = (sX2 - sX .^ 2 ./ n) ./ (n .- 1)
		var[var .< 0] .= 0
		return var
	end
end

function stdnan(X::AbstractArray; dims=nothing, func::Function=x->isnan(ismissing(isnothing(x))), kw...)
	return sqrt.(varnan(X; dims=dims, func=func, kw...))
end

function rmsenan(t::AbstractVector, o::AbstractVector; func::Function=x->isnan(ismissing(isnothing(x))))
	it = .!func.(t)
	ot = .!func.(o)
	ii = it .& ot
	return sqrt( sum( (t[ii] .- o[ii]) .^ 2.) ./ sum(ii) )
end

function l1nan(t::AbstractVector, o::AbstractVector; func::Function=x->isnan(ismissing(isnothing(x))))
	it = .!func.(t)
	ot = .!func.(o)
	ii = it .& ot
	return sum(abs.(t[ii] .- o[ii]))
end

function sortpermnan(v::AbstractVector; func::Function=x->isnan(ismissing(isnothing(x))), rev::Bool=false)
	v2 = sortperm(v; rev=rev)
	it = .!func.(v[v2])
	if rev
		v2 = [v2[it]; v2[.!it]]
	else
		v2 = [v2[.!it]; v2[it]]
	end
	return v2
end

function sortnan(v::AbstractVector; func::Function=x->isnan(ismissing(isnothing(x))), keepnan::Bool=true, kw...)
	it = .!func.(v)
	if keepnan
		v2 = sort(v[it]; kw...)
		v2 = [v2; fill(NaN, sum(.!it))...]
	else
		v2 = sort(v[it]; kw...)
	end
	return v2
end

function ssqrnan(t::AbstractVector, o::AbstractVector; func::Function=x->isnan(ismissing(isnothing(x)))) # Distances.euclidean(x, y)
	it = .!func.(t)
	ot = .!func.(o)
	ii = it .& ot
	return sqrt(sum( (t[ii] .- o[ii]) .^ 2.))
end

function ssqrnan(X::AbstractArray; func::Function=x->isnan(ismissing(isnothing(x))))
	sum(X[.!func.(X)].^2)
end

function normnan(X::AbstractArray; func::Function=x->isnan(ismissing(isnothing(x))))
	LinearAlgebra.norm(X[.!func.(X)])
end

function covnan(x::AbstractArray, y::AbstractArray; func::Function=x->isnan(ismissing(isnothing(x))))
	isn = .!(func.(x) .| func.(y))
	if length(x) > 0 && length(y) > 0 && sum(isn) > 1
		return Statistics.cov(x[isn], y[isn])
	else
		return NaN
	end
end

function cornan(x::AbstractArray, y::AbstractArray; func::Function=x->isnan(ismissing(isnothing(x))))
	isn = .!(func.(x) .| func.(y))
	if length(x) > 0 && length(y) > 0 && sum(isn) > 1
		return Statistics.cor(x[isn], y[isn])
	else
		return NaN
	end
end

function hardencodelength(x::AbstractVector{T}) where {T <: Number}
	u = unique(x)
	i = indexin(x, u)
	inan = indexin(true, isnan.(u))[1]
	d = !isnothing(inan) ? length(u) - 1 : d = length(u)
	return d
end

function hardencode(x::AbstractVector{T}, u::AbstractVector{T}=unique(x)) where {T <: Number}
	i = indexin(x, u)
	inan = indexin(true, isnan.(u))[1]
	d = !isnothing(inan) ? length(u) - 1 : d = length(u)
	m = zeros(length(x), d)
	for (j, k) in enumerate(i)
		if !isnothing(inan)
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

function hardencode(x::AbstractVector{T}, u::AbstractVector{T}=unique(x)) where {T <: Any}
	i = indexin(x, u)
	d = length(u)
	m = zeros(length(x), d)
	for (j, k) in enumerate(i)
		if !isnothing(k)
			m[j, k] = 1
		end
	end
	return m
end

function hardencode(x::AbstractMatrix{T}) where {T <: Number}
	hcat([hardencode(x[:,i]) for i in axes(x, 2)]...)
end

function gettypes(x::AbstractMatrix{T}, levels=[0.05,0.35]) where {T <: Number}
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

function harddecode(x::AbstractMatrix, h::AbstractMatrix{T}) where {T <: Number}
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

function movingwindow(A::AbstractArray{T, N}, windowsize::Number=1; func::Function=meannan) where {T <: Number, N}
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

function nanmask!(X::AbstractArray, mask::Union{Nothing,Number})
	if !isnothing(mask)
		X[X .<= mask] .= NaN
	end
	return nothing
end

function nanmask!(X::AbstractArray{T, N}, mask::BitArray{M}, dim::Integer) where {T, N, M}
	if N == M
		X[mask] .= NaN
	else
		X[remask(mask, size(X, dim))] .= NaN
	end
	return nothing
end

function nanmask!(X::AbstractArray{T, N}, mask::BitArray{M}) where {T, N, M}
	if N == M
		X[mask] .= NaN
	else
		msize = collect(size(mask))
		xsize = collect(size(X))
		ii = indexin(msize, xsize)
		imask = Vector{Int32}(undef, 0)
		for i = 1:N
			if i in ii
				continue
			else
				push!(imask, i)
			end
		end
		X[remask(mask, xsize[imask])] .= NaN
	end
	return nothing
end

function remask(sm, repeats::Integer=1)
	return reshape(repeat(sm, 1, repeats), (size(sm)..., repeats))
end

function remask(sm, repeats::Tuple)
	return reshape(repeat(sm, 1, *(repeats...)), (size(sm)..., repeats...))
end

function remask(sm, repeats::Union{AbstractVector{Int64},AbstractVector{Int32}})
	return reshape(repeat(sm, 1, *(repeats...)), (size(sm)..., repeats...))
end

function bincount(x::AbstractVector; cutoff=0)
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
	return -X .+ NMFk.maximumnan(X) .+ NMFk.minimumnan(X)
end

function estimateflip_permutedims(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, nNNF=10; save=false, method=:ipopt, regularizationweight=1e-8, kw...) where {T <: Number}
	@assert size(X, 2) == size(Y, 2)
	@assert size(B, 2) == size(A, 2)
	@assert size(X, 1) == size(A, 1)
	@assert size(Y, 1) == size(B, 1)
	nparam = size(X, 2)
	nk = size(X, 1)
	vflip = falses(nparam)
	for i = 1:nparam
		local H1
		Suppressor.@suppress W, H1, of, sil, aic = NMFk.execute(permutedims(Y[:,i]), nk, nNNF; Winit=permutedims(X[:,i]), Wfixed=true, save=save, method=method, regularizationweight=regularizationweight, kw...);
		a = NMFk.normnan(permutedims(A) .- (permutedims(B) * H1))
		local H2
		Suppressor.@suppress W, H2, of, sil, aic = NMFk.execute(permutedims(NMFk.flip(Y[:,i])), nk, nNNF; Winit=permutedims(NMFk.flip(X[:,i])), Wfixed=true, save=save, method=method, regularizationweight=regularizationweight, kw...);
		b = NMFk.normnan(permutedims(A) .- (permutedims(B) * H2))
		vflip[i] = a < b ? false : true
	end
	return vflip
end

function estimateflip(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, nNNF=10; save=false, method=:ipopt, regularizationweight=1e-8, kw...) where {T <: Number}
	@assert size(X, 1) == size(Y, 1)
	@assert size(B, 1) == size(A, 1)
	@assert size(X, 2) == size(A, 2)
	@assert size(Y, 2) == size(B, 2)
	nparam = size(X, 1)
	nk = size(X, 2)
	vflip = falses(nparam)
	for i = 1:nparam
		local H1
		Suppressor.@suppress W, H1, of, sil, aic = NMFk.execute(Y[i:i,:], nk, nNNF; Winit=X[i:i,:], Wfixed=true, save=save, method=method, regularizationweight=regularizationweight, kw...);
		a = NMFk.normnan(B .- (A * H1))
		local H2
		Suppressor.@suppress W, H2, of, sil, aic = NMFk.execute(NMFk.flip(Y[i:i,:]), nk, nNNF; Winit=NMFk.flip(X[i:i,:]), Wfixed=true, save=save, method=method, regularizationweight=regularizationweight, kw...);
		b = NMFk.normnan(B .- (A * H2))
		vflip[i] = a < b ? false : true
	end
	return vflip
end

function flatten(X::AbstractArray{T,N}, mask::BitArray{M}) where {T,N,M}
	@assert N - 1 == M
	sz = size(X)
	A = Matrix{T}(undef, sum(.!mask), sz[end])
	for i = 1:sz[end]
		nt = ntuple(k->(k == N ? i : Colon()), N)
		A[:, i] = X[nt...][.!mask]
	end
	return A
end

function flatten(X::AbstractArray{T,N}, dim::Number=1) where {T <: Number, N}
	sz = size(X)
	nt = Vector{Int64}(undef, 0)
	for k = 1:N
		if (k != dim)
			push!(nt, k)
		end
	end
	A = Matrix{T}(undef, *(sz[nt]...), sz[dim])
	for i = 1:sz[dim]
		nt = ntuple(k->(k == dim ? i : Colon()), N)
		A[:, i] = vec(X[nt...])
	end
	return A
end

function flattenindex(X::AbstractArray{T,N}, dim::Number=1; order=[1,2]) where {T <: Number, N}
	sz = size(X)
	nt = Vector{Int64}(undef, 0)
	for k = 1:N
		if (k != dim)
			push!(nt, k)
		end
	end
	if order == [1,2]
		I = repeat(1:sz[nt[1]], sz[nt[2]])
	elseif order == [2,1]
		I = sort(repeat(1:sz[nt[2]], sz[nt[1]]))
		# I = [repeat([i], sz[nt[1]]) for i=1:sz[nt[2]]]
	else
		error("Order must be [1,2] or [2,1]")
	end
	return I
end

function stringfix(str::AbstractString)
	replace(str, '&' => "&amp;", '(' => "[", ')' => "]", '<' => "≤", '–' => "-")
end

function remap(v::AbstractVector, vi::Union{AbstractVector,AbstractUnitRange,StepRangeLen}, ve::Union{AbstractVector,AbstractUnitRange,StepRangeLen}; nonneg::Bool=true, sp=[Interpolations.Gridded(Interpolations.Linear())], ep=[Interpolations.Line(Interpolations.OnGrid())])
	lv = length(v)
	li = length(vi)
	@assert lv == li
	f1 = Vector{Float64}(undef, li)
	isn = .!isnan.(v)
	itp = Interpolations.interpolate((vi[isn],), v[isn], sp...)
	etp = Interpolations.extrapolate(itp, ep...)
	f1 = etp.(ve)
	if nonneg
		f1[f1.<0] .= 0
	end
	return f1
end

function stringproduct(a::AbstractVector, b::AbstractVector)
	M = Matrix{String}(undef, length(a), length(b))
	for i = eachindex(a)
		for j = eachindex(b)
			M[i, j] = string(a[i]) * ":" * string(b[j])
		end
	end
	return M
end

function zerostoepsilon(X::AbstractArray)
	Xn = copy(X)
	zerostoepsilon!(Xn)
	return Xn
end

function zerostoepsilon!(X::AbstractArray)
	if eltype(X) <: Real
		e = eps(eltype(X)) ^ 2
		X[X .< e] .= e
	else
		@warn("Provided data are not numeric!")
	end
	return nothing
end

function aisnan(X::AbstractArray)
	Xn = copy(X)
	aisnan!(Xn)
	return Xn
end

function aisnan!(X::AbstractArray, l=1)
	X[isnan.(X)] .= l
	return nothing
end

"Convert `@Printf.sprintf` macro into `sprintf` function"
sprintf(args...) = eval(:@Printf.sprintf($(args...)))