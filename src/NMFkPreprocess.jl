import Dates
import DataFrames
import Statistics
import StatsBase

function log10s(x::AbstractArray; kw...)
	NMFk.log10s!(copy(x); kw...)
end

function log10s!(x::AbstractArray; offset::Number=1)
	iz = x .<= 0
	siz = sum(iz)
	siz > 0 && (x[iz] .= NaN)
	x .= log10.(x)
	if siz > 0
		min = minimumnan(x[.!iz]) - offset
		x[iz] .= min
	end
	return x
end

function datanalytics(v::AbstractVector; plothistogram::Bool=true, kw...)
	ig = .!isnan.(v)
	vn = v[ig]
	if length(vn) > 0
		plothistogram && NMFk.histogram(vn; kw...)
		return minimum(vn), maximum(vn), Statistics.std(vn), StatsBase.skewness(vn), sum(ig)
	else
		plothistogram && @warn("No data!")
		return -Inf, Inf, Inf, 0, 0
	end
end

function datanalytics(a::AbstractMatrix; dims::Integer=1, kw...)
	name = dims == 1 ? "Row" : "Column"
	names = ["$name $i" for i = 1:size(a, dims)]
	datanalytics(a, names; dims=dims, kw...)
end

function datanalytics(a::AbstractMatrix{T}, names::AbstractVector; dims::Integer=1, quiet::Bool=false, veryquiet::Bool=quiet, log::Bool=false, logv::AbstractVector=fill(log, length(names)), casefilename::AbstractString="", kw...) where {T <: Number}
	@assert length(names) == length(logv)
	@assert length(names) == size(a, dims)
	min = Vector{T}(undef, length(names))
	max = Vector{T}(undef, length(names))
	std = Vector{T}(undef, length(names))
	skewness = Vector{T}(undef, length(names))
	count = Vector{Int64}(undef, length(names))
	for (i, n) in enumerate(names)
		nt = ntuple(k->(k == dims ? i : Colon()), ndims(a))
		if logv[i]
			!veryquiet && @info("$n: log10-transformed")
			v = log10s(vec(a[nt...]))
		else
			!veryquiet && @info n
			v = vec(a[nt...])
		end
		if casefilename == ""
			filename = ""
		else
			if splitdir(casefilename)[end] == ""
				filename = casefilename * "histogram-$(n).png"
			else
				filename = casefilename * "-$(n).png"
			end
		end
		min[i], max[i], std[i], skewness[i], count[i] = datanalytics(v; filename=filename, kw..., title=n)
		!veryquiet && println("$n: Min $(min[i]) Max $(max[i]) StdDev $(std[i]) Skewness $(skewness[i]) Count $(count[i])")
	end
	if !quiet
		@info "Attributes"
		println("Name Min Max StdDev Count (non-NaN's)")
		for (i, n) in enumerate(names)
			println("$n $(min[i]) $(max[i]) $(std[i]) $(skewness[i]) $(count[i])")
		end
	end
	return min, max, std, skewness, count
end

function indicize(v::AbstractVector; rev::Bool=false, nbins::Integer=length(v), minvalue::Number=minimum(v), maxvalue::Number=maximum(v), stepvalue=nothing, granulate::Bool=true, quiet::Bool=false)
	if stepvalue !== nothing
		if granulate && !quiet
			@info("Initial: $minvalue $maxvalue")
		end
		if typeof(minvalue) <: Dates.DateTime
			if granulate
				maxvalue = ceil(maxvalue, stepvalue)
				minvalue = floor(minvalue, stepvalue)
			end
			nbins = convert(Int, (maxvalue - minvalue) / convert(Dates.Millisecond, stepvalue))
		elseif typeof(minvalue) <: Dates.Date
			if granulate
				maxvalue = ceil(maxvalue, stepvalue)
				minvalue = floor(minvalue, stepvalue)
			end
			nbins = -1
			date = minvalue
			while date <= maxvalue
				date += stepvalue
				nbins += 1
			end
		else
			if granulate
				maxvalue = ceil(maxvalue / stepvalue) * stepvalue
				minvalue = floor(minvalue / stepvalue) * stepvalue
			end
			nbins = convert(Int, ceil((maxvalue - minvalue) / float(stepvalue)))
		end
		if granulate && !quiet
			@info("Granulated: $minvalue $maxvalue")
		end
	end
	iv = convert(Vector{Int64}, ceil.((v .- minvalue) ./ (maxvalue - minvalue) .* nbins))
	i0 = iv .== 0
	if sum(i0) == 1
		iv[i0] .= 1
	elseif sum(i0) > 1
		iv .+= 1
	end
	if !quiet
		us = unique(sort(iv))
		nb = collect(1:nbins)
		for k in unique(sort([us; nb]))
			m = iv .== k
			s = sum(m)
			if s == 0
				@info("Bin $(lpad("$k", 3, " ")): count $(lpad("$(s)", 6, " "))")
			else
				@info("Bin $(lpad("$k", 3, " ")): count $(lpad("$(s)", 6, " ")) range $(minimum(v[m])) $(maximum(v[m]))")
			end
		end
		if length(us) != nbins
			@warn "There are empty bins ($(length(us)) vs $(nbins))"
		end
	end
	if rev == true
		iv = (nbins + 1) .- iv
	end
	@assert minimum(iv) >= 1
	@assert maximum(iv) <= nbins
	return iv, nbins, minvalue, maxvalue
end

function processdata(M::AbstractArray, type::DataType=Float32; kw...)
	Mn = processdata!(copy(M); kw...)
	Mn = convert(Array{type}, convert.(type, Mn))
	return Mn
end

function processdata!(M::AbstractArray; nanstring::AbstractString="NaN", negative::Bool=true)
	M[ismissing.(M)] .= NaN
	M[M .== ""] .= NaN
	M[M .== nanstring] .= NaN
	if !negative
		M[M .< 0] .= 0
	end
	return M
end

function griddata(x::AbstractVector, y::AbstractVector; stepvalue=nothing, nbins=nothing, xrev::Bool=false, xnbins::Integer=length(x), xminvalue=minimum(x), xmaxvalue=maximum(x), xstepvalue=stepvalue, yrev::Bool=false, ynbins=length(y), yminvalue=minimum(y), ymaxvalue=maximum(y), ystepvalue=stepvalue, granulate::Bool=true, quiet::Bool=true)
	if nbins !== nothing
		xnbins = nbins
		ynbins = nbins
	end
	ix, xbins, gxmin, gxmax = NMFk.indicize(x; rev=xrev, nbins=xnbins, minvalue=xminvalue, maxvalue=xmaxvalue, stepvalue=xstepvalue, granulate=granulate, quiet=quiet)
	iy, ybins, gymin, gymax = NMFk.indicize(y; rev=yrev, nbins=ynbins, minvalue=yminvalue, maxvalue=ymaxvalue, stepvalue=ystepvalue, granulate=granulate, quiet=quiet)
	return range(gxmin; stop=gxmax, length=xbins), range(gymin; stop=gymax, length=ybins)
end

function griddata(x::AbstractVector, y::AbstractVector, z::AbstractVector; kw...)
	return griddata(x, y, reshape(z, (length(z), 1)); kw...)
end

function griddata(x::AbstractVector, y::AbstractVector, z::AbstractMatrix; type::DataType=eltype(z), xrev::Bool=false, xnbins::Integer=length(x), xminvalue=minimum(x), xmaxvalue=maximum(x), xstepvalue=nothing, yrev::Bool=false, ynbins=length(y), yminvalue=minimum(y), ymaxvalue=maximum(y), ystepvalue=nothing, granulate::Bool=true, quiet::Bool=true)
	@assert length(x) == length(y)
	@assert length(x) == size(z, 1)
	zn = processdata(z, type)
	ix, xbins, gxmin, gxmax = NMFk.indicize(x; rev=xrev, nbins=xnbins, minvalue=xminvalue, maxvalue=xmaxvalue, stepvalue=xstepvalue, granulate=granulate, quiet=quiet)
	iy, ybins, gymin, gymax = NMFk.indicize(y; rev=yrev, nbins=ynbins, minvalue=yminvalue, maxvalue=ymaxvalue, stepvalue=ystepvalue, granulate=granulate, quiet=quiet)
	T = Array{type}(undef, xbins, ybins, size(z, 2))
	C = Array{Int32}(undef, xbins, ybins, size(z, 2))
	T .= 0
	C .= 0
	for i = 1:size(z, 2)
		for j = 1:length(ix)
			if !isnan(zn[j, i])
				T[ix[j], iy[j], i] += zn[j, i]
				C[ix[j], iy[j], i] += 1
			end
		end
	end
	@info("Maximum number of data overlaps $(maximum(C))")
	T ./= C
	return T
end

function bincoordinates(v::AbstractVector; rev::Bool=false, nbins=length(v), minvalue=minimum(v), maxvalue=maximum(v), stepvalue=nothing)
	if stepvalue !== nothing
		if typeof(minvalue) <: Dates.DateTime
			maxvalue = ceil(maxvalue, stepvalue)
			minvalue = floor(minvalue, stepvalue)
			nbins = convert(Int, (maxvalue - minvalue) / convert(Dates.Millisecond, stepvalue))
		elseif typeof(minvalue) <: Dates.Date
			maxvalue = ceil(maxvalue, stepvalue)
			minvalue = floor(minvalue, stepvalue)
			nbins = convert(Int, (maxvalue - minvalue) / Core.eval(Main, Meta.parse(stepvalue))(1))
		else
			granularity = -convert(Int, ceil(log10(stepvalue)))
			maxvalue = ceil(maxvalue, granularity)
			minvalue = floor(minvalue, granularity)
			nbins = convert(Int, ceil.((maxvalue - minvalue) / float(stepvalue)))
		end
	end
	if typeof(minvalue) <: Dates.DateTime || typeof(minvalue) <: Dates.Date
		stepv = (maxvalue - minvalue) / float(nbins)
		halfstepv = stepv / float(2)
		vs = collect(Base.range(minvalue + halfstepv, maxvalue - halfstepv; step=stepv))
	else
		halfstepv = (maxvalue - minvalue) / (2 * nbins)
		vs = collect(Base.range(minvalue + halfstepv, maxvalue - halfstepv; length=nbins))
	end
	if rev == true
		vs = reverse(vs)
	end
	return vs
end

function remap(v::AbstractVector{T}, mapping::Vector; func::Function=!isnothing) where {T <: Number}
	o = Vector{T}(undef, length(mapping))
	o .= NaN
	if typeof(T) <: Integer
		o .= 0
	else
		o .= NaN
	end
	i = func.(mapping)
	o[i] .= v[mapping[i]]
	return o
end

function remap(v::AbstractMatrix{T}, mapping::Vector; func::Function=!isnothing) where {T <: Number}
	o = Array{T}(undef, length(mapping), size(v, 2))
	o .= NaN
	if typeof(T) <: Integer
		o .= 0
	else
		o .= NaN
	end
	i = func.(mapping)
	o[i, :] .= v[mapping[i], :]
	return o
end

function slopes(v::AbstractVector)
	s = similar(v)
	s[1] = v[2] - v[1]
	s[end] = v[end] - v[end-1]
	for i = 2:length(v)-1
		s[i] = (v[i+1] - v[i-1]) ./ 2
	end
	return s
end

function getdatawindow(X::Array{T,N}, d::Integer; func::Function=i->i>0, funcfirst::Function=func, funclast::Function=func, start::Vector{Int64}=Vector{Int64}(undef, 0)) where {T <: Number, N}
	@assert d >= 1 && d <= N
	dd = size(X, d)
	if length(start) > 0
		@assert length(start) == dd
		endd = size(X)
	end
	afirstentry = Vector{Int64}(undef, dd)
	alastentry = Vector{Int64}(undef, dd)
	datasize = Vector{Int64}(undef, dd)
	for i = 1:dd
		if length(start) > 0 && start[i] > 0
			nt = ntuple(k->(k == d ? i : Base.Slice(start[i]:endd[k])), N)
		else
			nt = ntuple(k->(k == d ? i : Colon()), N)
		end
		firstentry = Base.findfirst(funcfirst.(X[nt...]))
		if firstentry !== nothing
			afirstentry[i] = firstentry
			lastentry = findlast(funclast.(X[nt...]))
			if lastentry !== nothing
				datasize[i] = lastentry - firstentry + 1
				alastentry[i] = lastentry
			else
				alastentry[i] = datasize[i] = 0
			end
		else
			afirstentry[i] = alastentry[i] = datasize[i] = 0
		end
	end
	return afirstentry, alastentry, datasize
end

function shiftarray(X::Array{T,N}, d::Integer, start::Vector{Int64}, finish::Vector{Int64}, datasize::Vector{Int64}) where {T <: Number, N}
	@assert d >= 1 && d <= N
	dd = size(X, d)
	@assert length(start) == dd
	@assert length(finish) == dd
	@assert length(datasize) == dd
	Y = Array{T}(undef, maximum(datasize), dd)
	Y .= NaN
	for i = 1:dd
		nty = ntuple(k->(k == d ? i : Base.Slice(1:datasize[i])), N)
		ntx = ntuple(k->(k == d ? i : Base.Slice(start[i]:finish[i])), N)
		Y[nty...] = X[ntx...]
	end
	return Y
end

"""
Extract a matrix from a dataframe
"""
function df2matrix(df::DataFrames.DataFrame, id::Vector, dates::Union{StepRange{Dates.Date,Dates.Month},Array{Dates.Date,1}}, dfattr::Symbol, dfdate::Symbol=:ReportDate, dfapi::Symbol=:API; addup::Bool=false, checkzero::Bool=true)
	nw = length(id)
	matrix = Array{Float32}(undef, length(dates), nw)
	matrix .= NaN32
	fwells = falses(nw)
	local k = 0
	for (i, w) in enumerate(id)
		iwell = df[!, dfapi] .== w
		attr = df[!, dfattr][iwell]
		innattr = .!isnan.(attr)
		welldates = df[!, dfdate][iwell][innattr]
		iwelldates = indexin(welldates, dates)
		iwelldates3 = .!isnothing.(iwelldates)
		if sum(iwelldates3) != 0 && (checkzero == false || sum(attr[innattr][iwelldates3]) > 0)
			fwells[i] = true
			k += 1
			!addup && (c = zeros(length(attr[innattr][iwelldates3])))
			matrix[iwelldates[iwelldates3], k] .= 0
			for (a, b) in enumerate(attr[innattr][iwelldates3])
				matrix[iwelldates[iwelldates3][a], k] += b
				!addup && (c[a] += 1)
			end
			!addup && (matrix[iwelldates[iwelldates3], k] ./= c)
		end
	end
	return matrix, fwells
end

"""
Extract a time shifted matrix from a dataframe
"""
function df2matrix_shifted(df::DataFrames.DataFrame, id::AbstractVector, dates::Union{StepRange{Dates.Date,Dates.Month},Array{Dates.Date,1}}, dfattr::Symbol, dfdate::Symbol=:ReportDate, dfapi::Symbol=:API; kw...)
	matrix, startdates, enddates = df2matrix_shifted(df, id, length(dates), dates, dfattr, dfdate, dfapi; kw...)
	recordlength = findlast(i->!isnan(i), NMFk.sumnan(matrix; dims=2))[1]
	matrixn = Array{Float32}(undef, recordlength, size(matrix, 2))
	matrixn .= matrix[1:recordlength, :]
	return matrixn, startdates, enddates
end
function df2matrix_shifted(df::DataFrames.DataFrame, id::AbstractVector, recordlength::Integer, dates::Union{StepRange{Dates.Date,Dates.Month},Array{Dates.Date,1}}, dfattr::Symbol, dfdate::Symbol=:ReportDate, dfapi::Symbol=:API; addup::Bool=false, checkzero::Bool=true)
	nw = length(id)
	matrix = Array{Float32}(undef, recordlength, nw)
	matrix .= NaN32
	startdates = Array{Dates.Date}(undef, nw)
	enddates = Array{Dates.Date}(undef, nw)
	for (i, w) in enumerate(id)
		iwell = df[!, dfapi] .== w
		attr = df[!, dfattr][iwell]
		innattr = .!isnan.(attr)
		welldates = df[!, dfdate][iwell][innattr]
		isortedwelldates = sortperm(welldates)
		iwelldates = indexin(welldates[isortedwelldates], dates)
		iwelldates3 = .!isnothing.(iwelldates)
		sumattr = sum(attr[innattr][isortedwelldates][iwelldates3])
		if checkzero && sumattr > 0
			iattrfirst = Base.findfirst(i->i>0, attr[innattr][isortedwelldates][iwelldates3])
			iattrlast = findlast(i->i>0, attr[innattr][isortedwelldates][iwelldates3])
		else
			if sumattr == 0
				@warn("Well $w: zero total ($(string(dfattr))) production!")
			end
			iattrfirst = Base.findfirst(i->!isnan(i), attr[innattr][isortedwelldates][iwelldates3])
			iattrlast = findlast(i->!isnan(i), attr[innattr][isortedwelldates][iwelldates3])
		end
		startdates[i] = welldates[isortedwelldates][iwelldates3][iattrfirst]
		enddates[i] = welldates[isortedwelldates][iwelldates3][iattrlast]
		iwelldates2 = iwelldates[iwelldates3][iattrfirst:iattrlast] .- iwelldates[iwelldates3][iattrfirst] .+ 1
		matrix[iwelldates2, i] .= 0
		!addup && (c = zeros(length(iattrfirst:iattrlast)))
		for (a, b) in enumerate(iattrfirst:iattrlast)
			matrix[iwelldates2[a], i] += attr[innattr][isortedwelldates][b]
			!addup && (c[a] += 1)
		end
		!addup && (matrix[iwelldates2, i] ./= c)
		de = length(iwelldates2) - length(unique(iwelldates2))
		er = abs(NMFk.sumnan(matrix[:, i]) - sumattr) ./ sumattr > eps(Float32)
		if de > 0 && er
			@info("Well $w: $(de) duplicate production entries")
			@info("Original  total production: $(sumattr)")
			@info("Processed total production: $(NMFk.sumnan(matrix[:, i]))")
		end
		if (addup || de == 0) && er
			@warn("Well $w (column $i): something is potentially wrong!")
			@info("Original  total production: $(sumattr)")
			@info("Processed total production: $(NMFk.sumnan(matrix[:, i]))")
			@show sum(matrix[:, i] .> 0)
			@show matrix[iwelldates2, i]

			@show sum(attr[innattr][isortedwelldates] .> 0)
			@show length(attr[innattr][isortedwelldates][iwelldates3])
			@show attr[innattr][isortedwelldates][iwelldates3]

			@show length(iwelldates2)
			@show iwelldates2
			@show length(welldates[isortedwelldates][iwelldates3])
			@show welldates[isortedwelldates][iwelldates3]
			@show iattrfirst:iattrlast

			# matrix[iwelldates2, i] .= 0
			# for (a, b) in enumerate(iattrfirst:iattrlast)
			# 	@show (a, b)
			# 	@show welldates[isortedwelldates][iwelldates3][a]
			# 	@show iwelldates2[a]
			# 	@show attr[innattr][isortedwelldates][b]
			# 	matrix[iwelldates2[a], i] += attr[innattr][isortedwelldates][b]
			# 	@show matrix[iwelldates2[a], i]
			# end
			# @show NMFk.sumnan(matrix[:, i])
			# @show sum(matrix[:, i] .> 0)
		end
	end
	return matrix, startdates, enddates
end

function moving_average(v::AbstractVector, window::Integer=3)
	wback = div(window, 2)
	wforw = isodd(window) ? div(window, 2) : div(window, 2) - 1
	lv = length(v)
	vs = similar(v)
	for i = 1:lv
		lo = max(1, i - wback)
		hi = min(lv, i + wforw)
		vs[i] = Statistics.mean(v[lo:hi])
	end
	return vs
end

function moving_average(m::AbstractMatrix, window::Integer=3; dims::Integer=2)
	ms = similar(m)
	for i = 1:size(m, dims)
		ms[:,i] = moving_average(m[:,i], window)
	end
	return ms
end