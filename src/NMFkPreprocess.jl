import Dates
import DataFrames
import Statistics

function log10s(v::AbstractVector; offset::Number=1)
	iz = v .<= 0
	siz = sum(iz)
	vm = copy(v)
	siz > 0 && (vm[iz] .= NaN)
	vm .= log10.(vm)
	if siz > 0
		m = minimumnan(vm[.!iz])
		vm[iz] .= m - offset
	end
	return vm
end

function datanalytics(v::AbstractVector; kw...)
	ig = .!isnan.(v)
	vn = v[ig]
	NMFk.histogram(vn; kw...)
	return minimum(vn), maximum(vn), Statistics.std(vn), sum(ig)
end

function datanalytics(a::AbstractMatrix; dims::Integer=1, kw...)
	name = dims == 1 ? "Row" : "Column"
	names = ["$name $i" for i = 1:size(a, dims)]
	datanalytics(a, names; dims=dims, kw...)
end

function datanalytics(a::AbstractMatrix{T}, names::AbstractVector; dims::Integer=1, log::Bool=false, logv::AbstractVector=fill(log, length(names)), casefilename::AbstractString="", kw...) where T
	@assert length(names) == length(logv)
	@assert length(names) == size(a, dims)
	min = Vector{T}(undef, length(names))
	max = Vector{T}(undef, length(names))
	std = Vector{T}(undef, length(names))
	c = Vector{Int64}(undef, length(names))
	for (i, n) in enumerate(names)
		nt = ntuple(k->(k == dims ? i : Colon()), ndims(a))
		if logv[i]
			@info("$n: log10-transformed")
			v = log10s(vec(a[nt...]))
		else
			@info n
			v = vec(a[nt...])
		end
		filename = casefilename == "" ? "" : casefilename * "-$(n).png"
		min[i], max[i], std[i], c[i] = datanalytics(v; filename=filename, kw...)
		@info "$n, $(min[i]), $(max[i]), $(std[i]), $(c[i])"
		println()
	end
	@info "Attributes"
	println("Name Min Max StdDev Count (non-NaN's")
	for (i, n) in enumerate(names)
		println("$n $(min[i]) $(max[i]) $(std[i]) $(c[i])")
	end
end

function indicize(v::AbstractVector; rev::Bool=false, nbins::Integer=length(v), minvalue::Number=minimum(v), maxvalue::Number=maximum(v), stepvalue=nothing, granulate::Bool=true)
	if stepvalue != nothing
		if granulate
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
		if granulate
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
	if rev == true
		iv = (nbins + 1) .- iv
	end
	@assert minimum(iv) >= 1
	@assert maximum(iv) <= nbins
	return iv, nbins, minvalue, maxvalue
end

function processdata(M::AbstractArray, at...; kw...)
	processdata!(copy(M), at...; kw...)
end

function processdata!(M::AbstractArray, type::DataType=Float32; nanstring::AbstractString="NaN", negative::Bool=true)
	M[ismissing.(M)] .= NaN
	M[M .== ""] .= NaN
	M[M .== nanstring] .= NaN
	if !negative
		M[M .< 0] .= 0
	end
	M .= convert.(type, M)
	M = convert(Array{type}, M)
	return M
end

function griddata(x::AbstractVector, y::AbstractVector, z::AbstractVector; kw...)
	return griddata(x, y, reshape(z, (length(z), 1)))
end

function griddata(x::AbstractVector, y::AbstractVector, z::AbstractMatrix; type::DataType=eltype(z), xrev::Bool=false, xnbins::Integer=length(x), xminvalue=minimum(x), xmaxvalue=maximum(x), xstepvalue=nothing, yrev::Bool=false, ynbins=length(y), yminvalue=minimum(y), ymaxvalue=maximum(y), ystepvalue=nothing, granulate::Bool=true)
	@assert length(x) == length(y)
	@assert length(x) == size(z, 1)
	zn = processdata(z, type)
	ix, xbins, gxmin, gxmax = NMFk.indicize(x; rev=xrev, nbins=xnbins, minvalue=xminvalue, maxvalue=xmaxvalue, stepvalue=xstepvalue, granulate=granulate)
	iy, ybins, gymin, gymax = NMFk.indicize(y; rev=yrev, nbins=ynbins, minvalue=yminvalue, maxvalue=ymaxvalue, stepvalue=ystepvalue, granulate=granulate)
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
	if stepvalue != nothing
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

function remap(v::AbstractVector{T}, mapping::Vector; func::Function=!isnothing) where {T}
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

function remap(v::AbstractMatrix{T}, mapping::Vector; func::Function=!isnothing) where {T, N}
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

function getdatawindow(X::Array{T,N}, d::Integer; func::Function=i->i>0, funcfirst::Function=func, funclast::Function=func, start::Vector{Int64}=Vector{Int64}(undef, 0)) where {T, N}
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
		if firstentry != nothing
			afirstentry[i] = firstentry
			lastentry = findlast(funclast.(X[nt...]))
			if lastentry != nothing
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

function shiftarray(X::Array{T,N}, d::Integer, start::Vector{Int64}, finish::Vector{Int64}, datasize::Vector{Int64}) where {T, N}
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
function df2matrix(df::DataFrames.DataFrame, id::Vector, dfattr::Symbol, dfdate::Symbol, dates::Union{StepRange{Dates.Date,Dates.Month},Array{Dates.Date,1}}; checkzero::Bool=true)
	nw = length(id)
	matrix = Array{Float32}(undef, length(dates), nw)
	matrix .= NaN32
	fwells = falses(nw)
	global k = 0
	for (i, w) in enumerate(id)
		iwell = findall((in)(w), df[!, :API])
		attr = df[!, dfattr][iwell]
		innattr = .!isnan.(attr)
		welldates = df[!, dfdate][iwell][innattr]
		iwelldates = indexin(welldates, dates)
		iwelldates3 = .!isnothing.(iwelldates)
		if sum(iwelldates3) != 0 && (checkzero==false || sum(attr[innattr][iwelldates3]) > 0)
			fwells[i] = true
			global k += 1
			matrix[iwelldates[iwelldates3], k] .= 0
			for (a, b) in enumerate(attr[innattr][iwelldates3])
				matrix[iwelldates[iwelldates3][a], k] += b
			end
		end
	end
	return matrix, fwells
end

"""
Extract a time shifted matrix from a dataframe
"""
function df2matrix_shifted(df::DataFrames.DataFrame, id::Vector, dfattr::Symbol, dfdate::Symbol, dates::Union{StepRange{Dates.Date,Dates.Month},Array{Dates.Date,1}}; checkzero::Bool=true)
	nw = length(id)
	matrix = Array{Float32}(undef, length(dates), nw)
	matrix .= NaN32
	startdates = Array{Dates.Date}(undef, nw)
	enddates = Array{Dates.Date}(undef, nw)
	for (i, w) in enumerate(id)
		iwell = findall((in)(w), df[!, :API])
		attr = df[!, dfattr][iwell]
		innattr = .!isnan.(attr)
		welldates = df[!, dfdate][iwell][innattr]
		isortedwelldates = sortperm(welldates)
		iwelldates = indexin(welldates[isortedwelldates], dates)
		iwelldates3 = .!isnothing.(iwelldates)
		if checkzero
			iattrfirst = Base.findfirst(i->i>0, attr[innattr][isortedwelldates][iwelldates3])
			iattrlast = findlast(i->i>0, attr[innattr][isortedwelldates][iwelldates3])
		else
			iattrfirst = Base.findfirst(i->i>=0, attr[innattr][isortedwelldates][iwelldates3])
			iattrlast = findlast(i->i>=0, attr[innattr][isortedwelldates][iwelldates3])
		end
		startdates[i] = welldates[isortedwelldates][iwelldates3][iattrfirst]
		enddates[i] = welldates[isortedwelldates][iwelldates3][iattrlast]
		iwelldates2 = iwelldates[iwelldates3][iattrfirst:end] .- iwelldates[iwelldates3][iattrfirst] .+ 1
		matrix[iwelldates2, i] .= 0
		for (a, b) in enumerate(iattrfirst:length(attr[innattr][isortedwelldates][iwelldates3]))
			matrix[iwelldates2[a], i] += attr[innattr][isortedwelldates][b]
		end
		if checkzero==true && (NMFk.sumnan(matrix[:, i]) == 0 || sum(matrix[:, i]) == NaN32)
			@show i
			@show w
			@show attr
			@show welldates
			@show iattrfirst iattrlast
			@show attr[innattr][isortedwelldates][iwelldates3][iattrfirst]
			@show welldates[isortedwelldates][iwelldates3]
			@show welldates[isortedwelldates][iwelldates3][iattrfirst]
			@show enddates[i]
			@show attr[innattr]
			@show attr[innattr][isortedwelldates][iwelldates3]
			@show attr[innattr][isortedwelldates][iwelldates3][iattrfirst:end]
			@show matrix[iwelldates2, i]
			error("Something went wrong")
		end
	end
	return matrix, startdates, enddates
end
