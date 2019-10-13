import Dates
import DataFrames

function remap(v::Vector{T}, mapping::Vector; func::Function=!isnothing) where {T}
	o = Vector{T}(undef, length(mapping))
	if typeof(T) <: Integer
		o .= 0
	else
		o .= NaN
	end
	i = func.(mapping)
	o[i] .= v[mapping[i]]
	return o
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
		firstentry = findfirst(funcfirst.(X[nt...]))
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
			iattrfirst = findfirst(i->i>0, attr[innattr][isortedwelldates][iwelldates3])
			iattrlast = findlast(i->i>0, attr[innattr][isortedwelldates][iwelldates3])
		else
			iattrfirst = findfirst(i->i>=0, attr[innattr][isortedwelldates][iwelldates3])
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
