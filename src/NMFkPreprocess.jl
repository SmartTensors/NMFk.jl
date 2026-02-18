import Dates
import DataFrames
import Statistics
import StatsBase
import Mads

function log10s(x::AbstractFloat; min::Number=log10(eps(typeof(x))))
	return x ≈ 0 ? min : log10(x)
end

function log10s(x::AbstractArray; kw...)
	return log10s!(copy(x); kw...)
end

function log10s!(x::AbstractArray; offset::Number=1)
	iz = x .<= 0
	siz = sum(iz)
	if siz == length(x)
		x = abs.(x)
	elseif siz > 0
		x[iz] .= NaN
	end
	x .= log10.(x)
	if (siz != length(x)) && (siz > 0)
		min = minimumnan(x[.!iz]) - offset
		x[iz] .= min
	end
	return x
end

function datanalytics(v::AbstractVector; plothistogram::Bool=true, logtest::Bool=false, log::Bool=false, kw...)
	ig = .!isnan.(v) .&& .!ismissing(v) .&& .!isinf.(v)
	vn = v[ig]
	if length(vn) > 0
		if !logtest && log
			vn = log10s(vn)
		end
		plothistogram && NMFk.histogram(vn; quiet=false, kw...)
		if logtest && !log
			NMFk.histogram(log10s(vn); quiet=false, kw..., title="Log test")
		end
		return minimum(vn), maximum(vn), Statistics.std(vn), StatsBase.skewness(vn), sum(ig)
	else
		return NaN, NaN, NaN, 0, 0
	end
end

function datanalytics(d::DataFrames.DataFrame; names::AbstractVector=names(d), log::Bool=false, logv::AbstractVector=fill(log, length(names)), kw...)
	ct = nonmissingtype.(eltype.(eachcol(d)))
	ci = ct .<: Number
	m = Float64.(Matrix(d[!, ci]))
	m[ismissing.(m)] .= NaN
	skipped_attributes = names[.!ci]
	if length(skipped_attributes) > 0
		@info("Attributes skipped ($(length(skipped_attributes))):")
		println.(skipped_attributes)
	end
	return datanalytics(m; dims=2, names=names[ci], logv=logv[ci], kw...)
end

function datanalytics(a::AbstractMatrix; dims::Integer=2, name=dims == 1 ? "Row" : "Column", names::AbstractVector=["$name $i" for i in axes(a, dims)], kw...)
	@assert length(names) == size(a, dims)
	return datanalytics(a, names; dims=dims, kw...)
end

function datanalytics(a::AbstractMatrix{T}, names::AbstractVector; dims::Integer=2, quiet::Bool=false, veryquiet::Bool=true, saveplot::Bool=false, logtest::Bool=false, log::Bool=false, logv::AbstractVector=fill(log, length(names)), casefilename::AbstractString="", kw...) where {T <: Number}
	@assert length(names) == length(logv)
	@assert length(names) == size(a, dims)
	names = String.(names)
	max_length = maximum(length.(names))
	min = Vector{T}(undef, length(names))
	max = Vector{T}(undef, length(names))
	std = Vector{T}(undef, length(names))
	skewness = Vector{T}(undef, length(names))
	count = Vector{Int64}(undef, length(names))
	for (i, n) in enumerate(names)
		nt = ntuple(k -> (k == dims ? i : Colon()), ndims(a))
		if !logtest && logv[i]
			v = log10s(vec(a[nt...]))
		else
			v = vec(a[nt...])
		end
		if saveplot
			if casefilename == ""
				filename = "histogram-$(n).png"
			else
				if last(splitdir(casefilename)) == ""
					filename = casefilename * "histogram-$(n).png"
				else
					filename = casefilename * "-$(n).png"
				end
			end
		else
			filename = ""
		end
		min[i], max[i], std[i], skewness[i], count[i] = datanalytics(v; filename_plot=filename, kw..., title=n)
		if !quiet
			print("$(Base.text_colors[:cyan])$(Base.text_colors[:bold])$(NMFk.sprintf("%-$(max_length)s", names[i])):$(Base.text_colors[:normal]) min: $(Printf.@sprintf("%12.7g", min[i])) max: $(Printf.@sprintf("%12.7g", max[i])) std.dev: $(Printf.@sprintf("%12.7g", std[i])) skewness: $(Printf.@sprintf("%12.7g", skewness[i])) count: $(Printf.@sprintf("%12d", count[i]))")
			if count[i] == 0
				print(" $(Base.text_colors[:red])<- no data$(Base.text_colors[:normal])")
			elseif count[i] < 4
				print(" $(Base.text_colors[:yellow])<- limited data$(Base.text_colors[:normal])")
			end
			if logv[i]
				print(" $(Base.text_colors[:blue])<- log-transformed$(Base.text_colors[:normal])")
			end
			println()
		end
		if logtest && logv[i]
			if saveplot
				if casefilename == ""
					filename = "histogram-$(n)-logtest.png"
				else
					if last(splitdir(casefilename)) == ""
						filename = casefilename * "histogram-$(n)-logtest.png"
					else
						filename = casefilename * "-$(n).png"
					end
				end
			else
				filename = ""
			end
			datanalytics(log10s(vec(a[nt...])); filename_plot=filename, kw..., title=n * " (log test)")
		end
	end
	return min, max, std, skewness, count
end

function indicize(v::AbstractVector; rev::Bool=false, nbins::Integer=length(v), minvalue::Number=minimum(v), maxvalue::Number=maximum(v), stepvalue=nothing, granulate::Bool=true, quiet::Bool=false)
	if !isnothing(stepvalue)
		if !quiet
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
		emptybins = false
		for k in unique(sort([us; nb]))
			m = iv .== k
			s = sum(m)
			if s == 0
				emptybins = true
				@info("Bin $(lpad("$k", 3, " ")): count $(lpad("$(s)", 6, " "))")
			else
				@info("Bin $(lpad("$k", 3, " ")): count $(lpad("$(s)", 6, " ")) range $(minimum(v[m])) $(maximum(v[m]))")
			end
		end
		if emptybins
			@warn("There are empty bins ...")
		end
	end
	if rev == true
		iv = (nbins + 1) .- iv
	end
	@assert minimum(iv) >= 1
	if granulate
		@assert maximum(iv) <= nbins
	else
		nbins += 1
		@assert maximum(iv) <= nbins
	end
	return iv, nbins, minvalue, maxvalue
end

function processdata(df::DataFrames.DataFrame, type::DataType=Float32; kw...)
	dfn = deepcopy(df)
	processdata!(dfn, type; kw...)
	return dfn
end

function processdata!(df::DataFrames.DataFrame, type::DataType=Float32; kw...)
	for i in axes(df, 2)
		v = processdata(df[!, i], type; kw...)
		if all(typeof.(v) .<: Number)
			v = convert(Vector{type}, convert.(type, v))
		end
		df[!, i] = convert(Vector{Union{unique(typeof.(v))...}}, v)
	end
	return df
end

function processdata(M::AbstractArray, type::DataType=Float32; kw...)
	Mn = processdata!(copy(M), type; kw...)
	return Mn
end

function processdata!(M::AbstractArray, type::DataType=Float32; nanstring::AbstractString="NaN", enforce_nan::Bool=true, nothing_ok::Bool=false, negative_ok::Bool=true, string_ok::Bool=true)
	if !(type <: Number)
		enforce_nan = false
	end
	# Choose the value used to represent "NaN" placeholders ("", nanstring) when targeting floats.
	# NOTE: Under current defaults, `missing` and `nothing` are converted to `missing` (not NaN).
	nan_value = (type <: AbstractFloat) && enforce_nan ? type(NaN) : missing

	# If input is a concrete string array but we're going to write non-strings
	# (NaN/missing), widen to an Any container.
	if (eltype(M) <: AbstractString) && (type <: Number)
		M = convert(Array{Any}, M)
	end
	# Replace string placeholders. Use NaN only for float outputs with enforce_nan=true.
	ie = coalesce.(M .== "", false)
	if any(ie)
		if nan_value === missing && !(eltype(M) >: Missing)
			M = convert(Array{Any}, M)
		end
		M[ie] .= nan_value
	end
	ie = coalesce.(M .== nanstring, false)
	if any(ie)
		if nan_value === missing && !(eltype(M) >: Missing)
			M = convert(Array{Any}, M)
		end
		M[ie] .= nan_value
	end
	if !nothing_ok
		inothing = isnothing.(M)
		if any(inothing)
			if !(eltype(M) >: Missing)
				M = convert(Array{Any}, M)
			end
			M[inothing] .= missing
		end
	end

	if type <: Number
		is = typeof.(M) .<: AbstractString
		if sum(is) > 0
			v = tryparse.(type, M[is])
			isn = isnothing.(v)
			if sum(isn) > 0
				if !string_ok
					if (type <: AbstractFloat) && enforce_nan
						v[isn] .= type(NaN)
					else
						v_any = convert(Vector{Any}, v)
						v_any[isn] .= missing
						v = v_any
					end
				else
					v_any = convert(Vector{Any}, v)
					v_any[isn] .= M[is][isn]
					v = v_any
				end
			end
			M[is] .= v
		end
		if !negative_ok
			for i in eachindex(M)
				if M[i] isa Number && M[i] < 0
					M[i] = type(0)
				end
			end
		end
		id = typeof.(M) .<: Number
		M[id] .= convert.(type, M[id])
	elseif type <: AbstractString
		id = typeof.(M) .<: Number
		M[id] .= string.(M[id])
	elseif type <: Dates.DateTime || type <: Dates.Date
		id = typeof.(M) .<: Dates.DateTime .|| typeof.(M) .<: Dates.Date
		M[id] .= type.(M[id])
	end
	M = convert(Array{Union{unique(typeof.(M))...}}, M)
	return M
end

function griddata(x::AbstractVector, y::AbstractVector; stepvalue=nothing, nbins=nothing, xrev::Bool=false, xnbins::Integer=length(x), xminvalue=minimumnan(x), xmaxvalue=maximumnan(x), xstepvalue=stepvalue, yrev::Bool=false, ynbins=length(y), yminvalue=minimumnan(y), ymaxvalue=maximumnan(y), ystepvalue=stepvalue, granulate::Bool=true, quiet::Bool=true)
	if !isnothing(nbins)
		xnbins = nbins
		ynbins = nbins
	end
	ix, xbins, gxmin, gxmax = NMFk.indicize(x; rev=xrev, nbins=xnbins, minvalue=xminvalue, maxvalue=xmaxvalue, stepvalue=xstepvalue, granulate=granulate, quiet=quiet)
	iy, ybins, gymin, gymax = NMFk.indicize(y; rev=yrev, nbins=ynbins, minvalue=yminvalue, maxvalue=ymaxvalue, stepvalue=ystepvalue, granulate=granulate, quiet=quiet)
	if .!isnothing(nbins)
		@info("Grid data: Step x=$(xstepvalue) y=$(ystepvalue) Number of bins x=$(xbins) y=$(ybins)")
		return range(gxmin, gxmax; length=xbins), range(gymin, gymax; length=ybins)
	elseif .!isnothing(xstepvalue) .&& .!isnothing(ystepvalue)
		@info("Grid data: Step x=$(xstepvalue) y=$(ystepvalue) Number of bins x=$(xbins) y=$(ybins)")
		return range(gxmin, gxmax; step=xstepvalue), range(gymin, gymax; step=ystepvalue)
	else
		@error("Something is wrong! Grid data failed ...")
		@show stepvalue, xstepvalue, ystepvalue
		@show nbins, xnbins, ynbins
		@show xminvalue, xmaxvalue
		@show yminvalue, ymaxvalue
		throw(ArgumentError("Something is wrong!"))
	end
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
	for i in axes(z, 2)
		for j in eachindex(ix)
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
	if !isnothing(stepvalue)
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

function remap(v::AbstractVector{T}, mapping::AbstractVector; func::Function=!isnothing) where {T <: Number}
	o = Vector{T}(undef, length(mapping))
	if T <: AbstractFloat
		o .= T(NaN)
	else
		o .= zero(T)
	end
	i = func.(mapping)
	o[i] .= v[mapping[i]]
	return o
end

function remap(v::AbstractMatrix{T}, mapping::AbstractVector; func::Function=!isnothing) where {T <: Number}
	o = Matrix{T}(undef, length(mapping), size(v, 2))
	if T <: AbstractFloat
		o .= T(NaN)
	else
		o .= zero(T)
	end
	i = func.(mapping)
	o[i, :] .= v[mapping[i], :]
	return o
end

function slopes(v::AbstractVector)
	s = similar(v)
	s[1] = v[2] - v[1]
	s[end] = v[end] - v[end - 1]
	for i in eachindex(v)[(begin + 1):(end - 1)]
		s[i] = (v[i + 1] - v[i - 1]) ./ 2
	end
	return s
end

function getdatawindow(X::AbstractArray{T, N}, d::Integer; func::Function=i -> i > 0, funcfirst::Function=func, funclast::Function=func, start::AbstractVector{Int64}=Vector{Int64}(undef, 0)) where {T <: Number, N}
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
			nt = ntuple(k -> (k == d ? i : Base.Slice(start[i]:endd[k])), N)
		else
			nt = ntuple(k -> (k == d ? i : Colon()), N)
		end
		firstentry = Base.findfirst(funcfirst.(X[nt...]))
		if !isnothing(firstentry)
			afirstentry[i] = firstentry
			lastentry = findlast(funclast.(X[nt...]))
			if !isnothing(lastentry)
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

function shiftarray(X::AbstractArray{T, N}, d::Integer, start::AbstractVector{Int64}, finish::AbstractVector{Int64}, datasize::AbstractVector{Int64}) where {T <: Number, N}
	@assert d >= 1 && d <= N
	dd = size(X, d)
	@assert length(start) == dd
	@assert length(finish) == dd
	@assert length(datasize) == dd
	Y = Matrix{T}(undef, maximum(datasize), dd)
	Y .= NaN
	for i = 1:dd
		nty = ntuple(k -> (k == d ? i : Base.Slice(1:datasize[i])), N)
		ntx = ntuple(k -> (k == d ? i : Base.Slice(start[i]:finish[i])), N)
		Y[nty...] = X[ntx...]
	end
	return Y
end

"""
Extract a matrix from a dataframe
"""
function df2matrix(df::DataFrames.DataFrame, id::AbstractVector, dates::Union{StepRange{Dates.Date, Dates.Month}, Vector{Dates.Date}}, dfattr::Symbol, dfdate::Symbol=:ReportDate, dfapi::Symbol=:API; addup::Bool=false, checkzero::Bool=true)
	nw = length(id)
	matrix = Matrix{Float32}(undef, length(dates), nw)
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
			if !addup
				c = zeros(length(attr[innattr][iwelldates3]))
			end
			matrix[iwelldates[iwelldates3], k] .= 0
			for (a, b) in enumerate(attr[innattr][iwelldates3])
				matrix[iwelldates[iwelldates3][a], k] += b
				if !addup
					c[a] += 1
				end
			end
			if !addup
				matrix[iwelldates[iwelldates3], k] ./= c
			end
		end
	end
	return matrix, fwells
end

"""
Extract a time shifted matrix from a dataframe
"""
function df2matrix_shifted(df::DataFrames.DataFrame, id::AbstractVector, dates::Union{StepRange{Dates.Date, Dates.Month}, Vector{Dates.Date}}, dfattr::Symbol, dfdate::Symbol=:ReportDate, dfapi::Symbol=:API; kw...)
	matrix, startdates, enddates = df2matrix_shifted(df, id, length(dates), dates, dfattr, dfdate, dfapi; kw...)
	recordlength = findlast(i -> !isnan(i), NMFk.sumnan(matrix; dims=2))[1]
	matrixn = Matrix{Float32}(undef, recordlength, size(matrix, 2))
	matrixn .= matrix[1:recordlength, :]
	return matrixn, startdates, enddates
end
function df2matrix_shifted(df::DataFrames.DataFrame, id::AbstractVector, recordlength::Integer, dates::Union{StepRange{Dates.Date, Dates.Month}, Vector{Dates.Date}}, dfattr::Symbol, dfdate::Symbol=:ReportDate, dfapi::Symbol=:API; addup::Bool=false, checkzero::Bool=true)
	nw = length(id)
	matrix = Matrix{Float32}(undef, recordlength, nw)
	matrix .= NaN32
	startdates = Vector{Dates.Date}(undef, nw)
	enddates = Vector{Dates.Date}(undef, nw)
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
			iattrfirst = Base.findfirst(i -> i > 0, attr[innattr][isortedwelldates][iwelldates3])
			iattrlast = findlast(i -> i > 0, attr[innattr][isortedwelldates][iwelldates3])
		else
			if sumattr == 0
				@warn("Well $w: zero total ($(string(dfattr))) production!")
			end
			iattrfirst = Base.findfirst(i -> !isnan(i), attr[innattr][isortedwelldates][iwelldates3])
			iattrlast = findlast(i -> !isnan(i), attr[innattr][isortedwelldates][iwelldates3])
		end
		startdates[i] = welldates[isortedwelldates][iwelldates3][iattrfirst]
		enddates[i] = welldates[isortedwelldates][iwelldates3][iattrlast]
		iwelldates2 = iwelldates[iwelldates3][iattrfirst:iattrlast] .- iwelldates[iwelldates3][iattrfirst] .+ 1
		matrix[iwelldates2, i] .= 0
		if !addup
			c = zeros(length(iattrfirst:iattrlast))
		end
		for (a, b) in enumerate(iattrfirst:iattrlast)
			matrix[iwelldates2[a], i] += attr[innattr][isortedwelldates][b]
			if !addup
				c[a] += 1
			end
		end
		if !addup
			matrix[iwelldates2, i] ./= c
		end
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
	for i in axes(m, dims)
		ms[:, i] = moving_average(m[:, i], window)
	end
	return ms
end

function minmax_dx(x::AbstractVector)
	minx = Inf
	maxx = -Inf
	for i in eachindex(x)[(begin + 1):end]
		dx = x[i] - x[i - 1]
		if dx < minx
			minx = dx
		elseif dx > maxx
			maxx = dx
		end
	end
	return minx, maxx, maxx - minx
end

function grid_reduction(lon::AbstractVector, lat::AbstractVector; skip::Int=1, sigdigits::Int=8)
	@assert length(lon) == length(lat)
	@assert skip >= 1
	lon_rounded = round.(lon; sigdigits=sigdigits)
	lat_rounded = round.(lat; sigdigits=sigdigits)
	@info("Number of original points       = $(length(lon_rounded))")
	lon_unique = unique(sort(lon_rounded))
	lat_unique = unique(sort(lat_rounded))

	@info("Number of Longitude unique grid points = $(length(lon_unique))")
	@info("Number of Latitude  unique grid points = $(length(lat_unique))")
	@info("Number of unique grid points = $(length(lon_unique) * length(lat_unique))")
	lon_grid = lon_unique[1:skip:end]
	lat_grid = lat_unique[1:skip:end]
	@info("Number of Longitude grid points = $(length(lon_grid))")
	@info("Number of Latitude  grid points = $(length(lat_grid))")
	@info("Number of grid points = $(length(lon_grid) * length(lat_grid))")
	skip_mask = falses(length(lon_rounded))
	for i in eachindex(lon_rounded)
		if lon_rounded[i] in lon_grid && lat_rounded[i] in lat_grid
			skip_mask[i] = true
		end
	end
	@info("Number of reduced points        = $(sum(skip_mask))")
	return skip_mask
end

function dataframe_rename!(df::DataFrames.DataFrame, df_names::AbstractDict; matchtype::Symbol=:exact)
	return dataframe_rename!(df, collect(keys(df_names)), collect(values(df_names)); matchtype=matchtype)
end

function dataframe_rename!(df::DataFrames.DataFrame, oldnames::AbstractVector, newnames::AbstractVector; matchtype::Symbol=:exact)
	@assert length(oldnames) == length(newnames)
	names_df = names(df)
	for i in eachindex(names_df)
		oldname = names_df[i]
		newname = oldname
		if matchtype == :exact
			m = oldnames .== names_df[i]
			if any(m)
				newname = first(newnames[m])
			end
		elseif matchtype == :startswith
			m = startswith.(names_df[i], oldnames)
			if any(m)
				j = sum(m) > 1 ? argmax(length.(oldnames[m])) : 1
				newname = replace(oldname, oldnames[m][j] => newnames[m][j])
			end
		elseif matchtype == :endswith
			m = endswith.(names_df[i], oldnames)
			if any(m)
				j = sum(m) > 1 ? argmax(length.(oldnames[m])) : 1
				newname = replace(oldname, oldnames[m][j] => newnames[m][j])
			end
		end
		if oldname != newname
			DataFrames.rename!(df, oldname => newname)
			println("Renamed: $(oldname) => $(newname)")
		end
	end
end