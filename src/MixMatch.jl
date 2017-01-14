module MixMatch

import JuMP
import Ipopt

JuMP.EnableNLPResolve()

const defaultregularizationweight = convert(Float32, 0)
const defaultmaxiter = 1000
const defaultverbosity = 0
const defaultratiosweight = convert(Float32, 1)
const defaultdeltasweight = convert(Float32, 1)

"Normalize matrix"
function normalizematrix(a::Matrix)
	min = minimum(a, 1)
	max = maximum(a, 1)
	dx = max - min
	i0 = dx .== 0 # check for zeros
	min[i0] = 0
	dx[i0] = max[i0]
	i0 = dx .== 0 # check for zeros again
	dx[i0] = 1
	a = (a .- min) ./ dx
	return a, min, max
end

"Denormalize matrix"
function denormalizematrix(a::Matrix, b::Matrix, min::Matrix, max::Matrix)
	a = a .* (max - min) + pinv(b) * repeat(min, outer=[size(b, 1), 1])
	return a
end

"Scale matrix (by rows)"
function scalematrix(a::Matrix)
	max = maximum(abs(a), 1)
	a = a ./ max
	return a, max
end

"Descale matrix (by rows)"
function descalematrix(a::Matrix, max::Matrix)
	a = a .* max
	return a
end

"Scale matrix (by columns)"
function scalematrix_col(a::Matrix)
	max = maximum(abs(a), 2)
	a = a ./ max
	return a, max
end

"Descale matrix (by columns)"
function descalematrix_col(a::Matrix, max::Matrix)
	a = a .* max
	return a
end

"Convert stable isotope deltas to concentrations"
function getisotopeconcentration(delta::Union{Number,Vector,Matrix}, deltastandard::Union{Number,Vector}, concentration_species::Union{Number,Vector,Matrix}, scalefactor::Union{Number,Vector}=ones(length(deltastandard)))
	lsd = length(size(delta))
	if lsd == 1 || (lsd == 2 && size(delta)[2] == 1)
		@assert size(delta)[1] == length(concentration_species)
		@assert length(deltastandard) == 1
	elseif lsd == 2
		@assert size(delta) == size(concentration_species)
		@assert size(delta)[2] == length(deltastandard)
	end
	if lsd > 0
		Adeltastandard = repeat(collect(deltastandard), outer=[1,size(delta)[1]])'
		Ascalefactor = repeat(collect(scalefactor), outer=[1,size(delta)[1]])'
	else
		Adeltastandard = deltastandard
		Ascalefactor = scalefactor
	end
	ratio = (delta / 1000 + 1) .* Adeltastandard
	concentration_isotope  = concentration_species .* ratio ./ (ratio + 1) .* Ascalefactor
end

"Convert stable isotope concentrations to deltas"
function getisotopedelta(concentration_isotope::Union{Number,Vector,Matrix}, deltastandard::Union{Number,Vector}, concentration_species::Union{Number,Vector,Matrix}, scalefactor::Union{Number,Vector}=ones(length(deltastandard)))
	lsd = length(size(concentration_isotope))
	if lsd == 1 || (lsd == 2 && size(concentration_isotope)[2] == 1)
		@assert size(concentration_isotope)[1] == length(concentration_species)
		@assert length(deltastandard) == 1
	elseif lsd == 2
		@assert size(concentration_isotope) == size(concentration_species)
		@assert size(concentration_isotope)[2] == length(deltastandard)
	end
	if lsd > 0
		Adeltastandard = repeat(collect(deltastandard), outer=[1,size(concentration_isotope)[1]])'
		Ascalefactor = repeat(collect(scalefactor), outer=[1,size(concentration_isotope)[1]])'
	else
		Adeltastandard = deltastandard
		Ascalefactor = scalefactor
	end
	ratio = (concentration_isotope .* Ascalefactor ) ./ (concentration_species .- concentration_isotope)
	delta_isotope = (ratio .- Adeltastandard) ./ Adeltastandard * 1000
end

"Compute deltas of mixtures (`compute_contributions` requires external normalization)"
function computedeltas(mixer::Matrix, buckets::Matrix, bucketdeltas::Matrix, deltaindices::Vector; compute_contributions::Bool=false)
	numwells = size(mixer, 1)
	numdeltas = length(deltaindices)
	deltas = Array(Float64, numwells, numdeltas)
	for i = 1:numwells
		for j = 1:numdeltas
			v = vec(mixer[i, :]) .* vec(buckets[:, deltaindices[j]])
			if compute_contributions
				deltas[i, j] = dot(v, bucketdeltas[:, j])
			else
				deltas[i, j] = dot(v, bucketdeltas[:, j]) / sum(v)
			end
		end
	end
	return deltas
end

"Match data with concentrations and an option for ratios (avoid using ratios; convert to concentrations)"
@generated function matchdata(concentrations_in::Matrix{Float32}, numbuckets::Int; normalize::Bool=false, scale::Bool=false, mixtures::Bool=true, ratios::Union{Void,Array{Float32, 2}}=nothing, ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array(Int, 0, 0), random::Bool=false, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Float32=defaultregularizationweight, ratiosweight::Float32=defaultratiosweight, weightinverse::Bool=false, initW::Matrix{Float32}=Array(Float32, 0, 0), initH::Matrix{Float32}=Array(Float32, 0, 0), tol::Float64=1e-3, maxouteriters::Int=10, quiet::Bool=true)
	if ratios != Void # ratios here is DataType
		extracodeforratios = quote
			numberrations = length(ratioindices[1,:])
	end
		ratiosterm = :(
			sum(
				sum(ratiosweight *
					(sum(mixer[i, k] * buckets[k, c1] for k=1:numbuckets) / sum(mixer[i, k] * buckets[k, c2] for k=1:numbuckets) -
					ratios[i, j])^2
					for i=1:nummixtures)
				for (j, c1, c2) in zip(1:numberrations, ratioindices[1,:], ratioindices[2,:]))
			)
	else
		extracodeforratios = :()
		ratiosterm = :(0)
	end
	q = quote
		concentrations = copy(concentrations_in) # we may overwrite some of the fields if there are NaN's, so make a copy
		if normalize
			concentrations, cmin, cmax = normalizematrix(concentrations)
		elseif scale
			concentrations, cmax = scalematrix(concentrations)
		end
		if weightinverse
			concweights = convert(Array{Float32,2}, 1. ./ concentrations)
			zis = concentrations .== 0
			concweights[zis] = maximum(concentrations[!zis]) * 10
		else
			concweights = ones(Float32, size(concentrations))
		end
		nans = isnan(concentrations)
		concentrations[nans] = 0
		concweights[nans] = 0
		nummixtures = size(concentrations, 1)
		numconstituents = size(concentrations, 2)
		if sizeof(initW) == 0
			if random
				initW = rand(Float32, nummixtures, numbuckets)
			else
				initW = ones(Float32, nummixtures, numbuckets) / numbuckets
			end
		end
		if sizeof(initH) == 0
			if random
				if scale || normalize
					initH = rand(Float32, numbuckets, numconstituents)
				else
					max = maximum(concentrations, 1) / 10
					initH = rand(Float32, numbuckets, numconstituents)
					for i=1:numbuckets
						initH[i:i,:] .*= max
					end
				end
			else
				if scale || normalize
					initH = ones(Float32, numbuckets, numconstituents) / 2
				else
					max = maximum(concentrations, 1)
					initH = Array(Float32, numbuckets, numconstituents)
					for i=1:numbuckets
						initH[i:i,:] = max
					end
				end
			end
		end
		m = JuMP.Model(solver=Ipopt.IpoptSolver(max_iter=maxiter, print_level=verbosity))
		@JuMP.variable(m, buckets[i=1:numbuckets, j=1:numconstituents], start = convert(Float32, initH[i, j]))
		@JuMP.variable(m, mixer[i=1:nummixtures, j=1:numbuckets], start = convert(Float32, initW[i, j]))
		if !normalize
			@JuMP.constraint(m, buckets .>= 0)
		end
		@JuMP.constraint(m, mixer .>= 0)
		if mixtures
			for i = 1:nummixtures
				@JuMP.constraint(m, sum(mixer[i, k] for k=1:numbuckets) == 1.)
			end
		end
		$extracodeforratios
		@JuMP.NLobjective(m, Min,
			regularizationweight * sum(sum(log(1. + buckets[i, j])^2 for i=1:numbuckets) for j=1:numconstituents) / numbuckets +
			sum(sum(concweights[i, j] * (concentrations[i, j] - sum(mixer[i, k] * buckets[k, j] for k=1:numbuckets))^2 for i=1:nummixtures) for j=1:numconstituents) +
			$ratiosterm)
		oldcolval = copy(m.colVal)
		JuMP.solve(m)
		mixerval = JuMP.getvalue(mixer)
		bucketval = JuMP.getvalue(buckets)
		of = JuMP.getobjectivevalue(m)
		!quiet && @show of
		of_best = of
		iters = 0
		while !(norm(oldcolval - m.colVal) < tol) && iters < 1 # keep doing the optimization until we really reach an optimum
			oldcolval = copy(m.colVal)
			JuMP.solve(m)
			of = JuMP.getobjectivevalue(m)
			!quiet && @show of
			if of < of_best
				mixerval = JuMP.getvalue(mixer)
				bucketval = JuMP.getvalue(buckets)
				of_best = of
			end
			iters += 1
		end
		!quiet && @show of_best
		fitquality = of_best - regularizationweight * sum(log(1. + bucketval).^2) / numbuckets
		if !quiet && typeof(ratios) != Void
			ratiosreconstruction = 0
			for (j, c1, c2) in zip(1:numberrations, ratioindices[1,:], ratioindices[2,:])
				for i = 1:nummixtures
					s1 = 0
					s2 = 0
					for k = 1:numbuckets
						s1 += mixerval[i, k] * bucketval[k, c1]
						s2 += mixerval[i, k] * bucketval[k, c2]
					end
					ratiosreconstruction2 += ratiosweight * (s1/s2 - ratios[i, j])^2
				end
			end
			@show ratiosreconstruction
		end
		if normalize
			bucketval = denormalizematrix(bucketval, mixerval, cmin, cmax)
		elseif scale
			bucketval = descalematrix(bucketval, cmax)
		end
		return mixerval, bucketval, fitquality
	end
	return q
end

"Match data with concentrations and deltas (avoid using deltas; convert to concentrations)"
function matchdata(concentrations_in::Matrix{Float32}, deltas_in::Matrix{Float32}, deltaindices::Vector{Int}, numbuckets::Int; normalize::Bool=false, scale::Bool=false, random::Bool=true, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Float32=defaultregularizationweight, deltasweight::Float32=defaultdeltasweight, weightinverse::Bool=false, initW::Matrix{Float32}=Array(Float32, 0, 0), initH::Matrix{Float32}=Array(Float32, 0, 0), initHd::Matrix{Float32}=Array(Float32, 0, 0), tol::Float64=1e-3, maxouteriters::Int=10, quiet::Bool=true)
	concentrations = copy(concentrations_in) # we may overwrite some of the fields if there are NaN's, so make a copy
	deltas = copy(deltas_in)
	numdeltas = size(deltas, 2)
	nummixtures = size(concentrations, 1)
	numconstituents = size(concentrations, 2)
	if weightinverse
		concweights = convert(Array{Float32,2}, 1. ./ concentrations)
		zis = concentrations .== 0
		concweights[zis] = maximum(concentrations[!zis]) * 10
	else
		concweights = ones(Float32, size(concentrations))
	end
	nans = isnan(concentrations)
	concentrations[nans] = 0
	concweights[nans] = 0
	deltasweights = convert(Array{Float32,2}, ones(Float32, size(deltas)) * deltasweight)
	nans = isnan(deltas)
	deltas[nans] = 0
	deltasweights[nans] = 0
	if normalize
		concentrations, cmax = scalematrix(concentrations)
		deltas, dmin, dmax = normalizematrix(deltas)
	elseif scale
		concentrations, cmax = scalematrix(concentrations)
		deltas, dmax = scalematrix(deltas)
	end
	if sizeof(initW) == 0
		if random
			initW = rand(Float32, nummixtures, numbuckets)
		else
			initW = ones(Float32, nummixtures, numbuckets) / numbuckets
		end
	end
	if sizeof(initH) == 0
		if random
			if scale
				initH = rand(Float32, numbuckets, numconstituents)
			else
				max = maximum(concentrations, 1) / 10
				initH = rand(numbuckets, numconstituents)
				for i=1:numbuckets
					initH[i,:] .*= max
				end
			end
		else
			if scale
				initH = ones(Float32, numbuckets, numconstituents) / 2
			else
				max = maximum(concentrations, 1)
				initH = Array(Float32, numbuckets, numconstituents)
				for i=1:numbuckets
					initH[i,:] = max
				end
			end
		end
	end
	if sizeof(initHd) == 0
		if random
			if scale
				initHd = rand(Float32, numbuckets, numdeltas)
			else
				max = maximum(deltas, 1) / 10
				initHd = rand(Float32, numbuckets, numdeltas)
				for i=1:numbuckets
					initHd[i,:] .*= max
				end
			end
		else
			if scale
				initHd = ones(Float32, numbuckets, numdeltas) / 2
			else
				max = maximum(deltas, 1)
				initHd = Array(Float32, numbuckets, numdeltas)
				for i=1:numbuckets
					initHd[i,:] = max
				end
			end
		end
	end
	m = JuMP.Model(solver=Ipopt.IpoptSolver(max_iter=maxiter, print_level=verbosity))
	@JuMP.variable(m, mixer[i=1:nummixtures, j=1:numbuckets], start = convert(Float32, initW[i, j]))
	@JuMP.variable(m, buckets[i=1:numbuckets, j=1:numconstituents], start = convert(Float32, initH[i, j]))
	@JuMP.variable(m, bucketdeltas[i=1:numbuckets, j=1:numdeltas], start = convert(Float32, initHd[i, j]))
	@JuMP.constraint(m, buckets .>= 0)
	@JuMP.constraint(m, mixer .>= 0)
	for i = 1:nummixtures
		@JuMP.constraint(m, sum(mixer[i, j] for j=1:numbuckets) == 1.)
	end
	#=
	for i = 1:numbuckets
		for j = 1:numconstituents
			if i != 1 || j != 1
				@JuMP.constraint(m, buckets[i, j] == initH[i, j]) # Fix buckets for testing
			end
		end
	end
	for i = 1:numbuckets
		for j = 1:numdeltas
			#if i != 1 || j != 1
				@JuMP.constraint(m, bucketdeltas[i, j] == initHd[i, j]) # Fix buckets for testing
			#end
		end
	end
	=#
	@JuMP.NLobjective(m, Min,
		regularizationweight * sum(sum(log(1. + buckets[i, j])^2 for i=1:numbuckets) for j=1:numconstituents) / numbuckets +
		regularizationweight * sum(sum(log(1. + abs(bucketdeltas[i, j]))^2, i=1:numbuckets) for j=1:numdeltas) / numbuckets +
		sum(sum(concweights[i, j] * (concentrations[i, j] - (sum(mixer[i, k] * buckets[k, j] for k=1:numbuckets)))^2 for i=1:nummixtures) for j=1:numconstituents) +
		sum(sum(deltasweights[i, di] * (deltas[i, di] - (sum(mixer[i, j] * buckets[j, deltaindices[di]] * bucketdeltas[j, di] for j=1:numbuckets) / sum(mixer[i, j] * buckets[j, deltaindices[di]] for j=1:numbuckets)))^2 for i = 1:nummixtures) for di=1:numdeltas)
		)
	oldcolval = copy(m.colVal)
	JuMP.solve(m)
	mixerval = JuMP.getvalue(mixer)
	bucketval = JuMP.getvalue(buckets)
	bucketdeltasval = JuMP.getvalue(bucketdeltas)
	of = JuMP.getobjectivevalue(m)
	!quiet && @show of
	of_best = of
	iters = 0
	while !(norm(oldcolval - m.colVal) < tol) && iters < maxouteriters # keep doing the optimization until we really reach an optimum
		oldcolval = copy(m.colVal)
		JuMP.solve(m)
		of = JuMP.getobjectivevalue(m)
		!quiet && @show of
		if of < of_best
			mixerval = JuMP.getvalue(mixer)
			bucketval = JuMP.getvalue(buckets)
			bucketdeltasval = JuMP.getvalue(bucketdeltas)
			of_best = of
		end
		iters += 1
	end
	!quiet && @show of_best
	fitquality = of_best - regularizationweight * sum(log(1. + bucketval).^2) / numbuckets - regularizationweight * sum(log(1. + abs(bucketdeltasval)).^2) / numbuckets
	if normalize
		bucketval = descalematrix(bucketval, cmax)
		bucketdeltasval = denormalizematrix(bucketdeltasval, mixerval, dmin, dmax)
	elseif scale
		bucketval = descalematrix(bucketval, cmax)
		bucketdeltasval = descalematrix(bucketdeltasval, dmax)
	end
	return mixerval, bucketval, bucketdeltasval, fitquality
end

"Match data with only deltas"
function matchwaterdeltas(deltas::Matrix{Float32}, numbuckets::Int; random::Bool=false, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Float32=defaultregularizationweight, maxdeltaguess::Float32=1000., bucketmeans::Matrix{Float32}=zeros(numbuckets, 2))
	deltas = copy(deltas) # we may overwrite some of the fields if there are NaN's, so make a copy
	nummixtures = size(deltas, 1)
	numconstituents = 2
	m = JuMP.Model(solver=Ipopt.IpoptSolver(max_iter=maxiter, print_level=verbosity))
	if random
		@JuMP.variable(m, mixer[1:nummixtures, 1:numbuckets] >= 0., start=randn(Float32))
		@JuMP.variable(m, buckets[1:numbuckets, 1:numconstituents], start=maxdeltaguess * rand(Float32))
	else
		@JuMP.variable(m, mixer[1:nummixtures, 1:numbuckets] >= 0.)
		@JuMP.variable(m, buckets[1:numbuckets, 1:numconstituents])
	end
	@JuMP.constraint(m, mixer .<= 1.)
	for i = 1:size(deltas, 1)
		@JuMP.constraint(m, sum(mixer[i, j] for j=1:numbuckets) == 1.)
	end
	concweights = ones(Float32, size(deltas))
	nans = isnan(deltas)
	deltas[nans] = 0
	concweights[nans] = 0
	@JuMP.NLobjective(m, Min,
		regularizationweight * sum(sum((buckets[i, j] - bucketmeans[i, j])^2, i=1:numbuckets) for j=1:numconstituents) / numbuckets +
		sum(sum(concweights[i, j] * (sum(mixer[i, k] * buckets[k, j] for k=1:numbuckets) - deltas[i, j])^2 for i=1:nummixtures) for j=1:numconstituents))
	JuMP.solve(m)
	mixerval = JuMP.getvalue(mixer)
	bucketval = JuMP.getvalue(buckets)
	fitquality = JuMP.getobjectivevalue(m) - regularizationweight * sum((bucketval - bucketmeans).^2) / numbuckets
	return mixerval, bucketval, fitquality
end

end