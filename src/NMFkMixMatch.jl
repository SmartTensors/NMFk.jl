import JuMP
import Ipopt

const defaultregularizationweight = convert(Float32, 0)
const defaultmaxiter = 1000
const defaultverbosity = 0
const defaultratiosweight = convert(Float32, 1)
const defaultdeltasweight = convert(Float32, 1)

"Match data with concentrations and an option for ratios (avoid using ratios; convert to concentrations)"
function mixmatchdata(concentrations_in::Matrix{Float32}, numbuckets::Int; normalize::Bool=false, scale::Bool=false, ratios::Union{Void,Array{Float32, 2}}=nothing, ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array{Int}(0, 0), seed::Number=-1, random::Bool=false, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Float32=defaultregularizationweight, ratiosweight::Float32=defaultratiosweight, weightinverse::Bool=false, initW::Matrix{Float32}=Array{Float32}(0, 0), initH::Matrix{Float32}=Array{Float32}(0, 0), tolX::Float64=1e-3, tol::Float64=1e-3, maxouteriters::Int=10, quiet::Bool=true, movie::Bool=false, moviename::String="", movieorder=1:numbuckets)
	if ratios != nothing
		numberrations = length(ratioindices[1,:])
	end
	if seed >= 0
		srand(seed)
	end
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
	nans = isnan.(concentrations)
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
				max = maximum(concentrations, 1)
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
				initH = Array{Float32}(numbuckets, numconstituents)
				for i=1:numbuckets
					initH[i:i,:] = max
				end
			end
		end
	end
	m = JuMP.Model(solver=Ipopt.IpoptSolver(max_iter=maxiter, print_level=verbosity))
	@JuMP.variable(m, mixer[i=1:nummixtures, j=1:numbuckets], start = convert(Float32, initW[i, j]))
	@JuMP.variable(m, buckets[i=1:numbuckets, j=1:numconstituents], start = convert(Float32, initH[i, j]))
	if !normalize
		@JuMP.constraint(m, buckets .>= 0)
	end
	@JuMP.constraint(m, mixer .>= 0)
	for i = 1:nummixtures
		@JuMP.constraint(m, sum(mixer[i, k] for k=1:numbuckets) == 1.)
	end
	if ratios == nothing
		@JuMP.NLobjective(m, Min,
			regularizationweight * sum(sum(log(1. + buckets[i, j])^2 for i=1:numbuckets) for j=1:numconstituents) / numbuckets +
			sum(sum(concweights[i, j] * (sum(mixer[i, k] * buckets[k, j] for k=1:numbuckets) - concentrations[i, j])^2 for i=1:nummixtures) for j=1:numconstituents))
	else
		@JuMP.NLobjective(m, Min,
			regularizationweight * sum(sum(log(1. + buckets[i, j])^2 for i=1:numbuckets) for j=1:numconstituents) / numbuckets +
			sum(sum(concweights[i, j] * (sum(mixer[i, k] * buckets[k, j] for k=1:numbuckets) - concentrations[i, j])^2 for i=1:nummixtures) for j=1:numconstituents) +
			sum(sum(ratiosweight *
					(sum(mixer[i, k] * buckets[k, c1] for k=1:numbuckets) / sum(mixer[i, k] * buckets[k, c2]
					for k=1:numbuckets) - ratios[i, j])^2 for i=1:nummixtures)
					for (j, c1, c2) in zip(1:numberrations, ratioindices[1,:], ratioindices[2,:])))
	end
	oldcolval = copy(m.colVal)
	if movie
		Xe = initW * initH
		NMFk.plotnmf(Xe, initW[:,movieorder], initH[movieorder,:]; movie=movie, filename=moviename, frame=1)
	end
	JuMP.solve(m)
	mixerval = JuMP.getvalue(mixer)
	bucketval = JuMP.getvalue(buckets)
	of = JuMP.getobjectivevalue(m)
	!quiet && @show of
	of_best = of
	iters = 0
	frame = 2
	while !(norm(oldcolval - m.colVal) < tolX) && !(of_best < tol)
		oldcolval = copy(m.colVal)
		JuMP.solve(m)
		if movie
			We = JuMP.getvalue(mixer)
			He = JuMP.getvalue(buckets)
			Xe = We * He
			NMFk.plotnmf(Xe, We[:,movieorder], He[movieorder,:]; movie=movie,filename=moviename, frame=frame)
			frame += 1
		end
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
	fitquality = of_best - regularizationweight * sum(log.(1. + bucketval).^2) / numbuckets
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
	if movie
		Xe = mixerval * bucketval
		NMFk.plotnmf(Xe, mixerval[:,movieorder], bucketval[movieorder,:]; movie=movie, filename=moviename, frame=frame)
	end
	return mixerval, bucketval, fitquality
end

"Match data with concentrations and deltas (avoid using deltas; convert to concentrations)"
function mixmatchdeltas(concentrations_in::Matrix{Float32}, deltas_in::Matrix{Float32}, deltaindices::Vector{Int}, numbuckets::Int; normalize::Bool=false, scale::Bool=false, random::Bool=true, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Float32=defaultregularizationweight, deltasweight::Float32=defaultdeltasweight, weightinverse::Bool=false, initW::Matrix{Float32}=Array{Float32}(0, 0), initH::Matrix{Float32}=Array{Float32}(0, 0), initHd::Matrix{Float32}=Array{Float32}(0, 0), tol::Float64=1e-3, maxouteriters::Int=10, quiet::Bool=true)
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
				initH = Array{Float32}(numbuckets, numconstituents)
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
				initHd = Array{Float32}(numbuckets, numdeltas)
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
function mixmatchwaterdeltas(deltas::Matrix{Float32}, numbuckets::Int; random::Bool=false, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Float32=defaultregularizationweight, maxdeltaguess::Float32=1000., bucketmeans::Matrix{Float32}=zeros(numbuckets, 2))
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
