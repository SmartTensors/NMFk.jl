import JuMP
import Ipopt

"Match data with concentrations and an option for ratios (avoid using ratios; convert to concentrations)"
function mixmatchdata(concentrations_in::Array{Float32, 3}, numbuckets::Int; method::Symbol=:ipopt, algorithm::Symbol=:LD_SLSQP, normalize::Bool=false, scale::Bool=false, ratios::Array{Float32, 2}=Array{Float32}(0, 0), ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array{Int}(0, 0), seed::Number=-1, random::Bool=false, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Float32=defaultregularizationweight, ratiosweight::Float32=defaultratiosweight, weightinverse::Bool=false, initW::Matrix{Float32}=Array{Float32}(0, 0), initH::Matrix{Float32}=Array{Float32}(0, 0), tolX::Float64=1e-3, tol::Float64=1e-3, maxouteriters::Int=10, quiet::Bool=true, movie::Bool=false, moviename::String="", movieorder=1:numbuckets)
	if seed >= 0
		srand(seed)
	end
	concentrations = copy(concentrations_in)
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
	nummixtures, numconstituents = size(concentrations)
	nans = isnan.(concentrations)
	concweights[nans] = 0
	if sizeof(ratios) == 0
		concentrations[nans] = 0
	else
		sr = size(ratioindices)
		if length(sr) == 1
			numberofpairs = sr[1]
			numberofratios = 1
		else
			numberofpairs, numberofratios = sr
		end
		@assert numberofpairs == 2
		for i=1:nummixtures
			for j=1:numberofratios
				r1 = ratioindices[1, j]
				r2 = ratioindices[2, j]
				if isnan(concentrations[i, r1]) && isnan(concentrations[i, r2])
					concentrations[i, r1] = ratios[i,j]
					concentrations[i, r2] = 1
				elseif isnan(concentrations[i, r2])
					concentrations[i, r2] = concentrations[i, r1] / ratios[i,j]
				elseif isnan(concentrations[i, r1])
					concentrations[i, r1] = concentrations[i, r2] * ratios[i,j]
				end
			end
		end
		nans = isnan.(concentrations)
		concentrations[nans] = 0
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
	if method == :ipopt
		m = JuMP.Model(solver=Ipopt.IpoptSolver(max_iter=maxiter, print_level=verbosity)) # tol here is something else
	elseif method == :nlopt
		m = JuMP.Model(solver=NLopt.NLoptSolver(algorithm=algorithm, maxeval=maxiter)) # xtol_abs=tolX, ftol_abs=tol
	end
	@JuMP.variable(m, mixer[i=1:nummixtures, j=1:numbuckets], start = convert(Float32, initW[i, j]))
	@JuMP.variable(m, buckets[i=1:numbuckets, j=1:numconstituents], start = convert(Float32, initH[i, j]))
	if !normalize
		@JuMP.constraint(m, buckets .>= 0)
	end
	@JuMP.constraint(m, mixer .>= 0)
	for i = 1:nummixtures
		@JuMP.constraint(m, sum(mixer[i, k] for k=1:numbuckets) == 1.)
	end
	if sizeof(ratios) == 0
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
					for (j, c1, c2) in zip(1:numberofratios, ratioindices[1,:], ratioindices[2,:])))
	end
	oldcolval = copy(m.colVal)
	if movie
		Xe = initW * initH
		NMFk.plotnmf(Xe, initW[:,movieorder], initH[movieorder,:]; movie=movie, filename=moviename, frame=1)
	end
	status = JuMP.solve(m)
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
	if !quiet && sizeod(ratios) > 0
		ratiosreconstruction = 0
		for (j, c1, c2) in zip(1:numberofratios, ratioindices[1,:], ratioindices[2,:])
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