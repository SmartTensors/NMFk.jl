import JuMP
import Ipopt
import LinearAlgebra
import Suppressor

const defaultregularizationweight = 0.
const defaultmaxiter = 1000
const defaultverbosity = 0
const defaultratiosweight = 1.
const defaultdeltasweight = 1.

"Match data with concentrations and an option for ratios (avoid using ratios; convert to concentrations)"
function mixmatchdata(concentrations_in::AbstractMatrix{T}, numbuckets::Int; method::Symbol=:ipopt, algorithm::Symbol=:LD_SLSQP, normalize::Bool=false, scale::Bool=false, maxH::Bool=false, ratios::Array{T, 2}=Array{T}(undef, 0, 0), ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array{Int}(undef, 0, 0), seed::Number=-1, random::Bool=true, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::T=convert(T, defaultregularizationweight), ratiosweight::T=convert(T, defaultratiosweight), weightinverse::Bool=false, Winit::AbstractMatrix{T}=Array{T}(undef, 0, 0), Hinit::AbstractMatrix{T}=Array{T}(undef, 0, 0), tolX::Float64=1e-3, tol::Float64=1e-3, tolOF::Float64=1e-3, maxreattempts::Int=1, maxbaditers::Int=5, quiet::Bool=NMFk.quiet, movie::Bool=false, moviename::AbstractString="", movieorder=1:numbuckets) where {T <: Number}
	if seed >= 0
		Random.seed!(seed)
	end
	concentrations = copy(concentrations_in)
	if weightinverse
		concweights = convert(Array{T, 2}, 1 ./ concentrations)
		zis = concentrations .== 0
		concweights[zis] .= maximum(concentrations[!zis]) * 10
	else
		concweights = ones(T, size(concentrations))
	end
	nummixtures, numconstituents = size(concentrations)
	nans = isnan.(concentrations)
	concweights[nans] .= 0
	if normalize
		concentrations, cmin, cmax = normalizematrix_row!(concentrations)
	elseif scale
		concentrations, cmax = scalematrix_row!(concentrations)
	end
	if sizeof(ratios) == 0
		concentrations[nans] .= 0
	else
		sr = size(ratioindices)
		if length(sr) == 1
			numberofpairs = 1
			numberofratios = sr[1]
		else
			numberofratios, numberofpairs = sr
		end
		@assert numberofpairs == 2
		isn = isnan
		ratiosweightmatrix = similar(ratios)
		ratiosweightmatrix .= ratiosweight
		for i=1:nummixtures
			for j=1:numberofratios
				r1 = ratioindices[j, 1]
				r2 = ratioindices[j, 2]
				if isnan(ratios[i, j]) || ratios[i, j] == 0
					ratiosweightmatrix[i, j] = 0
					ratios[i, j] = 0
					concentrations[i, r1] = 1
					concentrations[i, r2] = 1
				elseif isnan(concentrations[i, r1]) && isnan(concentrations[i, r2])
					concentrations[i, r1] = ratios[i, j]
					concentrations[i, r2] = 1
				elseif isnan(concentrations[i, r2])
					concentrations[i, r2] = concentrations[i, r1] / ratios[i, j]
				elseif isnan(concentrations[i, r1])
					concentrations[i, r1] = concentrations[i, r2] * ratios[i, j]
				end
			end
		end
		nans = isnan.(concentrations)
		concentrations[nans] .= 0
	end
	if sizeof(Winit) == 0
		if random
			Winit = rand(T, nummixtures, numbuckets)
		else
			Winit = ones(T, nummixtures, numbuckets) / numbuckets
		end
	end
	if sizeof(Hinit) == 0
		if random
			Hinit = rand(T, numbuckets, numconstituents)
		else
			Hinit = ones(T, numbuckets, numconstituents) / 2
		end
		if maxH
			maxc = maximum(concentrations; dims=1)
			for i=1:numbuckets
				Hinit[i:i,:] .*= maxc
			end
		end
	end
	if method == :ipopt
		m = JuMP.Model(Ipopt.Optimizer)
		JuMP.set_optimizer_attributes(m, "max_iter" => maxiter, "print_level" => verbosity, "tol" => tol) # tol here is something else
	elseif method == :nlopt
		m = JuMP.Model(NLopt.Optimizer)
		JuMP.set_optimizer_attributes(m, "algorithm" => algorithm, "maxeval" => maxiter) # "xtol_abs" => tolX, "ftol_abs" => tol
	end
	JuMP.@variable(m, mixer[i=1:nummixtures, j=1:numbuckets], start = convert(T, Winit[i, j]))
	JuMP.@variable(m, buckets[i=1:numbuckets, j=1:numconstituents], start = convert(T, Hinit[i, j]))
	if !normalize
		JuMP.@constraint(m, buckets .>= 0)
	end
	JuMP.@constraint(m, mixer .>= 0)
	for i = 1:nummixtures
		JuMP.@constraint(m, sum(mixer[i, j] for j=1:numbuckets) == 1.)
	end
	if sizeof(ratios) == 0
		JuMP.@NLobjective(m, Min,
			regularizationweight * sum(sum(log(1. + buckets[i, j])^2 for i=1:numbuckets) for j=1:numconstituents) / numbuckets +
			sum(sum(concweights[i, j] * (sum(mixer[i, k] * buckets[k, j] for k=1:numbuckets) - concentrations[i, j])^2 for i=1:nummixtures) for j=1:numconstituents))
	else
		JuMP.@NLobjective(m, Min,
			regularizationweight * sum(sum(log(1. + buckets[i, j])^2 for i=1:numbuckets) for j=1:numconstituents) / numbuckets +
			sum(sum(concweights[i, j] * (sum(mixer[i, k] * buckets[k, j] for k=1:numbuckets) - concentrations[i, j])^2 for i=1:nummixtures) for j=1:numconstituents) +
			sum(sum(ratiosweightmatrix[i, j] *
					(sum(mixer[i, k] * buckets[k, c1] for k=1:numbuckets) / sum(mixer[i, k] * buckets[k, c2]
					for k=1:numbuckets) - ratios[i, j])^2 for i=1:nummixtures)
					for (j, c1, c2) in zip(1:numberofratios, ratioindices[:,1], ratioindices[:,2])))
	end
	if movie
		Xe = Winit * Hinit
		NMFk.plotnmf(Xe, Winit[:,movieorder], Hinit[movieorder,:]; movie=movie, filename=moviename, frame=1)
	end
	jumpvariables = JuMP.all_variables(m)
	jumpvalues = JuMP.start_value.(jumpvariables)
	if quiet
		Suppressor.@suppress JuMP.optimize!(m)
	else
		JuMP.optimize!(m)
	end
	mixerval = convert(Array{T, 2}, JuMP.value.(mixer))
	bucketval = convert(Array{T, 2}, JuMP.value.(buckets))
	of = JuMP.objective_value(m)
	if movie
		Xe = mixerval * bucketval
		NMFk.plotnmf(Xe, We[:,movieorder], He[movieorder,:]; movie=movie,filename=moviename, frame=2)
		frame += 1
	end
	ofbest = of
	iters = 1
	baditers = 0
	reattempts = 0
	frame = 3
	!quiet && @info("Iteration: $iters Resets: $reattempts Objective function: $of Best: $ofbest")
	while LinearAlgebra.norm(jumpvalues - JuMP.value.(jumpvariables)) > tolX && ofbest > tol && baditers < maxbaditers && reattempts < maxreattempts
		jumpvalues = JuMP.value.(jumpvariables)
		if quiet
			Suppressor.@suppress JuMP.optimize!(m)
		else
			JuMP.optimize!(m)
		end
		if movie
			We = convert(Array{T, 2}, JuMP.value.(mixer))
			He = convert(Array{T, 2}, JuMP.value.(buckets))
			Xe = We * He
			NMFk.plotnmf(Xe, We[:,movieorder], He[movieorder,:]; movie=movie,filename=moviename, frame=frame)
			frame += 1
		end
		of = JuMP.objective_value(m)
		if of < ofbest
			if (ofbest - of) < tolOF
				baditers += 1
			else
				!quiet && @info("Objective function improved substantially (more than $tolOF; $objvalue < $objvalue_best); bad iteration counter reset ...")
				baditers = 0
			end
			mixerval = convert(Array{T, 2}, JuMP.value.(mixer))
			bucketval = convert(Array{T, 2}, JuMP.value.(buckets))
			ofbest = of
		else
			baditers += 1
		end
		if baditers >= maxbaditers
			reattempts += 1
			if reattempts >= maxreattempts
				!quiet && @info("Objective function does not improve substantially! Maximum number of reattempts ($maxreattempts) has been reached; quit!")
			end
			baditers = 0
		else
			!quiet && @info("Objective function does not improve substantially! Reattempts $reattempts Bad iterations $baditers")
		end
		iters += 1
		!quiet && @info("Iteration: $iters Resets: $reattempts Objective function: $of Best: $ofbest")
	end
	isnm = isnan.(mixerval)
	isnb = isnan.(bucketval)
	if sum(isnm) > 0
		@warn("There are NaN's in the W matrix!")
		mixerval[isnm] .= 0
	end
	if sum(isnb) > 0
		@warn("There are NaN's in the H matrix!")
		bucketval[isnb] .= 0
	end
	if sum(isnm) > 0 || sum(isnb) > 0
		@warn("norm: $(LinearAlgebra.norm(concentrations - mixerval * bucketval)) OF: $(ofbest)")
	end
	penalty = regularizationweight * sum(log.(1. .+ bucketval).^2) / numbuckets
	fitquality = ofbest - penalty
	if !quiet
		@info("Final objective function: $ofbest")
		(regularizationweight > 0) && (@info("Final penalty: $penalty"))
		@info("Final fit: $fitquality")
	end
	if !quiet && sizeof(ratios) > 0
		ratiosreconstruction = 0
		for (j, c1, c2) in zip(1:numberofratios, ratioindices[:, 1], ratioindices[:, 2])
			for i = 1:nummixtures
				s1 = 0
				s2 = 0
				for k = 1:numbuckets
					s1 += mixerval[i, k] * bucketval[k, c1]
					s2 += mixerval[i, k] * bucketval[k, c2]
				end
				ratiosreconstruction += ratiosweight * (s1/s2 - ratios[i, j])^2
			end
		end
		!quiet && println("Ratio reconstruction = $ratiosreconstruction")
	end
	if sizeof(ratios) > 0
		ratios[ratios.==0] .= NaN32
	end
	if normalize
		bucketval = denormalizematrix_col!(bucketval, cmin, cmax)
	elseif scale
		bucketval = descalematrix!(bucketval, cmax)
	end
	if movie
		Xe = mixerval * bucketval
		NMFk.plotnmf(Xe, mixerval[:,movieorder], bucketval[movieorder,:]; movie=movie, filename=moviename, frame=frame)
	end
	return mixerval, bucketval, fitquality
end

"Match data with concentrations and deltas (avoid using deltas; convert to concentrations)"
function mixmatchdeltas(concentrations_in::AbstractMatrix{T}, deltas_in::AbstractMatrix{T}, deltaindices::AbstractVector{Int}, numbuckets::Int; method::Symbol=:ipopt, algorithm::Symbol=:LD_LBFGS, normalize::Bool=false, scale::Bool=false, maxH::Bool=false, random::Bool=true, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::T=convert(T, defaultregularizationweight), deltasweight::T=convert(T, defaultdeltasweight), weightinverse::Bool=false, Winit::AbstractMatrix{T}=Array{T}(undef, 0, 0), Hinit::AbstractMatrix{T}=Array{T}(undef, 0, 0), Hinitd::AbstractMatrix{T}=Array{T}(undef, 0, 0), tol::Float64=1e-3, maxbaditers::Int=10, quiet::Bool=NMFk.quiet) where {T <: Number}
	concentrations = copy(concentrations_in) # we may overwrite some of the fields if there are NaN's, so make a copy
	deltas = copy(deltas_in)
	numdeltas = size(deltas, 2)
	nummixtures = size(concentrations, 1)
	numconstituents = size(concentrations, 2)
	if weightinverse
		concweights = convert(Array{T, 2}, 1. ./ concentrations)
		zis = concentrations .== 0
		concweights[zis] = maximum(concentrations[!zis]) * 10
	else
		concweights = ones(T, size(concentrations))
	end
	nans = isnan.(concentrations)
	concentrations[nans] .= 0
	concweights[nans] .= 0
	deltasweights = convert(Array{T, 2}, ones(T, size(deltas)) * deltasweight)
	nans = isnan.(deltas)
	deltas[nans] .= 0
	deltasweights[nans] .= 0
	if normalize
		concentrations, cmax, cmin = normalizematrix_row!(concentrations)
		deltas, dmin, dmax = normalizematrix_row!(deltas)
	elseif scale
		concentrations, cmax = scalematrix_row!(concentrations)
		deltas, dmax = scalematrix_row!(deltas)
	end
	if sizeof(Winit) == 0
		if random
			Winit = rand(T, nummixtures, numbuckets)
		else
			Winit = ones(T, nummixtures, numbuckets) / numbuckets
		end
	end
	if sizeof(Hinit) == 0
		if random
			Hinit = rand(T, numbuckets, numconstituents)
		else
			Hinit = ones(T, numbuckets, numconstituents) / 2
		end
		if maxH
			maxc = maximum(concentrations; dims=1)
			for i=1:numbuckets
				Hinit[i:i,:] .*= maxc
			end
		end
	end
	if sizeof(Hinitd) == 0
		if random
			if scale
				Hinitd = rand(T, numbuckets, numdeltas)
			else
				maxr = vec(maximum(deltas; dims=1) ./ 10)
				Hinitd = rand(T, numbuckets, numdeltas)
				for i=1:numbuckets
					Hinitd[i, :] = Hinitd[i, :] .* maxr
				end
			end
		else
			if scale
				Hinitd = ones(T, numbuckets, numdeltas) / 2
			else
				maxr = vec(maximum(deltas; dims=1))
				Hinitd = Array{T}(undef, numbuckets, numdeltas)
				for i=1:numbuckets
					Hinitd[i,:] = maxr
				end
			end
		end
	end
	if method == :ipopt
		m = JuMP.Model(Ipopt.Optimizer)
		JuMP.set_optimizer_attributes(m, "max_iter" => maxiter, "print_level" => verbosity, "tol" => tol)
	elseif method == :nlopt
		m = JuMP.Model(NLopt.Optimizer)
		JuMP.set_optimizer_attributes(m, "algorithm" => algorithm, "maxeval" => maxiter, "xtol_abs" => tolX, "ftol_abs" => tol)
	end
	JuMP.@variable(m, mixer[i=1:nummixtures, j=1:numbuckets], start = convert(T, Winit[i, j]))
	JuMP.@variable(m, buckets[i=1:numbuckets, j=1:numconstituents], start = convert(T, Hinit[i, j]))
	JuMP.@variable(m, bucketdeltas[i=1:numbuckets, j=1:numdeltas], start = convert(T, Hinitd[i, j]))
	JuMP.@constraint(m, buckets .>= 0)
	JuMP.@constraint(m, mixer .>= 0)
	for i = 1:nummixtures
		JuMP.@constraint(m, sum(mixer[i, j] for j=1:numbuckets) == 1.)
	end
	#=
	for i = 1:numbuckets
		for j = 1:numconstituents
			if i != 1 || j != 1
				JuMP.@constraint(m, buckets[i, j] == Hinit[i, j]) # Fix buckets for testing
			end
		end
	end
	for i = 1:numbuckets
		for j = 1:numdeltas
			#if i != 1 || j != 1
				JuMP.@constraint(m, bucketdeltas[i, j] == Hinitd[i, j]) # Fix buckets for testing
			#end
		end
	end
	=#
	JuMP.@NLobjective(m, Min,
		regularizationweight * sum(sum(log(1. + buckets[i, j])^2 for i=1:numbuckets) for j=1:numconstituents) / numbuckets +
		regularizationweight * sum(sum(log(1. + abs(bucketdeltas[i, j]))^2 for i=1:numbuckets) for j=1:numdeltas) / numbuckets +
		sum(sum(concweights[i, j] * (concentrations[i, j] - (sum(mixer[i, k] * buckets[k, j] for k=1:numbuckets)))^2 for i=1:nummixtures) for j=1:numconstituents) +
		sum(sum(deltasweights[i, di] * (deltas[i, di] - (sum(mixer[i, j] * buckets[j, deltaindices[di]] * bucketdeltas[j, di] for j=1:numbuckets) / sum(mixer[i, j] * buckets[j, deltaindices[di]] for j=1:numbuckets)))^2 for i = 1:nummixtures) for di=1:numdeltas)
		)
	jumpvariables = JuMP.all_variables(m)
	jumpvalues = JuMP.start_value.(jumpvariables)
	JuMP.optimize!(m)
	mixerval = convert(Array{T, 2}, JuMP.value.(mixer))
	bucketval = convert(Array{T, 2}, JuMP.value.(buckets))
	bucketdeltasval = convert(Array{T, 2}, JuMP.value.(bucketdeltas))
	of = JuMP.objective_value(m)
	ofbest = of
	iters = 1
	!quiet && @info("Iteration: $iters Objective function: $of Best: $ofbest")
	while !(LinearAlgebra.norm(jumpvalues - JuMP.value.(jumpvariables)) < tol) && iters < maxbaditers # keep doing the optimization until we really reach an optimum
		jumpvalues = JuMP.value.(jumpvariables)
		JuMP.optimize!(m)
		of = JuMP.objective_value(m)
		!quiet && @info("Iteration: $iters Objective function: $of Best: $ofbest")
		if of < ofbest
			iters = 0
			mixerval = convert(Array{T, 2}, JuMP.value.(mixer))
			bucketval = convert(Array{T, 2}, JuMP.value.(buckets))
			bucketdeltasval = convert(Array{T, 2}, JuMP.value.(bucketdeltas))
			ofbest = of
		end
		iters += 1
	end
	!quiet && @info("Iteration: $iters Objective function: $of Best: $ofbest")
	fitquality = ofbest - regularizationweight * sum(log.(1. .+ bucketval).^2) / numbuckets - regularizationweight * sum(log.(1. .+ abs.(bucketdeltasval)).^2) / numbuckets
	if normalize
		bucketval = denormalizematrix_cool!(bucketval, cmin, cmax)
		bucketdeltasval = denormalizematrix_col!(bucketdeltasval, dmin, dmax)
	elseif scale
		bucketval = descalematrix!(bucketval, cmax)
		bucketdeltasval = descalematrix!(bucketdeltasval, dmax)
	end
	return mixerval, bucketval, bucketdeltasval, fitquality
end

"Match data with only deltas"
function mixmatchwaterdeltas(deltas::AbstractMatrix{T}, numbuckets::Int; method::Symbol=:ipopt, algorithm::Symbol=:LD_LBFGS, random::Bool=true, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::T=convert(T, defaultregularizationweight), maxdeltaguess::T=convert(T, 1000.), bucketmeans::AbstractMatrix{T}=zeros(numbuckets, 2)) where {T <: Number}
	deltas = copy(deltas) # we may overwrite some of the fields if there are NaN's, so make a copy
	nummixtures = size(deltas, 1)
	numconstituents = 2
	m = JuMP.Model(Ipopt.Optimizer)
	JuMP.set_optimizer_attributes(m, "max_iter" => maxiter, "print_level" => verbosity)
	if random
		JuMP.@variable(m, mixer[1:nummixtures, 1:numbuckets] >= 0., start=randn(T))
		JuMP.@variable(m, buckets[1:numbuckets, 1:numconstituents], start=maxdeltaguess * rand(T))
	else
		JuMP.@variable(m, mixer[1:nummixtures, 1:numbuckets] >= 0.)
		JuMP.@variable(m, buckets[1:numbuckets, 1:numconstituents])
	end
	JuMP.@constraint(m, mixer .<= 1.)
	for i in axes(deltas, 1)
		JuMP.@constraint(m, sum(mixer[i, j] for j=1:numbuckets) == 1.)
	end
	concweights = ones(T, size(deltas))
	nans = isnan(deltas)
	deltas[nans] = 0
	concweights[nans] = 0
	JuMP.@NLobjective(m, Min,
		regularizationweight * sum(sum((buckets[i, j] - bucketmeans[i, j])^2 for i=1:numbuckets) for j=1:numconstituents) / numbuckets +
		sum(sum(concweights[i, j] * (sum(mixer[i, k] * buckets[k, j] for k=1:numbuckets) - deltas[i, j])^2 for i=1:nummixtures) for j=1:numconstituents))
	if method == :ipopt
		m = JuMP.Model(Ipopt.Optimizer)
		JuMP.set_optimizer_attributes(m, "max_iter" => maxiter, "print_level" => verbosity, "tol" => tol)
	elseif method == :nlopt
		m = JuMP.Model(NLopt.Optimizer)
		JuMP.set_optimizer_attributes(m, "algorithm" => algorithm, "maxeval" => maxiter, "xtol_abs" => tolX, "ftol_abs" => tol)
	end
	JuMP.optimize!(m)
	mixerval = convert(Array{T,2 }, JuMP.value.(mixer))
	bucketval = convert(Array{T, 2}, JuMP.value.(buckets))
	fitquality = JuMP.objective_value(m) - regularizationweight * sum((bucketval - bucketmeans).^2) / numbuckets
	return mixerval, bucketval, fitquality
end
