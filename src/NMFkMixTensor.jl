import JuMP
import Ipopt
import LinearAlgebra
import Suppressor

"Match data with concentrations and an option for ratios (avoid using ratios; convert to concentrations)"
function mixmatchdata(concentrations::AbstractArray{T, 3}, numbuckets::Int; method::Symbol=:ipopt, algorithm::Symbol=:LD_SLSQP, normalize::Bool=false, scale::Bool=false, maxH::Bool=false, ratios::AbstractArray{T, 2}=Array{T}(undef, 0, 0), ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array{Int}(undef, 0, 0), seed::Number=-1, random::Bool=true, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::T=convert(T, defaultregularizationweight), ratiosweight::T=convert(T, defaultratiosweight), weightinverse::Bool=false, Winit::Matrix{T}=Array{T}(undef, 0, 0), Hinit::Matrix{T}=Array{T}(undef, 0, 0), tolX::Float64=1e-3, tol::Float64=1e-3, tolOF::Float64=1e-3, maxreattempts::Int=1, maxbaditers::Int=5, quiet::Bool=NMFk.quiet, movie::Bool=false, moviename::AbstractString="", movieorder=1:numbuckets) where {T <: Float32}
	if seed >= 0
		Random.seed!(seed)
	end
	if weightinverse
		concweights = convert(Array{T,2}, 1. ./ concentrations)
		zis = concentrations .== 0
		concweights[zis] = maximum(concentrations[!zis]) * 10
	else
		concweights = ones(T, size(concentrations))
	end
	nummixtures, numconstituents, ntimes = size(concentrations)
	nans = isnan.(concentrations)
	concweights[nans] .= 0
	concentrations[nans] .= 0
	if normalize
		concentrations, cmin, cmax = normalizearrar!(concentrations)
	elseif scale
		concentrations, cmax = scalearray!(concentrations)
	end
	if sizeof(Winit) == 0
		if random
			Winit = rand(T, nummixtures, numbuckets, ntimes)
		else
			Winit = ones(T, nummixtures, numbuckets, ntimes) / numbuckets
		end
	end
	if sizeof(Hinit) == 0
		if random
			Hinit = rand(T, numbuckets, numconstituents)
		else
			Hinit = ones(T, numbuckets, numconstituents) / 2
		end
		if maxH
			maxconc = permutedims(vec(maximum(concentrations, dims=(1,3))))
			for i=1:numbuckets
				Hinit[i:i, :] .*= maxconc
			end
		end
	end
	if method == :ipopt
		m = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer; max_iter=maxiter, print_level=verbosity)) # tol here is something else
	elseif method == :nlopt
		m = JuMP.Model(JuMP.with_optimizer(NLopt.Optimizer; algorithm=algorithm, maxeval=maxiter)) # xtol_abs=tolX, ftol_abs=tol
	end
	@JuMP.variable(m, mixer[i=1:nummixtures, j=1:numbuckets, k=1:ntimes], start = convert(T, Winit[i, j, k]))
	@JuMP.variable(m, buckets[i=1:numbuckets, j=1:numconstituents], start = convert(T, Hinit[i, j]))
	if !normalize
		@JuMP.constraint(m, buckets .>= 0)
	end
	@JuMP.constraint(m, mixer .>= 0)
	for k = 1:ntimes
		for i = 1:nummixtures
			@JuMP.constraint(m, sum(mixer[i, j, k] for j=1:numbuckets) == 1.)
		end
	end
	@JuMP.NLobjective(m, Min,
		regularizationweight * sum(sum(log(1. + buckets[i, j])^2 for i=1:numbuckets) for j=1:numconstituents) / numbuckets +
		sum(sum(sum(concweights[i, j, t] * (sum(mixer[i, k, t] * buckets[k, j] for k=1:numbuckets) - concentrations[i, j, t])^2 for i=1:nummixtures) for j=1:numconstituents) for t=1:ntimes))
	jumpvariables = JuMP.all_variables(m)
	jumpvalues = JuMP.start_value.(jumpvariables)
	if quiet
		@Suppressor.suppress JuMP.optimize!(m)
	else
		JuMP.optimize!(m)
	end
	W = convert(Array{T, 3}, JuMP.value.(mixer))
	H = convert(Array{T, 2}, JuMP.value.(buckets))
	of = JuMP.objective_value(m)
	ofbest = of
	iters = 1
	baditers = 0
	reattempts = 0
	frame = 2
	!quiet && @info("Iteration: $iters Resets: $reattempts Objective function: $of Best: $ofbest")
	while LinearAlgebra.norm(jumpvalues - JuMP.value.(jumpvariables)) > tolX && ofbest > tol && baditers < maxbaditers && reattempts < maxreattempts
		jumpvalues = JuMP.value.(jumpvariables)
		if quiet
			@Suppressor.suppress JuMP.optimize!(m)
		else
			JuMP.optimize!(m)
		end
		of = JuMP.objective_value(m)
		iters += 1
		if of < ofbest
			if (ofbest - of) < tolOF
				baditers += 1
			else
				!quiet && @info("Objective function improved substantially (more than $tolOF; $objvalue < $objvalue_best); bad iteration counter reset ...")
				baditers = 0
			end
			W = convert(Array{T, 3}, JuMP.value.(mixer))
			H = convert(Array{T, 2}, JuMP.value.(buckets))
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
		!quiet && @info("Iteration: $iters Resets: $reattempts Objective function: $of Best: $ofbest")
	end
	concentrations[nans] .= NaN
	fitquality = ofbest - regularizationweight * sum(log.(1. .+ H).^2) / numbuckets
	# setbadmixerelements!(concentrations, W, H) this is not needed
	if normalize
		H = denormalizematrix!(H, W, cmin, cmax)
	elseif scale
		H = descalearray!(H, cmax)
	end
	return abs.(W), abs.(H), fitquality
end

function setbadmixerelements!(X::AbstractArray, W::AbstractArray, H::AbstractArray) # this function is not needed
	nw, nc, nt = size(X)
	for t = 1:nt
		for w = 1:nw
			@show X[w, :, t]
			Xe = mixmatchcompute(X, W, H)
			@show Xe[w, :, t]
			if any(.!isnan.(X[w, :, t]))
				W[w, :, t] .= NaN
			end
		end
	end
end

function mixmatchcompute(X::AbstractArray{T, 3}, W::AbstractArray{T, 3}, H::AbstractArray{T, 2}, isn=isnan.(X)) where {T}
	nummixtures, numconstituents, ntimes = size(X)
	nummixtures2, numbuckets, ntimes2 = size(W)
	numbuckets2, numconstituents2 = size(H)
	@assert nummixtures == nummixtures2
	@assert numconstituents == numconstituents2
	@assert ntimes == ntimes2
	@assert numbuckets == numbuckets2
	Xe = zeros(nummixtures, numconstituents, ntimes)
	for t=1:ntimes
		for j=1:numconstituents
			for i=1:nummixtures
				for k=1:numbuckets
					Xe[i, j, t] +=  W[i, k, t] * H[k, j]
				end
			end
		end
	end
	Xe[isn] .= NaN
	return convert(AbstractArray{T, 3}, Xe)
end

function mixmatchcompute(W::AbstractArray{T, 3}, H::AbstractArray{T, 2}) where {T}
	nummixtures, numbuckets, ntimes = size(W)
	numbuckets2, numconstituents = size(H)
	@assert numbuckets == numbuckets2
	Xe = zeros(nummixtures, numconstituents, ntimes)
	for t=1:ntimes
		for j=1:numconstituents
			for i=1:nummixtures
				for k=1:numbuckets
					Xe[i, j, t] +=  W[i, k, t] * H[k, j]
				end
			end
		end
	end
	return convert(Array{T, 3}, Xe)
end

function fixmixers!(X::AbstractArray{T, 3}, W::AbstractArray{T, 3}) where {T}
	nw, nc, nt = size(X)
	for t = 1:nt
		for w = 1:nw
			if !any(.!isnan.(X[w,:,t]))
				W[w,:,t] .= NaN
			end
		end
	end
end