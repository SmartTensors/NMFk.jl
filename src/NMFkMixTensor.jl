import JuMP
import Ipopt
import Suppressor

"Match data with concentrations and an option for ratios (avoid using ratios; convert to concentrations)"
function mixmatchdata(concentrations::AbstractArray{T, 3}, numbuckets::Int; method::Symbol=:ipopt, algorithm::Symbol=:LD_SLSQP, normalize::Bool=false, scale::Bool=false, maxH::Bool=false, ratios::AbstractArray{T, 2}=Array{T}(undef, 0, 0), ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array{Int}(undef. 0, 0), seed::Number=-1, random::Bool=true, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::T=defaultregularizationweight, ratiosweight::T=defaultratiosweight, weightinverse::Bool=false, initW::Matrix{T}=Array{T}(undef, 0, 0), initH::Matrix{T}=Array{T}(undef, 0, 0), tolX::Float64=1e-3, tol::Float64=1e-3, tolOF::Float64=1e-3, maxresets::Int=-1, maxouteriters::Int=10, quiet::Bool=NMFk.quiet, movie::Bool=false, moviename::AbstractString="", movieorder=1:numbuckets) where {T <:Float32}
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
	concweights[nans] = 0
	concentrations[nans] = 0
	if normalize
		concentrations, cmin, cmax = normalizearrar!(concentrations)
	elseif scale
		concentrations, cmax = scalearray!(concentrations)
	end
	if sizeof(initW) == 0
		if random
			initW = rand(T, nummixtures, numbuckets, ntimes)
		else
			initW = ones(T, nummixtures, numbuckets, ntimes) / numbuckets
		end
	end
	if sizeof(initH) == 0
		if random
			initH = rand(T, numbuckets, numconstituents)
		else
			initH = ones(T, numbuckets, numconstituents) / 2
		end
		if maxH
			maxconc = permutedims(vec(maximum(concentrations, dims=(1,3))))
			for i=1:numbuckets
				initH[i:i, :] .*= maxconc
			end
		end
	end
	if method == :ipopt
		m = JuMP.Model(solver=Ipopt.IpoptSolver(max_iter=maxiter, print_level=verbosity)) # tol here is something else
	elseif method == :nlopt
		m = JuMP.Model(solver=NLopt.NLoptSolver(algorithm=algorithm, maxeval=maxiter)) # xtol_abs=tolX, ftol_abs=tol
	end
	@JuMP.variable(m, mixer[i=1:nummixtures, j=1:numbuckets, k=1:ntimes], start = convert(T, initW[i, j, k]))
	@JuMP.variable(m, buckets[i=1:numbuckets, j=1:numconstituents], start = convert(T, initH[i, j]))
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
	of = JuMP.getobjectivevalue(m)
	ofbest = of
	iters = 1
	outiters = 0
	resets = 0
	frame = 2
	!quiet && @info("Iteration: $iters Resets: $resets Objective function: $of Best: $ofbest")
	while norm(jumpvalues - JuMP.value.(jumpvariables)) > tolX && ofbest > tol && outiters < maxouteriters && resets <= maxresets
		jumpvalues = JuMP.value.(jumpvariables)
		if quiet
			@Suppressor.suppress JuMP.optimize!(m)
		else
			JuMP.optimize!(m)
		end
		of = JuMP.getobjectivevalue(m)
		iters += 1
		outiters += 1
		if of < ofbest
			if (ofbest - of) > tolOF
				resets += 1
				if resets > maxresets
					@warn("Maximum number of resets has been reached; quit!")
				else
					@warn("Objective function improved substantially (more than $tolOF; $of < $ofbest); iteration counter reset ...")
					outiters = 0
				end
			end
			W = convert(Array{T, 3}, JuMP.value.(mixer))
			H = convert(Array{T, 2}, JuMP.value.(buckets))
			ofbest = of
		else
			outiters = maxouteriters + 1
		end
		!quiet && @info("Iteration: $iters Resets: $resets Objective function: $of Best: $ofbest")
	end
	concentrations[nans] = NaN
	fitquality = ofbest - regularizationweight * sum(log.(1. + H).^2) / numbuckets
	setbadmixerelements!(concentrations, W)
	if normalize
		H = denormalizematrix!(H, W, cmin, cmax)
	elseif scale
		H = descalearray!(H, cmax)
	end
	return abs.(W), abs.(H), fitquality
end

function setbadmixerelements!(X::AbstractArray, W::AbstractArray)
	nw, nc, nt = size(X)
	for t = 1:nt
		for w = 1:nw
			if !any(.!isnan.(X[w, :, t]))
				W[w, :, t] = NaN
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
	Xe[isn] = NaN
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
				W[w,:,t] = NaN
			end
		end
	end
end