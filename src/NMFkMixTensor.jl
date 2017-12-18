import JuMP
import Ipopt

"Match data with concentrations and an option for ratios (avoid using ratios; convert to concentrations)"
function mixmatchdata(concentrations::Array{Float32, 3}, numbuckets::Int; method::Symbol=:ipopt, algorithm::Symbol=:LD_SLSQP, normalize::Bool=false, scale::Bool=false, ratios::Array{Float32, 2}=Array{Float32}(0, 0), ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array{Int}(0, 0), seed::Number=-1, random::Bool=false, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Float32=defaultregularizationweight, ratiosweight::Float32=defaultratiosweight, weightinverse::Bool=false, initW::Matrix{Float32}=Array{Float32}(0, 0), initH::Matrix{Float32}=Array{Float32}(0, 0), tolX::Float64=1e-3, tol::Float64=1e-3, maxouteriters::Int=10, quiet::Bool=true, movie::Bool=false, moviename::AbstractString="", movieorder=1:numbuckets)
	if seed >= 0
		srand(seed)
	end
	if weightinverse
		concweights = convert(Array{Float32,2}, 1. ./ concentrations)
		zis = concentrations .== 0
		concweights[zis] = maximum(concentrations[!zis]) * 10
	else
		concweights = ones(Float32, size(concentrations))
	end
	nummixtures, numconstituents, ntimes = size(concentrations)
	nans = isnan.(concentrations)
	concweights[nans] = 0
	concentrations[nans] = 0
	if normalize
		concentrations, cmin, cmax = normalizearrar(concentrations)
	elseif scale
		concentrations, cmax = scalearray(concentrations)
	end
	if sizeof(initW) == 0
		if random
			initW = rand(Float32, nummixtures, numbuckets, ntimes)
		else
			initW = ones(Float32, nummixtures, numbuckets, ntimes) / numbuckets
		end
	end
	if sizeof(initH) == 0
		if random
			if scale || normalize
				initH = rand(Float32, numbuckets, numconstituents)
			else
				maxconc = vec(maximum(concentrations, (1,3)))
				initH = rand(Float32, numbuckets, numconstituents)
				for i=1:numbuckets
					initH[i:i, :] .*= maxconc
				end
			end
		else
			if scale || normalize
				initH = ones(Float32, numbuckets, numconstituents) / 2
			else
				maxconc = vec(maximum(concentrations, (1,3)))
				initH = Array{Float32}(numbuckets, numconstituents)
				for i=1:numbuckets
					initH[i:i, :] = maxconc
				end
			end
		end
	end
	if method == :ipopt
		m = JuMP.Model(solver=Ipopt.IpoptSolver(max_iter=maxiter, print_level=verbosity)) # tol here is something else
	elseif method == :nlopt
		m = JuMP.Model(solver=NLopt.NLoptSolver(algorithm=algorithm, maxeval=maxiter)) # xtol_abs=tolX, ftol_abs=tol
	end
	@JuMP.variable(m, mixer[i=1:nummixtures, j=1:numbuckets, k=1:ntimes], start = convert(Float32, initW[i, j, k]))
	@JuMP.variable(m, buckets[i=1:numbuckets, j=1:numconstituents], start = convert(Float32, initH[i, j]))
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
	oldcolval = copy(m.colVal)
	if movie
		Xe = initW * initH
		NMFk.plotnmf(Xe, initW[:,movieorder], initH[movieorder,:]; movie=movie, filename=moviename, frame=1)
	end
	status = JuMP.solve(m)
	W = JuMP.getvalue(mixer)
	H = JuMP.getvalue(buckets)
	of = JuMP.getobjectivevalue(m)
	!quiet && @show of
	of_best = of
	iters = 0
	frame = 2
	while !(norm(oldcolval - m.colVal) < tolX) && !(of_best < tol) && iters < maxouteriters
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
			iters = 0
			W = JuMP.getvalue(mixer)
			H = JuMP.getvalue(buckets)
			of_best = of
		end
		iters += 1
	end
	!quiet && @show of_best
	fitquality = of_best - regularizationweight * sum(log.(1. + H).^2) / numbuckets
	concentrations[nans] = NaN32
	setbadmixerelements!(concentrations, W)
	if normalize
		H = denormalizematrix(H, W, cmin, cmax)
	elseif scale
		H = descalearray(H, cmax)
	end
	if movie
		NMFk.plotnmf(Xe, W[:,movieorder], H[movieorder,:]; movie=movie, filename=moviename, frame=frame)
	end
	return abs.(convert(Array{Float32, 3}, W)), abs.(convert(Array{Float32, 2}, H)), fitquality
end

function setbadmixerelements!(X::Array, W::Array)
	nw, nc, nt = size(X)
	for t = 1:nt
		for w = 1:nw
			if !any(.!isnan.(X[w, :, t]))
				W[w, :, t] = NaN32
			end
		end
	end
end

function mixmatchcompute(X::Array{Float32, 3}, W::Array{Float32, 3}, H::Array{Float32, 2})
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
	Xe[isnan.(X)] = NaN
	return Xe
end