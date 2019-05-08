import JuMP
import Ipopt
import NLopt
import Suppressor

const defaultregularizationweight = convert(Float32, 0)
const defaultmaxiter = 1000
const defaultverbosity = 0

"Iterative factorization of matrix X (X = W * H) using Ipopt fixing W and H matrices"
function jumpiter(X::AbstractMatrix, nk::Int; kw...)
	m, n = size(X)
	jumpiter(convert(Array{Float32, 2}, X), nk, convert(Array{Float32, 2}, rand(m, nk)), convert(Array{Float32, 2}, rand(nk, n)); kw...)
end
function jumpiter(X::AbstractMatrix{Float32}, nk::Int, W::AbstractMatrix{Float32}, H::AbstractMatrix{Float32}; iter::Int=100, tolerance::Float64=1e-2, quiet::Bool=NMFk.quiet, kw...)
	m, n = size(X)
	mw, k = size(W)
	k, nh = size(H)
	@assert m == mw
	@assert n == nh
	@assert k == nk
	fit = 0
	W, H, oldfit = NMFk.jump(X, nk; initW=W, initH=H, fixH=true, quiet=true, kw...)
	!quiet && println("of: $(oldfit)")
	for i = 1:iter
		W, H, fit = NMFk.jump(X, nk; initW=convert(Array{Float32, 2}, W), initH=convert(Array{Float32, 2}, H), fixW=true, quiet=true, kw...)
		!quiet && println("of: $(fit)")
		W, H, fit = NMFk.jump(X, nk; initW=convert(Array{Float32, 2}, W), initH=convert(Array{Float32, 2}, H), fixH=true, quiet=true, kw...)
		!quiet && println("of: $(fit)")
		if oldfit - fit > tolerance
			oldfit = fit
		else
			break
		end
	end
	return W, H, fit
end

"Factorize matrix X (X = W * H) using Ipopt for each row of X/H"
function jumpHrows(X::AbstractMatrix{Float32}, nk::Int, W::AbstractMatrix{Float32}, H::AbstractMatrix{Float32}; quiet::Bool=NMFk.quiet, kw...)
	fit = 0
	for i = 1:size(X, 2)
		fitrowold = sum((X[:,i] .- W * H[:,i]).^2)
		W, H[:,i], fitrow = NMFk.jump(X[:,i], nk; initW=convert(Array{Float32, 2}, W), initH=convert(Array{Float32, 1}, H[:,i]), fixW=true, quiet=true, kw...)
		!quiet && println("of: $(fitrowold) -> $(fitrow)")
		fit += fitrow
	end
	return W, H, fit
end

"Factorize matrix X (X = W * H) using Ipopt"
function jump(X::AbstractMatrix{Float64}, nk::Int; kw...)
	jump(convert(Array{Float32, 2}, X), nk; kw...)
end
function jump(X::AbstractArray{Float32}, nk::Int; method::Symbol=:nlopt, algorithm::Symbol=:LD_LBFGS, maxW::Bool=false, maxH::Bool=false, random::Bool=true, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Number=defaultregularizationweight, weightinverse::Bool=false, initW::AbstractMatrix=Array{Float32}(undef, 0, 0), initH::AbstractArray=Array{Float32}(undef, 0, 0), tolX::Float64=1e-3, tol::Float64=1e-3, tolOF::Float64=1e-3, maxresets::Int=-1, maxouteriters::Int=10, quiet::Bool=NMFk.quiet, kullbackleibler=false, fixW::Bool=false, fixH::Bool=false, seed::Number=-1, nonnegW::Bool=true, nonnegH::Bool=true, constrainW::Bool=false, constrainH::Bool=false, movie::Bool=false, moviename::AbstractString="", movieorder=1:nk, moviecheat::Integer=0, cheatlevel::Number=1)
	if seed >= 0
		Random.seed!(seed)
	end
	if weightinverse
		obsweights = convert(Array{Float32,2}, 1. ./ X)
		zis = X .== 0
		obsweights[zis] = maximum(X[!zis]) * 10
	else
		obsweights = ones(Float32, size(X))
	end
	nans = isnan.(X)
	X[nans] .= 0
	obsweights[nans] .= 0
	nummixtures = size(X, 1)
	numconstituents = size(X, 2)
	if sizeof(initW) == 0
		fixW = false
		if random
			initW = rand(Float32, nummixtures, nk)
		else
			initW = ones(Float32, nummixtures, nk) / nk
		end
		if maxW
			maxx = maximum(X; dims=2)
			for i=1:nk
				initW[:,i:i] .*= maxx
			end
		end
		nansw = 0
	else
		@assert size(initW) == (nummixtures, nk)
		nansw = isnan.(initW)
		initW[nansw] .= 0
	end
	if sizeof(initH) == 0
		fixH = false
		if random
			initH = rand(Float32, nk, numconstituents)
		else
			initH = ones(Float32, nk, numconstituents) / 2
		end
		if maxH
			maxx = maximum(X; dims=1)
			for i=1:nk
				initH[i:i,:] .*= maxx
			end
		end
		nansh = 0
	else
		@assert size(initH) == (nk, numconstituents)
		nansh = isnan.(initH)
		initH[nansh] .= 0
	end
	if method == :ipopt
		m = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer, max_iter=maxiter, print_level=verbosity, tol=tol))
	elseif method == :nlopt
		@warn "NLopt does not work with JuMP; Ipopt will be used!"
		m = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer, algorithm=algorithm, maxeval=maxiter, xtol_abs=tolX, ftol_abs=tol))
	end
	#IMPORTANT the order at which parameters are defined is very important
	if fixW
		W = initW
	else
		constrainW && normalizematrix!(initW)
		@JuMP.variable(m, W[i=1:nummixtures, j=1:nk], start=convert(Float32, initW[i, j]))
		nonnegW && @JuMP.constraint(m, W .>= 0)
		constrainW && @JuMP.constraint(m, W .<= 1)
	end
	if fixH
		H = initH
	else
		constrainH && normalizematrix!(initH)
		@JuMP.variable(m, H[i=1:nk, j=1:numconstituents], start=convert(Float32, initH[i, j]))
		nonnegH && @JuMP.constraint(m, H .>= 0)
		constrainH && @JuMP.constraint(m, H .<= 1)
	end
	if kullbackleibler
		smallnumber = eps(Float64)
		@JuMP.NLobjective(m, Min, sum(X[i, j] * (log(smallnumber + X[i, j]) - log(smallnumber + sum(W[i, k] * H[k, j] for k = 1:nk))) - X[i, j] + sum(W[i, k] * H[k, j] for k = 1:nk) for i=1:nummixtures, j=1:numconstituents))
	else
		if regularizationweight == 0.
			@JuMP.NLobjective(m, Min,
				sum(sum(obsweights[i, j] * (sum(W[i, k] * H[k, j] for k=1:nk) - X[i, j])^2 for i=1:nummixtures) for j=1:numconstituents))
		else
			if fixH
				@JuMP.NLobjective(m, Min,
					regularizationweight * sum(sum(log(1. + W[i, j])^2 for i=1:numconstituents) for j=1:nk) / nk +
					sum(sum(obsweights[i, j] * (sum(W[i, k] * H[k, j] for k=1:nk) - X[i, j])^2 for i=1:nummixtures) for j=1:numconstituents))
			else
				@JuMP.NLobjective(m, Min,
					regularizationweight * sum(sum(log(1. + H[i, j])^2 for i=1:nk) for j=1:numconstituents) / nk +
					sum(sum(obsweights[i, j] * (sum(W[i, k] * H[k, j] for k=1:nk) - X[i, j])^2 for i=1:nummixtures) for j=1:numconstituents))
			end
		end
	end
	if movie
		Xe = initW * initH
		NMFk.plotnmf(Xe, initW[:,movieorder], initH[movieorder,:]; movie=movie, filename=moviename, frame=1)
	end
	jumpvariables = JuMP.all_variables(m)
	jumpvalues = JuMP.start_value.(jumpvariables)
	if quiet
		@Suppressor.suppress JuMP.optimize!(m)
	else
		JuMP.optimize!(m)
	end

	Wbest = (fixW) ? W : JuMP.value.(W)
	Hbest = (fixH) ? H : JuMP.value.(H)
	of = JuMP.objective_value(m)
	!quiet && @info("Objective function $of")
	ofbest = of
	objvalue = ofbest - regularizationweight * sum(log.(1. .+ Hbest).^2) / nk
	frame = 2
	iters = 1
	outiters = 0
	resets = 0
	#TODO this does not work; JuMP fails when restarted
	while !(norm(jumpvalues - JuMP.value.(jumpvariables)) < tolX) && !(objvalue < tol) && outiters < maxouteriters && resets <= maxresets
		jumpvalues = JuMP.value.(jumpvariables)
		if movie
			for mcheat = 1:moviecheat
				We = JuMP.value.(W)
				We .+= rand(size(We)...) .* cheatlevel
				He = JuMP.value.(H)
				He .+= rand(size(He)...) .* cheatlevel
				Xe = We * He
				NMFk.plotnmf(Xe, We[:,movieorder], He[movieorder,:]; movie=movie, filename=moviename, frame=frame)
				frame += 1
			end
			!fixW && (We = JuMP.value.(W))
			!fixH && (He = JuMP.value.(H))
			Xe = We * He
			NMFk.plotnmf(Xe, We[:,movieorder], He[movieorder,:]; movie=movie, filename=moviename, frame=frame)
			frame += 1
		end
		if quiet
			@Suppressor.suppress JuMP.optimize!(m)
		else
			JuMP.optimize!(m)
		end
		of = JuMP.objective_value(m)
		outiters += 1
		iters += 1
		if of < ofbest
			if (ofbest - of) > tolOF
				resets += 1
				if resets > maxresets
					!quiet && @warn("Maximum number of resets has been reached; quit!")
				else
					!quiet && @warn("Objective function improved substantially (more than $tolOF; $of < $ofbest); iteration counter reset ...")
					outiters = 0
				end
			end
			!fixW && (Wbest = JuMP.value.(W))
			!fixH && (Hbest = JuMP.value.(H))
			ofbest = of
			objvalue = ofbest - regularizationweight * sum(log.(1. + Hbest).^2) / nk
		end
		if !quiet
			@info("Iteration $iters")
			@info("Objective function $of")
			(regularizationweight > 0.) && @info("Objective function - regularization penalty $objvalue")
			@info("Parameter norm: $(norm(jumpvalues - JuMP.value.(jumpvariables)))")
		end
	end
	X[nans] .= NaN
	isnm = isnan.(Wbest)
	isnb = isnan.(Hbest)
	if sum(isnm) > 0
		@warn("There are NaN's in the W matrix!")
		Wbest[isnm] .= 0
	end
	if sum(isnb) > 0
		@warn("There are NaN's in the H matrix!")
		Hbest[isnb] .= 0
	end
	penalty = regularizationweight * sum(log.(1. .+ Hbest).^2) / nk
	fitquality = ofbest - penalty
	if sum(isnm) > 0 || sum(isnb) > 0
		@warn("SSQR: $(ssqrnan(X - Wbest * Hbest)) OF: $(fitquality)")
	end
	if !quiet
		@info("Final objective function: $ofbest")
		(regularizationweight > 0.) && @info("Final penalty: $penalty")
		@info("Final fit: $fitquality")
	end
	if movie
		Xe = Wbest * Hbest
		NMFk.plotnmf(Xe, Wbest[:,movieorder], Hbest[movieorder,:]; movie=movie, filename=moviename, frame=frame)
	end
	if sum(nansw) > 0
		initW[nansw] .= NaN
		Wbest[nansw] .= NaN
	end
	if sum(nansh) > 0
		initH[nansh] .= NaN
		Hbest[nansh] .= NaN
	end
	return Wbest, Hbest, fitquality
end
