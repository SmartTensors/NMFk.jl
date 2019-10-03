import JuMP
import Ipopt
import NLopt
import Suppressor
import LinearAlgebra

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
	W, H, oldfit = NMFk.jump(X, nk; Winit=W, Hinit=H, Hfixed=true, quiet=true, kw...)
	!quiet && println("of: $(oldfit)")
	for i = 1:iter
		W, H, fit = NMFk.jump(X, nk; Winit=convert(Array{Float32, 2}, W), Hinit=convert(Array{Float32, 2}, H), Wfixed=true, quiet=true, kw...)
		!quiet && println("of: $(fit)")
		W, H, fit = NMFk.jump(X, nk; Winit=convert(Array{Float32, 2}, W), Hinit=convert(Array{Float32, 2}, H), Hfixed=true, quiet=true, kw...)
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
		W, H[:,i], fitrow = NMFk.jump(X[:,i], nk; Winit=convert(Array{Float32, 2}, W), Hinit=convert(Array{Float32, 1}, H[:,i]), Wfixed=true, quiet=true, kw...)
		!quiet && println("of: $(fitrowold) -> $(fitrow)")
		fit += fitrow
	end
	return W, H, fit
end

"Factorize matrix X (X = W * H) using Ipopt"
function jump(X::AbstractMatrix{Float64}, nk::Int; kw...)
	jump(convert(Array{Float32, 2}, X), nk; kw...)
end
function jump(X::AbstractArray{Float32}, nk::Int; method::Symbol=:nlopt, algorithm::Symbol=:LD_LBFGS, maxW::Bool=false, maxH::Bool=false, random::Bool=true, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Number=defaultregularizationweight, weightinverse::Bool=false, Winit::AbstractMatrix=Array{Float32}(undef, 0, 0), Hinit::AbstractArray=Array{Float32}(undef, 0, 0), tolX::Float64=1e-3, tol::Float64=1e-3, tolOF::Float64=1e-3, maxreattempts::Int=1, maxbaditers::Int=5, quiet::Bool=NMFk.quiet, kullbackleibler=false, Wfixed::Bool=false, Hfixed::Bool=false, seed::Number=-1, Wnonneg::Bool=true, Hnonneg::Bool=true, constrainW::Bool=false, constrainH::Bool=false, movie::Bool=false, moviename::AbstractString="", movieorder=1:nk, moviecheat::Integer=0, cheatlevel::Number=1)
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
	if sizeof(Winit) == 0
		Wfixed = false
		if random
			Winit = rand(Float32, nummixtures, nk)
		else
			Winit = ones(Float32, nummixtures, nk) / nk
		end
		if maxW
			maxx = maximum(X; dims=2)
			for i=1:nk
				Winit[:,i:i] .*= maxx
			end
		end
		nansw = 0
	else
		@assert size(Winit) == (nummixtures, nk)
		nansw = isnan.(Winit)
		Winit[nansw] .= 0
	end
	if sizeof(Hinit) == 0
		Hfixed = false
		if random
			Hinit = rand(Float32, nk, numconstituents)
		else
			Hinit = ones(Float32, nk, numconstituents) / 2
		end
		if maxH
			maxx = maximum(X; dims=1)
			for i=1:nk
				Hinit[i:i,:] .*= maxx
			end
		end
		nansh = 0
	else
		@assert size(Hinit) == (nk, numconstituents)
		nansh = isnan.(Hinit)
		Hinit[nansh] .= 0
	end
	if method == :ipopt
		m = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer, max_iter=maxiter, print_level=verbosity, tol=tol))
	elseif method == :nlopt
		@warn "NLopt does not work with JuMP; Ipopt will be used!"
		# m = JuMP.Model(JuMP.with_optimizer(NLopt.Optimizer, algorithm=algorithm, maxeval=maxiter, xtol_abs=tolX, ftol_abs=tol))
		m = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer, max_iter=maxiter, print_level=verbosity, tol=tol))
	end
	#IMPORTANT the order at which parameters are defined is very important
	if Wfixed
		W = Winit
	else
		constrainW && normalizematrix!(Winit)
		@JuMP.variable(m, W[i=1:nummixtures, j=1:nk], start=convert(Float32, Winit[i, j]))
		Wnonneg && @JuMP.constraint(m, W .>= 0)
		constrainW && @JuMP.constraint(m, W .<= 1)
	end
	if Hfixed
		H = Hinit
	else
		constrainH && normalizematrix!(Hinit)
		@JuMP.variable(m, H[i=1:nk, j=1:numconstituents], start=convert(Float32, Hinit[i, j]))
		Hnonneg && @JuMP.constraint(m, H .>= 0)
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
			if Hfixed
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
		Xe = Winit * Hinit
		NMFk.plotnmf(Xe, Winit[:,movieorder], Hinit[movieorder,:]; movie=movie, filename=moviename, frame=1)
	end
	jumpvariables = JuMP.all_variables(m)
	jumpvalues = JuMP.start_value.(jumpvariables)
	if quiet
		@Suppressor.suppress JuMP.optimize!(m)
	else
		JuMP.optimize!(m)
	end

	Wbest = (Wfixed) ? W : JuMP.value.(W)
	Hbest = (Hfixed) ? H : JuMP.value.(H)
	of = JuMP.objective_value(m)
	!quiet && @info("Objective function $of")
	ofbest = of
	objvalue = regularizationweight > 0. ? ofbest - regularizationweight * sum(log.(1. .+ Hbest).^2) / nk : ofbest
	frame = 2
	iters = 1
	baditers = 0
	reattempts = 0
	while !(LinearAlgebra.norm(jumpvalues - JuMP.value.(jumpvariables)) < tolX) && !(objvalue < tol) && baditers < maxbaditers && reattempts < maxreattempts
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
			!Wfixed && (We = JuMP.value.(W))
			!Hfixed && (He = JuMP.value.(H))
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
		iters += 1
		if of < ofbest
			if (ofbest - of) < tolOF
				baditers += 1
			else
				!quiet && @info("Objective function improved substantially (more than $tolOF; $objvalue < $objvalue_best); bad iteration counter reset ...")
				baditers = 0
			end
			!Wfixed && (Wbest = JuMP.value.(W))
			!Hfixed && (Hbest = JuMP.value.(H))
			ofbest = of
			objvalue = regularizationweight > 0. ? ofbest - regularizationweight * sum(log.(1. .+ Hbest).^2) / nk : ofbest
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
		if !quiet
			@info("Iteration $iters")
			@info("Objective function $of")
			(regularizationweight > 0.) && @info("Objective function - regularization penalty $objvalue")
			@info("Parameter norm: $(LinearAlgebra.norm(jumpvalues - JuMP.value.(jumpvariables)))")
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
	penalty = regularizationweight > 0. ? regularizationweight * sum(log.(1. .+ Hbest).^2) / nk : 0
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
		Winit[nansw] .= NaN
		Wbest[nansw] .= NaN
	end
	if sum(nansh) > 0
		Hinit[nansh] .= NaN
		Hbest[nansh] .= NaN
	end
	Hnonneg && (Hbest[Hbest .< 0] .= 0)
	Wnonneg && (Wbest[Wbest .< 0] .= 0)
	return Wbest, Hbest, fitquality
end
