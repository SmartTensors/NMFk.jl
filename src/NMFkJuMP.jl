import JuMP
import Ipopt
import NLopt
import Suppressor

const defaultregularizationweight = convert(Float32, 0)
const defaultmaxiter = 1000
const defaultverbosity = 0

"Iterative factorization of matrix X (X = W * H) using Ipopt fixing W and H matrices"
function jumpiter(X::Matrix, nk::Int; kw...)
	m, n = size(X)
	jumpiter(convert(Array{Float32, 2}, X), nk, convert(Array{Float32, 2}, rand(m, nk)), convert(Array{Float32, 2}, rand(nk, n)); kw...)
end
function jumpiter(X::Matrix{Float32}, nk::Int, W::Matrix{Float32}, H::Matrix{Float32}; iter::Int=100, tolerance::Float64=1e-2, quiet::Bool=NMFk.quiet, kw...)
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
function jumpHrows(X::Matrix{Float32}, nk::Int, W::Matrix{Float32}, H::Matrix{Float32}; quiet::Bool=NMFk.quiet, kw...)
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
function jump(X::Matrix{Float64}, nk::Int; kw...)
	jump(convert(Array{Float32, 2}, X), nk; kw...)
end
function jump(X::Array{Float32}, nk::Int; method::Symbol=:nlopt, algorithm::Symbol=:LD_LBFGS, normalize::Bool=false, scale::Bool=false, maxW::Bool=false, maxH::Bool=false, random::Bool=true, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Float32=defaultregularizationweight, weightinverse::Bool=false, initW::Matrix{Float32}=Array{Float32}(0, 0), initH::Array{Float32}=Array{Float32}(0, 0), tolX::Float64=1e-3, tol::Float64=1e-3, tolOF::Float64=1e-3, maxresets::Int=3, maxouteriters::Int=10, quiet::Bool=NMFk.quiet, kullbackleibler=false, fixW::Bool=false, fixH::Bool=false, seed::Number=-1, constrainW::Bool=true, movie::Bool=false, moviename::AbstractString="", movieorder=1:nk, moviecheat::Integer=0)
	if seed >= 0
		srand(seed)
	end
	if normalize
		X, cmin, cmax = normalizematrix!(X)
	elseif scale
		X, cmax = scalematrix!(X)
	end
	if weightinverse
		obsweights = convert(Array{Float32,2}, 1. ./ X)
		zis = X .== 0
		obsweights[zis] = maximum(X[!zis]) * 10
	else
		obsweights = ones(Float32, size(X))
	end
	nans = isnan.(X)
	X[nans] = 0
	obsweights[nans] = 0
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
			maxx = maximum(X, 2)
			for i=1:nk
				initW[:,i:i] .*= maxx
			end
		end
	end
	if sizeof(initH) == 0
		fixH = false
		if random
			initH = rand(Float32, nk, numconstituents)
		else
			initH = ones(Float32, nk, numconstituents) / 2
		end
		if maxH
			maxx = maximum(X, 1)
			for i=1:nk
				initH[i:i,:] .*= maxx
			end
		end
	end
	if method == :ipopt
		m = JuMP.Model(solver=Ipopt.IpoptSolver(max_iter=maxiter, print_level=verbosity, tol=tol))
	elseif method == :nlopt
		m = JuMP.Model(solver=NLopt.NLoptSolver(algorithm=algorithm, maxeval=maxiter, xtol_abs=tolX, ftol_abs=tol))
	end
	#IMPORTANT the order at which parameters are defined is very important
	if fixW
		W = initW
	else
		@JuMP.variable(m, W[i=1:nummixtures, j=1:nk] >= 0., start = convert(Float32, initW[i, j]))
		!constrainW && @JuMP.constraint(m, W .<= 1) # this is very important constraint to make optimization faster
	end
	if fixH
		H = initH
	else
		@JuMP.variable(m, H[i=1:nk, j=1:numconstituents] >= 0., start = convert(Float32, initH[i, j]))
	end
	if kullbackleibler
		smallnumber = eps(Float64)
		@JuMP.NLobjective(m, Min, sum(X[i, j] * (log(smallnumber + X[i, j]) - log(smallnumber + sum(W[i, k] * H[k, j] for k = 1:nk))) - X[i, j] + sum(W[i, k] * H[k, j] for k = 1:nk) for i=1:nummixtures, j=1:numconstituents))
	else
		@JuMP.NLobjective(m, Min,
			regularizationweight * sum(sum(log(1. + H[i, j])^2 for i=1:nk) for j=1:numconstituents) / nk +
			sum(sum(obsweights[i, j] * (sum(W[i, k] * H[k, j] for k=1:nk) - X[i, j])^2 for i=1:nummixtures) for j=1:numconstituents))
	end
	oldcolval = copy(m.colVal)
	if movie
		Xe = initW * initH
		NMFk.plotnmf(Xe, initW[:,movieorder], initH[movieorder,:]; movie=movie, filename=moviename, frame=1)
	end
	if quiet
		@Suppressor.suppress JuMP.solve(m)
	else
		JuMP.solve(m)
	end
	if fixW
		Wbest = W
	else
		Wbest = JuMP.getvalue(W)
	end
	if fixH
		Hbest = H
	else
		Hbest = JuMP.getvalue(H)
	end
	of = JuMP.getobjectivevalue(m)
	!quiet && info("Initial objective function $of")
	ofbest = of
	objvalue = ofbest - regularizationweight * sum(log.(1. + Hbest).^2) / nk
	frame = 2
	iters = 1
	outiters = 0
	resets = 0
	while !(norm(oldcolval - m.colVal) < tolX) && !(objvalue < tol) && outiters < maxouteriters && resets <= maxresets
		oldcolval = copy(m.colVal)
		if movie
			mcheat = 1
			while mcheat <= moviecheat
				We = JuMP.getvalue(W)
				c = (moviecheat - mcheat) / moviecheat + 0.1
				We += rand(similar(We)) .* c
				He = JuMP.getvalue(H)
				He += rand(similar(He)) .* c / 10
				Xe = We * He
				NMFk.plotnmf(Xe, We[:,movieorder], He[movieorder,:]; movie=movie,filename=moviename, frame=frame)
				frame += 1
				mcheat += 1
			end
			We = JuMP.getvalue(W)
			He = JuMP.getvalue(H)
			Xe = We * He
			NMFk.plotnmf(Xe, We[:,movieorder], He[movieorder,:]; movie=movie,filename=moviename, frame=frame)
			frame += 1
		end
		if quiet
			@Suppressor.suppress JuMP.solve(m)
		else
			JuMP.solve(m)
		end
		of = JuMP.getobjectivevalue(m)
		outiters += 1
		iters += 1
		if of < ofbest
			if (ofbest - of) > tolOF
				resets += 1
				if resets > maxresets
					warn("Maximum number of resets has been reached; quit!")
				else
					warn("Objective function improved substantially (more than $tolOF; $of < $ofbest); iteration counter reset ...")
					outiters = 0
				end
			end
			!fixW && (Wbest = JuMP.getvalue(W))
			!fixH && (Hbest = JuMP.getvalue(H))
			ofbest = of
		end
		objvalue = ofbest - regularizationweight * sum(log.(1. + Hbest).^2) / nk
		if !quiet
			info("Iteration $iters")
			info("Objective function $of")
			(regularizationweight > 0.) && info("Objective function + regularization penalty $objvalue")
			info("Parameter norm: $(norm(oldcolval - m.colVal))")
		end
	end
	isnm = isnan.(Wbest)
	isnb = isnan.(Hbest)
	if sum(isnm) > 0
		warn("There are NaN's in the W matrix!")
		Wbest[isnm] .= 0
	end
	if sum(isnb) > 0
		warn("There are NaN's in the H matrix!")
		Hbest[isnb] .= 0
	end
	if sum(isnm) > 0 || sum(isnb) > 0
		warn("Vecnorm: $(sqrt(vecnorm(X - Wbest * Hbest))) OF: $(ofbest)")
	end
	penalty = regularizationweight * sum(log.(1. + Hbest).^2) / nk
	fitquality = ofbest - penalty
	if !quiet
		info("Final objective function: $ofbest")
		(regularizationweight > 0.) && iinfo("Final penalty: $penalty")
		info("Final fit: $fitquality")
	end
	if normalize
		Hbest = denormalizematrix!(Hbest, Wbest, cmin, cmax)
	elseif scale
		Hbest = descalematrix!(Hbest, cmax)
	end
	if movie
		Xe = Wbest * Hbest
		NMFk.plotnmf(Xe, Wbest[:,movieorder], Hbest[movieorder,:]; movie=movie, filename=moviename, frame=frame)
	end
	X[nans] = NaN
	return Wbest, Hbest, objvalue
end
