import JuMP
import Ipopt
import NLopt

const defaultregularizationweight = convert(Float32, 0)
const defaultmaxiter = 1000
const defaultverbosity = 0

"Iterative factorization of matrix X (X = W * H) using Ipopt fixing W and H matrices"
function jumpiter(X::Matrix, nk::Int; kw...)
	m, n = size(X)
	jumpiter(convert(Array{Float32, 2}, X), nk, convert(Array{Float32, 2}, rand(m, nk)), convert(Array{Float32, 2}, rand(nk, n)); kw...)
end
function jumpiter(X::Matrix{Float32}, nk::Int, W::Matrix{Float32}, H::Matrix{Float32}; iter::Int=100, tolerance::Float64=1e-2, quiet::Bool=true, kw...)
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
function jumpHrows(X::Matrix{Float32}, nk::Int, W::Matrix{Float32}, H::Matrix{Float32}; quiet::Bool=true, kw...)
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
function jump(X_in::Matrix{Float64}, nk::Int; kw...)
	jump(convert(Array{Float32, 2}, X_in), nk; kw...)
end
function jump(X_in::Array{Float32}, nk::Int; method::Symbol=:nlopt, algorithm::Symbol=:LD_LBFGS, normalize::Bool=false, scale::Bool=false, random::Bool=true, maxiter::Int=defaultmaxiter, verbosity::Int=defaultverbosity, regularizationweight::Float32=defaultregularizationweight, weightinverse::Bool=false, initW::Matrix{Float32}=Array{Float32}(0, 0), initH::Array{Float32}=Array{Float32}(0, 0), tolX::Float64=1e-3, tol::Float64=1e-3, maxouteriters::Int=10, quiet::Bool=true, kullbackleibler=false, fixW::Bool=false, fixH::Bool=false, seed::Number=-1, constrainW::Bool=true, movie::Bool=false, moviename::AbstractString="", movieorder=1:nk, moviecheat::Integer=0)
	if seed >= 0
		srand(seed)
	end
	X = copy(X_in) # we may overwrite some of the fields if there are NaN's, so make a copy
	if normalize
		X, cmin, cmax = normalizematrix(X)
	elseif scale
		X, cmax = scalematrix(X)
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
	end
	if sizeof(initH) == 0
		fixH = false
		if random
			if !scale || !normalize
				initH = rand(Float32, nk, numconstituents)
			else
				max = maximum(X, 1)
				initH = rand(Float32, nk, numconstituents)
				for i=1:nk
					initH[i:i,:] .*= max
				end
			end
		else
			if !scale || !normalize
				initH = ones(Float32, nk, numconstituents) / 2
			else
				max = maximum(X, 1)
				initH = Array{Float32}(nk, numconstituents)
				for i=1:nk
					initH[i:i,:] = max
				end
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
	JuMP.solve(m)
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
	!quiet && @show of
	ofbest = of
	objvalue = ofbest - regularizationweight * sum(log.(1. + Hbest).^2) / nk
	frame = 2
	while !(norm(oldcolval - m.colVal) < tolX) && !(objvalue < tol)
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
		JuMP.solve(m)
		of = JuMP.getobjectivevalue(m)
		if of < ofbest
			!fixW && (Wbest = JuMP.getvalue(W))
			!fixH && (Hbest = JuMP.getvalue(H))
			ofbest = of
		end
		objvalue = ofbest - regularizationweight * sum(log.(1. + Hbest).^2) / nk
		!quiet && @show of, norm(oldcolval - m.colVal), objvalue
	end
	!quiet && @show ofbest
	objvalue = ofbest - regularizationweight * sum(log.(1. + Hbest).^2) / nk
	if normalize
		Hbest = denormalizematrix(Hbest, Wbest, cmin, cmax)
	elseif scale
		Hbest = descalematrix(Hbest, cmax)
	end
	if movie
		Xe = Wbest * Hbest
		NMFk.plotnmf(Xe, Wbest[:,movieorder], Hbest[movieorder,:]; movie=movie, filename=moviename, frame=frame)
	end
	return Wbest, Hbest, objvalue
end
