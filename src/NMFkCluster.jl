import Distances
import Suppressor
import Clustering
import GaussianMixtures

Base.@kwdef struct GMMClusterResult{TG,TM}
	model::TG
	assignments::Vector{Int}
	responsibilities::TM
	loglik::Float64
	bic::Float64
	aic::Float64
	k::Int
end

function _gmm_num_params(k::Int, d::Int; kind::Symbol=:full)
	if kind == :diag
		# weights (k-1) + means (k*d) + diagonal variances (k*d)
		return (k - 1) + k * d + k * d
	elseif kind == :full
		# weights (k-1) + means (k*d) + full covariances (k*d*(d+1)/2)
		return (k - 1) + k * d + k * (d * (d + 1) ÷ 2)
	else
		throw(ArgumentError("Unknown covariance kind=$kind (expected :diag or :full)"))
	end
end

function robustbgmm(X::AbstractMatrix, krange::Union{AbstractUnitRange{Int},AbstractVector{<:Integer}}, repeats::Int=20;
		criterion::Symbol=:bic,
		kind::Symbol=:full,
		nInit::Int=1,
		nIter::Int=50,
		nFinal::Int=10,
		sparse::Int=0,
		parallel::Bool=false)
	ks = sort!(collect(krange))
	if isempty(ks)
		throw(ArgumentError("krange is empty"))
	end
	# Clustering.kmeans uses observations in columns; GaussianMixtures expects samples in rows.
	Xn = zerostoepsilon(X)
	Xn[isnan.(Xn)] .= 0
	x = Matrix{Float64}(permutedims(Xn))
	n = size(x, 1)
	d = size(x, 2)

	best_gmm = nothing
	best_k = 0
	best_ll = -Inf
	best_bic = Inf
	best_aic = Inf
	best_score = Inf

	for k in ks
		if k >= n
			@info("$k: cannot be computed (k >= number of samples; $k >= $n)")
			continue
		end
		local best_gmm_k = nothing
		local best_ll_k = -Inf
		for _ = 1:repeats
			local gmm
			Suppressor.@suppress begin
				gmm = GaussianMixtures.GMM(k, x; kind=kind, nInit=nInit, nIter=nIter, nFinal=nFinal, sparse=sparse, parallel=parallel)
			end
			ll = GaussianMixtures.avll(gmm, x) * n
			if ll > best_ll_k
				best_ll_k = ll
				best_gmm_k = gmm
			end
		end
		p = _gmm_num_params(k, d; kind=kind)
		bic = -2 * best_ll_k + p * log(n)
		aic = -2 * best_ll_k + 2 * p
		score = criterion == :bic ? bic : criterion == :aic ? aic : throw(ArgumentError("criterion must be :bic or :aic"))
		@info("$k: GMM loglik=$(best_ll_k) BIC=$(bic) AIC=$(aic)")
		if score < best_score
			best_score = score
			best_gmm = best_gmm_k
			best_k = k
			best_ll = best_ll_k
			best_bic = bic
			best_aic = aic
		end
	end
	if isnothing(best_gmm)
		return nothing
	end
	resp, _ = GaussianMixtures.gmmposterior(best_gmm, x)
	assignments = Vector{Int}(undef, n)
	@inbounds for i in 1:n
		_, j = findmax(@view resp[i, :])
		assignments[i] = j
	end
	assignments = remap2count(assignments)
	@info("Best k=$(best_k) using $(criterion): score=$(best_score)")
	return GMMClusterResult(model=best_gmm, assignments=assignments, responsibilities=resp, loglik=best_ll, bic=best_bic, aic=best_aic, k=best_k)
end

function robustcluster(X::AbstractMatrix, krange::Union{AbstractUnitRange{Int},AbstractVector{<:Integer}}, repeats::Int=1000;
		method::Symbol=:kmeans, kw...)
	if method == :kmeans
		return robustkmeans(X, krange, repeats; kw...)
	elseif method == :bgmm
		return robustbgmm(X, krange, repeats; kw...)
	else
		throw(ArgumentError("Unknown method=$method (expected :kmeans or :bgmm)"))
	end
end

"""
Given a vector of classifiers, return a vector where
the highest count is now classifier #1, and the lowest count is
classifier #N

params:
	assignments (Clustering.assignments): a vector containing assignments

returns:
	remapped_assignments: a vector containing "sorted" assignments
"""
function remap2count(assignments)
	# Count how many of each classifier are in the vector
	assignment_count = [(i, count(x->x==i, assignments)) for i in unique(assignments)]

	# Sort the vector by highest -> lowest count
	sort!(assignment_count; by=x->x[2], rev=true)

	# Now, the highest count classifier will be first entry in assignment_count.
	# Create a dictionary that maps "1" => "N", where N is the most populated classifier
	remapper = Dict([(assignment_count[i][1],i) for i = eachindex(unique(assignments))])

	# Finally, map the assignments to the new ordering and return
	mfunc(i) = remapper[i]
	return map(mfunc, assignments)
end

function robustkmeans(X::AbstractMatrix, krange::Union{AbstractUnitRange{Int},AbstractVector{Int64}}, repeats::Int=1000; best_method::Symbol=:worst_cliff, distance=Distances.CosineDist(), kw...)
	if krange[1] >= size(X, 2)
		@info("Cannot be computed (min range is greater than or equal to size(X,2); $krange[1] >= $(size(X, 2)))")
		return nothing
	end
	totalcosts = Vector{Float64}(undef, length(krange))
	mean_silhouette = Vector{Float64}(undef, length(krange))
	worst_silhouette = Vector{Float64}(undef, length(krange))
	cresult = Vector{Any}(undef, length(krange))
	cluster_silhouettes = Vector{Any}(undef, length(krange))
	for (i, k) in enumerate(krange)
		if k >= size(X, 2)
			@info("$k: cannot be computed (k is greater than or equal to size(X,2); $k >= $(size(X, 2)))")
			continue
		end
		cresult[i], silhouettes = robustkmeans(X, k, repeats; distance=distance, kw..., compute_silhouettes_flag=true)
		totalcosts[i] = cresult[i].totalcost
		mean_silhouette[i] = Statistics.mean(silhouettes)
		cluster_silhouettes[i] = map(j->Statistics.mean(silhouettes[cresult[i].assignments .== j]), unique(cresult[i].assignments))
		worst_silhouette[i] = minimum(silhouettes)
		@info("$k: OF: $(totalcosts[i]) Mean Silhouette: $(mean_silhouette[i]) Worst Silhouette: $(worst_silhouette[i]) Cluster Count: $(map(j->sum(cresult[i].assignments .== j), unique(cresult[i].assignments))) Cluster Silhouettes: $(cluster_silhouettes[i])")
	end
	if best_method == :worst_cliff
		ki = last(findmax(map(i->worst_silhouette[i] - worst_silhouette[i+1], eachindex(krange)[begin:end-1]))) + 1
	elseif best_method == :worst_cluster_cliff
		ki = last(findmax(map(i->minimum(cluster_silhouettes[i]) - minimum(cluster_silhouettes[i+1]), eachindex(krange)[begin:end-1]))) + 1
	else
		@error("Unknown method: best_method must be :worst_cliff or :worst_cluster_cliff")
	end
	k = krange[ki]
	@info("Best $k - OF: $(totalcosts[ki]) Mean Silhouette: $(mean_silhouette[ki]) Worst Silhouette: $(worst_silhouette[ki]) Cluster Count: $(map(i->sum(cresult[ki].assignments .== i), unique(cresult[ki].assignments))) Cluster Silhouettes: $(cluster_silhouettes[ki])")
	return cresult[ki]
end

function robustkmeans(X::AbstractMatrix, k::Integer, repeats::Integer=1000; maxiter::Integer=1000, tol::Number=1e-32, display=:none, distance=Distances.CosineDist(), resultdir::AbstractString=".", casefilename::AbstractString="assignments", load::Bool=false, save::Bool=false, compute_silhouettes_flag::Bool=false)
	if load && casefilename != ""
		filename = joinpathcheck(resultdir, "$casefilename-$k-$(join(size(X), '_'))-$repeats.jld")
		if isfile(filename)
			try
				if compute_silhouettes_flag
					sc  = JLD.load(filename, "assignments")
					@info("Robust k-means analysis results are loaded from file $(filename)!")
					if length(best_silhouettes) == size(X, 2)
						return sc
					else
						@warn("File $(filename) does not contain correct information! Robust k-means analysis will be executed ...")
					end
				else
					sc = JLD.load(filename, "assignments")
					@info("Robust k-means analysis results are loaded from file $(filename)!")
					return sc
				end
			catch err
				@warn("Failed to load robust k-means results from $(filename) ($(typeof(err))); Robust k-means analysis will be executed ...")
			end
		else
			@info("File $(filename) does not exist! Robust k-means analysis will be executed ...")
		end
	end
	local c = nothing
	local best_totalcost = Inf
	local best_mean_silhouette = Inf
	local best_silhouettes = zeros(size(X, 2))
	Xn = zerostoepsilon(X)
	Xd = nothing
	if compute_silhouettes_flag
		Xd = Distances.pairwise(distance, Xn; dims=2)
	end
	for i = 1:repeats
		local c_new
		Suppressor.@suppress begin
			c_new = Clustering.kmeans(X, k; maxiter=maxiter, tol=tol, display=display, distance=distance)
		end
		if compute_silhouettes_flag
			if maximum(c_new.assignments) >= 2
				silhouettes = Clustering.silhouettes(c_new, Xd)
			else
				@warn("Only one cluster found during k-means clustering; silhouettes set to zero.")
				silhouettes = zeros(size(X, 2))
			end
		else
			silhouettes = nothing
		end
		if i == 1 || c_new.totalcost < best_totalcost
			c = deepcopy(c_new)
			best_totalcost = c_new.totalcost
			if compute_silhouettes_flag
				best_mean_silhouette = Statistics.mean(silhouettes)
				best_silhouettes = silhouettes
			end
		end
	end
	if length(unique(c.assignments)) < k
		@warn("Robust k-means analysis could not find $k clusters! Only $(length(unique(c.assignments))) clusters were found.")
	end
	sc = sortclustering(c)
	if save && casefilename != ""
		filename = joinpathcheck(resultdir, "$casefilename-$k-$(join(size(X), '_'))-$repeats.jld")
		JLD.save(filename, "assignments", sc, "best_silhouettes", best_silhouettes)
		@info("Robust k-means analysis results are saved in file $(filename)!")
	end
	if compute_silhouettes_flag
		return sc
	else
		return sc
	end
end

function sortclustering(c::AbstractVector; rev=true)
	j = unique(c)
	counts = Vector{Int64}(undef, length(j))
	for (k, a) in enumerate(j)
		ic = c .== a
		c[ic] .= k
		counts[k] = sum(ic)
	end
	i = sortperm(counts[j]; rev=rev)
	cnew = similar(c)
	for (k, a) in enumerate(i)
		cnew[c .== a] .= k
	end
	return cnew
end

function sortclustering(c::Clustering.KmeansResult; rev=true)
	cassignments = similar(c.assignments)
	j = unique(c.assignments)
	# @show j
	# for j = eachindex(j)
	# 	@show sum(c.assignments .== j)
	# end
	for (k, a) in enumerate(j)
		cassignments[c.assignments .== a] .= k
	end
	# @show unique(cassignments)
	# for j = eachindex(j)
	# 	@show sum(cassignments .== j)
	# end
	i = sortperm(c.counts[j]; rev=rev)
	cassignments2 = similar(c.assignments)
	for (k, a) in enumerate(i)
		cassignments2[cassignments .== a] .= k
	end
	# @show unique(cassignments2)
	# for j = eachindex(i)
	# 	@show sum(cassignments2 .== j)
	# end
	r = j[i]
	return Clustering.KmeansResult(c.centers[:,r], cassignments2, c.costs, c.counts[r], c.wcounts[r], c.totalcost, c.iterations, c.converged)
end

function labelassignements(c::AbstractVector)
	t = unique(c)
	nc = length(c)
	nt = length(t)
	if nt < 31
		types = collect(range('A'; length=nt))
		cassignments = Vector{Char}(undef, nc)
	else
		nz = convert(Int64, ceil(log10(nt)))
		types = ["T$(lpad(i, nz, '0'))" for i=1:nt]
		cassignments = Vector{String}(undef, nc)
	end
	for a in t
		cassignments[c .== a] .= types[a]
	end
	@assert sort(types) == sort(unique(cassignments))
	return cassignments
end

function finduniquesignals(X::AbstractMatrix; quiet::Bool=false)
	k = size(X, 1)
	@assert k == size(X, 2)
	signalmap = zeros(Int64, k)
	Xc = abs.(copy(X)) # absolute value ensures we can handle negative values in the input
	Xc[isnan.(Xc)] .= 0.
	failed = false
	maxiters = 2 * length(Xc)
	iters = 0
	while any(signalmap .== 0)
		iters += 1
		if iters > maxiters
			!quiet && @warn("Procedure to find unique signals exceeded iteration budget $(maxiters); aborting...")
			failed = true
			break
		end
		if all(Xc .== 0.)
			!quiet && @warn("Procedure to find unique signals could not identify an optimal solution ...")
			failed = true
			break
		end
		maxvalue, rc = findmax(Xc)
		if isnan(maxvalue) || maxvalue == 0.
			!quiet && @warn("Procedure to find unique signals could not identify an optimal solution ...")
			failed = true
			break
		end
		Xc[rc] = 0.
		if signalmap[rc[1]] == 0 && !any(signalmap .== rc[2])
			signalmap[rc[1]] = rc[2]
		end
	end
	o = 0
	if failed
		@warn("Procedure to find unique signals failed to find a valid signal map.")
		@info("Max value in Xc at failure: $(maximum(Xc))")
		@info("Max rows: $(maximumnan(Xc; dims=1))")
		@info("Max columns: $(maximumnan(Xc; dims=2))")
		@info("Signal map at failure: $(signalmap)")
	else
		@info("Signal map: $(signalmap)")
		for i = 1:k
			o += X[i,signalmap[i]]
		end
	end
	return o, signalmap
end

function finduniquesignalsbest(X::AbstractMatrix)
	o, signalmap = finduniquesignals(X)
	k = size(X, 1)
	obest = o
	signalmapbest = signalmap
	for i = 1:k
		Xc = copy(X)
		Xc[i, signalmap[i]] = 0.
		o, signalmap = finduniquesignals(Xc; quiet=true)
		if o > obest
			obest = o
			signalmapbest = signalmap
		end
	end
	return signalmapbest
end

function signalassignments(X::AbstractMatrix{T}, c::AbstractVector; dims=1, clusterlabels=nothing) where {T <: Number}
	if isnothing(clusterlabels)
		clusterlabels = sort(unique(c))
	end
	d = dims == 1 ? 2 : 1
	k = size(X, d)
	Ms = Matrix{T}(undef, k, k)
	for (j, i) in enumerate(clusterlabels)
		# nt1 = ntuple(k->(k != dims ? j : Colon()), 2)
		nt2 = ntuple(k->(k == dims ? c .== i : Colon()), 2)
		Ms[j,:] .= vec(Statistics.mean(X[nt2...]; dims=dims))
	end
	return NMFk.finduniquesignalsbest(Ms)
end

function clustersolutions(factors::AbstractVector, clusterWmatrix::Bool=false)
	if !clusterWmatrix
		factors = [permutedims(f) for f in factors]
	end
	# invariant: we can now assume that our matrices are n x k
	numFactors = length(factors)
	numTrials = numFactors
	r, k = size(factors[1])
	for w in factors
		@assert size(w) == (r, k)
	end
	# fix zero case
	needZeroFix = false
	for i in 1:numFactors
		factor = factors[i]
		if minimum(sum(factor; dims=1)) == 0 # if we have a zero column
			needZeroFix = true
			break
		end
	end
	if needZeroFix
		biasRow = permutedims(ones(k))
		for i in 1:numFactors
			factors[i] = vcat(factors[i], biasRow)
		end
	end
	# invariant: at this point, no factor has a column with all zeros

	centSeeds = factors[1]
	# when we label a factor column, we are going to keep a sum
	newClusterCenters = factors[1]
	# clusterLabels[a, b] = c --> factors[b][:, a] belongs to cluster c
	clusterLabels = zeros(Int, k, numTrials)
	# note: all clusterLabels should be in [1, k] upon return

	# by definition, the columns of the first solution belong to their own cluster.
	clusterLabels[:, 1] = [i for i in 1:k]

	clusterDistances = Matrix{eltype(factors[1])}(undef, k, k)
	for trial in 2:numTrials
		W = factors[trial]
		# clusterDistances[a, b] = c --> dist(W[:,a], centSeeds[:,b]) = c
		for centroidIdx in 1:k
			centroid = centSeeds[:, centroidIdx]
			for factorColIdx in 1:k
				clusterDistances[factorColIdx, centroidIdx] = Distances.cosine_dist(W[:, factorColIdx], centroid)
			end
		end
		clusterDistances[isnan.(clusterDistances)] .= 0
		while minimum(clusterDistances) < Inf
			# get the row and column of the smallest distance
			selectFactorCol, selectCentIdx = Tuple(CartesianIndices(clusterDistances)[argmin(clusterDistances)])
			# save that col in trial belongs to centroid's cluster
			# println("Assigned: Trial: $trial, Factor: $selectFactorCol, Centroid: $selectCentIdx")
			clusterLabels[selectFactorCol, trial] = selectCentIdx
			# this factor cannot belong to other centSeeds
			clusterDistances[selectFactorCol, :] .+= Inf
			# this cluster cannot collect more factor columns
			clusterDistances[:, selectCentIdx] .+= Inf
			newClusterCenters[:, selectCentIdx] .+= W[:, selectFactorCol]
		end
	end
	while minimum(clusterLabels) == 0
		idx, trial = Tuple(CartesianIndices(clusterLabels)[argmin(clusterLabels)])
		if sum(clusterLabels[:, trial]) == 0
			@warn("Solution $trial was not assigned to any of the cluster!")
			clusterLabels[:, trial] = [i for i in 1:k]
		else
			@warn("Parameter $idx in solution $trial was not assigned a cluster!")
			clusterLabels[idx, trial] = idx
		end
	end
	if minimum(clusterLabels) <= 0
		@warn("Minimum assignments should be greater than 1: $(minimum(clusterLabels))")
	end
	if maximum(clusterLabels) > k
		@warn("Maximum assignments should be less than $k: $(maximum(clusterLabels))")
	end
	for i in 1:k
		for j in 1:numTrials
			l = length(findall((in)(i), clusterLabels[:, j]))
			if l != 1
				@warn("Cluster $i does not appear only once in column $j; it appears $l times!")
			end
		end
	end

	newClusterCenters ./= numTrials
	if !clusterWmatrix
		factors = [permutedims(f) for f in factors]
	end
	return clusterLabels, permutedims(newClusterCenters)
end

"Cluster NMFk solutions"
function clustersolutions_old(W::AbstractVector, clusterWmatrix::Bool=false)
	nNMF = length(W)
	nc, nr = size(W[1])
	nk = clusterWmatrix ? nr : nc

	centroids = W[1]
	idx = Matrix{Int}(undef, nk, nNMF)

	for clusterIt = 1:nNMF
		for globalIterID = 1:nNMF
			processesTaken = falses(nk)
			centroidsTaken = falses(nk)
			for currentProcessID = 1:nk
				distMatrix = ones(nk, nk) + Inf
				for processID = 1:nk
					if !processesTaken[processID]
						for centroidID = 1:nk
							if !centroidsTaken[centroidID]
								if clusterWmatrix
									distMatrix[processID, centroidID] = Distances.cosine_dist(W[globalIterID][:, processID], centroids[:, centroidID])
								else
									distMatrix[processID, centroidID] = Distances.cosine_dist(W[globalIterID][processID, :], centroids[centroidID, :])
								end
							end
						end
					end
				end
				minProcess, minCentroid = Tuple(CartesianIndices(size(distMatrix))[argmin(distMatrix)])
				processesTaken[minProcess] = true
				centroidsTaken[minCentroid] = true
				idx[minProcess, globalIterID] = minCentroid
			end
		end
		centroids = zeros(size(W[1]))
		for centroidID = 1:nk
			for globalIterID = 1:nNMF
				if clusterWmatrix
					centroids[:, centroidID] += W[globalIterID][:, findall((in)(centroidID), idx[:, globalIterID])]
				else
					centroids[centroidID:centroidID, :] += W[globalIterID][findall((in)(centroidID), idx[:, globalIterID]), :]
				end
			end
		end
		centroids ./= nNMF
	end
	return idx, permutedims(centroids)
end

function clustersolutions_old(W::AbstractMatrix, nNMF::Integer)
	nP, nT = size(W) # number of observations (components/transients), number of total number of signals to cluster
	nk = convert(Int, nT / nNMF)

	centroids = W[:, 1:nk]
	idx = Matrix{Int}(undef, nk, nNMF)

	for clusterIt = 1:nNMF
		for globalIterID = 1:nNMF
			processesTaken = falses(nk)
			centroidsTaken = falses(nk)
			for currentProcessID = 1:nk
				distMatrix = ones(nk, nk) + Inf
				for processID = 1:nk
					if !processesTaken[processID]
						for centroidID = 1:nk
							if !centroidsTaken[centroidID]
								distMatrix[processID, centroidID] = Distances.cosine_dist(W[:,processID + (globalIterID - 1) * nk], centroids[:,centroidID])
							end
						end
					end
				end
				minProcess, minCentroid = Tuple(CartesianIndices(size(distMatrix))[argmin(distMatrix)])
				processesTaken[minProcess] = true
				centroidsTaken[minCentroid] = true
				idx[minProcess, globalIterID] = minCentroid
			end
		end
		centroids = zeros(nP, nk)
		for centroidID = 1:nk
			for globalIterID = 1:nNMF
				centroids[:, centroidID] += W[:, findall((in)(centroidID), idx[:, globalIterID]) + (globalIterID - 1) * nk]
			end
		end
		centroids ./= nNMF
	end
	return idx, permutedims(centroids)
end

struct WeightedPeriodicMinkowski{T, W, P <: Real} <: Distances.UnionMetric
	weights:: W
	periods:: T
	p:: P
end

Distances.parameters(wpm::WeightedPeriodicMinkowski) = (wpm.periods, wpm.weights)

@inline function Distances.eval_op(d::WeightedPeriodicMinkowski, ai, bi, Ti, wi)
	s1 = abs(ai - bi)
	s2 = mod(s1, Ti)
	abs(min(s2, Ti - s2))^d.p * wi
end

# @inline Distances.eval_end(d::WeightedPeriodicMinkowski, s) = s^(1/d.p)
# wpminkowski(a, b, w, T, p) = WeightedPeriodicMinkowski(T, w, p)(a, b)
# (w::WeightedPeriodicMinkowski)(a,b) = Distances._evaluate(w, a, b)

# Distances.result_type(dist::Distances.UnionMetrics, ::Type{Ta}, ::Type{Tb}, (p1, p2)) where {Ta,Tb} = typeof(Distances._evaluate(dist, oneunit(Ta), oneunit(Tb), oneunit(eltype(p1)), oneunit(eltype(p2))))

# function Distances._evaluate(dist::Distances.UnionMetrics, a::Number, b::Number, p1::Number, p2::Number)
# 	Distances.eval_end(dist, Distances.eval_op(dist, a, b, p1, p2))
# end

# Base.@propagate_inbounds function Distances._evaluate(d::Distances.UnionMetrics, a::AbstractArray, b::AbstractArray, (p1, p2)::Tuple{AbstractArray, AbstractArray})
# 	@boundscheck if length(a) != length(b)
# 		throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
# 	end
# 	@boundscheck if length(a) != length(p1)
# 		throw(DimensionMismatch("arrays have length $(length(a)) but parameter 1 has length $(length(p1))."))
# 	end
# 	@boundscheck if length(a) != length(p2)
# 		throw(DimensionMismatch("arrays have length $(length(a)) but parameter 2 has length $(length(p2))."))
# 	end
# 	if length(a) == 0
# 		return zero(result_type(d, a, b))
# 	end
# 	@inbounds begin
# 		s = eval_start(d, a, b)
# 		if (IndexStyle(a, b, p1, p2) === IndexLinear() && eachindex(a) == eachindex(b) == eachindex(p1)) == eachindex(p2)||
# 				axes(a) == axes(b) == axes(p) == axes(p2)
# 			@simd for I in eachindex(a, b, p1, p2)
# 				ai = a[I]
# 				bi = b[I]
# 				p1i = p1[I]
# 				p2i = p2[I]
# 				s = eval_reduce(d, s, eval_op(d, ai, bi, p1i, p2i))
# 			end
# 		else
# 			for (ai, bi, p1i, p2i) in zip(a, b, p1, p2)
# 				s = eval_reduce(d, s, eval_op(d, ai, bi, p1i, p2i))
# 			end
# 		end
# 		return eval_end(d, s)
# 	end
# end