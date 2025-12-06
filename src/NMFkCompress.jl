import Random
import Clustering
import Distances
import OrderedCollections
import Statistics

struct MatrixCompressionResult
	compressed_matrix::Matrix{Float64}
	original_to_group::Vector{Int}
	group_members::Vector{Vector{Int}}
	representative_indices::Vector{Int}
	group_means::Matrix{Float64}
	group_variances::Matrix{Float64}
	nan_group_id::Union{Int,Nothing}
	selected_k::Int
	silhouette_by_k::OrderedCollections.OrderedDict{Int,Float64}
end

function fill_nan_with_means!(mat::Matrix{Float64})
	nrows, ncols = size(mat)
	for j in 1:ncols
		sum_val = 0.0
		count = 0
		for i in 1:nrows
			v = mat[i, j]
			if !isnan(v)
				sum_val += v
				count += 1
			end
		end
		mean_val = count == 0 ? 0.0 : sum_val / count
		for i in 1:nrows
			if isnan(mat[i, j])
				mat[i, j] = mean_val
			end
		end
	end
	return mat
end

function medoid_index(data::Matrix{Float64}, members::Vector{Int})
	block = @view data[members, :]
	dmat = Distances.pairwise(Distances.SqEuclidean(), block; dims=1)
	sums = vec(sum(dmat; dims=2))
	minpos = argmin(sums)
	return members[minpos]
end

function prepare_matrix_clustering(X::AbstractMatrix)
	original = Matrix{Float64}(X)
	nrows, ncols = size(original)
	nan_mask = [all(isnan, @view original[i, :]) for i in 1:nrows]
	valid_idx = findall(x -> !x, nan_mask)
	nan_idx = findall(x -> x, nan_mask)
	work = isempty(valid_idx) ? Matrix{Float64}(undef, 0, ncols) : copy(original[valid_idx, :])
	isempty(valid_idx) || fill_nan_with_means!(work)
	return original, work, valid_idx, nan_idx
end

function build_compression_result(original::Matrix{Float64}, work::Matrix{Float64}, valid_idx::Vector{Int}, nan_idx::Vector{Int}, labels::Vector{Int}, best_k::Int, scores::OrderedCollections.OrderedDict{Int,Float64}, quiet::Bool)
	nrows, ncols = size(original)
	original_to_group = zeros(Int, nrows)
	group_members = Vector{Vector{Int}}()
	representatives = Int[]
	group_mean_rows = Vector{Vector{Float64}}()
	group_var_rows = Vector{Vector{Float64}}()
	if !isempty(valid_idx)
		if isempty(labels)
			best_k = max(best_k, 1)
			labels = ones(Int, length(valid_idx))
			scores[best_k] = get(scores, best_k, 0.0)
		end
		for g in unique(labels)
			members_local = findall(==(g), labels)
			isempty(members_local) && continue
			orig_members = valid_idx[members_local]
			push!(group_members, orig_members)
			medoid_local = medoid_index(work, members_local)
			rep_idx = valid_idx[medoid_local]
			push!(representatives, rep_idx)
			cluster_block = work[members_local, :]
			mean_vec = vec(Statistics.mean(cluster_block; dims=1))
			var_vec = vec(Statistics.var(cluster_block; dims=1, corrected=false))
			push!(group_mean_rows, mean_vec)
			push!(group_var_rows, var_vec)
			group_id = length(group_members)
			for idx in orig_members
				original_to_group[idx] = group_id
			end
		end
	end
	nan_group_id = nothing
	if !isempty(nan_idx)
		push!(group_members, nan_idx)
		push!(representatives, nan_idx[1])
		push!(group_mean_rows, fill(NaN, ncols))
		push!(group_var_rows, fill(NaN, ncols))
		nan_group_id = length(group_members)
		for idx in nan_idx
			original_to_group[idx] = nan_group_id
		end
	end
	n_groups = length(representatives)
	compressed = Matrix{Float64}(undef, n_groups, ncols)
	group_means = Matrix{Float64}(undef, n_groups, ncols)
	group_variances = Matrix{Float64}(undef, n_groups, ncols)
	for (i, rep_idx) in enumerate(representatives)
		compressed[i, :] = original[rep_idx, :]
		group_means[i, :] = group_mean_rows[i]
		group_variances[i, :] = group_var_rows[i]
	end
	selected_effective = nan_group_id === nothing ? n_groups : n_groups - 1
	quiet || @info("compress_similar_rows identified $(n_groups) groups (data clusters=$(selected_effective))")
	return MatrixCompressionResult(compressed, original_to_group, group_members, representatives, group_means, group_variances, nan_group_id, selected_effective, scores)
end

function cluster_rows(work::Matrix{Float64}, k_range::Vector{Int}; max_iter::Int, n_restarts::Int, rng::Random.AbstractRNG)
	valid_count = size(work, 1)
	scores = OrderedCollections.OrderedDict{Int, Float64}()
	best_labels = Int[]
	best_k = 0
	if valid_count == 0
		return best_labels, best_k, scores
	elseif valid_count == 1
		return ones(Int, 1), 1, OrderedCollections.OrderedDict(1 => 0.0)
	end
	data_t = permutedims(work)
	best_score = -Inf
	n_restarts = max(1, n_restarts)
	for k in k_range
		k < 1 && continue
		k > valid_count && continue
		best_k_score = -Inf
		best_k_labels = Int[]
		for restart in 1:n_restarts
			restart_seed = Random.rand(rng, UInt)
			restart_rng = Random.TaskLocalRNG()
			Random.seed!(restart_rng, restart_seed)
			result = Clustering.kmeans(data_t, k; maxiter=max_iter, init=:rand, rng=restart_rng, display=:none)
			sil = Clustering.silhouettes(result.assignments, data_t; metric=Distances.SqEuclidean())
			score = Statistics.mean(sil)
			if score > best_k_score
				best_k_score = score
				best_k_labels = copy(result.assignments)
			end
		end
		scores[k] = best_k_score
		if best_k_score > best_score
			best_score = best_k_score
			best_labels = best_k_labels
			best_k = k
		end
	end
	if best_k == 0
		best_k = min(valid_count, maximum(k_range))
		best_labels = ones(Int, valid_count)
		scores[best_k] = get(scores, best_k, 0.0)
	end
	return best_labels, best_k, scores
end

"""compress_similar_rows(X; k_range=2:20, max_iter=100, n_restarts=1, quiet=true)

Cluster rows of `X` (columns in [0, 1]) into groups using Clustering.jl's k-means
implementation with silhouette-based model selection. Each candidate ``k`` is run
`n_restarts` times with different initial centers and the best-silhouette run is kept.
Rows of all `NaN`s are assigned to a dedicated group. Returns a `RowCompressionResult`
for forward/backward mapping between original and compressed matrices.
"""
function compress_rows(X::AbstractMatrix;
	k_range::UnitRange{Int} = max(2,Int(round(floor(size(X, 1)/200); sigdigits=1))):max(2,Int(round(floor(size(X, 1)/20); sigdigits=1))),
	n_restarts::Int = 1,
	max_iter::Int = 100,
	rng::Random.AbstractRNG = Random.default_rng(),
	quiet::Bool = true,
)
	original, work, valid_idx, nan_idx = prepare_matrix_clustering(X)
	k_candidates = collect(k_range)
	k_candidates = isempty(k_candidates) ? [1] : k_candidates
	labels = Int[]
	best_k = 0
	scores = OrderedCollections.OrderedDict{Int, Float64}()
	if !isempty(valid_idx)
		labels, best_k, scores = cluster_rows(work, k_candidates; max_iter=max_iter, n_restarts=n_restarts, rng=rng)
	end
	return build_compression_result(original, work, valid_idx, nan_idx, labels, best_k, scores, quiet)
end

"""decompress_rows(result; mode=:representative, missing_value=NaN, add_noise=false)

Reconstruct an approximation of the original matrix from `result::RowCompressionResult`
by expanding each cluster template (either medoid representatives or per-cluster
means) back to its assigned rows. Optionally perturb the mean-based reconstruction by
sampling Gaussian noise with cluster variances for additional diversity. Rows never
assigned to any cluster (group id ``0``) are filled with `missing_value`.
"""
function decompress_rows(X::AbstractMatrix, result::MatrixCompressionResult;
	mode::Symbol = :representative,
	missing_value::Union{Float64, Missing} = NaN,
	add_noise::Bool = false,
	rng::Random.AbstractRNG = Random.default_rng(),
)
	@assert size(X) == size(result.compressed_matrix) "Matrix size must match compression setup"
	nrows = length(result.original_to_group)
	ncols = size(result.compressed_matrix, 2)
	reconstructed = Matrix{Float64}(undef, nrows, ncols)
	add_noise = add_noise && mode === :mean
	for (row_idx, group_id) in enumerate(result.original_to_group)
		if group_id <= 0
			reconstructed[row_idx, :] .= missing_value
		else
			@views reconstructed[row_idx, :] .= X[group_id, :]
			if add_noise
				vars = result.group_variances[group_id, :]
				for j in 1:ncols
					std = sqrt(max(vars[j], 0.0))
					std == 0 && continue
					reconstructed[row_idx, j] += std * randn(rng)
				end
			end
		end
	end
	return reconstructed
end

"""evaluate_compression(original, reconstructed; ignore_nans=true)

Return basic error metrics comparing `reconstructed` against the original matrix.
Computes MAE, RMSE, max absolute error, and R² using only entries that are finite in
both matrices (configurable via `ignore_nans`)."""
function evaluate_compression(original::AbstractMatrix, reconstructed::AbstractMatrix;
	ignore_nans::Bool = true,
)
	size(original) == size(reconstructed) || throw(ArgumentError("matrix dimensions must match"))
	if ignore_nans
		mask = .!(isnan.(original) .| isnan.(reconstructed))
	else
		mask = trues(size(original))
	end
	n_valid = count(mask)
	n_valid == 0 && throw(ArgumentError("no valid entries remain for comparison"))
	orig_vals = original[mask]
	recon_vals = reconstructed[mask]
	diffs = orig_vals .- recon_vals
	mae = Statistics.mean(abs.(diffs))
	rmse = sqrt(Statistics.mean(diffs .^ 2))
	max_abs = maximum(abs.(diffs))
	denom = sum((orig_vals .- Statistics.mean(orig_vals)) .^ 2)
	r2 = denom == 0 ? NaN : 1 - sum(diffs .^ 2) / denom
	return (mae=mae, rmse=rmse, max_abs=max_abs, r2=r2, n=n_valid)
end