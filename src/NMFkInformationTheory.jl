import Statistics
import Gadfly
import LinearAlgebra

# This module provides functions for computing information-theoretic measures on quantized tensors, including discrete entropy, Huffman encoding, residual coding information, and axis-specific information.

# Internal helper functions for information-theoretic computations.
# These functions are intended for internal use within the module and are not part of the public API.

# Compute the discrete entropy of a vector of integer symbols.
function _discrete_entropy(symbols::AbstractVector{<:Integer})::Float64
	observation_count::Int = length(symbols)
	observation_count == 0 && return 0.0
	counts::Dict{Int, Int} = Dict{Int, Int}()
	for symbol::Integer in symbols
		symbol_value::Int = Int(symbol)
		counts[symbol_value] = get(counts, symbol_value, 0) + 1
	end
	probabilities::Vector{Float64} = Float64.(collect(values(counts))) ./ observation_count
	return -sum(probabilities .* log2.(probabilities))
end

# Compute the number of bits required to encode a vector of integer symbols using Huffman coding.
function _huffman_encoded_bits(symbols::AbstractVector{<:Integer})::Int
	isempty(symbols) && return 0
	counts::Dict{Int, Int} = Dict{Int, Int}()
	for symbol::Integer in symbols
		symbol_value::Int = Int(symbol)
		counts[symbol_value] = get(counts, symbol_value, 0) + 1
	end
	weights::Vector{Int} = collect(values(counts))
	length(weights) == 1 && return 0
	encoded_bits::Int = 0
	while length(weights) > 1
		sort!(weights; rev=true)
		first_weight::Int = pop!(weights)
		second_weight::Int = pop!(weights)
		combined_weight::Int = first_weight + second_weight
		encoded_bits += combined_weight
		push!(weights, combined_weight)
	end
	return encoded_bits
end

# Compute the residual coding information for a vector of residual symbols given the number of bins and the coding method.
function _residual_coding_information(residual_symbols::Vector{Int}, bin_count::Int, residual_coding::Symbol)::NamedTuple
	pair_count::Int = length(residual_symbols)
	fixed_bits_per_symbol::Int = ceil(Int, log2(Float64(2 * bin_count - 1)))
	fixed_width_bits::Int = pair_count * fixed_bits_per_symbol
	residual_entropy::Float64 = _discrete_entropy(residual_symbols)
	encoded_bits::Float64 = if residual_coding == :none
		Float64(fixed_width_bits)
	elseif residual_coding == :shannon
		pair_count * residual_entropy
	else
		Float64(_huffman_encoded_bits(residual_symbols))
	end
	bits_per_symbol::Float64 = pair_count > 0 ? encoded_bits / pair_count : 0.0
	compression_ratio::Float64 = encoded_bits > 0.0 ? fixed_width_bits / encoded_bits : (fixed_width_bits > 0 ? Inf : 1.0)
	coding_efficiency::Float64 = bits_per_symbol > 0.0 ? residual_entropy / bits_per_symbol : (residual_entropy == 0.0 ? 1.0 : 0.0)
	return (
		method=residual_coding,
		encoded_bits=encoded_bits,
		bits_per_symbol=bits_per_symbol,
		fixed_width_bits=fixed_width_bits,
		compression_ratio=compression_ratio,
		coding_efficiency=coding_efficiency
	)
end

# Quantize the values in a tensor into discrete bins, returning the quantized tensor and a mask of valid entries.
function _quantize_tensor(tensor::AbstractArray{T}, valid_mask::AbstractArray{Bool}, bin_count::Int)::Tuple{Array{Int}, BitArray} where {T <: Real}
	if size(valid_mask) != size(tensor)
		throw(DimensionMismatch("The valid mask and tensor must have the same size!"))
	end
	if bin_count < 2
		throw(ArgumentError("The number of quantization bins must be at least 2!"))
	end
	finite_mask::BitArray = BitArray(valid_mask .& isfinite.(tensor))
	values::Vector{Float64} = Float64.(vec(tensor[finite_mask]))
	quantized::Array{Int} = zeros(Int, size(tensor))
	isempty(values) && return quantized, finite_mask
	minimum_value::Float64 = minimum(values)
	maximum_value::Float64 = maximum(values)
	if maximum_value == minimum_value
		quantized[finite_mask] .= 1
		return quantized, finite_mask
	end
	scale::Float64 = bin_count / (maximum_value - minimum_value)
	quantized_values::Vector{Int} = clamp.(floor.(Int, (values .- minimum_value) .* scale) .+ 1, 1, bin_count)
	quantized[finite_mask] .= quantized_values
	return quantized, finite_mask
end

# Compute the information-theoretic measures for a specific axis of a quantized tensor, including residual coding information.
function _axis_information(quantized::Array{Int}, valid_mask::BitArray, axis::Int, role::Symbol, bin_count::Int, residual_coding::Symbol)::NamedTuple
	left_symbols::Vector{Int} = Int[]
	right_symbols::Vector{Int} = Int[]
	residual_symbols::Vector{Int} = Int[]
	abs_difference_total::Float64 = 0.0
	axis_offset::CartesianIndex = CartesianIndex(ntuple(dimension::Int -> dimension == axis ? 1 : 0, ndims(quantized)))
	for index::CartesianIndex in CartesianIndices(quantized)
		index[axis] == size(quantized, axis) && continue
		neighbor_index::CartesianIndex = index + axis_offset
		(valid_mask[index] && valid_mask[neighbor_index]) || continue
		left_symbol::Int = quantized[index]
		right_symbol::Int = quantized[neighbor_index]
		push!(left_symbols, left_symbol)
		push!(right_symbols, right_symbol)
		push!(residual_symbols, right_symbol - left_symbol)
		abs_difference_total += abs(right_symbol - left_symbol)
	end
	pair_count::Int = length(left_symbols)
	left_entropy::Float64 = _discrete_entropy(left_symbols)
	right_entropy::Float64 = _discrete_entropy(right_symbols)
	joint_symbols::Vector{Int} = (left_symbols .- 1) .* bin_count .+ right_symbols
	joint_entropy::Float64 = _discrete_entropy(joint_symbols)
	mutual_information::Float64 = max(0.0, left_entropy + right_entropy - joint_entropy)
	conditional_entropy::Float64 = max(0.0, joint_entropy - left_entropy)
	residual_entropy::Float64 = _discrete_entropy(residual_symbols)
	coding_information::NamedTuple = _residual_coding_information(residual_symbols, bin_count, residual_coding)
	normalizer::Float64 = max(left_entropy, right_entropy)
	normalized_mutual_information::Float64 = normalizer > 0.0 ? mutual_information / normalizer : 0.0
	mean_normalized_difference::Float64 = pair_count > 0 ? abs_difference_total / (pair_count * (bin_count - 1)) : 0.0
	predictive_gain_bits::Float64 = right_entropy - residual_entropy
	return (
		axis=axis,
		role=role,
		pair_count=pair_count,
		mutual_information_bits=mutual_information,
		normalized_mutual_information=normalized_mutual_information,
		conditional_entropy_bits=conditional_entropy,
		residual_entropy_bits=residual_entropy,
		predictive_gain_bits=predictive_gain_bits,
		mean_normalized_difference=mean_normalized_difference,
		residual_coding=coding_information
	)
end

# Compute the spectral information for a specific axis of a tensor, including spectral entropy and effective rank.
function _spectral_axis_information(tensor::AbstractArray{T}, valid_mask::AbstractArray{Bool}, axis::Int)::NamedTuple where {T <: Real}
	dimension_order::Vector{Int} = vcat(axis, [dimension::Int for dimension::Int = 1:ndims(tensor) if dimension != axis])
	working_tensor::Array{Float64} = Float64.(tensor)
	working_tensor[.!valid_mask .| .!isfinite.(working_tensor)] .= 0.0
	unfolding::Matrix{Float64} = reshape(permutedims(working_tensor, dimension_order), size(tensor, axis), :)
	singular_values::Vector{Float64} = LinearAlgebra.svdvals(unfolding)
	energy::Vector{Float64} = singular_values .^ 2
	total_energy::Float64 = sum(energy)
	maximum_rank::Int = min(size(unfolding)...)
	if total_energy == 0.0
		return (axis=axis, spectral_entropy_bits=0.0, normalized_spectral_entropy=0.0, effective_rank=0.0, maximum_rank=maximum_rank)
	end
	positive_energy::Vector{Float64} = energy[energy .> 0.0]
	probabilities::Vector{Float64} = positive_energy ./ total_energy
	spectral_entropy::Float64 = -sum(probabilities .* log2.(probabilities))
	maximum_entropy::Float64 = maximum_rank > 1 ? log2(Float64(maximum_rank)) : 0.0
	normalized_entropy::Float64 = maximum_entropy > 0.0 ? spectral_entropy / maximum_entropy : 0.0
	return (axis=axis, spectral_entropy_bits=spectral_entropy, normalized_spectral_entropy=normalized_entropy, effective_rank=exp2(spectral_entropy), maximum_rank=maximum_rank)
end

"""
structure_information(tensor; valid_mask=trues(size(tensor)), bins=16, temporal_dim=ndims(tensor), residual_coding=:shannon, compute_spectral=true)

Measure information that depends on tensor structure rather than only on flattened
cell weights.

Values are quantized globally before neighboring cells are compared.

The returned `axis_information` reports mutual information, conditional entropy, normalized variation, and predictive-residual entropy along every axis. The temporal axis uses the same inter-frame residual principle as lossless movie compression.

`spectral_information` reports entropy and effective rank of every tensor unfolding.

`residual_coding` selects fixed-width residuals (`:none`), the ideal Shannon limit (`:shannon`), or a realizable binary Huffman code (`:huffman`).

Coding results are reported as `residual_coding` within each entry of `axis_information`.

Set `compute_spectral=false` for tensors whose unfoldings are too large for SVD.
"""
function structure_information(tensor::AbstractArray{T}; valid_mask::AbstractArray{Bool}=trues(size(tensor)), bins::Integer=16, temporal_dim::Union{Nothing, Integer}=ndims(tensor), residual_coding::Symbol=:shannon, compute_spectral::Bool=true)::NamedTuple where {T <: Real}
	bin_count::Int = Int(bins)
	if temporal_dim !== nothing && !(1 <= temporal_dim <= ndims(tensor))
		throw(ArgumentError("The temporal dimension must identify a tensor dimension or be nothing!"))
	end
	if !(residual_coding in (:none, :shannon, :huffman))
		throw(ArgumentError("Residual coding must be :none, :shannon, or :huffman!"))
	end
	# Quantize the tensor and determine the finite mask for valid entries.
	quantized::Array{Int}, finite_mask::BitArray = _quantize_tensor(tensor, valid_mask, bin_count)
	valid_symbols::Vector{Int} = vec(quantized[finite_mask])
	value_entropy::Float64 = _discrete_entropy(valid_symbols)
	maximum_value_entropy::Float64 = log2(Float64(bin_count))
	normalized_value_entropy::Float64 = maximum_value_entropy > 0.0 ? value_entropy / maximum_value_entropy : 0.0
	axis_information::Vector{NamedTuple} = NamedTuple[]
	spectral_information::Vector{NamedTuple} = NamedTuple[]
	for axis::Int = 1:ndims(tensor)
		role::Symbol = temporal_dim === axis ? :temporal : :spatial
		push!(axis_information, _axis_information(quantized, finite_mask, axis, role, bin_count, residual_coding))
		if compute_spectral
			push!(spectral_information, _spectral_axis_information(tensor, finite_mask, axis))
		end
	end
	spatial_metrics::Vector{NamedTuple} = filter(metric::NamedTuple -> metric.role == :spatial && metric.pair_count > 0, axis_information)
	temporal_metrics::Vector{NamedTuple} = filter(metric::NamedTuple -> metric.role == :temporal && metric.pair_count > 0, axis_information)
	spatial_dependence::Float64 = isempty(spatial_metrics) ? 0.0 : Statistics.mean(metric.normalized_mutual_information for metric::NamedTuple in spatial_metrics)
	spatial_variation::Float64 = isempty(spatial_metrics) ? 0.0 : Statistics.mean(metric.mean_normalized_difference for metric::NamedTuple in spatial_metrics)
	temporal_predictive_gain::Float64 = isempty(temporal_metrics) ? 0.0 : Statistics.mean(metric.predictive_gain_bits for metric::NamedTuple in temporal_metrics)
	temporal_variation::Float64 = isempty(temporal_metrics) ? 0.0 : Statistics.mean(metric.mean_normalized_difference for metric::NamedTuple in temporal_metrics)
	return (
		value_entropy_bits=value_entropy,
		normalized_value_entropy=normalized_value_entropy,
		spatial_dependence=spatial_dependence,
		spatial_variation=spatial_variation,
		temporal_predictive_gain_bits=temporal_predictive_gain,
		temporal_variation=temporal_variation,
		axis_information=axis_information,
		spectral_information=spectral_information,
		bins=bin_count,
		residual_coding=residual_coding,
		spectral_computed=compute_spectral,
		valid_cell_count=count(finite_mask)
	)
end

# Compute the information-theoretic measures for an entire tensor, including entropy, normalized entropy, effective cells, occupancy, divergence from uniform, and resolution score.
function tensor_information(tensor::AbstractArray{T}; valid_mask::AbstractArray{Bool}=trues(size(tensor)))::NamedTuple where {T <: Real}
	if size(valid_mask) != size(tensor)
		throw(DimensionMismatch("The valid mask and tensor must have the same size!"))
	end

	values::Vector{Float64} = Float64.(vec(tensor[valid_mask]))
	values[isnan.(values)] .= 0.0
	cell_count::Int = length(values)

	if any(!isfinite, values)
		throw(ArgumentError("Data input values must be finite or NaN!"))
	end
	if any(values .< 0.0)
		throw(ArgumentError("Data input values must be nonnegative!"))
	end

	if cell_count == 0
		return (
			entropy_bits=0.0,
			normalized_entropy=0.0,
			effective_cells=0.0,
			effective_fraction=0.0,
			occupancy=0.0,
			divergence_from_uniform_bits=0.0,
			normalized_divergence=0.0,
			resolution_score=0.0,
			occupied_cell_count=0,
			cell_count=cell_count
		)
	end

	maximum_value::Float64 = maximum(values)
	if maximum_value > 0.0
		values ./= maximum_value
	end

	positive_count::Int = count(>(0.0), values)
	total::Float64 = sum(values)

	if total == 0.0
		return (
			entropy_bits=0.0,
			normalized_entropy=0.0,
			effective_cells=0.0,
			effective_fraction=0.0,
			occupancy=0.0,
			divergence_from_uniform_bits=0.0,
			normalized_divergence=0.0,
			resolution_score=0.0,
			occupied_cell_count=positive_count,
			cell_count=cell_count
		)
	end

	probabilities::Vector{Float64} = values[values .> 0.0] ./ total
	entropy_bits::Float64 = -sum(probabilities .* log2.(probabilities))
	maximum_entropy::Float64 = cell_count > 1 ? log2(Float64(cell_count)) : 0.0
	normalized_entropy::Float64 = maximum_entropy > 0.0 ? entropy_bits / maximum_entropy : 0.0
	effective_cells::Float64 = exp2(entropy_bits)
	effective_fraction::Float64 = effective_cells / cell_count
	occupancy::Float64 = positive_count / cell_count
	normalized_divergence::Float64 = maximum_entropy > 0.0 ? (maximum_entropy - entropy_bits) / maximum_entropy : 0.0
	resolution_score::Float64 = effective_fraction * occupancy

	return (
		entropy_bits=entropy_bits,
		normalized_entropy=normalized_entropy,
		effective_cells=effective_cells,
		effective_fraction=effective_fraction,
		occupancy=occupancy,
		divergence_from_uniform_bits=maximum_entropy - entropy_bits,
		normalized_divergence=normalized_divergence,
		resolution_score=resolution_score,
		occupied_cell_count=positive_count,
		cell_count=cell_count
	)
end

# Plot the information-theoretic measures for a series of binning resolutions, with options for x-axis, normalization, and title.
function plot_information(information_steps::AbstractVector{<:NamedTuple}, steps::AbstractVector, filename::AbstractString=""; xaxis::Symbol=:steps, normalize::Symbol=:range, title_extra::AbstractString="")::Gadfly.Plot
	if isempty(information_steps)
		throw(ArgumentError("Information steps must not be empty!"))
	end
	if length(information_steps) != length(steps)
		throw(DimensionMismatch("Information steps and resolution labels must have the same length!"))
	end
	if !(xaxis in (:steps, :cells))
		throw(ArgumentError("The xaxis option must be :steps or :cells!"))
	end
	if !(normalize in (:range, :intrinsic))
		throw(ArgumentError("The normalize option must be :range or :intrinsic!"))
	end

	title = "Information utilization by binning resolution"
	if title_extra != ""
		title *= "$(title_extra)"
	end
	required_keys::Vector{Symbol} = [:entropy_bits, :effective_cells, :normalized_entropy, :occupancy]
	plotted_keys::Vector{Symbol} = normalize == :range ? required_keys : [:normalized_entropy, :occupancy]
	attribute_colors::Vector{String} = String.(NMFk.colors[1:length(plotted_keys)])
	attribute_labels::Vector{String} = normalize == :range ?
									   ["Entropy Bits (Normalized)", "Effective Cells (Normalized)", "Normalized Entropy", "Occupancy"] :
									   ["Normalized Entropy", "Occupancy"]
	attribute_label_by_key::Dict{Symbol, String} = Dict(zip(plotted_keys, attribute_labels))
	if !all(k -> all(haskey(info, k) for info in information_steps), required_keys)
		throw(ArgumentError("Each information entry must contain $(join(required_keys, ", ")). Recompute entries with tensor_information."))
	end

	xvals::Vector{Float64} = Float64[]
	step_xvals::Vector{String} = String[]
	yvals::Vector{Float64} = Float64[]
	metrics::Vector{String} = String[]
	cell_counts::Vector{Float64} = Float64[]
	label_yvals::Vector{Float64} = Float64[]
	step_order::Vector{Int} = sortperm([Int(info.cell_count) for info in information_steps])
	step_labels::Vector{String} = string.(steps[step_order])
	values_by_key::Dict{Symbol, Vector{Float64}} = Dict{Symbol, Vector{Float64}}()
	for key in plotted_keys
		raw_values::Vector{Float64} = [Float64(getfield(information_steps[index], key)) for index in step_order]
		if normalize == :range && key in (:entropy_bits, :effective_cells)
			minimum_value::Float64 = minimum(raw_values)
			maximum_value::Float64 = maximum(raw_values)
			values_by_key[key] = maximum_value > minimum_value ?
								 (raw_values .- minimum_value) ./ (maximum_value - minimum_value) : zeros(Float64, length(raw_values))
		else
			values_by_key[key] = raw_values
		end
	end

	for (position, index) in enumerate(step_order)
		info::NamedTuple = information_steps[index]
		cell_count::Float64 = Float64(info.cell_count)
		if cell_count <= 0.0
			throw(ArgumentError("Cell counts must be positive for logarithmic resolution comparison!"))
		end
		push!(cell_counts, cell_count)
		push!(label_yvals, Float64(info.normalized_entropy))
		for k in plotted_keys
			push!(xvals, cell_count)
			push!(step_xvals, step_labels[length(cell_counts)])
			push!(yvals, values_by_key[k][position])
			push!(metrics, attribute_label_by_key[k])
		end
	end

	metric_layers::Vector{Gadfly.Layer} = Gadfly.Layer[]
	label_layers::Vector{Gadfly.Layer} = Gadfly.Layer[]
	no_highlight(::Any)::Nothing = nothing
	plot_theme::Gadfly.Theme = Gadfly.Theme(; highlight_width=0Gadfly.pt, discrete_highlight_color=no_highlight)
	information_plot::Gadfly.Plot = if xaxis == :steps
		metric_layers = Gadfly.layer(
			Gadfly.Geom.line,
			Gadfly.Geom.point,
			Gadfly.Theme(; highlight_width=0Gadfly.pt, discrete_highlight_color=no_highlight);
			x=step_xvals,
			y=yvals,
			color=metrics
		)
		cell_count_labels::Vector{String} = string.(Int.(cell_counts))
		label_layers = Gadfly.layer(
			Gadfly.Geom.label(; position=:dynamic, hide_overlaps=false);
			x=step_labels,
			y=label_yvals,
			label=cell_count_labels
		)
		Gadfly.plot(
			metric_layers...,
			label_layers...,
			plot_theme,
			Gadfly.Scale.x_discrete,
			Gadfly.Scale.color_discrete_manual(attribute_colors...),
			Gadfly.Guide.colorkey(; title="", labels=attribute_labels),
			Gadfly.Coord.cartesian(; ymin=0.0, ymax=1.0),
			Gadfly.Guide.xlabel("Resolution size"),
			Gadfly.Guide.ylabel(normalize == :range ? "Comparable normalized metric" : "Intrinsic normalized metric"),
			Gadfly.Guide.title(title)
		)
	else
		metric_layers = Gadfly.layer(
			Gadfly.Geom.line,
			Gadfly.Geom.point,
			Gadfly.Theme(; highlight_width=0Gadfly.pt, discrete_highlight_color=no_highlight);
			x=xvals,
			y=yvals,
			color=metrics
		)
		label_layers = Gadfly.layer(
			Gadfly.Geom.label(; position=:dynamic, hide_overlaps=false);
			x=cell_counts,
			y=label_yvals,
			label=step_labels
		)
		# format_log_tick(exponent::Real)::String = isinteger(exponent) ? "10^$(Int(round(exponent)))" : "10^$(round(exponent; digits=2))"
		Gadfly.plot(
			metric_layers...,
			label_layers...,
			plot_theme,
			Gadfly.Scale.x_log10(),
			# Gadfly.Scale.x_log10(labels=format_log_tick),
			Gadfly.Scale.color_discrete_manual(attribute_colors...),
			Gadfly.Guide.colorkey(; title="", labels=attribute_labels),
			Gadfly.Coord.cartesian(; ymin=0.0, ymax=1.0),
			Gadfly.Guide.xlabel("Number of valid cells (log scale)"),
			Gadfly.Guide.ylabel(normalize == :range ? "Comparable normalized metric" : "Intrinsic normalized metric"),
			Gadfly.Guide.title(title)
		)
	end
	if filename != ""
		@info("Saving information plot to file: $filename")
		Mads.plotfileformat(information_plot, filename, 8Gadfly.inch, 4Gadfly.inch)
	end
	return information_plot
end

# Compute a summary of the temporal coding efficiency from the information-theoretic measures.
function _temporal_coding_summary(information::NamedTuple)::NamedTuple
	temporal_metrics::Vector{NamedTuple} = filter(metric::NamedTuple -> metric.role == :temporal, information.axis_information)
	pair_count::Int = sum(metric.pair_count for metric::NamedTuple in temporal_metrics; init=0)
	fixed_width_bits::Int = sum(metric.residual_coding.fixed_width_bits for metric::NamedTuple in temporal_metrics; init=0)
	encoded_bits::Float64 = sum(metric.residual_coding.encoded_bits for metric::NamedTuple in temporal_metrics; init=0.0)
	shannon_bits::Float64 = sum(metric.pair_count * metric.residual_entropy_bits for metric::NamedTuple in temporal_metrics; init=0.0)
	fixed_bits_per_symbol::Float64 = pair_count > 0 ? fixed_width_bits / pair_count : 0.0
	encoded_bits_per_symbol::Float64 = pair_count > 0 ? encoded_bits / pair_count : 0.0
	shannon_bits_per_symbol::Float64 = pair_count > 0 ? shannon_bits / pair_count : 0.0
	coding_savings::Float64 = fixed_width_bits > 0 ? clamp(1.0 - encoded_bits / fixed_width_bits, 0.0, 1.0) : 0.0
	return (
		pair_count=pair_count,
		fixed_bits_per_symbol=fixed_bits_per_symbol,
		shannon_bits_per_symbol=shannon_bits_per_symbol,
		encoded_bits_per_symbol=encoded_bits_per_symbol,
		coding_savings=coding_savings
	)
end

# Compute a summary of the spectral compactness from the information-theoretic measures.
function _spectral_compactness(information::NamedTuple)::Float64
	spectral_metrics::Vector{NamedTuple} = information.spectral_information
	isempty(spectral_metrics) && return NaN
	normalized_entropy::Float64 = Statistics.mean(metric.normalized_spectral_entropy for metric::NamedTuple in spectral_metrics)
	return clamp(1.0 - normalized_entropy, 0.0, 1.0)
end

"""
plot_structure_information(information_steps, steps, filename=""; xaxis=:steps, title_extra="")

Compare normalized structural-information metrics across tensor discretizations.
All displayed metrics use a higher-is-more-informative-or-structured orientation.
"""
function plot_structure_information(information_steps::AbstractVector{<:NamedTuple}, steps::AbstractVector, filename::AbstractString=""; xaxis::Symbol=:steps, title_extra::AbstractString="")::Gadfly.Plot
	if isempty(information_steps)
		throw(ArgumentError("Information steps must not be empty!"))
	end
	if length(information_steps) != length(steps)
		throw(DimensionMismatch("Information steps and resolution labels must have the same length!"))
	end
	if !(xaxis in (:steps, :cells))
		throw(ArgumentError("The xaxis option must be :steps or :cells!"))
	end
	include_spectral::Bool = all(!isempty(information.spectral_information) for information::NamedTuple in information_steps)
	metric_labels::Vector{String} = [
		"Value entropy",
		"Spatial dependence",
		"Spatial coherence",
		"Temporal coherence",
		"Residual coding savings"
	]
	metric_colors::Vector{String} = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#17becf"]
	if include_spectral
		insert!(metric_labels, 5, "Spectral compactness")
		insert!(metric_colors, 5, "#9467bd")
	end
	step_labels::Vector{String} = string.(steps)
	x_step_values::Vector{String} = String[]
	x_cell_values::Vector{Float64} = Float64[]
	y_values::Vector{Float64} = Float64[]
	labels::Vector{String} = String[]
	for (step_index::Int, information::NamedTuple) in enumerate(information_steps)
		coding_summary::NamedTuple = _temporal_coding_summary(information)
		cell_count::Float64 = Float64(information.valid_cell_count)
		cell_count > 0.0 || throw(ArgumentError("Valid cell counts must be positive!"))
		metric_values::Vector{Float64} = [
			clamp(Float64(information.normalized_value_entropy), 0.0, 1.0),
			clamp(Float64(information.spatial_dependence), 0.0, 1.0),
			clamp(1.0 - Float64(information.spatial_variation), 0.0, 1.0),
			clamp(1.0 - Float64(information.temporal_variation), 0.0, 1.0),
			coding_summary.coding_savings
		]
		if include_spectral
			insert!(metric_values, 5, _spectral_compactness(information))
		end
		for metric_index::Int in eachindex(metric_labels)
			push!(x_step_values, step_labels[step_index])
			push!(x_cell_values, cell_count)
			push!(y_values, metric_values[metric_index])
			push!(labels, metric_labels[metric_index])
		end
	end
	title::String = "Structure-aware information by binning resolution$(title_extra)"
	structure_plot::Gadfly.Plot = if xaxis == :steps
		Gadfly.plot(
			Gadfly.layer(Gadfly.Geom.line, Gadfly.Geom.point; x=x_step_values, y=y_values, color=labels),
			Gadfly.Scale.x_discrete,
			Gadfly.Scale.color_discrete_manual(metric_colors...),
			Gadfly.Coord.cartesian(; ymin=0.0, ymax=1.0),
			Gadfly.Guide.xlabel("Resolution size"),
			Gadfly.Guide.ylabel("Normalized metric"),
			Gadfly.Guide.title(title)
		)
	else
		Gadfly.plot(
			Gadfly.layer(Gadfly.Geom.line, Gadfly.Geom.point; x=x_cell_values, y=y_values, color=labels),
			Gadfly.Scale.x_log10(),
			Gadfly.Scale.color_discrete_manual(metric_colors...),
			Gadfly.Coord.cartesian(; ymin=0.0, ymax=1.0),
			Gadfly.Guide.xlabel("Number of valid cells (log scale)"),
			Gadfly.Guide.ylabel("Normalized metric"),
			Gadfly.Guide.title(title)
		)
	end
	if filename != ""
		Mads.plotfileformat(structure_plot, filename, 9Gadfly.inch, 5Gadfly.inch)
	end
	return structure_plot
end

"""
plot_residual_coding(information_steps, steps, filename=""; xaxis=:steps, title_extra="")

Compare fixed-width residual storage, the Shannon limit, and the selected residual
coding method across temporal discretizations. Values are bits per prediction residual.
"""
function plot_residual_coding(information_steps::AbstractVector{<:NamedTuple}, steps::AbstractVector, filename::AbstractString=""; xaxis::Symbol=:steps, title_extra::AbstractString="")::Gadfly.Plot
	if isempty(information_steps)
		throw(ArgumentError("Information steps must not be empty!"))
	end
	if length(information_steps) != length(steps)
		throw(DimensionMismatch("Information steps and resolution labels must have the same length!"))
	end
	if !(xaxis in (:steps, :cells))
		throw(ArgumentError("The xaxis option must be :steps or :cells!"))
	end
	coding_method::String = string(information_steps[1].residual_coding)
	metric_labels::Vector{String} = ["Fixed width", "Shannon limit", "Selected coding ($(coding_method))"]
	metric_colors::Vector{String} = ["#7f7f7f", "#1f77b4", "#d62728"]
	step_labels::Vector{String} = string.(steps)
	x_step_values::Vector{String} = String[]
	x_cell_values::Vector{Float64} = Float64[]
	y_values::Vector{Float64} = Float64[]
	labels::Vector{String} = String[]
	for (step_index::Int, information::NamedTuple) in enumerate(information_steps)
		coding_summary::NamedTuple = _temporal_coding_summary(information)
		cell_count::Float64 = Float64(information.valid_cell_count)
		metric_values::Vector{Float64} = [coding_summary.fixed_bits_per_symbol, coding_summary.shannon_bits_per_symbol, coding_summary.encoded_bits_per_symbol]
		for metric_index::Int in eachindex(metric_labels)
			push!(x_step_values, step_labels[step_index])
			push!(x_cell_values, cell_count)
			push!(y_values, metric_values[metric_index])
			push!(labels, metric_labels[metric_index])
		end
	end
	title::String = "Temporal prediction-residual coding by binning resolution$(title_extra)"
	coding_plot::Gadfly.Plot = if xaxis == :steps
		Gadfly.plot(
			Gadfly.layer(Gadfly.Geom.line, Gadfly.Geom.point; x=x_step_values, y=y_values, color=labels),
			Gadfly.Scale.x_discrete,
			Gadfly.Scale.color_discrete_manual(metric_colors...),
			Gadfly.Guide.xlabel("Resolution size"),
			Gadfly.Guide.ylabel("Bits per temporal residual"),
			Gadfly.Guide.title(title)
		)
	else
		Gadfly.plot(
			Gadfly.layer(Gadfly.Geom.line, Gadfly.Geom.point; x=x_cell_values, y=y_values, color=labels),
			Gadfly.Scale.x_log10(),
			Gadfly.Scale.color_discrete_manual(metric_colors...),
			Gadfly.Guide.xlabel("Number of valid cells (log scale)"),
			Gadfly.Guide.ylabel("Bits per temporal residual"),
			Gadfly.Guide.title(title)
		)
	end
	if filename != ""
		Mads.plotfileformat(coding_plot, filename, 9Gadfly.inch, 5Gadfly.inch)
	end
	return coding_plot
end
