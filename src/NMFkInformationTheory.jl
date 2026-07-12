import Statistics
import Gadfly

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
		["Entropy Bits (Range Normalized)", "Effective Cells (Range Normalized)", "Normalized Entropy", "Occupancy"] :
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
	plot_theme::Gadfly.Theme = Gadfly.Theme(highlight_width=0Gadfly.pt, discrete_highlight_color=no_highlight)
	information_plot::Gadfly.Plot = if xaxis == :steps
		metric_layers = Gadfly.layer(
			Gadfly.Geom.line,
			Gadfly.Geom.point,
			Gadfly.Theme(highlight_width=0Gadfly.pt, discrete_highlight_color=no_highlight),
			x=step_xvals,
			y=yvals,
			color=metrics
		)
		cell_count_labels::Vector{String} = string.(Int.(cell_counts))
		label_layers = Gadfly.layer(
			Gadfly.Geom.label(position=:dynamic, hide_overlaps=false),
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
			Gadfly.Guide.colorkey(title="", labels=attribute_labels),
			Gadfly.Coord.cartesian(ymin=0.0, ymax=1.0),
			Gadfly.Guide.xlabel("Resolution size"),
			Gadfly.Guide.ylabel(normalize == :range ? "Comparable normalized metric" : "Intrinsic normalized metric"),
			Gadfly.Guide.title(title)
		)
	else
		metric_layers = Gadfly.layer(
			Gadfly.Geom.line,
			Gadfly.Geom.point,
			Gadfly.Theme(highlight_width=0Gadfly.pt, discrete_highlight_color=no_highlight),
			x=xvals,
			y=yvals,
			color=metrics
		)
		label_layers = Gadfly.layer(
			Gadfly.Geom.label(position=:dynamic, hide_overlaps=false),
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
			Gadfly.Guide.colorkey(title="", labels=attribute_labels),
			Gadfly.Coord.cartesian(ymin=0.0, ymax=1.0),
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
