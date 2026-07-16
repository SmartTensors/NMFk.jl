import Statistics
import Gadfly
import LinearAlgebra
import Dates
import SHA
import Serialization

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

# Compute empirical entropy when observations carry nonnegative weights.
function _weighted_discrete_entropy(symbols::AbstractVector{<:Integer}, weights::AbstractVector{<:Real})::Float64
    if length(symbols) != length(weights)
        throw(DimensionMismatch("Symbols and observation weights must have the same length!"))
    end
    maximum_weight::Float64 = maximum(Float64(weight) for weight::Real in weights; init=0.0)
    maximum_weight > 0.0 || throw(ArgumentError("At least one observation weight must be positive!"))
    scaled_weights::Vector{Float64} = Float64.(weights) ./ maximum_weight
    total_weight::Float64 = sum(scaled_weights)
    state_weights::Dict{Int, Float64} = Dict{Int, Float64}()
    for observation_index::Int in eachindex(symbols)
        weight::Float64 = scaled_weights[observation_index]
        weight == 0.0 && continue
        symbol::Int = Int(symbols[observation_index])
        state_weights[symbol] = get(state_weights, symbol, 0.0) + weight
    end
    probabilities::Vector{Float64} = collect(values(state_weights)) ./ total_weight
    return -sum(probabilities .* log2.(probabilities))
end

function _raw_value_is_valid(value::Missing)::Bool
    return false
end

function _raw_value_is_valid(value::Nothing)::Bool
    return false
end

function _raw_value_is_valid(value::Real)::Bool
    return isfinite(value)
end

function _raw_value_is_valid(value::Any)::Bool
    return true
end

function _raw_datetime_milliseconds(value::Dates.DateTime)::Int64
    epoch::Dates.DateTime = Dates.DateTime(1970, 1, 1)
    return Int64(Dates.value(value - epoch))
end

function _raw_datetime_milliseconds(value::Dates.Date)::Int64
    return _raw_datetime_milliseconds(Dates.DateTime(value))
end

function _raw_period_milliseconds(precision::Dates.FixedPeriod)::Int64
    precision_milliseconds::Int64 = Int64(Dates.value(Dates.Millisecond(precision)))
    precision_milliseconds > 0 || throw(ArgumentError("Date/time precision must be positive!"))
    return precision_milliseconds
end

function _raw_period_milliseconds(precision::Union{Dates.Microsecond,Dates.Nanosecond})::Int64
    throw(ArgumentError("Date and DateTime values have millisecond resolution; use Millisecond or a coarser fixed period!"))
end

function _raw_floor_symbol(value::Float64, precision::Float64, origin::Float64)::Int
    scaled_value::Float64 = (value - origin) / precision
    isfinite(scaled_value) || throw(ArgumentError("Raw-data value cannot be represented at the requested precision!"))
    input_resolution::Float64 = max(eps(abs(value)), eps(abs(origin)), eps(precision))
    if precision <= 8.0 * input_resolution
        throw(ArgumentError("Requested raw-data precision is finer than the numerical resolution of the input value!"))
    end
    nearest_integer::Float64 = round(scaled_value)
    rounding_tolerance::Float64 = 8.0 * eps(abs(scaled_value))
    stable_scaled_value::Float64 =
        rounding_tolerance <= 1.0e-9 &&
        abs(scaled_value - nearest_integer) <= rounding_tolerance ?
        nearest_integer : scaled_value
    return floor(Int, stable_scaled_value)
end

function _raw_numeric_symbol(value::Integer, precision::Integer, origin::Integer)::Int
    bin_coordinate::BigInt = fld(BigInt(value) - BigInt(origin), BigInt(precision))
    if bin_coordinate < typemin(Int) || bin_coordinate > typemax(Int)
        throw(ArgumentError("The requested integer raw-data bin coordinate exceeds the supported Int range!"))
    end
    return Int(bin_coordinate)
end

function _raw_numeric_symbol(value::Real, precision::Real, origin::Real)::Int
    return _raw_floor_symbol(Float64(value), Float64(precision), Float64(origin))
end

function _raw_quantize_column(
    column::AbstractVector,
    valid_indices::Vector{Int},
    precision::Real,
    origin::Any,
)::Tuple{Vector{Int},Any,Symbol,Bool}
    precision_value::Float64 = Float64(precision)
    if !isfinite(precision_value) || precision_value <= 0.0
        throw(ArgumentError("Numeric raw-data precision must be finite and positive!"))
    end
    if origin !== nothing && !(origin isa Real)
        throw(ArgumentError("A numeric raw-data origin must be real!"))
    end
    origin_value::Real = origin === nothing ? zero(precision) : origin
    isfinite(origin_value) || throw(ArgumentError("Numeric raw-data origin must be finite!"))
    symbols::Vector{Int} = Vector{Int}(undef, length(valid_indices))
    for (position::Int, observation_index::Int) in enumerate(valid_indices)
        value::Any = column[observation_index]
        value isa Real || throw(ArgumentError("A numeric precision requires a real-valued raw-data column!"))
        symbols[position] = _raw_numeric_symbol(value, precision, origin_value)
    end
    return symbols, origin_value, :numeric, true
end

function _raw_quantize_column(
    column::AbstractVector,
    valid_indices::Vector{Int},
    precision::Dates.FixedPeriod,
    origin::Any,
)::Tuple{Vector{Int},Any,Symbol,Bool}
    precision_milliseconds::Int64 = _raw_period_milliseconds(precision)
    if origin !== nothing && !(origin isa Dates.Date || origin isa Dates.DateTime)
        throw(ArgumentError("A date/time raw-data origin must be a Date or DateTime!"))
    end
    origin_value::Union{Dates.Date,Dates.DateTime} = origin === nothing ?
        Dates.DateTime(1970, 1, 1) : origin
    origin_milliseconds::Int64 = _raw_datetime_milliseconds(origin_value)
    symbols::Vector{Int} = Vector{Int}(undef, length(valid_indices))
    for (position::Int, observation_index::Int) in enumerate(valid_indices)
        value::Any = column[observation_index]
        if !(value isa Dates.Date || value isa Dates.DateTime)
            throw(ArgumentError("A date/time precision requires a Date or DateTime raw-data column!"))
        end
        value_milliseconds::Int64 = _raw_datetime_milliseconds(value)
        symbols[position] = fld(value_milliseconds - origin_milliseconds, precision_milliseconds)
    end
    return symbols, origin_value, :datetime, true
end

function _raw_quantize_column(
    column::AbstractVector,
    valid_indices::Vector{Int},
    precision::Dates.Period,
    origin::Any,
)::Tuple{Vector{Int},Any,Symbol,Bool}
    throw(ArgumentError("Date/time raw-data precision must be fixed (Week through Millisecond); Month, Quarter, and Year are not fixed durations!"))
end

function _raw_quantize_column(
    column::AbstractVector,
    valid_indices::Vector{Int},
    precision::Nothing,
    origin::Any,
)::Tuple{Vector{Int},Any,Symbol,Bool}
    origin === nothing || throw(ArgumentError("Exact/categorical raw-data columns do not use an origin!"))
    state_lookup::Dict{Any, Int} = Dict{Any, Int}()
    symbols::Vector{Int} = Vector{Int}(undef, length(valid_indices))
    for (position::Int, observation_index::Int) in enumerate(valid_indices)
        value::Any = column[observation_index]
        if !haskey(state_lookup, value)
            state_lookup[value] = length(state_lookup) + 1
        end
        symbols[position] = state_lookup[value]
    end
    return symbols, nothing, :exact, false
end

function _raw_joint_symbols(columns::Vector{Vector{Int}})::Vector{Int}
    isempty(columns) && throw(ArgumentError("At least one raw-data state feature is required!"))
    observation_count::Int = length(first(columns))
    all(length(column) == observation_count for column::Vector{Int} in columns) ||
        throw(DimensionMismatch("All raw-data state columns must have the same length!"))
    state_lookup::Dict{Tuple, Int} = Dict{Tuple, Int}()
    symbols::Vector{Int} = Vector{Int}(undef, observation_count)
    for observation_index::Int = 1:observation_count
        state::Tuple = Tuple(column[observation_index] for column::Vector{Int} in columns)
        if !haskey(state_lookup, state)
            state_lookup[state] = length(state_lookup) + 1
        end
        symbols[observation_index] = state_lookup[state]
    end
    return symbols
end

function _raw_residual_information(
    symbols::Vector{Int},
    ordered_positions::Vector{Int},
    residual_coding::Symbol,
    applicable::Bool,
)::NamedTuple
    if !applicable
        return (
            applicable=false,
            weighting=:unweighted_sequence,
            pair_count=0,
            residual_entropy_bits=0.0,
            predictive_gain_bits=0.0,
            method=residual_coding,
            encoded_bits=0.0,
            bits_per_residual=0.0,
            fixed_width_bits=0,
            fixed_bits_per_residual=0.0,
            coding_savings=0.0,
        )
    end
    ordered_symbols::Vector{Int} = symbols[ordered_positions]
    pair_count::Int = max(length(ordered_symbols) - 1, 0)
    residual_values::Vector{BigInt} = BigInt[]
    for pair_index::Int = 1:pair_count
        push!(
            residual_values,
            BigInt(ordered_symbols[pair_index + 1]) - BigInt(ordered_symbols[pair_index]),
        )
    end
    residual_lookup::Dict{BigInt, Int} = Dict{BigInt, Int}()
    residual_symbols::Vector{Int} = Vector{Int}(undef, pair_count)
    for residual_index::Int in eachindex(residual_values)
        residual_value::BigInt = residual_values[residual_index]
        if !haskey(residual_lookup, residual_value)
            residual_lookup[residual_value] = length(residual_lookup) + 1
        end
        residual_symbols[residual_index] = residual_lookup[residual_value]
    end
    source_span::BigInt = isempty(ordered_symbols) ? BigInt(1) :
        BigInt(maximum(ordered_symbols)) - BigInt(minimum(ordered_symbols)) + 1
    possible_residual_count::BigInt = 2 * source_span - 1
    fixed_bits_per_symbol::Int = possible_residual_count > 1 ?
        ndigits(possible_residual_count - 1; base=2) : 0
    fixed_width_bits::Int = Base.checked_mul(pair_count, fixed_bits_per_symbol)
    right_symbols::Vector{Int} = pair_count > 0 ? ordered_symbols[2:end] : Int[]
    residual_entropy::Float64 = _discrete_entropy(residual_symbols)
    right_entropy::Float64 = _discrete_entropy(right_symbols)
    encoded_bits::Float64 = if residual_coding == :none
        Float64(fixed_width_bits)
    elseif residual_coding == :shannon
        pair_count * residual_entropy
    else
        Float64(_huffman_encoded_bits(residual_symbols))
    end
    coding_savings::Float64 = fixed_width_bits > 0 ?
        clamp(1.0 - encoded_bits / fixed_width_bits, 0.0, 1.0) : 0.0
    fixed_bits_per_residual::Float64 = pair_count > 0 ? fixed_width_bits / pair_count : 0.0
    return (
        applicable=true,
        weighting=:unweighted_sequence,
        pair_count=pair_count,
        residual_entropy_bits=residual_entropy,
        predictive_gain_bits=right_entropy - residual_entropy,
        method=residual_coding,
        encoded_bits=encoded_bits,
        bits_per_residual=pair_count > 0 ? encoded_bits / pair_count : 0.0,
        fixed_width_bits=fixed_width_bits,
        fixed_bits_per_residual=fixed_bits_per_residual,
        coding_savings=coding_savings,
    )
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
    temporal_dependence::Float64 = isempty(temporal_metrics) ? 0.0 : Statistics.mean(metric.normalized_mutual_information for metric::NamedTuple in temporal_metrics)
    temporal_predictive_gain::Float64 = isempty(temporal_metrics) ? 0.0 : Statistics.mean(metric.predictive_gain_bits for metric::NamedTuple in temporal_metrics)
    temporal_variation::Float64 = isempty(temporal_metrics) ? 0.0 : Statistics.mean(metric.mean_normalized_difference for metric::NamedTuple in temporal_metrics)
    return (
        value_entropy_bits=value_entropy,
        normalized_value_entropy=normalized_value_entropy,
        spatial_dependence=spatial_dependence,
        spatial_variation=spatial_variation,
        temporal_dependence=temporal_dependence,
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

function _raw_observation_count(data::NamedTuple)::Int
    feature_names::Vector{Symbol} = collect(keys(data))
    isempty(feature_names) && throw(ArgumentError("Raw data must contain at least one feature column!"))
    first_column::Any = getfield(data, first(feature_names))
    first_column isa AbstractVector || throw(ArgumentError("Every raw-data feature must be an observation vector!"))
    observation_count::Int = length(first_column)
    for feature_name::Symbol in feature_names
        column::Any = getfield(data, feature_name)
        column isa AbstractVector || throw(ArgumentError("Every raw-data feature must be an observation vector!"))
        length(column) == observation_count ||
            throw(DimensionMismatch("All raw-data feature columns must have the same length!"))
    end
    return observation_count
end

function _raw_state_diagnostics(symbols::Vector{Int})::NamedTuple
    counts::Dict{Int, Int} = Dict{Int, Int}()
    for symbol::Int in symbols
        counts[symbol] = get(counts, symbol, 0) + 1
    end
    state_count::Int = length(counts)
    singleton_count::Int = count(==(1), values(counts))
    observation_count::Int = length(symbols)
    return (
        state_count=state_count,
        singleton_count=singleton_count,
        singleton_fraction=observation_count > 0 ? singleton_count / observation_count : 0.0,
        unique_state_fraction=observation_count > 0 ? state_count / observation_count : 0.0,
    )
end

function _raw_state_fingerprint(
    state_feature_names::Vector{Symbol},
    state_columns::Vector{Vector{Int}},
    state_feature_information::Vector{NamedTuple},
    state_exact_values::Vector{Vector{Any}},
    observation_indices::Vector{Int},
    observation_weights::Vector{Float64},
)::String
    feature_count::Int = length(state_feature_names)
    length(state_columns) == feature_count ||
        throw(DimensionMismatch("Raw-state fingerprint columns must match the state features!"))
    length(state_feature_information) == feature_count ||
        throw(DimensionMismatch("Raw-state fingerprint metadata must match the state features!"))
    length(state_exact_values) == feature_count ||
        throw(DimensionMismatch("Raw-state fingerprint exact values must match the state features!"))
    fingerprint_buffer::IOBuffer = IOBuffer()
    write(fingerprint_buffer, Int64(feature_count))
    for feature_index::Int in eachindex(state_feature_names)
        feature_name::Symbol = state_feature_names[feature_index]
        feature_information::NamedTuple = state_feature_information[feature_index]
        feature_name_bytes::Vector{UInt8} = collect(codeunits(String(feature_name)))
        write(fingerprint_buffer, Int64(length(feature_name_bytes)))
        write(fingerprint_buffer, feature_name_bytes)
        Serialization.serialize(fingerprint_buffer, feature_information.kind)
        Serialization.serialize(fingerprint_buffer, feature_information.precision)
        Serialization.serialize(fingerprint_buffer, feature_information.origin)
        if feature_information.kind == :exact
            exact_values::Vector{Any} = state_exact_values[feature_index]
            length(exact_values) == length(observation_indices) ||
                throw(DimensionMismatch("Exact raw-state fingerprint values must match the observations!"))
            for exact_value::Any in exact_values
                Serialization.serialize(fingerprint_buffer, exact_value)
            end
        end
    end
    for observation_index::Int in observation_indices
        write(fingerprint_buffer, Int64(observation_index))
    end
    for state_column::Vector{Int} in state_columns
        for symbol::Int in state_column
            write(fingerprint_buffer, Int64(symbol))
        end
    end
    for observation_weight::Float64 in observation_weights
        write(fingerprint_buffer, observation_weight)
    end
    return bytes2hex(SHA.sha256(take!(fingerprint_buffer)))
end

function _raw_comparison_fingerprint(
    raw_fingerprint::String,
    observation_indices::Vector{Int},
    raw_state_symbols::Vector{Int},
    observation_weights::Vector{Float64},
)::String
    fingerprint_buffer::IOBuffer = IOBuffer()
    write(fingerprint_buffer, collect(codeunits(raw_fingerprint)))
    for observation_index::Int in observation_indices
        write(fingerprint_buffer, Int64(observation_index))
    end
    for raw_state_symbol::Int in raw_state_symbols
        write(fingerprint_buffer, Int64(raw_state_symbol))
    end
    for observation_weight::Float64 in observation_weights
        write(fingerprint_buffer, observation_weight)
    end
    return bytes2hex(SHA.sha256(take!(fingerprint_buffer)))
end

function _raw_mapping_conflict_information(
    raw_state_symbols::Vector{Int},
    grid_state_symbols::Vector{Int},
    observation_weights::Vector{Float64},
)::NamedTuple
    length(raw_state_symbols) == length(grid_state_symbols) ||
        throw(DimensionMismatch("Raw and grid states must have the same length!"))
    length(raw_state_symbols) == length(observation_weights) ||
        throw(DimensionMismatch("Raw states and weights must have the same length!"))
    first_grid_by_raw_state::Dict{Int,Int} = Dict{Int,Int}()
    conflicting_raw_states::Set{Int} = Set{Int}()
    for observation_index::Int in eachindex(raw_state_symbols)
        raw_state::Int = raw_state_symbols[observation_index]
        grid_state::Int = grid_state_symbols[observation_index]
        if haskey(first_grid_by_raw_state, raw_state)
            first_grid_by_raw_state[raw_state] == grid_state ||
                push!(conflicting_raw_states, raw_state)
        else
            first_grid_by_raw_state[raw_state] = grid_state
        end
    end
    conflict_observation_count::Int = 0
    conflict_weight::Float64 = 0.0
    for observation_index::Int in eachindex(raw_state_symbols)
        if raw_state_symbols[observation_index] in conflicting_raw_states
            conflict_observation_count += 1
            conflict_weight += observation_weights[observation_index]
        end
    end
    total_weight::Float64 = sum(observation_weights)
    return (
        mapping_is_deterministic=isempty(conflicting_raw_states),
        mapping_determinism_scope=:observed_support,
        conflicting_raw_state_count=length(conflicting_raw_states),
        conflict_observation_count=conflict_observation_count,
        conflict_weight_fraction=total_weight > 0.0 ? conflict_weight / total_weight : 0.0,
    )
end

"""
rawdata_information(data; precisions, origins=nothing, state_features=keys(data), ...)

Measure the empirical information in unbinned observation records. `data` is a
named tuple of equal-length feature vectors. Numeric and date/time features are
quantized on fixed, origin-anchored lattices; pass `precision=nothing` for an
already discrete or categorical feature.

The joint `state_entropy_bits` is computed from `state_features`, not by adding
their marginal entropies. Optional nonnegative `weights` define observation
probabilities. Prediction residuals are computed separately for each ordered
numeric/date feature and never from arbitrary joint-state identifiers.
"""
function rawdata_information(
    data::NamedTuple;
    precisions::NamedTuple,
    origins::Union{Nothing,NamedTuple}=nothing,
    state_features::Union{Tuple,AbstractVector{Symbol}}=Tuple(keys(data)),
    valid_mask::AbstractVector{Bool}=trues(_raw_observation_count(data)),
    weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
    sequence_order::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    residual_coding::Symbol=:huffman,
)::NamedTuple
    observation_count::Int = _raw_observation_count(data)
    feature_names::Vector{Symbol} = collect(keys(data))
    length(valid_mask) == observation_count ||
        throw(DimensionMismatch("The raw-data valid mask must match the observation count!"))
    all(haskey(precisions, feature_name) for feature_name::Symbol in feature_names) ||
        throw(ArgumentError("A precision must be supplied for every raw-data feature!"))
    length(keys(precisions)) == length(feature_names) ||
        throw(ArgumentError("Raw-data precisions contain unknown feature names!"))
    if origins !== nothing
        all(haskey(origins, feature_name) for feature_name::Symbol in feature_names) ||
            throw(ArgumentError("When origins are supplied, every raw-data feature needs an origin!"))
        length(keys(origins)) == length(feature_names) ||
            throw(ArgumentError("Raw-data origins contain unknown feature names!"))
    end
    if !(residual_coding in (:none, :shannon, :huffman))
        throw(ArgumentError("Residual coding must be :none, :shannon, or :huffman!"))
    end
    state_feature_names::Vector{Symbol} = Symbol.(collect(state_features))
    isempty(state_feature_names) && throw(ArgumentError("At least one state feature is required!"))
    all(feature_name in feature_names for feature_name::Symbol in state_feature_names) ||
        throw(ArgumentError("Every state feature must identify a raw-data feature column!"))
    length(unique(state_feature_names)) == length(state_feature_names) ||
        throw(ArgumentError("Raw-data state features must be unique!"))

    input_weights::Vector{Float64} = weights === nothing ?
        ones(Float64, observation_count) : Float64.(weights)
    length(input_weights) == observation_count ||
        throw(DimensionMismatch("Raw-data weights must match the observation count!"))
    if any(weight::Float64 -> !isfinite(weight) || weight < 0.0, input_weights)
        throw(ArgumentError("Raw-data weights must be finite and nonnegative!"))
    end
    columns::Vector{AbstractVector} = AbstractVector[]
    for feature_name::Symbol in feature_names
        push!(columns, getfield(data, feature_name))
    end
    finite_mask::BitVector = BitVector(valid_mask)
    for observation_index::Int = 1:observation_count
        if finite_mask[observation_index]
            finite_mask[observation_index] = all(
                _raw_value_is_valid(column[observation_index]) for column::AbstractVector in columns
            )
        end
    end
    candidate_indices::Vector{Int} = findall(finite_mask)
    isempty(candidate_indices) && throw(ArgumentError("Raw data contain no valid observations!"))
    weight_scale::Float64 = maximum(input_weights[candidate_indices]; init=0.0)
    weight_scale > 0.0 || throw(ArgumentError("At least one valid raw-data observation weight must be positive!"))
    input_weights ./= weight_scale
    for observation_index::Int in candidate_indices
        if input_weights[observation_index] == 0.0
            finite_mask[observation_index] = false
        end
    end
    valid_indices::Vector{Int} = findall(finite_mask)
    isempty(valid_indices) && throw(ArgumentError("Raw data contain no valid observations!"))

    observation_weights::Vector{Float64} = input_weights[valid_indices]
    total_weight::Float64 = sum(observation_weights)
    total_weight > 0.0 || throw(ArgumentError("At least one valid raw-data observation must have positive weight!"))

    input_sequence_order::Vector{Int} = sequence_order === nothing ?
        collect(1:observation_count) : Int.(sequence_order)
    if length(input_sequence_order) != observation_count ||
       sort(input_sequence_order) != collect(1:observation_count)
        throw(ArgumentError("Raw-data sequence_order must be a permutation of all input observation indices!"))
    end
    valid_position_by_input::Vector{Int} = zeros(Int, observation_count)
    for (valid_position::Int, observation_index::Int) in enumerate(valid_indices)
        valid_position_by_input[observation_index] = valid_position
    end
    ordered_positions::Vector{Int} = Int[]
    ordered_observation_indices::Vector{Int} = Int[]
    for observation_index::Int in input_sequence_order
        valid_position::Int = valid_position_by_input[observation_index]
        if valid_position > 0
            push!(ordered_positions, valid_position)
            push!(ordered_observation_indices, observation_index)
        end
    end

    quantized_columns::Vector{Vector{Int}} = Vector{Int}[]
    feature_information::Vector{NamedTuple} = NamedTuple[]
    resolved_origins::Vector{Any} = Any[]
    for feature_index::Int in eachindex(feature_names)
        feature_name::Symbol = feature_names[feature_index]
        precision::Any = getfield(precisions, feature_name)
        supplied_origin::Any = origins === nothing ? nothing : getfield(origins, feature_name)
        symbols::Vector{Int}, resolved_origin::Any, feature_kind::Symbol, ordered::Bool =
            _raw_quantize_column(columns[feature_index], valid_indices, precision, supplied_origin)
        push!(quantized_columns, symbols)
        push!(resolved_origins, resolved_origin)
        entropy_bits::Float64 = _weighted_discrete_entropy(symbols, observation_weights)
        diagnostics::NamedTuple = _raw_state_diagnostics(symbols)
        residual_information::NamedTuple = _raw_residual_information(
            symbols,
            ordered_positions,
            residual_coding,
            ordered,
        )
        push!(feature_information, (
            name=feature_name,
            kind=feature_kind,
            precision=precision,
            origin=resolved_origin,
            state_feature=feature_name in state_feature_names,
            entropy_bits=entropy_bits,
            effective_states=exp2(entropy_bits),
            state_count=diagnostics.state_count,
            singleton_count=diagnostics.singleton_count,
            singleton_fraction=diagnostics.singleton_fraction,
            residual_information=residual_information,
        ))
    end

    state_columns::Vector{Vector{Int}} = Vector{Int}[]
    state_feature_information::Vector{NamedTuple} = NamedTuple[]
    state_exact_values::Vector{Vector{Any}} = Vector{Any}[]
    for state_feature_name::Symbol in state_feature_names
        feature_index::Int = Base.findfirst(==(state_feature_name), feature_names)
        push!(state_columns, quantized_columns[feature_index])
        selected_feature_information::NamedTuple = feature_information[feature_index]
        push!(state_feature_information, selected_feature_information)
        exact_values::Vector{Any} = selected_feature_information.kind == :exact ?
            Any[columns[feature_index][observation_index] for observation_index::Int in valid_indices] :
            Any[]
        push!(state_exact_values, exact_values)
    end
    state_symbols::Vector{Int} = _raw_joint_symbols(state_columns)
    state_fingerprint::String = _raw_state_fingerprint(
        state_feature_names,
        state_columns,
        state_feature_information,
        state_exact_values,
        valid_indices,
        observation_weights,
    )
    state_entropy_bits::Float64 = _weighted_discrete_entropy(state_symbols, observation_weights)
    record_symbols::Vector{Int} = collect(1:length(valid_indices))
    record_entropy_bits::Float64 = _weighted_discrete_entropy(record_symbols, observation_weights)
    state_diagnostics::NamedTuple = _raw_state_diagnostics(state_symbols)
    normalized_state_entropy::Float64 = record_entropy_bits > 0.0 ?
        state_entropy_bits / record_entropy_bits : NaN
    resolved_origins_named::NamedTuple =
        NamedTuple{Tuple(feature_names)}(Tuple(resolved_origins))
    return (
        estimator=:empirical_plugin,
        feature_names=feature_names,
        state_features=state_feature_names,
        precisions=precisions,
        origins=resolved_origins_named,
        residual_coding=residual_coding,
        input_observation_count=observation_count,
        valid_observation_count=length(valid_indices),
        excluded_observation_count=observation_count - length(valid_indices),
        scaled_total_weight=total_weight,
        weight_scale=weight_scale,
        weighted=weights !== nothing,
        observation_indices=valid_indices,
        observation_weights=observation_weights,
        state_symbols=state_symbols,
        state_fingerprint=state_fingerprint,
        state_entropy_bits=state_entropy_bits,
        record_entropy_bits=record_entropy_bits,
        normalized_state_entropy=normalized_state_entropy,
        effective_state_count=exp2(state_entropy_bits),
        effective_record_count=exp2(record_entropy_bits),
        state_count=state_diagnostics.state_count,
        singleton_state_count=state_diagnostics.singleton_count,
        singleton_fraction=state_diagnostics.singleton_fraction,
        unique_state_fraction=state_diagnostics.unique_state_fraction,
        feature_information=feature_information,
        sequence_order_explicit=sequence_order !== nothing,
        ordered_observation_indices=ordered_observation_indices,
    )
end

function rawdata_information(
    data::AbstractVector;
    precision::Any,
    origin::Any=nothing,
    feature_name::Symbol=:value,
    valid_mask::AbstractVector{Bool}=trues(length(data)),
    weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
    sequence_order::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    residual_coding::Symbol=:huffman,
)::NamedTuple
    names::Tuple{Symbol} = (feature_name,)
    named_data::NamedTuple = NamedTuple{names}((data,))
    precisions::NamedTuple = NamedTuple{names}((precision,))
    origins::NamedTuple = NamedTuple{names}((origin,))
    return rawdata_information(
        named_data;
        precisions=precisions,
        origins=origins,
        valid_mask=valid_mask,
        weights=weights,
        sequence_order=sequence_order,
        residual_coding=residual_coding,
    )
end

function rawdata_information(
    data::AbstractMatrix;
    precisions::AbstractVector,
    origins::Union{Nothing,AbstractVector}=nothing,
    feature_names::Union{Nothing,AbstractVector{Symbol}}=nothing,
    observation_dim::Int=1,
    state_features::Union{Nothing,Tuple,AbstractVector{Symbol}}=nothing,
    valid_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
    sequence_order::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    residual_coding::Symbol=:huffman,
)::NamedTuple
    observation_dim in (1, 2) || throw(ArgumentError("observation_dim must be 1 or 2!"))
    feature_count::Int = size(data, 3 - observation_dim)
    observation_count::Int = size(data, observation_dim)
    length(precisions) == feature_count ||
        throw(DimensionMismatch("The precision vector must match the number of matrix features!"))
    if origins !== nothing && length(origins) != feature_count
        throw(DimensionMismatch("The origin vector must match the number of matrix features!"))
    end
    names::Vector{Symbol} = feature_names === nothing ?
        [Symbol("feature_$(feature_index)") for feature_index::Int = 1:feature_count] :
        collect(feature_names)
    length(names) == feature_count ||
        throw(DimensionMismatch("The feature-name vector must match the number of matrix features!"))
    length(unique(names)) == feature_count || throw(ArgumentError("Matrix feature names must be unique!"))
    columns::Vector{AbstractVector} = AbstractVector[]
    for feature_index::Int = 1:feature_count
        column::AbstractVector = observation_dim == 1 ?
            collect(view(data, :, feature_index)) : collect(view(data, feature_index, :))
        push!(columns, column)
    end
    name_tuple::Tuple = Tuple(names)
    named_data::NamedTuple = NamedTuple{name_tuple}(Tuple(columns))
    named_precisions::NamedTuple = NamedTuple{name_tuple}(Tuple(precisions))
    origin_values::Tuple = origins === nothing ? ntuple(feature_index::Int -> nothing, feature_count) : Tuple(origins)
    named_origins::NamedTuple = NamedTuple{name_tuple}(origin_values)
    selected_state_features::Union{Tuple,AbstractVector{Symbol}} =
        state_features === nothing ? name_tuple : state_features
    selected_mask::AbstractVector{Bool} = valid_mask === nothing ? trues(observation_count) : valid_mask
    return rawdata_information(
        named_data;
        precisions=named_precisions,
        origins=named_origins,
        state_features=selected_state_features,
        valid_mask=selected_mask,
        weights=weights,
        sequence_order=sequence_order,
        residual_coding=residual_coding,
    )
end

function _raw_collision_probability(symbols::Vector{Int}, weights::Vector{Float64})::Float64
    total_weight::Float64 = sum(weights)
    state_weights::Dict{Int, Float64} = Dict{Int, Float64}()
    for observation_index::Int in eachindex(symbols)
        symbol::Int = symbols[observation_index]
        state_weights[symbol] = get(state_weights, symbol, 0.0) + weights[observation_index]
    end
    return sum((state_weight / total_weight)^2 for state_weight::Float64 in values(state_weights))
end

"""
compare_rawdata_grid(raw_information, grid_assignments; grid_cell_count=nothing)

Compare paired raw observation states with their grid assignments. The retained
empirical state distinguishability is mutual information `I(raw; grid)` and the
merging loss is conditional entropy `H(raw | grid)`. This is an observed-support
collision measure, not physical reconstruction or numeric precision loss.
`mapping_uncertainty_bits = H(grid | raw)` accompanies an exact observed-support
mapping-conflict diagnostic.
"""
function compare_rawdata_grid(
    raw_information::NamedTuple,
    grid_assignments::NamedTuple;
    valid_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    grid_cell_count::Union{Nothing,Integer}=nothing,
)::NamedTuple
    required_keys::Tuple = (
        :input_observation_count,
        :observation_indices,
        :observation_weights,
        :state_symbols,
        :state_fingerprint,
        :weighted,
        :weight_scale,
    )
    all(haskey(raw_information, key) for key::Symbol in required_keys) ||
        throw(ArgumentError("raw_information must be produced by rawdata_information!"))
    input_observation_count::Int = Int(raw_information.input_observation_count)
    selected_valid_mask::AbstractVector{Bool} = valid_mask === nothing ?
        trues(input_observation_count) : valid_mask
    length(selected_valid_mask) == input_observation_count ||
        throw(DimensionMismatch("The grid valid mask must match the raw input observation count!"))
    grid_observation_count::Int = _raw_observation_count(grid_assignments)
    grid_observation_count == input_observation_count ||
        throw(DimensionMismatch("Grid assignments and raw data must describe the same input observations!"))
    grid_feature_names::Vector{Symbol} = collect(keys(grid_assignments))
    grid_columns::Vector{AbstractVector} = AbstractVector[]
    for feature_name::Symbol in grid_feature_names
        push!(grid_columns, getfield(grid_assignments, feature_name))
    end

    raw_observation_indices::Vector{Int} = Int.(raw_information.observation_indices)
    raw_state_symbols_all::Vector{Int} = Int.(raw_information.state_symbols)
    raw_weights_all::Vector{Float64} = Float64.(raw_information.observation_weights)
    comparison_positions::Vector{Int} = Int[]
    comparison_observation_indices::Vector{Int} = Int[]
    for raw_position::Int in eachindex(raw_observation_indices)
        observation_index::Int = raw_observation_indices[raw_position]
        grid_valid::Bool = selected_valid_mask[observation_index] && all(
            _raw_value_is_valid(column[observation_index]) for column::AbstractVector in grid_columns
        )
        if grid_valid
            push!(comparison_positions, raw_position)
            push!(comparison_observation_indices, observation_index)
        end
    end
    isempty(comparison_positions) && throw(ArgumentError("No observations are valid in both raw data and grid assignments!"))
    raw_state_symbols::Vector{Int} = raw_state_symbols_all[comparison_positions]
    observation_weights::Vector{Float64} = raw_weights_all[comparison_positions]
    cohort_fingerprint::String = _raw_comparison_fingerprint(
        String(raw_information.state_fingerprint),
        comparison_observation_indices,
        raw_state_symbols,
        observation_weights,
    )
    encoded_grid_columns::Vector{Vector{Int}} = Vector{Int}[]
    for column::AbstractVector in grid_columns
        grid_symbols::Vector{Int}, resolved_origin::Any, feature_kind::Symbol, ordered::Bool =
            _raw_quantize_column(column, comparison_observation_indices, nothing, nothing)
        push!(encoded_grid_columns, grid_symbols)
    end
    grid_state_symbols::Vector{Int} = _raw_joint_symbols(encoded_grid_columns)
    joint_symbols::Vector{Int} = _raw_joint_symbols([raw_state_symbols, grid_state_symbols])
    raw_entropy_bits::Float64 = _weighted_discrete_entropy(raw_state_symbols, observation_weights)
    grid_entropy_bits::Float64 = _weighted_discrete_entropy(grid_state_symbols, observation_weights)
    joint_entropy_bits::Float64 = _weighted_discrete_entropy(joint_symbols, observation_weights)
    retained_information_bits::Float64 = clamp(
        raw_entropy_bits + grid_entropy_bits - joint_entropy_bits,
        0.0,
        min(raw_entropy_bits, grid_entropy_bits),
    )
    lost_information_bits::Float64 = max(0.0, joint_entropy_bits - grid_entropy_bits)
    mapping_uncertainty_bits::Float64 = max(0.0, joint_entropy_bits - raw_entropy_bits)
    retention_fraction::Float64 = raw_entropy_bits > 0.0 ?
        clamp(retained_information_bits / raw_entropy_bits, 0.0, 1.0) : NaN
    loss_fraction::Float64 = raw_entropy_bits > 0.0 ?
        clamp(lost_information_bits / raw_entropy_bits, 0.0, 1.0) : NaN
    mapping_conflicts::NamedTuple = _raw_mapping_conflict_information(
        raw_state_symbols,
        grid_state_symbols,
        observation_weights,
    )
    raw_diagnostics::NamedTuple = _raw_state_diagnostics(raw_state_symbols)
    grid_diagnostics::NamedTuple = _raw_state_diagnostics(grid_state_symbols)
    occupied_grid_state_count::Int = grid_diagnostics.state_count
    grid_cell_count_supplied::Bool = grid_cell_count !== nothing
    possible_grid_cell_count::Int = grid_cell_count === nothing ?
        occupied_grid_state_count : Int(grid_cell_count)
    possible_grid_cell_count >= occupied_grid_state_count ||
        throw(ArgumentError("grid_cell_count cannot be smaller than the number of occupied grid states!"))

    record_symbols::Vector{Int} = collect(1:length(comparison_positions))
    record_entropy_bits::Float64 = _weighted_discrete_entropy(record_symbols, observation_weights)
    record_lost_information_bits::Float64 = max(0.0, record_entropy_bits - grid_entropy_bits)
    record_retention_fraction::Float64 = record_entropy_bits > 0.0 ?
        clamp(grid_entropy_bits / record_entropy_bits, 0.0, 1.0) : NaN
    record_loss_fraction::Float64 = record_entropy_bits > 0.0 ?
        clamp(record_lost_information_bits / record_entropy_bits, 0.0, 1.0) : NaN
    raw_collision_probability::Float64 = _raw_collision_probability(raw_state_symbols, observation_weights)
    grid_collision_probability::Float64 = _raw_collision_probability(grid_state_symbols, observation_weights)
    collision_amplification::Float64 = raw_collision_probability > 0.0 ?
        grid_collision_probability / raw_collision_probability : Inf
    comparison_count::Int = length(comparison_positions)
    return (
        estimator=:empirical_plugin,
        interpretation=:observed_state_distinguishability,
        observation_count=comparison_count,
        scaled_total_weight=sum(observation_weights),
        weight_scale=Float64(raw_information.weight_scale),
        weighted=Bool(raw_information.weighted),
        cohort_fingerprint=cohort_fingerprint,
        raw_entropy_bits=raw_entropy_bits,
        grid_entropy_bits=grid_entropy_bits,
        joint_entropy_bits=joint_entropy_bits,
        retained_information_bits=retained_information_bits,
        lost_information_bits=lost_information_bits,
        retention_fraction=retention_fraction,
        loss_fraction=loss_fraction,
        retained_distinguishability_bits=retained_information_bits,
        merging_loss_bits=lost_information_bits,
        distinguishability_retention_fraction=retention_fraction,
        merging_loss_fraction=loss_fraction,
        effective_raw_states=exp2(raw_entropy_bits),
        effective_grid_states=exp2(grid_entropy_bits),
        effective_ambiguity=exp2(lost_information_bits),
        effective_raw_states_per_grid_label=exp2(lost_information_bits),
        mapping_uncertainty_bits=mapping_uncertainty_bits,
        mapping_is_deterministic=mapping_conflicts.mapping_is_deterministic,
        mapping_determinism_scope=mapping_conflicts.mapping_determinism_scope,
        conflicting_raw_state_count=mapping_conflicts.conflicting_raw_state_count,
        mapping_conflict_observation_count=mapping_conflicts.conflict_observation_count,
        mapping_conflict_weight_fraction=mapping_conflicts.conflict_weight_fraction,
        raw_state_count=raw_diagnostics.state_count,
        occupied_grid_state_count=occupied_grid_state_count,
        grid_cell_count=possible_grid_cell_count,
        grid_cell_count_supplied=grid_cell_count_supplied,
        grid_occupancy=occupied_grid_state_count / possible_grid_cell_count,
        raw_singleton_fraction=raw_diagnostics.singleton_fraction,
        grid_singleton_fraction=grid_diagnostics.singleton_fraction,
        collision_amplification=collision_amplification,
        record_entropy_bits=record_entropy_bits,
        record_retained_information_bits=grid_entropy_bits,
        record_lost_information_bits=record_lost_information_bits,
        record_retention_fraction=record_retention_fraction,
        record_loss_fraction=record_loss_fraction,
        record_retained_distinguishability_bits=grid_entropy_bits,
        record_merging_loss_bits=record_lost_information_bits,
        record_distinguishability_retention_fraction=record_retention_fraction,
        record_merging_loss_fraction=record_loss_fraction,
        effective_record_count=exp2(record_entropy_bits),
        effective_record_ambiguity=exp2(record_lost_information_bits),
        record_merge_fraction=max(0.0, 1.0 - occupied_grid_state_count / comparison_count),
    )
end

function compare_rawdata_grid(
    raw_information::NamedTuple,
    grid_assignments::AbstractVector;
    valid_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
    grid_cell_count::Union{Nothing,Integer}=nothing,
)::NamedTuple
    return compare_rawdata_grid(
        raw_information,
        (grid=grid_assignments,);
        valid_mask=valid_mask,
        grid_cell_count=grid_cell_count,
    )
end

function _rawdata_grid_plot_values(
    comparisons::AbstractVector{<:NamedTuple},
    steps::AbstractVector;
    xaxis::Symbol,
    normalize::Symbol,
    baseline::Symbol,
)::NamedTuple
    isempty(comparisons) && throw(ArgumentError("Raw-data grid comparisons must not be empty!"))
    length(comparisons) == length(steps) ||
        throw(DimensionMismatch("Raw-data grid comparisons and resolution labels must have the same length!"))
    xaxis in (:steps, :cells) || throw(ArgumentError("The xaxis option must be :steps or :cells!"))
    normalize in (:fraction, :bits) ||
        throw(ArgumentError("Raw-data grid normalization must be :fraction or :bits!"))
    baseline in (:states, :records) ||
        throw(ArgumentError("Raw-data grid baseline must be :states or :records!"))
    required_keys::Vector{Symbol} = [
        :observation_count,
        :scaled_total_weight,
        :weighted,
        :cohort_fingerprint,
        :raw_entropy_bits,
        :retained_information_bits,
        :lost_information_bits,
        :retention_fraction,
        :loss_fraction,
        :effective_ambiguity,
        :record_entropy_bits,
        :record_retained_information_bits,
        :record_lost_information_bits,
        :record_retention_fraction,
        :record_loss_fraction,
        :effective_record_ambiguity,
        :mapping_is_deterministic,
        :grid_cell_count,
        :grid_cell_count_supplied,
    ]
    all(all(haskey(comparison, key) for key::Symbol in required_keys) for comparison::NamedTuple in comparisons) ||
        throw(ArgumentError("Each entry must be produced by compare_rawdata_grid!"))
    reference_comparison::NamedTuple = first(comparisons)
    reference_observation_count::Int = Int(reference_comparison.observation_count)
    reference_total_weight::Float64 = Float64(reference_comparison.scaled_total_weight)
    reference_cohort_fingerprint::String = String(reference_comparison.cohort_fingerprint)
    all(Int(comparison.observation_count) == reference_observation_count for comparison::NamedTuple in comparisons) ||
        throw(ArgumentError("All plotted grid comparisons must use the same observations!"))
    all(
        String(comparison.cohort_fingerprint) == reference_cohort_fingerprint
        for comparison::NamedTuple in comparisons
    ) || throw(ArgumentError("All plotted grid comparisons must use the same raw-data cohort!"))
    all(
        isapprox(Float64(comparison.scaled_total_weight), reference_total_weight; atol=1.0e-12, rtol=1.0e-12)
        for comparison::NamedTuple in comparisons
    ) || throw(ArgumentError("All plotted grid comparisons must use the same total observation weight!"))
    reference_raw_entropy::Float64 = baseline == :states ?
        Float64(reference_comparison.raw_entropy_bits) : Float64(reference_comparison.record_entropy_bits)
    reference_raw_entropy > 0.0 || throw(ArgumentError(
        "The selected raw baseline has zero entropy, so a retained fraction is undefined; use a finer raw precision or a nondegenerate record baseline!",
    ))
    all(
        isapprox(
            baseline == :states ? Float64(comparison.raw_entropy_bits) : Float64(comparison.record_entropy_bits),
            reference_raw_entropy;
            atol=1.0e-12,
            rtol=1.0e-12,
        ) for comparison::NamedTuple in comparisons
    ) || throw(ArgumentError("All plotted grid comparisons must use the same raw-information baseline!"))
    cell_counts_are_possible::Bool =
        all(Bool(comparison.grid_cell_count_supplied) for comparison::NamedTuple in comparisons)
    order::Vector{Int} = xaxis == :steps ?
        collect(eachindex(comparisons)) :
        sortperm([Int(comparison.grid_cell_count) for comparison::NamedTuple in comparisons])
    step_labels::Vector{String} = string.(steps)
    tick_labels::Vector{String} = String[]
    retained_values::Vector{Float64} = Float64[]
    lost_values::Vector{Float64} = Float64[]
    raw_values::Vector{Float64} = Float64[]
    retention_fractions::Vector{Float64} = Float64[]
    lost_bits::Vector{Float64} = Float64[]
    ambiguities::Vector{Float64} = Float64[]
    for comparison_index::Int in order
        comparison::NamedTuple = comparisons[comparison_index]
        cell_count::Int = Int(comparison.grid_cell_count)
        cell_label::String = Bool(comparison.grid_cell_count_supplied) ? "cells" : "occupied states"
        tick_label::String = xaxis == :steps ?
            "$(step_labels[comparison_index])\n$(cell_count) $(cell_label)" :
            "$(cell_count)\nscale=$(step_labels[comparison_index])"
        push!(tick_labels, tick_label)
        if baseline == :states
            push!(raw_values, Float64(comparison.raw_entropy_bits))
            push!(retention_fractions, Float64(comparison.retention_fraction))
            push!(lost_bits, Float64(comparison.lost_information_bits))
            push!(ambiguities, Float64(comparison.effective_ambiguity))
            if normalize == :fraction
                push!(retained_values, Float64(comparison.retention_fraction))
                push!(lost_values, Float64(comparison.loss_fraction))
            else
                push!(retained_values, Float64(comparison.retained_information_bits))
                push!(lost_values, Float64(comparison.lost_information_bits))
            end
        else
            push!(raw_values, Float64(comparison.record_entropy_bits))
            push!(retention_fractions, Float64(comparison.record_retention_fraction))
            push!(lost_bits, Float64(comparison.record_lost_information_bits))
            push!(ambiguities, Float64(comparison.effective_record_ambiguity))
            if normalize == :fraction
                push!(retained_values, Float64(comparison.record_retention_fraction))
                push!(lost_values, Float64(comparison.record_loss_fraction))
            else
                push!(retained_values, Float64(comparison.record_retained_information_bits))
                push!(lost_values, Float64(comparison.record_lost_information_bits))
            end
        end
    end
    bar_x_values::Vector{String} = String[]
    bar_y_values::Vector{Float64} = Float64[]
    bar_labels::Vector{String} = String[]
    annotation_labels::Vector{String} = String[]
    for position::Int in eachindex(tick_labels)
        push!(bar_x_values, tick_labels[position], tick_labels[position])
        push!(bar_y_values, retained_values[position], lost_values[position])
        push!(bar_labels, "Retained", "Lost")
        ambiguity_label::String = ambiguities[position] < 1000.0 ?
            string(round(ambiguities[position]; digits=2)) :
            string(round(ambiguities[position]; sigdigits=3))
        mapping_warning::String = Bool(comparisons[order[position]].mapping_is_deterministic) ?
            "" : "\nobserved-state mapping conflict"
        push!(annotation_labels,
            "$(round(100.0 * retention_fractions[position]; digits=1))% retained\n" *
            "$(round(lost_bits[position]; digits=2)) bits merging loss\n" *
            "$(ambiguity_label)x ambiguity$(mapping_warning)")
    end
    plotted_raw_values::Vector{Float64} = normalize == :fraction ?
        ones(Float64, length(raw_values)) : raw_values
    return (
        tick_labels=tick_labels,
        bar_x_values=bar_x_values,
        bar_y_values=bar_y_values,
        bar_labels=bar_labels,
        raw_values=plotted_raw_values,
        annotation_labels=annotation_labels,
        annotation_values=plotted_raw_values,
        cell_count_description=cell_counts_are_possible ? "possible grid cells" : "occupied grid states",
        weighted=Bool(reference_comparison.weighted),
    )
end

"""
plot_rawdata_grid_information(comparisons, steps, filename="";
    xaxis=:steps, normalize=:fraction, baseline=:states, title_extra="")

Plot an intrinsic stacked decomposition of empirical raw-state distinguishability
into a part retained by grid labels and a merging loss. Fraction plots always use
the selected empirical raw baseline as 100%; they are never min-max normalized.
"""
function plot_rawdata_grid_information(
    comparisons::AbstractVector{<:NamedTuple},
    steps::AbstractVector,
    filename::AbstractString="";
    xaxis::Symbol=:steps,
    normalize::Symbol=:fraction,
    baseline::Symbol=:states,
    title_extra::AbstractString="",
)::Gadfly.Plot
    values::NamedTuple = _rawdata_grid_plot_values(
        comparisons,
        steps;
        xaxis=xaxis,
        normalize=normalize,
        baseline=baseline,
    )
    maximum_raw_value::Float64 = maximum(values.raw_values)
    y_maximum::Float64 = maximum_raw_value > 0.0 ? 1.26 * maximum_raw_value : 1.0
    annotation_y_values::Vector{Float64} = [
        min(1.10 * raw_value, 0.94 * y_maximum) for raw_value::Float64 in values.annotation_values
    ]
    baseline_description::String = baseline == :states ? "raw states" : "raw records"
    draw_description::String = values.weighted ? "weighted draw" : "observation"
    normalization_description::String = normalize == :fraction ?
        "fraction of observed $(baseline_description) distinguishability" :
        "distinguishability bits per $(draw_description)"
    title::String =
        "Observed raw-state distinguishability after grid merging: $(baseline_description)$(title_extra)"
    raw_grid_plot::Gadfly.Plot = Gadfly.plot(
        Gadfly.layer(
            Gadfly.Geom.label(; position=:dynamic, hide_overlaps=false);
            x=values.tick_labels,
            y=annotation_y_values,
            label=values.annotation_labels,
        ),
        Gadfly.layer(
            Gadfly.Geom.line,
            Gadfly.Theme(; default_color="black", line_width=2Gadfly.pt);
            x=values.tick_labels,
            y=values.raw_values,
        ),
        Gadfly.layer(
            Gadfly.Geom.bar(; position=:stack);
            x=values.bar_x_values,
            y=values.bar_y_values,
            color=values.bar_labels,
        ),
        Gadfly.Scale.x_discrete,
        Gadfly.Scale.color_discrete_manual("#16a085", "#e74c3c"),
        Gadfly.Guide.colorkey(; title="", labels=["Retained", "Lost"]),
        Gadfly.Coord.cartesian(; ymin=0.0, ymax=y_maximum),
        Gadfly.Guide.xlabel(
            xaxis == :steps ?
            "Grid scale ($(values.cell_count_description) shown below)" :
            "$(values.cell_count_description) (scale shown below)",
        ),
        Gadfly.Guide.ylabel(normalization_description),
        Gadfly.Guide.title(title),
        Gadfly.Theme(; bar_spacing=1Gadfly.mm, key_position=:right),
    )
    if filename != ""
        @info("Saving raw-data grid-information plot to file: $filename")
        Mads.plotfileformat(raw_grid_plot, filename, 10Gadfly.inch, 5Gadfly.inch)
    end
    return raw_grid_plot
end

"""
plot_rawdata_grid_heatmap(comparisons, x_steps, y_steps, filename="";
    baseline=:states, quantity=:retained, x_label="Grid resolution 1",
    y_label="Grid resolution 2", title_extra="")

Plot the fraction of empirical raw-state distinguishability retained or merged
across a two-dimensional resolution sweep. Matrix rows correspond to `y_steps`
and columns correspond to `x_steps`. `quantity=:lost` uses green for low merging
loss and red for high merging loss.
"""
function plot_rawdata_grid_heatmap(
    comparisons::AbstractMatrix{<:NamedTuple},
    x_steps::AbstractVector,
    y_steps::AbstractVector,
    filename::AbstractString="";
    baseline::Symbol=:states,
    quantity::Symbol=:retained,
    x_label::AbstractString="Grid resolution 1",
    y_label::AbstractString="Grid resolution 2",
    title_extra::AbstractString="",
)::Gadfly.Plot
    size(comparisons, 2) == length(x_steps) ||
        throw(DimensionMismatch("Heatmap columns must match the x-axis resolution labels!"))
    size(comparisons, 1) == length(y_steps) ||
        throw(DimensionMismatch("Heatmap rows must match the y-axis resolution labels!"))
    isempty(comparisons) && throw(ArgumentError("Raw-data grid heatmap comparisons must not be empty!"))
    quantity in (:retained, :lost) ||
        throw(ArgumentError("Raw-data heatmap quantity must be :retained or :lost!"))
    flattened_comparisons::Vector{NamedTuple} =
        NamedTuple[comparison for comparison::NamedTuple in vec(comparisons)]
    validation_steps::Vector{Int} = collect(eachindex(flattened_comparisons))
    _rawdata_grid_plot_values(
        flattened_comparisons,
        validation_steps;
        xaxis=:steps,
        normalize=:fraction,
        baseline=baseline,
    )
    x_labels::Vector{String} = string.(x_steps)
    y_labels::Vector{String} = string.(y_steps)
    length(unique(x_labels)) == length(x_labels) ||
        throw(ArgumentError("Heatmap x-axis resolution labels must be unique after conversion to text!"))
    length(unique(y_labels)) == length(y_labels) ||
        throw(ArgumentError("Heatmap y-axis resolution labels must be unique after conversion to text!"))
    x_values::Vector{String} = String[]
    y_values::Vector{String} = String[]
    fraction_values::Vector{Float64} = Float64[]
    fraction_labels::Vector{String} = String[]
    mapping_warning_present::Bool = false
    for y_index::Int in eachindex(y_steps)
        for x_index::Int in eachindex(x_steps)
            comparison::NamedTuple = comparisons[y_index, x_index]
            retained_fraction::Float64 = baseline == :states ?
                Float64(comparison.retention_fraction) :
                Float64(comparison.record_retention_fraction)
            lost_fraction::Float64 = baseline == :states ?
                Float64(comparison.loss_fraction) : Float64(comparison.record_loss_fraction)
            plotted_fraction::Float64 = quantity == :retained ? retained_fraction : lost_fraction
            mapping_is_deterministic::Bool = Bool(comparison.mapping_is_deterministic)
            mapping_warning_present = mapping_warning_present || !mapping_is_deterministic
            mapping_suffix::String = mapping_is_deterministic ? "" : "*"
            push!(x_values, x_labels[x_index])
            push!(y_values, y_labels[y_index])
            push!(fraction_values, plotted_fraction)
            push!(fraction_labels, "$(round(100.0 * plotted_fraction; digits=1))%$(mapping_suffix)")
        end
    end
    baseline_description::String = baseline == :states ? "raw states" : "raw records"
    quantity_description::String = quantity == :retained ?
        "retained by grid labels" : "lost by grid merging"
    color_key_title::String = quantity == :retained ?
        "Retained fraction" : "Merging-loss fraction"
    color_map::Function = quantity == :retained ?
        Gadfly.Scale.lab_gradient("#e74c3c", "#f1c40f", "#16a085") :
        Gadfly.Scale.lab_gradient("#16a085", "#f1c40f", "#e74c3c")
    mapping_warning::String = mapping_warning_present ?
        " (* observed-state mapping conflict)" : ""
    title::String =
        "Observed raw-state distinguishability $(quantity_description): $(baseline_description)$(mapping_warning)$(title_extra)"
    rawdata_heatmap::Gadfly.Plot = Gadfly.plot(
        Gadfly.layer(
            Gadfly.Geom.label(; position=:centered, hide_overlaps=false),
            Gadfly.Theme(; point_label_color="#1b1b1b");
            x=x_values,
            y=y_values,
            label=fraction_labels,
        ),
        Gadfly.layer(
            Gadfly.Geom.rectbin;
            x=x_values,
            y=y_values,
            color=fraction_values,
        ),
        Gadfly.Scale.x_discrete,
        Gadfly.Scale.y_discrete,
        Gadfly.Scale.color_continuous(
            ;
            minvalue=0.0,
            maxvalue=1.0,
            colormap=color_map,
        ),
        Gadfly.Guide.colorkey(; title=color_key_title),
        Gadfly.Guide.xlabel(x_label),
        Gadfly.Guide.ylabel(y_label),
        Gadfly.Guide.title(title),
        Gadfly.Theme(; key_position=:right),
    )
    if filename != ""
        @info("Saving raw-data grid-information heatmap to file: $filename")
        Mads.plotfileformat(rawdata_heatmap, filename, 10Gadfly.inch, 6Gadfly.inch)
    end
    return rawdata_heatmap
end

function _range_normalize_plot_values!(values::Vector{Float64}, labels::Vector{String}, metric_labels::Vector{String})::Nothing
    for metric_label::String in metric_labels
        metric_indices::Vector{Int} = findall(==(metric_label), labels)
        metric_values::Vector{Float64} = values[metric_indices]
        minimum_value::Float64 = minimum(metric_values)
        maximum_value::Float64 = maximum(metric_values)
        if maximum_value > minimum_value
            values[metric_indices] .= (metric_values .- minimum_value) ./ (maximum_value - minimum_value)
        else
            values[metric_indices] .= 0.5
        end
    end
    return nothing
end

function _temporal_dependence(information::NamedTuple)::Float64
    temporal_metrics::Vector{NamedTuple} = filter(metric::NamedTuple -> metric.role == :temporal && metric.pair_count > 0, information.axis_information)
    return isempty(temporal_metrics) ? 0.0 : Statistics.mean(metric.normalized_mutual_information for metric::NamedTuple in temporal_metrics)
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
plot_structure_information(information_steps, steps, filename=""; xaxis=:steps, normalize=:intrinsic, title_extra="")

Compare normalized structural-information metrics across tensor discretizations.
All displayed metrics use a higher-is-more-informative-or-structured orientation.
"""
function plot_structure_information(information_steps::AbstractVector{<:NamedTuple}, steps::AbstractVector, filename::AbstractString=""; xaxis::Symbol=:steps, normalize::Symbol=:intrinsic, title_extra::AbstractString="")::Gadfly.Plot
    if isempty(information_steps)
        throw(ArgumentError("Information steps must not be empty!"))
    end
    if length(information_steps) != length(steps)
        throw(DimensionMismatch("Information steps and resolution labels must have the same length!"))
    end
    if !(xaxis in (:steps, :cells))
        throw(ArgumentError("The xaxis option must be :steps or :cells!"))
    end
    if !(normalize in (:intrinsic, :range))
        throw(ArgumentError("The normalize option must be :intrinsic or :range!"))
    end
    include_spectral::Bool = all(!isempty(information.spectral_information) for information::NamedTuple in information_steps)
    metric_labels::Vector{String} = [
        "Value entropy",
        "Spatial dependence",
        "Spatial coherence",
        "Temporal dependence",
        "Temporal coherence",
        "Residual coding savings"
    ]
    metric_colors::Vector{String} = ["#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b", "#d62728", "#17becf"]
    if include_spectral
        insert!(metric_labels, 6, "Spectral compactness")
        insert!(metric_colors, 6, "#9467bd")
    end
    step_labels::Vector{String} = string.(steps)
    x_step_values::Vector{String} = String[]
    x_cell_values::Vector{Float64} = Float64[]
    y_values::Vector{Float64} = Float64[]
    labels::Vector{String} = String[]
    step_cell_counts::Vector{Float64} = Float64[]
    for (step_index::Int, information::NamedTuple) in enumerate(information_steps)
        coding_summary::NamedTuple = _temporal_coding_summary(information)
        cell_count::Float64 = Float64(information.valid_cell_count)
        cell_count > 0.0 || throw(ArgumentError("Valid cell counts must be positive!"))
        push!(step_cell_counts, cell_count)
        metric_values::Vector{Float64} = [
            clamp(Float64(information.normalized_value_entropy), 0.0, 1.0),
            clamp(Float64(information.spatial_dependence), 0.0, 1.0),
            clamp(1.0 - Float64(information.spatial_variation), 0.0, 1.0),
            clamp(_temporal_dependence(information), 0.0, 1.0),
            clamp(1.0 - Float64(information.temporal_variation), 0.0, 1.0),
            coding_summary.coding_savings
        ]
        if include_spectral
            insert!(metric_values, 6, _spectral_compactness(information))
        end
        for metric_index::Int in eachindex(metric_labels)
            push!(x_step_values, step_labels[step_index])
            push!(x_cell_values, cell_count)
            push!(y_values, metric_values[metric_index])
            push!(labels, metric_labels[metric_index])
        end
    end
    if normalize == :range
        _range_normalize_plot_values!(y_values, labels, metric_labels)
    end
    metric_count::Int = length(metric_labels)
    label_y_values::Vector{Float64} = Float64[]
    for step_index::Int in eachindex(information_steps)
        first_metric_index::Int = (step_index - 1) * metric_count + 1
        last_metric_index::Int = first_metric_index + metric_count - 1
        maximum_step_value::Float64 = maximum(y_values[first_metric_index:last_metric_index])
        push!(label_y_values, min(maximum_step_value + 0.04, 0.98))
    end
    cell_count_labels::Vector{String} = ["cells=$(Int(round(cell_count)))" for cell_count::Float64 in step_cell_counts]
    scale_labels::Vector{String} = ["scale=$(step_label)" for step_label::String in step_labels]
    title::String = "Structure-aware information by binning resolution$(normalize == :range ? " (range normalized)" : "")$(title_extra)"
    y_label::String = normalize == :range ? "Within-metric normalized range" : "Intrinsic normalized metric"
    structure_plot::Gadfly.Plot = if xaxis == :steps
        Gadfly.plot(
            Gadfly.layer(Gadfly.Geom.line, Gadfly.Geom.point; x=x_step_values, y=y_values, color=labels),
            Gadfly.layer(Gadfly.Geom.label(; position=:dynamic, hide_overlaps=false); x=step_labels, y=label_y_values, label=cell_count_labels),
            Gadfly.Scale.x_discrete,
            Gadfly.Scale.color_discrete_manual(metric_colors...),
            Gadfly.Coord.cartesian(; ymin=0.0, ymax=1.0),
            Gadfly.Guide.xlabel("Resolution size"),
            Gadfly.Guide.ylabel(y_label),
            Gadfly.Guide.title(title)
        )
    else
        Gadfly.plot(
            Gadfly.layer(Gadfly.Geom.line, Gadfly.Geom.point; x=x_cell_values, y=y_values, color=labels),
            Gadfly.layer(Gadfly.Geom.label(; position=:dynamic, hide_overlaps=false); x=step_cell_counts, y=label_y_values, label=scale_labels),
            Gadfly.Scale.x_log10(),
            Gadfly.Scale.color_discrete_manual(metric_colors...),
            Gadfly.Coord.cartesian(; ymin=0.0, ymax=1.0),
            Gadfly.Guide.xlabel("Number of valid cells (log scale)"),
            Gadfly.Guide.ylabel(y_label),
            Gadfly.Guide.title(title)
        )
    end
    if filename != ""
        Mads.plotfileformat(structure_plot, filename, 9Gadfly.inch, 5Gadfly.inch)
    end
    return structure_plot
end

"""
plot_residual_coding(information_steps, steps, filename=""; xaxis=:steps, normalize=:none, title_extra="")

Compare fixed-width residual storage, the Shannon limit, and the selected residual
coding method across temporal discretizations. Values are bits per prediction residual.
"""
function plot_residual_coding(information_steps::AbstractVector{<:NamedTuple}, steps::AbstractVector, filename::AbstractString=""; xaxis::Symbol=:steps, normalize::Symbol=:none, title_extra::AbstractString="")::Gadfly.Plot
    if isempty(information_steps)
        throw(ArgumentError("Information steps must not be empty!"))
    end
    if length(information_steps) != length(steps)
        throw(DimensionMismatch("Information steps and resolution labels must have the same length!"))
    end
    if !(xaxis in (:steps, :cells))
        throw(ArgumentError("The xaxis option must be :steps or :cells!"))
    end
    if !(normalize in (:none, :fixed, :range))
        throw(ArgumentError("The normalize option must be :none, :fixed, or :range!"))
    end
    coding_method::String = string(information_steps[1].residual_coding)
    metric_labels::Vector{String} = ["Fixed width", "Shannon limit", "Selected coding ($(coding_method))"]
    metric_colors::Vector{String} = ["#7f7f7f", "#1f77b4", "#d62728"]
    step_labels::Vector{String} = string.(steps)
    x_step_values::Vector{String} = String[]
    x_cell_values::Vector{Float64} = Float64[]
    y_values::Vector{Float64} = Float64[]
    labels::Vector{String} = String[]
    step_cell_counts::Vector{Float64} = Float64[]
    for (step_index::Int, information::NamedTuple) in enumerate(information_steps)
        coding_summary::NamedTuple = _temporal_coding_summary(information)
        cell_count::Float64 = Float64(information.valid_cell_count)
        push!(step_cell_counts, cell_count)
        metric_values::Vector{Float64} = [coding_summary.fixed_bits_per_symbol, coding_summary.shannon_bits_per_symbol, coding_summary.encoded_bits_per_symbol]
        if normalize == :fixed && coding_summary.fixed_bits_per_symbol > 0.0
            metric_values ./= coding_summary.fixed_bits_per_symbol
        end
        for metric_index::Int in eachindex(metric_labels)
            push!(x_step_values, step_labels[step_index])
            push!(x_cell_values, cell_count)
            push!(y_values, metric_values[metric_index])
            push!(labels, metric_labels[metric_index])
        end
    end
    if normalize == :range
        _range_normalize_plot_values!(y_values, labels, metric_labels)
    end
    metric_count::Int = length(metric_labels)
    label_y_values::Vector{Float64} = Float64[]
    for step_index::Int in eachindex(information_steps)
        first_metric_index::Int = (step_index - 1) * metric_count + 1
        last_metric_index::Int = first_metric_index + metric_count - 1
        maximum_step_value::Float64 = maximum(y_values[first_metric_index:last_metric_index])
        label_offset::Float64 = normalize == :none ? max(0.04 * maximum_step_value, 0.05) : 0.04
        push!(label_y_values, maximum_step_value + label_offset)
    end
    cell_count_labels::Vector{String} = ["cells=$(Int(round(cell_count)))" for cell_count::Float64 in step_cell_counts]
    scale_labels::Vector{String} = ["scale=$(step_label)" for step_label::String in step_labels]
    normalization_text::String = normalize == :none ? "" : normalize == :fixed ? " (fraction of fixed-width cost)" : " (range normalized)"
    title::String = "Temporal prediction-residual coding by binning resolution$(normalization_text)$(title_extra)"
    y_label::String = normalize == :none ? "Bits per temporal residual" : normalize == :fixed ? "Fraction of fixed-width bits" : "Within-series normalized range"
    coding_plot::Gadfly.Plot = if xaxis == :steps
        Gadfly.plot(
            Gadfly.layer(Gadfly.Geom.line, Gadfly.Geom.point; x=x_step_values, y=y_values, color=labels),
            Gadfly.layer(Gadfly.Geom.label(; position=:dynamic, hide_overlaps=false); x=step_labels, y=label_y_values, label=cell_count_labels),
            Gadfly.Scale.x_discrete,
            Gadfly.Scale.color_discrete_manual(metric_colors...),
            Gadfly.Guide.xlabel("Resolution size"),
            Gadfly.Guide.ylabel(y_label),
            Gadfly.Guide.title(title)
        )
    else
        Gadfly.plot(
            Gadfly.layer(Gadfly.Geom.line, Gadfly.Geom.point; x=x_cell_values, y=y_values, color=labels),
            Gadfly.layer(Gadfly.Geom.label(; position=:dynamic, hide_overlaps=false); x=step_cell_counts, y=label_y_values, label=scale_labels),
            Gadfly.Scale.x_log10(),
            Gadfly.Scale.color_discrete_manual(metric_colors...),
            Gadfly.Guide.xlabel("Number of valid cells (log scale)"),
            Gadfly.Guide.ylabel(y_label),
            Gadfly.Guide.title(title)
        )
    end
    if filename != ""
        Mads.plotfileformat(coding_plot, filename, 9Gadfly.inch, 5Gadfly.inch)
    end
    return coding_plot
end
