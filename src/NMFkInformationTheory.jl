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

# Increment a histogram entry while keeping zero-count entries out of the
# dictionary. Negative increments are used by the sparse baseline-pair path.
function _increment_information_count!(counts::Dict{K,Int}, symbol::K, increment::Int=1)::Nothing where {K}
    updated_count::Int = get(counts, symbol, 0) + increment
    updated_count >= 0 || throw(ArgumentError("Information histogram counts cannot be negative!"))
    if updated_count == 0
        delete!(counts, symbol)
    else
        counts[symbol] = updated_count
    end
    return nothing
end

# Compute empirical entropy directly from a histogram. This avoids materializing
# one symbol vector per lag, which is prohibitive for large tensors.
function _discrete_entropy_from_counts(counts::AbstractDict, observation_count::Int)::Float64
    observation_count == 0 && return 0.0
    entropy_bits::Float64 = 0.0
    counted_observations::Int = 0
    for count_value_any::Any in values(counts)
        count_value::Int = Int(count_value_any)
        count_value > 0 || continue
        probability::Float64 = count_value / observation_count
        entropy_bits -= probability * log2(probability)
        counted_observations = Base.checked_add(counted_observations, count_value)
    end
    counted_observations == observation_count || throw(ArgumentError("Information histogram counts do not match the observation count!"))
    return entropy_bits
end

# Compute the encoded size of a Huffman code directly from symbol counts.
function _huffman_encoded_bits_from_counts(counts::AbstractDict)::Int
    weights::Vector{Int} = Int[]
    for count_value_any::Any in values(counts)
        count_value::Int = Int(count_value_any)
        count_value > 0 && push!(weights, count_value)
    end
    isempty(weights) && return 0
    length(weights) == 1 && return 0
    encoded_bits::Int = 0
    while length(weights) > 1
        sort!(weights; rev=true)
        first_weight::Int = pop!(weights)
        second_weight::Int = pop!(weights)
        combined_weight::Int = Base.checked_add(first_weight, second_weight)
        encoded_bits = Base.checked_add(encoded_bits, combined_weight)
        push!(weights, combined_weight)
    end
    return encoded_bits
end

# Compute the number of bits required to encode a vector of integer symbols using Huffman coding.
function _huffman_encoded_bits(symbols::AbstractVector{<:Integer})::Int
    isempty(symbols) && return 0
    counts::Dict{Int, Int} = Dict{Int, Int}()
    for symbol::Integer in symbols
        symbol_value::Int = Int(symbol)
        counts[symbol_value] = get(counts, symbol_value, 0) + 1
    end
    return _huffman_encoded_bits_from_counts(counts)
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

function _raw_scaled_coordinate_tolerance(
    value::Float64,
    precision::Float64,
    origin::Float64,
    scaled_value::Float64,
)::Float64
    absolute_precision::Float64 = abs(precision)
    input_resolution_component::Float64 =
        (0.5 * eps(abs(value)) + 0.5 * eps(abs(origin))) / absolute_precision
    precision_resolution_component::Float64 =
        abs(scaled_value) * 0.5 * eps(absolute_precision) / absolute_precision
    division_resolution_component::Float64 = 0.5 * eps(abs(scaled_value))
    return input_resolution_component +
        precision_resolution_component +
        division_resolution_component
end

function _raw_floor_symbol(value::Float64, precision::Float64, origin::Float64)::Int
    scaled_value::Float64 = (value - origin) / precision
    isfinite(scaled_value) || throw(ArgumentError("Raw-data value cannot be represented at the requested precision!"))
    input_resolution::Float64 = max(eps(abs(value)), eps(abs(origin)), eps(precision))
    if precision <= 8.0 * input_resolution
        throw(ArgumentError("Requested raw-data precision is finer than the numerical resolution of the input value!"))
    end
    nearest_integer::Float64 = round(scaled_value)
    rounding_tolerance::Float64 = _raw_scaled_coordinate_tolerance(
        value,
        precision,
        origin,
        scaled_value,
    )
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
    residual_counts::Dict{Int,Int} = Dict{Int,Int}()
    for residual_symbol::Int in residual_symbols
        _increment_information_count!(residual_counts, residual_symbol)
    end
    return _residual_coding_information(residual_counts, length(residual_symbols), bin_count, residual_coding)
end

# Compute residual-coding measures from a histogram rather than a materialized
# residual stream.
function _residual_coding_information(residual_counts::Dict{Int,Int}, pair_count::Int, bin_count::Int, residual_coding::Symbol)::NamedTuple
    fixed_bits_per_symbol::Int = ceil(Int, log2(Float64(2 * bin_count - 1)))
    fixed_width_bits::Int = Base.checked_mul(pair_count, fixed_bits_per_symbol)
    residual_entropy::Float64 = _discrete_entropy_from_counts(residual_counts, pair_count)
    encoded_bits::Float64 = if residual_coding == :none
        Float64(fixed_width_bits)
    elseif residual_coding == :shannon
        pair_count * residual_entropy
    else
        Float64(_huffman_encoded_bits_from_counts(residual_counts))
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
function _quantize_tensor(
    tensor::AbstractArray{T},
    valid_mask::AbstractArray{Bool},
    bin_count::Int,
    quantization::Symbol=:linear,
)::Tuple{Array{Int}, BitArray} where {T <: Real}
    if size(valid_mask) != size(tensor)
        throw(DimensionMismatch("The valid mask and tensor must have the same size!"))
    end
    if bin_count < 2
        throw(ArgumentError("The number of quantization bins must be at least 2!"))
    end
    quantization in (:linear, :zero_preserving) || throw(ArgumentError(
        "Tensor quantization must be :linear or :zero_preserving!",
    ))
    finite_mask::BitArray = falses(size(tensor))
    quantized::Array{Int} = zeros(Int, size(tensor))
    minimum_value::Float64 = Inf
    maximum_value::Float64 = -Inf
    minimum_nonzero_value::Float64 = Inf
    maximum_nonzero_value::Float64 = -Inf
    valid_cell_count::Int = 0
    valid_nonzero_cell_count::Int = 0
    for index::CartesianIndex in CartesianIndices(tensor)
        value::T = tensor[index]
        cell_is_valid::Bool = valid_mask[index] && isfinite(value)
        finite_mask[index] = cell_is_valid
        cell_is_valid || continue
        value_float::Float64 = Float64(value)
        minimum_value = min(minimum_value, value_float)
        maximum_value = max(maximum_value, value_float)
        valid_cell_count = Base.checked_add(valid_cell_count, 1)
        if !iszero(value_float)
            minimum_nonzero_value = min(minimum_nonzero_value, value_float)
            maximum_nonzero_value = max(maximum_nonzero_value, value_float)
            valid_nonzero_cell_count = Base.checked_add(valid_nonzero_cell_count, 1)
        end
    end
    valid_cell_count == 0 && return quantized, finite_mask
    if quantization == :zero_preserving
        minimum_value >= 0.0 || throw(ArgumentError(
            "Zero-preserving quantization requires nonnegative valid tensor values!",
        ))
        if valid_nonzero_cell_count == 0
            quantized[finite_mask] .= 1
            return quantized, finite_mask
        end
        if maximum_nonzero_value == minimum_nonzero_value
            for index::CartesianIndex in CartesianIndices(tensor)
                finite_mask[index] || continue
                quantized[index] = iszero(Float64(tensor[index])) ? 1 : 2
            end
            return quantized, finite_mask
        end
        nonzero_scale::Float64 =
            (bin_count - 1) / (maximum_nonzero_value - minimum_nonzero_value)
        for index::CartesianIndex in CartesianIndices(tensor)
            finite_mask[index] || continue
            value_float::Float64 = Float64(tensor[index])
            quantized[index] = if iszero(value_float)
                1
            else
                clamp(
                    floor(Int, (value_float - minimum_nonzero_value) * nonzero_scale) + 2,
                    2,
                    bin_count,
                )
            end
        end
        return quantized, finite_mask
    end
    if maximum_value == minimum_value
        quantized[finite_mask] .= 1
        return quantized, finite_mask
    end
    scale::Float64 = bin_count / (maximum_value - minimum_value)
    for index::CartesianIndex in CartesianIndices(tensor)
        finite_mask[index] || continue
        value_float::Float64 = Float64(tensor[index])
        quantized[index] = clamp(floor(Int, (value_float - minimum_value) * scale) + 1, 1, bin_count)
    end
    return quantized, finite_mask
end

function _lag_candidate_ranges(tensor_size::NTuple{N,Int}, offset::NTuple{N,Int})::NTuple{N,UnitRange{Int}} where {N}
    return ntuple(
        dimension::Int -> begin
            dimension_size::Int = tensor_size[dimension]
            if abs(offset[dimension]) >= dimension_size
                return 1:0
            end
            first_index::Int = 1 + max(0, -offset[dimension])
            last_index::Int = dimension_size - max(0, offset[dimension])
            first_index:last_index
        end,
        N,
    )
end

function _lag_candidate_count(tensor_size::NTuple{N,Int}, offset::NTuple{N,Int})::Int where {N}
    candidate_count::Int = 1
    for dimension::Int = 1:N
        dimension_count::Int = max(tensor_size[dimension] - abs(offset[dimension]), 0)
        candidate_count = Base.checked_mul(candidate_count, dimension_count)
    end
    return candidate_count
end

function _lag_origin_in_ranges(index::CartesianIndex{N}, ranges::NTuple{N,UnitRange{Int}})::Bool where {N}
    for dimension::Int = 1:N
        index[dimension] in ranges[dimension] || return false
    end
    return true
end

function _dense_lag_histograms(
    quantized::Array{Int,N},
    valid_mask::BitArray{N},
    offset::NTuple{N,Int},
    ranges::NTuple{N,UnitRange{Int}},
)::NamedTuple where {N}
    left_counts::Dict{Int,Int} = Dict{Int,Int}()
    right_counts::Dict{Int,Int} = Dict{Int,Int}()
    joint_counts::Dict{Tuple{Int,Int},Int} = Dict{Tuple{Int,Int},Int}()
    residual_counts::Dict{Int,Int} = Dict{Int,Int}()
    pair_count::Int = 0
    abs_difference_total::Float64 = 0.0
    lag_index::CartesianIndex{N} = CartesianIndex(offset)
    for index::CartesianIndex{N} in CartesianIndices(ranges)
        neighbor_index::CartesianIndex{N} = index + lag_index
        (valid_mask[index] && valid_mask[neighbor_index]) || continue
        left_symbol::Int = quantized[index]
        right_symbol::Int = quantized[neighbor_index]
        residual_symbol::Int = right_symbol - left_symbol
        _increment_information_count!(left_counts, left_symbol)
        _increment_information_count!(right_counts, right_symbol)
        _increment_information_count!(joint_counts, (left_symbol, right_symbol))
        _increment_information_count!(residual_counts, residual_symbol)
        pair_count = Base.checked_add(pair_count, 1)
        abs_difference_total += abs(residual_symbol)
    end
    return (
        left_counts=left_counts,
        right_counts=right_counts,
        joint_counts=joint_counts,
        residual_counts=residual_counts,
        pair_count=pair_count,
        abs_difference_total=abs_difference_total,
        method=:dense,
    )
end

# When every tensor cell is valid and one symbol dominates, all candidate pairs
# can initially be counted as baseline-baseline. Only origins touching a
# nonbaseline cell need dictionary lookups, making sparse count grids exact in
# O(number of nonbaseline cells) work per lag.
function _sparse_baseline_lag_histograms(
    quantized::Array{Int,N},
    offset::NTuple{N,Int},
    ranges::NTuple{N,UnitRange{Int}},
    candidate_pair_count::Int,
    baseline_symbol::Int,
    nonbaseline_indices::Vector{CartesianIndex{N}},
)::NamedTuple where {N}
    left_counts::Dict{Int,Int} = Dict{Int,Int}()
    right_counts::Dict{Int,Int} = Dict{Int,Int}()
    joint_counts::Dict{Tuple{Int,Int},Int} = Dict{Tuple{Int,Int},Int}()
    residual_counts::Dict{Int,Int} = Dict{Int,Int}()
    if candidate_pair_count > 0
        left_counts[baseline_symbol] = candidate_pair_count
        right_counts[baseline_symbol] = candidate_pair_count
        joint_counts[(baseline_symbol, baseline_symbol)] = candidate_pair_count
        residual_counts[0] = candidate_pair_count
    end
    lag_index::CartesianIndex{N} = CartesianIndex(offset)
    impacted_origins::Set{CartesianIndex{N}} = Set{CartesianIndex{N}}()
    for nonbaseline_index::CartesianIndex{N} in nonbaseline_indices
        if _lag_origin_in_ranges(nonbaseline_index, ranges)
            push!(impacted_origins, nonbaseline_index)
        end
        preceding_index::CartesianIndex{N} = nonbaseline_index - lag_index
        if _lag_origin_in_ranges(preceding_index, ranges)
            push!(impacted_origins, preceding_index)
        end
    end
    abs_difference_total::Float64 = 0.0
    for index::CartesianIndex{N} in impacted_origins
        neighbor_index::CartesianIndex{N} = index + lag_index
        left_symbol::Int = quantized[index]
        right_symbol::Int = quantized[neighbor_index]
        residual_symbol::Int = right_symbol - left_symbol
        _increment_information_count!(left_counts, baseline_symbol, -1)
        _increment_information_count!(right_counts, baseline_symbol, -1)
        _increment_information_count!(joint_counts, (baseline_symbol, baseline_symbol), -1)
        _increment_information_count!(residual_counts, 0, -1)
        _increment_information_count!(left_counts, left_symbol)
        _increment_information_count!(right_counts, right_symbol)
        _increment_information_count!(joint_counts, (left_symbol, right_symbol))
        _increment_information_count!(residual_counts, residual_symbol)
        abs_difference_total += abs(residual_symbol)
    end
    return (
        left_counts=left_counts,
        right_counts=right_counts,
        joint_counts=joint_counts,
        residual_counts=residual_counts,
        pair_count=candidate_pair_count,
        abs_difference_total=abs_difference_total,
        method=:sparse_baseline,
    )
end

function _lag_scaled_step(step::Real, offset::Int)::Real
    return step * offset
end

function _lag_scaled_step(step::Dates.Period, offset::Int)::Dates.Period
    return step * offset
end

function _lag_display_label(
    offset::NTuple{N,Int},
    dimension_names::NTuple{N,Symbol},
    coordinate_offset::Tuple,
    dimension_units::NTuple{N,String},
)::String where {N}
    component_labels::Vector{String} = String[]
    for dimension::Int = 1:N
        offset[dimension] == 0 && continue
        coordinate_value::Any = coordinate_offset[dimension]
        coordinate_label::String = string(coordinate_value)
        unit_label::String = dimension_units[dimension]
        suffix::String = coordinate_value isa Dates.Period || isempty(unit_label) ? "" : " $(unit_label)"
        push!(component_labels, "$(dimension_names[dimension])=$(coordinate_label)$(suffix)")
    end
    return join(component_labels, ", ")
end

# Compute information-theoretic measures for an arbitrary N-dimensional lag.
function _lag_information(
    quantized::Array{Int,N},
    valid_mask::BitArray{N},
    offset::NTuple{N,Int},
    role::Symbol,
    bin_count::Int,
    residual_coding::Symbol,
    dimension_names::NTuple{N,Symbol},
    dimension_steps::Tuple,
    dimension_units::NTuple{N,String},
    dimension_roles::NTuple{N,Symbol},
    baseline_symbol::Union{Nothing,Int},
    nonbaseline_indices::Union{Nothing,Vector{CartesianIndex{N}}},
)::NamedTuple where {N}
    tensor_size::NTuple{N,Int} = size(quantized)
    ranges::NTuple{N,UnitRange{Int}} = _lag_candidate_ranges(tensor_size, offset)
    candidate_pair_count::Int = _lag_candidate_count(tensor_size, offset)
    histograms::NamedTuple = if baseline_symbol !== nothing && nonbaseline_indices !== nothing
        _sparse_baseline_lag_histograms(
            quantized,
            offset,
            ranges,
            candidate_pair_count,
            baseline_symbol,
            nonbaseline_indices,
        )
    else
        _dense_lag_histograms(quantized, valid_mask, offset, ranges)
    end
    pair_count::Int = histograms.pair_count
    left_entropy::Float64 = _discrete_entropy_from_counts(histograms.left_counts, pair_count)
    right_entropy::Float64 = _discrete_entropy_from_counts(histograms.right_counts, pair_count)
    joint_entropy::Float64 = _discrete_entropy_from_counts(histograms.joint_counts, pair_count)
    mutual_information::Float64 = max(0.0, left_entropy + right_entropy - joint_entropy)
    conditional_entropy::Float64 = max(0.0, joint_entropy - left_entropy)
    reverse_conditional_entropy::Float64 = max(0.0, joint_entropy - right_entropy)
    residual_entropy::Float64 = _discrete_entropy_from_counts(histograms.residual_counts, pair_count)
    coding_information::NamedTuple = _residual_coding_information(histograms.residual_counts, pair_count, bin_count, residual_coding)
    normalizer::Float64 = max(left_entropy, right_entropy)
    normalized_mutual_information::Float64 = normalizer > 0.0 ? mutual_information / normalizer : 0.0
    mean_absolute_difference_bins::Float64 = pair_count > 0 ? histograms.abs_difference_total / pair_count : 0.0
    mean_normalized_difference::Float64 = mean_absolute_difference_bins / (bin_count - 1)
    active_dimensions::Tuple{Vararg{Int}} = Tuple(findall(!iszero, offset))
    active_dimension_roles::Tuple = Tuple(dimension_roles[dimension] for dimension::Int in active_dimensions)
    axis::Union{Nothing,Int} = length(active_dimensions) == 1 ? first(active_dimensions) : nothing
    coordinate_offset::Tuple = ntuple(
        dimension::Int -> _lag_scaled_step(dimension_steps[dimension], offset[dimension]),
        N,
    )
    grid_index_norm::Float64 = sqrt(sum(Float64(component)^2 for component::Int in offset))
    display_label::String = _lag_display_label(offset, dimension_names, coordinate_offset, dimension_units)
    valid_pair_fraction::Float64 = candidate_pair_count > 0 ? pair_count / candidate_pair_count : 0.0
    return (
        offset=offset,
        axis=axis,
        role=role,
        active_dimensions=active_dimensions,
        active_dimension_roles=active_dimension_roles,
        pair_scope=:both_endpoints_valid,
        applicable=pair_count > 0,
        candidate_pair_count=candidate_pair_count,
        pair_count=pair_count,
        valid_pair_fraction=valid_pair_fraction,
        left_entropy_bits=left_entropy,
        right_entropy_bits=right_entropy,
        joint_entropy_bits=joint_entropy,
        mutual_information_bits=mutual_information,
        normalized_mutual_information=normalized_mutual_information,
        conditional_entropy_bits=conditional_entropy,
        reverse_conditional_entropy_bits=reverse_conditional_entropy,
        residual_entropy_bits=residual_entropy,
        predictive_gain_bits=right_entropy - residual_entropy,
        reverse_predictive_gain_bits=left_entropy - residual_entropy,
        mean_absolute_difference_bins=mean_absolute_difference_bins,
        mean_normalized_difference=mean_normalized_difference,
        residual_coding=coding_information,
        coordinate_offset=coordinate_offset,
        grid_index_norm=grid_index_norm,
        display_label=display_label,
        histogram_method=histograms.method,
    )
end

# Preserve the original internal unit-axis entry point.
function _axis_information(quantized::Array{Int,N}, valid_mask::BitArray{N}, axis::Int, role::Symbol, bin_count::Int, residual_coding::Symbol)::NamedTuple where {N}
    offset::NTuple{N,Int} = ntuple(dimension::Int -> dimension == axis ? 1 : 0, N)
    dimension_names::NTuple{N,Symbol} = ntuple(dimension::Int -> Symbol("dim$(dimension)"), N)
    dimension_steps::NTuple{N,Int} = ntuple(dimension::Int -> 1, N)
    dimension_units::NTuple{N,String} = ntuple(dimension::Int -> "bins", N)
    dimension_roles::NTuple{N,Symbol} = ntuple(dimension::Int -> role, N)
    return _lag_information(
        quantized,
        valid_mask,
        offset,
        role,
        bin_count,
        residual_coding,
        dimension_names,
        dimension_steps,
        dimension_units,
        dimension_roles,
        nothing,
        nothing,
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

function _lag_role(offset::NTuple{N,Int}, dimension_roles::NTuple{N,Symbol})::Symbol where {N}
    active_roles::Vector{Symbol} = unique(Symbol[
        dimension_roles[dimension]
        for dimension::Int = 1:N
        if offset[dimension] != 0
    ])
    temporal_component_present::Bool = :temporal in active_roles
    if temporal_component_present
        return length(active_roles) == 1 ? :temporal : :spatiotemporal
    end
    depth_component_present::Bool = :depth in active_roles
    horizontal_component_present::Bool = any(
        role::Symbol -> role in (:horizontal, :spatial),
        active_roles,
    )
    if depth_component_present && horizontal_component_present
        return :spatial_depth
    end
    return depth_component_present ? :depth : :spatial
end

function _canonical_lag_offset(offset::NTuple{N,Int}, temporal_dim::Union{Nothing,Int})::NTuple{N,Int} where {N}
    sign_dimension::Int = if temporal_dim !== nothing && offset[temporal_dim] != 0
        temporal_dim
    else
        first(findall(!iszero, offset))
    end
    multiplier::Int = offset[sign_dimension] > 0 ? 1 : -1
    return ntuple(dimension::Int -> Base.checked_mul(multiplier, offset[dimension]), N)
end

function _normalize_lag_offset(offset_value::Any, dimension_count::Int)::Tuple
    raw_offset::Tuple = if offset_value isa CartesianIndex
        Tuple(offset_value)
    elseif offset_value isa Tuple
        offset_value
    else
        throw(ArgumentError("Every lag offset must be a Tuple or CartesianIndex!"))
    end
    length(raw_offset) == dimension_count || throw(DimensionMismatch("Every lag offset must have one component per tensor dimension!"))
    normalized_offset::Tuple = ntuple(
        dimension::Int -> begin
            component::Any = raw_offset[dimension]
            component isa Integer || throw(ArgumentError("Lag-offset components must be integers!"))
            component_big::BigInt = BigInt(component)
            if component_big < typemin(Int) || component_big > typemax(Int)
                throw(ArgumentError("A lag-offset component exceeds the supported Int range!"))
            end
            component_int::Int = Int(component)
            component_int == typemin(Int) && throw(ArgumentError("A lag-offset component is too negative to canonicalize safely!"))
            component_int
        end,
        dimension_count,
    )
    any(!iszero, normalized_offset) || throw(ArgumentError("The all-zero lag offset is not a valid cell pair!"))
    return normalized_offset
end

function _normalize_lag_offsets(
    lag_offsets::Union{Nothing,AbstractVector},
    dimension_count::Int,
    temporal_dim::Union{Nothing,Int},
    lag_sign::Symbol,
)::Vector{<:Tuple}
    lag_sign in (:canonical, :directed) || throw(ArgumentError("The lag_sign option must be :canonical or :directed!"))
    default_offsets::Vector{Tuple} = Tuple[
        ntuple(candidate_dimension::Int -> candidate_dimension == dimension ? 1 : 0, dimension_count)
        for dimension::Int = 1:dimension_count
    ]
    raw_offsets::AbstractVector = lag_offsets === nothing ? default_offsets : lag_offsets
    isempty(raw_offsets) && throw(ArgumentError("Lag offsets must not be empty!"))
    normalized_offsets::Vector{Tuple} = Tuple[]
    observed_offsets::Set{Tuple} = Set{Tuple}()
    for offset_value::Any in raw_offsets
        offset::Tuple = _normalize_lag_offset(offset_value, dimension_count)
        normalized_offset::Tuple = lag_sign == :canonical ?
            _canonical_lag_offset(offset, temporal_dim) : offset
        if normalized_offset in observed_offsets
            duplicate_message::String = lag_sign == :canonical ?
                "Lag offsets contain duplicate directions after canonicalization; remove one direction or use lag_sign=:directed!" :
                "Lag offsets must not contain exact duplicates!"
            throw(ArgumentError(duplicate_message))
        end
        push!(observed_offsets, normalized_offset)
        push!(normalized_offsets, normalized_offset)
    end
    return normalized_offsets
end

function _normalize_dimension_metadata(
    dimension_count::Int,
    temporal_dim::Union{Nothing,Int},
    dimension_names::Union{Nothing,Tuple},
    dimension_steps::Union{Nothing,Tuple},
    dimension_units::Union{Nothing,Tuple},
    dimension_roles::Union{Nothing,Tuple},
)::NamedTuple
    dimension_names !== nothing && length(dimension_names) != dimension_count &&
        throw(DimensionMismatch("Dimension names must match the number of tensor dimensions!"))
    dimension_steps !== nothing && length(dimension_steps) != dimension_count &&
        throw(DimensionMismatch("Dimension steps must match the number of tensor dimensions!"))
    dimension_units !== nothing && length(dimension_units) != dimension_count &&
        throw(DimensionMismatch("Dimension units must match the number of tensor dimensions!"))
    dimension_roles !== nothing && length(dimension_roles) != dimension_count &&
        throw(DimensionMismatch("Dimension roles must match the number of tensor dimensions!"))
    names::Tuple = ntuple(
        dimension::Int -> begin
            if dimension_names === nothing
                Symbol("dim$(dimension)")
            else
                name_value::Any = dimension_names[dimension]
                name_value isa Symbol || throw(ArgumentError("Dimension names must be Symbols!"))
                name_value
            end
        end,
        dimension_count,
    )
    steps::Tuple = ntuple(
        dimension::Int -> begin
            step_value::Any = dimension_steps === nothing ? 1 : dimension_steps[dimension]
            if step_value isa Real
                isfinite(step_value) && step_value > 0 || throw(ArgumentError("Numeric dimension steps must be finite and positive!"))
            elseif step_value isa Dates.Period
                Dates.value(step_value) > 0 || throw(ArgumentError("Date/time dimension steps must be positive!"))
            else
                throw(ArgumentError("Dimension steps must be real numbers or Dates.Period values!"))
            end
            step_value
        end,
        dimension_count,
    )
    units::Tuple = ntuple(
        dimension::Int -> begin
            unit_value::Any = dimension_units === nothing ? "bins" : dimension_units[dimension]
            (unit_value isa AbstractString || unit_value isa Symbol) ||
                throw(ArgumentError("Dimension units must be strings or Symbols!"))
            string(unit_value)
        end,
        dimension_count,
    )
    roles::Tuple = ntuple(
        dimension::Int -> begin
            role_value::Any = dimension_roles === nothing ?
                (temporal_dim === dimension ? :temporal : :spatial) :
                dimension_roles[dimension]
            role_value isa Symbol || throw(ArgumentError("Dimension roles must be Symbols!"))
            role::Symbol = role_value
            role in (:horizontal, :depth, :spatial, :temporal) || throw(ArgumentError(
                "Dimension roles must be :horizontal, :depth, :spatial, or :temporal!",
            ))
            role
        end,
        dimension_count,
    )
    if temporal_dim === nothing
        :temporal in roles && throw(ArgumentError(
            "A :temporal dimension role requires temporal_dim to identify that dimension!",
        ))
    else
        roles[temporal_dim] == :temporal || throw(ArgumentError(
            "The dimension identified by temporal_dim must have the :temporal role!",
        ))
        count(==(:temporal), roles) == 1 || throw(ArgumentError(
            "Exactly one dimension can have the :temporal role!",
        ))
    end
    return (names=names, steps=steps, units=units, roles=roles)
end

function _tensor_symbol_counts(quantized::Array{Int,N}, valid_mask::BitArray{N})::Dict{Int,Int} where {N}
    symbol_counts::Dict{Int,Int} = Dict{Int,Int}()
    for index::CartesianIndex{N} in CartesianIndices(quantized)
        valid_mask[index] || continue
        _increment_information_count!(symbol_counts, quantized[index])
    end
    return symbol_counts
end

function _sparse_baseline_information(
    quantized::Array{Int,N},
    valid_cell_count::Int,
    symbol_counts::Dict{Int,Int},
)::NamedTuple where {N}
    if valid_cell_count != length(quantized) || isempty(symbol_counts)
        return (symbol=nothing, indices=nothing)
    end
    baseline_symbol::Int = first(keys(symbol_counts))
    baseline_count::Int = symbol_counts[baseline_symbol]
    for (symbol::Int, symbol_count::Int) in symbol_counts
        if symbol_count > baseline_count || (symbol_count == baseline_count && symbol < baseline_symbol)
            baseline_symbol = symbol
            baseline_count = symbol_count
        end
    end
    nonbaseline_count::Int = valid_cell_count - baseline_count
    if nonbaseline_count > div(valid_cell_count, 4)
        return (symbol=nothing, indices=nothing)
    end
    nonbaseline_indices::Vector{CartesianIndex{N}} = findall(
        symbol::Int -> symbol != baseline_symbol,
        quantized,
    )
    return (symbol=baseline_symbol, indices=nonbaseline_indices)
end

"""
structure_information(tensor; valid_mask=trues(size(tensor)), bins=16, temporal_dim=ndims(tensor), residual_coding=:shannon, compute_spectral=true, quantization=:linear, lag_offsets=nothing, lag_sign=:canonical, dimension_names=nothing, dimension_steps=nothing, dimension_units=nothing, dimension_roles=nothing)

Measure information that depends on tensor structure rather than only on flattened
cell weights.

Values are quantized globally before cells at the requested integer `lag_offsets`
are compared. `nothing` preserves the original positive unit-axis comparisons.
In canonical mode, opposite purely spatial directions are equivalent and every
temporal offset points forward. Use `lag_sign=:directed` to retain input signs.

The default `quantization=:linear` preserves the original global linear bins.
Use `quantization=:zero_preserving` for nonnegative sparse count or mass fields
when exact zeros must remain distinct from every positive cell; the remaining
bins are assigned linearly across the positive values. Negative valid values are
rejected in this mode so symbol order and prediction residuals remain monotone.

The returned `axis_information` reports mutual information, conditional entropy, normalized variation, and predictive-residual entropy along every axis. The temporal axis uses the same inter-frame residual principle as lossless movie compression.

`spectral_information` reports entropy and effective rank of every tensor unfolding.

`residual_coding` selects fixed-width residuals (`:none`), the ideal Shannon limit (`:shannon`), or a realizable binary Huffman code (`:huffman`).

Coding results are reported as `residual_coding` within each entry of `axis_information`.

`lag_information` reports arbitrary axis-aligned, diagonal, or mixed
space-time offsets. Existing `axis_information` and top-level summaries always
remain based on positive unit-axis offsets for backward compatibility.

Use explicit `dimension_roles`, such as
`(:horizontal, :horizontal, :depth, :temporal)`, to prevent a depth axis from
being averaged into the legacy horizontal spatial summary.

Set `compute_spectral=false` for tensors whose unfoldings are too large for SVD.
"""
function structure_information(
    tensor::AbstractArray{T};
    valid_mask::AbstractArray{Bool}=trues(size(tensor)),
    bins::Integer=16,
    temporal_dim::Union{Nothing,Integer}=ndims(tensor),
    residual_coding::Symbol=:shannon,
    compute_spectral::Bool=true,
    quantization::Symbol=:linear,
    lag_offsets::Union{Nothing,AbstractVector}=nothing,
    lag_sign::Symbol=:canonical,
    dimension_names::Union{Nothing,Tuple}=nothing,
    dimension_steps::Union{Nothing,Tuple}=nothing,
    dimension_units::Union{Nothing,Tuple}=nothing,
    dimension_roles::Union{Nothing,Tuple}=nothing,
)::NamedTuple where {T <: Real}
    bin_count::Int = Int(bins)
    if temporal_dim !== nothing && !(1 <= temporal_dim <= ndims(tensor))
        throw(ArgumentError("The temporal dimension must identify a tensor dimension or be nothing!"))
    end
    normalized_temporal_dim::Union{Nothing,Int} =
        temporal_dim === nothing ? nothing : Int(temporal_dim)
    if !(residual_coding in (:none, :shannon, :huffman))
        throw(ArgumentError("Residual coding must be :none, :shannon, or :huffman!"))
    end
    quantization in (:linear, :zero_preserving) || throw(ArgumentError(
        "Tensor quantization must be :linear or :zero_preserving!",
    ))
    dimension_count::Int = ndims(tensor)
    metadata::NamedTuple = _normalize_dimension_metadata(
        dimension_count,
        normalized_temporal_dim,
        dimension_names,
        dimension_steps,
        dimension_units,
        dimension_roles,
    )
    normalized_lag_offsets_untyped::Vector{<:Tuple} = _normalize_lag_offsets(
        lag_offsets,
        dimension_count,
        normalized_temporal_dim,
        lag_sign,
    )
    normalized_lag_offsets::Vector{NTuple{dimension_count,Int}} =
        NTuple{dimension_count,Int}[offset for offset::Tuple in normalized_lag_offsets_untyped]
    # Quantize the tensor and determine the finite mask for valid entries.
    quantized::Array{Int}, finite_mask::BitArray =
        _quantize_tensor(tensor, valid_mask, bin_count, quantization)
    valid_cell_count::Int = count(finite_mask)
    symbol_counts::Dict{Int,Int} = _tensor_symbol_counts(quantized, finite_mask)
    value_entropy::Float64 = _discrete_entropy_from_counts(symbol_counts, valid_cell_count)
    maximum_value_entropy::Float64 = log2(Float64(bin_count))
    normalized_value_entropy::Float64 = maximum_value_entropy > 0.0 ? value_entropy / maximum_value_entropy : 0.0
    baseline_information::NamedTuple = _sparse_baseline_information(
        quantized,
        valid_cell_count,
        symbol_counts,
    )
    baseline_symbol::Union{Nothing,Int} = baseline_information.symbol
    nonbaseline_indices::Union{Nothing,Vector{CartesianIndex{dimension_count}}} = baseline_information.indices
    axis_offsets::Vector{NTuple{dimension_count,Int}} = NTuple{dimension_count,Int}[
        ntuple(candidate_dimension::Int -> candidate_dimension == axis ? 1 : 0, dimension_count)
        for axis::Int = 1:dimension_count
    ]
    information_cache::Dict{NTuple{dimension_count,Int},NamedTuple} = Dict{NTuple{dimension_count,Int},NamedTuple}()
    axis_information::Vector{NamedTuple} = NamedTuple[]
    spectral_information::Vector{NamedTuple} = NamedTuple[]
    for axis::Int = 1:dimension_count
        offset::NTuple{dimension_count,Int} = axis_offsets[axis]
        role::Symbol = _lag_role(offset, metadata.roles)
        metric::NamedTuple = _lag_information(
            quantized,
            finite_mask,
            offset,
            role,
            bin_count,
            residual_coding,
            metadata.names,
            metadata.steps,
            metadata.units,
            metadata.roles,
            baseline_symbol,
            nonbaseline_indices,
        )
        information_cache[offset] = metric
        push!(axis_information, metric)
        if compute_spectral
            push!(spectral_information, _spectral_axis_information(tensor, finite_mask, axis))
        end
    end
    lag_information::Vector{NamedTuple} = NamedTuple[]
    for offset::NTuple{dimension_count,Int} in normalized_lag_offsets
        if !haskey(information_cache, offset)
            role::Symbol = _lag_role(offset, metadata.roles)
            information_cache[offset] = _lag_information(
                quantized,
                finite_mask,
                offset,
                role,
                bin_count,
                residual_coding,
                metadata.names,
                metadata.steps,
                metadata.units,
                metadata.roles,
                baseline_symbol,
                nonbaseline_indices,
            )
        end
        push!(lag_information, information_cache[offset])
    end
    spatial_metrics::Vector{NamedTuple} = filter(metric::NamedTuple -> metric.role == :spatial && metric.pair_count > 0, axis_information)
    depth_metrics::Vector{NamedTuple} = filter(metric::NamedTuple -> metric.role == :depth && metric.pair_count > 0, axis_information)
    temporal_metrics::Vector{NamedTuple} = filter(metric::NamedTuple -> metric.role == :temporal && metric.pair_count > 0, axis_information)
    spatial_dependence::Float64 = isempty(spatial_metrics) ? 0.0 : Statistics.mean(metric.normalized_mutual_information for metric::NamedTuple in spatial_metrics)
    spatial_variation::Float64 = isempty(spatial_metrics) ? 0.0 : Statistics.mean(metric.mean_normalized_difference for metric::NamedTuple in spatial_metrics)
    depth_dependence::Float64 = isempty(depth_metrics) ? 0.0 : Statistics.mean(metric.normalized_mutual_information for metric::NamedTuple in depth_metrics)
    depth_variation::Float64 = isempty(depth_metrics) ? 0.0 : Statistics.mean(metric.mean_normalized_difference for metric::NamedTuple in depth_metrics)
    temporal_dependence::Float64 = isempty(temporal_metrics) ? 0.0 : Statistics.mean(metric.normalized_mutual_information for metric::NamedTuple in temporal_metrics)
    temporal_predictive_gain::Float64 = isempty(temporal_metrics) ? 0.0 : Statistics.mean(metric.predictive_gain_bits for metric::NamedTuple in temporal_metrics)
    temporal_variation::Float64 = isempty(temporal_metrics) ? 0.0 : Statistics.mean(metric.mean_normalized_difference for metric::NamedTuple in temporal_metrics)
    return (
        value_entropy_bits=value_entropy,
        normalized_value_entropy=normalized_value_entropy,
        spatial_dependence=spatial_dependence,
        spatial_variation=spatial_variation,
        depth_dependence=depth_dependence,
        depth_variation=depth_variation,
        temporal_dependence=temporal_dependence,
        temporal_predictive_gain_bits=temporal_predictive_gain,
        temporal_variation=temporal_variation,
        axis_information=axis_information,
        lag_information=lag_information,
        spectral_information=spectral_information,
        bins=bin_count,
        quantization=quantization,
        residual_coding=residual_coding,
        lag_offsets=normalized_lag_offsets,
        lag_sign=lag_sign,
        lag_pair_scope=:both_endpoints_valid,
        dimension_metadata=metadata,
        spectral_computed=compute_spectral,
        valid_cell_count=valid_cell_count,
    )
end

"""
plot_lag_information(information, filename=""; normalize=:intrinsic, title_extra="")

Plot direction-resolved dependence, coherence, and residual-coding savings for
the explicit lag vectors returned by `structure_information`. Lag vectors are
kept separate; the plot does not average axial and diagonal directions or mix
space, depth, and time into one physical distance.
"""
function plot_lag_information(
    information::NamedTuple,
    filename::AbstractString="";
    normalize::Symbol=:intrinsic,
    title_extra::AbstractString="",
)::Gadfly.Plot
    haskey(information, :lag_information) || throw(ArgumentError(
        "The input must be produced by structure_information with lag support!",
    ))
    normalize in (:intrinsic, :range) || throw(ArgumentError(
        "The lag-plot normalization must be :intrinsic or :range!",
    ))
    requested_lag_count::Int = length(information.lag_information)
    lag_metrics::Vector{NamedTuple} = filter(
        metric::NamedTuple -> Bool(metric.applicable),
        information.lag_information,
    )
    isempty(lag_metrics) && throw(ArgumentError("No requested lag has a valid cell pair!"))
    omitted_lag_count::Int = requested_lag_count - length(lag_metrics)
    if omitted_lag_count > 0
        @warn(
            "Omitting requested lags with no valid cell pairs from the lag plot",
            omitted_lag_count=omitted_lag_count,
            requested_lag_count=requested_lag_count,
        )
    end
    if normalize == :range && length(lag_metrics) < 2
        throw(ArgumentError("Range normalization requires at least two applicable lag vectors!"))
    end
    metric_labels::Vector{String} = [
        "Normalized mutual information",
        "Symbol coherence",
        "Residual coding savings",
    ]
    metric_colors::Vector{String} = ["#1f77b4", "#2ca02c", "#d62728"]
    lag_labels::Vector{String} = String[]
    pair_labels::Vector{String} = String[]
    label_y_values::Vector{Float64} = Float64[]
    x_values::Vector{String} = String[]
    y_values::Vector{Float64} = Float64[]
    series_labels::Vector{String} = String[]
    for lag_metric::NamedTuple in lag_metrics
        coordinate_label::String = replace(
            String(lag_metric.display_label),
            ", " => "\n",
            "latitude" => "lat",
            "longitude" => "lon",
            "depth" => "z",
            "time" => "t",
            " deg" => "°",
        )
        lag_label::String = coordinate_label
        push!(lag_labels, lag_label)
        fixed_width_bits::Int = Int(lag_metric.residual_coding.fixed_width_bits)
        encoded_bits::Float64 = Float64(lag_metric.residual_coding.encoded_bits)
        coding_savings::Float64 = fixed_width_bits > 0 ?
            clamp(1.0 - encoded_bits / fixed_width_bits, 0.0, 1.0) : 0.0
        metric_values::Vector{Float64} = [
            clamp(Float64(lag_metric.normalized_mutual_information), 0.0, 1.0),
            clamp(1.0 - Float64(lag_metric.mean_normalized_difference), 0.0, 1.0),
            coding_savings,
        ]
        for metric_index::Int in eachindex(metric_labels)
            push!(x_values, lag_label)
            push!(y_values, metric_values[metric_index])
            push!(series_labels, metric_labels[metric_index])
        end
        push!(pair_labels, "n=$(lag_metric.pair_count)")
        push!(label_y_values, min(maximum(metric_values) + 0.04, 0.98))
    end
    if normalize == :range
        _range_normalize_plot_values!(y_values, series_labels, metric_labels)
        empty!(label_y_values)
        metric_count::Int = length(metric_labels)
        for lag_index::Int in eachindex(lag_metrics)
            first_index::Int = (lag_index - 1) * metric_count + 1
            last_index::Int = first_index + metric_count - 1
            push!(label_y_values, min(maximum(y_values[first_index:last_index]) + 0.04, 0.98))
        end
    end
    normalization_text::String = normalize == :range ? " (range normalized)" : ""
    y_label::String = normalize == :range ?
        "Within-metric normalized range" : "Intrinsic normalized metric"
    title::String = "Lag-resolved structure information$(normalization_text)$(title_extra)"
    lag_plot::Gadfly.Plot = Gadfly.plot(
        Gadfly.layer(Gadfly.Geom.point; x=x_values, y=y_values, color=series_labels),
        Gadfly.layer(
            Gadfly.Geom.label(; position=:dynamic, hide_overlaps=false);
            x=lag_labels,
            y=label_y_values,
            label=pair_labels,
        ),
        Gadfly.Scale.x_discrete,
        Gadfly.Scale.color_discrete_manual(metric_colors...),
        Gadfly.Guide.colorkey(; title="", labels=metric_labels),
        Gadfly.Coord.cartesian(; ymin=0.0, ymax=1.0),
        Gadfly.Guide.xticks(; orientation=:horizontal),
        Gadfly.Guide.xlabel("Lag vector (coordinate offsets; wrapped by dimension)"),
        Gadfly.Guide.ylabel(y_label),
        Gadfly.Guide.title(title),
        Gadfly.Theme(;
            key_position=:right,
            background_color="white",
            minor_label_font_size=8Gadfly.pt,
        ),
    )
    if filename != ""
        @info("Saving lag-information plot to file: $filename")
        plot_width_inches::Float64 = clamp(0.9 * length(lag_metrics), 15.0, 30.0)
        Mads.plotfileformat(
            lag_plot,
            filename,
            plot_width_inches * Gadfly.inch,
            6Gadfly.inch,
        )
    end
    return lag_plot
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

function _magnitude_grid_label_is_valid(label::Tuple)::Bool
    for component::Any in label
        _magnitude_grid_label_is_valid(component) || return false
    end
    return true
end

function _magnitude_grid_label_is_valid(label::NamedTuple)::Bool
    for component::Any in values(label)
        _magnitude_grid_label_is_valid(component) || return false
    end
    return true
end

function _magnitude_grid_label_is_valid(label::Any)::Bool
    return _raw_value_is_valid(label)
end

function _magnitude_grid_label_data(
    grid_labels::AbstractVector,
    observation_count::Int,
)::NamedTuple
    length(grid_labels) == observation_count ||
        throw(DimensionMismatch("Grid labels and magnitudes must have the same length!"))
    labels::Vector{Any} = Any[grid_labels[index] for index::Int in eachindex(grid_labels)]
    label_valid_mask::BitVector = BitVector(
        _magnitude_grid_label_is_valid(label) for label::Any in labels
    )
    return (
        labels=labels,
        valid_mask=label_valid_mask,
        feature_names=(:grid_label,),
        representation=:supplied_labels,
    )
end

function _magnitude_grid_label_data(
    grid_assignments::NamedTuple,
    observation_count::Int,
)::NamedTuple
    feature_names::Tuple = Tuple(keys(grid_assignments))
    isempty(feature_names) && throw(ArgumentError("At least one grid-assignment feature is required!"))
    columns::Vector{AbstractVector} = AbstractVector[]
    for feature_name::Symbol in feature_names
        column::Any = getfield(grid_assignments, feature_name)
        column isa AbstractVector ||
            throw(ArgumentError("Every grid-assignment feature must be a vector!"))
        length(column) == observation_count ||
            throw(DimensionMismatch("Every grid-assignment feature must match the number of magnitudes!"))
        push!(columns, column)
    end
    labels::Vector{Any} = Vector{Any}(undef, observation_count)
    label_valid_mask::BitVector = trues(observation_count)
    for observation_index::Int = 1:observation_count
        components::Tuple = Tuple(
            column[observation_index] for column::AbstractVector in columns
        )
        label::NamedTuple = NamedTuple{feature_names}(components)
        labels[observation_index] = label
        label_valid_mask[observation_index] = _magnitude_grid_label_is_valid(label)
    end
    return (
        labels=labels,
        valid_mask=label_valid_mask,
        feature_names=feature_names,
        representation=:joint_feature_labels,
    )
end

function _magnitude_predictor_symbol(
    prediction::Float64,
    precision::Float64,
    origin::Float64,
)::Int
    coordinate::Float64 = (prediction - origin) / precision
    isfinite(coordinate) ||
        throw(ArgumentError("A magnitude predictor cannot be represented at the requested precision!"))
    doubled_coordinate::Float64 = 2.0 * coordinate
    nearest_half_integer::Float64 = round(doubled_coordinate)
    nearest_half_coordinate::Float64 = nearest_half_integer / 2.0
    half_integer_tolerance::Float64 = _raw_scaled_coordinate_tolerance(
        prediction,
        precision,
        origin,
        coordinate,
    )
    stable_coordinate::Float64 =
        half_integer_tolerance <= 1.0e-9 &&
        abs(coordinate - nearest_half_coordinate) <= half_integer_tolerance ?
        nearest_half_coordinate : coordinate
    rounded_coordinate::Float64 = round(stable_coordinate)
    if rounded_coordinate < typemin(Int) || rounded_coordinate > typemax(Int)
        throw(ArgumentError("A magnitude predictor symbol exceeds the supported Int range!"))
    end
    return Int(rounded_coordinate)
end

function _magnitude_reconstruction_value(values::Vector{Float64}, method::Symbol)::Float64
    if method == :maximum
        return maximum(values)
    elseif method == :mean
        return Statistics.mean(values)
    end
    return Statistics.median(values)
end

function _magnitude_reconstruction_information(
    magnitude_values::Vector{Float64},
    magnitude_symbols::Vector{Int},
    grid_symbols::Vector{Int},
    cell_values::Dict{Int,Vector{Float64}},
    method::Symbol,
    residual_coding::Symbol,
    magnitude_precision::Float64,
    magnitude_origin::Float64,
    source_fixed_bits_per_symbol::Int,
    expected_conditional_entropy_bits::Float64,
)::NamedTuple
    predictor_values::Dict{Int,Float64} = Dict{Int,Float64}()
    predictor_symbols::Dict{Int,Int} = Dict{Int,Int}()
    ordered_cell_symbols::Vector{Int} = sort!(collect(keys(cell_values)))
    ordered_predictor_values::Vector{Float64} = Float64[]
    ordered_predictor_symbols::Vector{Int} = Int[]
    for grid_symbol::Int in ordered_cell_symbols
        prediction::Float64 = _magnitude_reconstruction_value(cell_values[grid_symbol], method)
        predictor_symbol::Int = _magnitude_predictor_symbol(
            prediction,
            magnitude_precision,
            magnitude_origin,
        )
        predictor_values[grid_symbol] = prediction
        predictor_symbols[grid_symbol] = predictor_symbol
        push!(ordered_predictor_values, prediction)
        push!(ordered_predictor_symbols, predictor_symbol)
    end

    observation_count::Int = length(magnitude_values)
    residual_counts::Dict{Int,Int} = Dict{Int,Int}()
    joint_residual_grid_counts::Dict{Tuple{Int,Int},Int} = Dict{Tuple{Int,Int},Int}()
    reconstruction_grid_counts::Dict{Int,Int} = Dict{Int,Int}()
    cell_residual_counts::Dict{Int,Dict{Int,Int}} = Dict{Int,Dict{Int,Int}}()
    physical_residuals::Vector{Float64} = Vector{Float64}(undef, observation_count)
    minimum_residual_symbol::Int = typemax(Int)
    maximum_residual_symbol::Int = typemin(Int)
    for observation_position::Int = 1:observation_count
        grid_symbol::Int = grid_symbols[observation_position]
        residual_symbol_big::BigInt =
            BigInt(magnitude_symbols[observation_position]) - BigInt(predictor_symbols[grid_symbol])
        if residual_symbol_big < typemin(Int) || residual_symbol_big > typemax(Int)
            throw(ArgumentError("A quantized magnitude residual exceeds the supported Int range!"))
        end
        residual_symbol::Int = Int(residual_symbol_big)
        _increment_information_count!(residual_counts, residual_symbol)
        _increment_information_count!(joint_residual_grid_counts, (residual_symbol, grid_symbol))
        _increment_information_count!(reconstruction_grid_counts, grid_symbol)
        if !haskey(cell_residual_counts, grid_symbol)
            cell_residual_counts[grid_symbol] = Dict{Int,Int}()
        end
        _increment_information_count!(cell_residual_counts[grid_symbol], residual_symbol)
        minimum_residual_symbol = min(minimum_residual_symbol, residual_symbol)
        maximum_residual_symbol = max(maximum_residual_symbol, residual_symbol)
        physical_residuals[observation_position] =
            magnitude_values[observation_position] - predictor_values[grid_symbol]
    end

    residual_entropy_bits::Float64 =
        _discrete_entropy_from_counts(residual_counts, observation_count)
    residual_grid_joint_entropy_bits::Float64 =
        _discrete_entropy_from_counts(joint_residual_grid_counts, observation_count)
    reconstruction_grid_entropy_bits::Float64 =
        _discrete_entropy_from_counts(reconstruction_grid_counts, observation_count)
    joint_difference_conditional_residual_entropy_bits::Float64 = clamp(
        residual_grid_joint_entropy_bits - reconstruction_grid_entropy_bits,
        0.0,
        residual_entropy_bits,
    )
    conditional_residual_entropy_bits::Float64 = 0.0
    for grid_symbol::Int in sort!(collect(keys(cell_residual_counts)))
        cell_counts::Dict{Int,Int} = cell_residual_counts[grid_symbol]
        cell_observation_count::Int = sum(values(cell_counts))
        cell_entropy_bits::Float64 =
            _discrete_entropy_from_counts(cell_counts, cell_observation_count)
        conditional_residual_entropy_bits +=
            (cell_observation_count / observation_count) * cell_entropy_bits
    end
    conditional_residual_entropy_bits = clamp(
        conditional_residual_entropy_bits,
        0.0,
        residual_entropy_bits,
    )
    entropy_identity_tolerance::Float64 = max(
        1.0e-12,
        16.0 * observation_count * eps(max(
            expected_conditional_entropy_bits,
            conditional_residual_entropy_bits,
            1.0,
        )),
    )
    conditional_entropy_matches_magnitude::Bool = isapprox(
        conditional_residual_entropy_bits,
        expected_conditional_entropy_bits;
        atol=entropy_identity_tolerance,
        rtol=16.0 * eps(Float64),
    )
    conditional_entropy_matches_magnitude || throw(ArgumentError(
        "Cell-conditioned residual entropy must equal quantized magnitude heterogeneity!",
    ))
    shannon_encoded_bits::Float64 = observation_count * residual_entropy_bits
    huffman_encoded_bits::Int = _huffman_encoded_bits_from_counts(residual_counts)
    conditional_shannon_encoded_bits::Float64 =
        observation_count * conditional_residual_entropy_bits
    conditional_huffman_encoded_bits::Int = sum(
        _huffman_encoded_bits_from_counts(counts)
        for counts::Dict{Int,Int} in values(cell_residual_counts);
        init=0,
    )
    residual_support_span::BigInt = observation_count == 0 ? BigInt(0) :
        BigInt(maximum_residual_symbol) - BigInt(minimum_residual_symbol) + 1
    residual_range_fixed_bits_per_symbol::Int = residual_support_span > 1 ?
        ndigits(residual_support_span - 1; base=2) : 0
    residual_range_fixed_width_bits::Int =
        Base.checked_mul(observation_count, residual_range_fixed_bits_per_symbol)
    fixed_width_bits::Int =
        Base.checked_mul(observation_count, source_fixed_bits_per_symbol)
    encoded_bits::Float64 = residual_coding == :shannon ?
        shannon_encoded_bits : Float64(huffman_encoded_bits)
    coding_savings::Float64 = fixed_width_bits > 0 ?
        1.0 - encoded_bits / fixed_width_bits : NaN
    residual_range_coding_savings::Float64 = residual_range_fixed_width_bits > 0 ?
        1.0 - encoded_bits / residual_range_fixed_width_bits : NaN
    mae::Float64 = Statistics.mean(abs.(physical_residuals))
    rmse::Float64 = sqrt(Statistics.mean(abs2.(physical_residuals)))
    bias::Float64 = Statistics.mean(physical_residuals)
    ordered_residual_symbols::Vector{Int} = sort!(collect(keys(residual_counts)))
    ordered_residual_counts::Vector{Int} =
        Int[residual_counts[symbol] for symbol::Int in ordered_residual_symbols]
    return (
        method=method,
        estimator_optimality=method == :mean ? :minimum_squared_error_within_cell :
            (method == :median ? :minimum_absolute_error_within_cell :
             :preserves_cell_peak_not_error_optimal),
        observation_count=observation_count,
        predictor_symbol_rule=:nearest_lattice_coordinate_ties_to_even,
        residual_definition=:magnitude_symbol_minus_cell_predictor_symbol,
        residual_transform_exact=true,
        ordered_residual_stream_retained=false,
        residual_symbols=ordered_residual_symbols,
        residual_counts=ordered_residual_counts,
        residual_symbol_count=length(residual_counts),
        residual_symbol_minimum=minimum_residual_symbol,
        residual_symbol_maximum=maximum_residual_symbol,
        residual_entropy_bits=residual_entropy_bits,
        pooled_residual_entropy_bits=residual_entropy_bits,
        conditional_residual_entropy_bits=conditional_residual_entropy_bits,
        joint_difference_conditional_residual_entropy_bits=
            joint_difference_conditional_residual_entropy_bits,
        conditional_residual_entropy_diagnostic_difference_bits=
            joint_difference_conditional_residual_entropy_bits -
            conditional_residual_entropy_bits,
        entropy_identity_tolerance=entropy_identity_tolerance,
        conditional_entropy_matches_magnitude_heterogeneity=
            conditional_entropy_matches_magnitude,
        residual_entropy_excess_bits=
            max(0.0, residual_entropy_bits - conditional_residual_entropy_bits),
        shannon_encoded_bits=shannon_encoded_bits,
        huffman_encoded_bits=huffman_encoded_bits,
        conditional_shannon_encoded_bits=conditional_shannon_encoded_bits,
        conditional_huffman_encoded_bits=conditional_huffman_encoded_bits,
        selected_coding=residual_coding,
        encoded_bits=encoded_bits,
        bits_per_residual=observation_count > 0 ? encoded_bits / observation_count : 0.0,
        fixed_width_bits=fixed_width_bits,
        fixed_bits_per_residual=Float64(source_fixed_bits_per_symbol),
        fixed_width_reference=:global_observed_magnitude_symbol_span,
        coding_savings=coding_savings,
        residual_range_fixed_width_bits=residual_range_fixed_width_bits,
        residual_range_fixed_bits_per_residual=
            Float64(residual_range_fixed_bits_per_symbol),
        residual_range_fixed_width_reference=
            :observed_integer_residual_range_without_side_information,
        residual_range_coding_savings=residual_range_coding_savings,
        predictor_overhead_included=false,
        grid_label_overhead_included=false,
        codebook_overhead_included=false,
        singleton_payload_caveat=
            :zero_residual_payload_can_move_a_singleton_mark_into_its_cell_predictor,
        physical_residual_definition=:magnitude_minus_cell_prediction,
        mae=mae,
        rmse=rmse,
        bias=bias,
        cell_predictors=(
            grid_symbols=ordered_cell_symbols,
            predictions=ordered_predictor_values,
            predictor_symbols=ordered_predictor_symbols,
        ),
    )
end

"""
    magnitude_aggregation_information(magnitudes, grid_assignments;
        magnitude_precision, magnitude_origin=0.0,
        valid_mask=trues(length(magnitudes)),
        reconstructions=(:maximum, :mean, :median),
        residual_coding=:huffman)

Measure information lost when continuous magnitude marks are aggregated into
deterministic grid cells. Magnitudes are quantized on an explicit
origin-anchored floor lattice before computing
`magnitude_conditional_entropy_bits = H(M | G)`. The returned
`cell_histograms` retains only occupied cells and nonzero symbol counts in a
compact compressed-sparse-row representation. For cell `j`, its histogram is
stored at `offsets[j]:(offsets[j + 1] - 1)`.

For each requested cell reconstruction, physical residuals `M - Mhat_G`
produce MAE, RMSE, and bias. Coding residuals are instead the exactly
reconstructable integers `q - qhat_G`, where `q` is the floor-lattice magnitude
symbol and `qhat_G` is the nearest lattice coordinate to the physical cell
predictor (ties to even). Shannon and binary-Huffman sizes exclude predictor,
grid-label, codebook, and container overhead. The common fixed-width reference
uses the global observed magnitude-symbol span, so coding savings are comparable
across grids and reconstructors. A second adaptive reference uses the smallest
integer width covering the observed pooled residual range. Neither reference
includes its range as side information.

The pooled residual entropy `H(R)` can exceed the cell-conditioned entropy
`H(R | G)`. Because subtracting one predictor symbol per cell is bijective,
`H(R | G) = H(M | G)`; this identity is checked explicitly. A zero payload for
a singleton cell only moves its magnitude mark into the uncounted predictor.

`magnitude_location_dependence_fraction = I(M;G) / H(M)` measures the fraction
of quantized magnitude entropy statistically associated with the grid labels.
The compatibility field `magnitude_retention_fraction` is an exact alias; it is
not retention by the maximum, mean, or median cell reconstruction. All entropy
values are empirical plug-in estimates and can be optimistic on sparse,
singleton-heavy grids.

`grid_assignments` may be one vector of arbitrary deterministic labels or a
named tuple of equal-length coordinate-label vectors.
"""
function magnitude_aggregation_information(
    magnitudes::AbstractVector,
    grid_assignments::Union{AbstractVector,NamedTuple};
    magnitude_precision::Real,
    magnitude_origin::Real=0.0,
    valid_mask::AbstractVector{Bool}=trues(length(magnitudes)),
    reconstructions::Union{Tuple{Vararg{Symbol}},AbstractVector{Symbol}}=
        (:maximum, :mean, :median),
    residual_coding::Symbol=:huffman,
)::NamedTuple
    observation_count::Int = length(magnitudes)
    length(valid_mask) == observation_count ||
        throw(DimensionMismatch("The magnitude valid mask must match the observation count!"))
    precision_value::Float64 = Float64(magnitude_precision)
    origin_value::Float64 = Float64(magnitude_origin)
    isfinite(precision_value) && precision_value > 0.0 ||
        throw(ArgumentError("Magnitude precision must be finite and positive!"))
    isfinite(origin_value) || throw(ArgumentError("Magnitude origin must be finite!"))
    residual_coding in (:shannon, :huffman) ||
        throw(ArgumentError("Magnitude residual coding must be :shannon or :huffman!"))
    reconstruction_methods::Vector{Symbol} = collect(reconstructions)
    isempty(reconstruction_methods) &&
        throw(ArgumentError("At least one magnitude reconstruction method is required!"))
    length(unique(reconstruction_methods)) == length(reconstruction_methods) ||
        throw(ArgumentError("Magnitude reconstruction methods must be unique!"))
    all(method in (:maximum, :mean, :median) for method::Symbol in reconstruction_methods) ||
        throw(ArgumentError("Magnitude reconstruction methods must be :maximum, :mean, or :median!"))

    grid_label_data::NamedTuple =
        _magnitude_grid_label_data(grid_assignments, observation_count)
    valid_indices::Vector{Int} = Int[]
    for observation_index::Int = 1:observation_count
        magnitude::Any = magnitudes[observation_index]
        magnitude_is_valid::Bool = magnitude isa Real && isfinite(magnitude)
        if valid_mask[observation_index] && grid_label_data.valid_mask[observation_index] &&
           magnitude_is_valid
            push!(valid_indices, observation_index)
        end
    end
    isempty(valid_indices) &&
        throw(ArgumentError("At least one finite magnitude with a valid grid label is required!"))

    magnitude_symbols::Vector{Int}, resolved_origin::Any, feature_kind::Symbol, ordered::Bool =
        _raw_quantize_column(
            magnitudes,
            valid_indices,
            magnitude_precision,
            magnitude_origin,
        )
    label_symbols::Vector{Int}, label_origin::Any, label_kind::Symbol, label_ordered::Bool =
        _raw_quantize_column(
            grid_label_data.labels,
            valid_indices,
            nothing,
            nothing,
        )
    magnitude_values::Vector{Float64} = Float64[magnitudes[index] for index::Int in valid_indices]
    valid_observation_count::Int = length(valid_indices)
    magnitude_counts::Dict{Int,Int} = Dict{Int,Int}()
    grid_counts::Dict{Int,Int} = Dict{Int,Int}()
    joint_counts::Dict{Tuple{Int,Int},Int} = Dict{Tuple{Int,Int},Int}()
    cell_magnitude_counts::Dict{Int,Dict{Int,Int}} = Dict{Int,Dict{Int,Int}}()
    cell_values::Dict{Int,Vector{Float64}} = Dict{Int,Vector{Float64}}()
    cell_labels::Dict{Int,Any} = Dict{Int,Any}()
    for observation_position::Int = 1:valid_observation_count
        magnitude_symbol::Int = magnitude_symbols[observation_position]
        grid_symbol::Int = label_symbols[observation_position]
        _increment_information_count!(magnitude_counts, magnitude_symbol)
        _increment_information_count!(grid_counts, grid_symbol)
        _increment_information_count!(joint_counts, (magnitude_symbol, grid_symbol))
        if !haskey(cell_magnitude_counts, grid_symbol)
            cell_magnitude_counts[grid_symbol] = Dict{Int,Int}()
            cell_values[grid_symbol] = Float64[]
            cell_labels[grid_symbol] = grid_label_data.labels[valid_indices[observation_position]]
        end
        _increment_information_count!(
            cell_magnitude_counts[grid_symbol],
            magnitude_symbol,
        )
        push!(cell_values[grid_symbol], magnitude_values[observation_position])
    end

    magnitude_entropy_bits::Float64 =
        _discrete_entropy_from_counts(magnitude_counts, valid_observation_count)
    grid_entropy_bits::Float64 =
        _discrete_entropy_from_counts(grid_counts, valid_observation_count)
    joint_entropy_bits::Float64 =
        _discrete_entropy_from_counts(joint_counts, valid_observation_count)
    joint_difference_magnitude_conditional_entropy_bits::Float64 = clamp(
        joint_entropy_bits - grid_entropy_bits,
        0.0,
        magnitude_entropy_bits,
    )

    histogram_grid_symbols::Vector{Int} = Int[]
    histogram_grid_labels::Vector{Any} = Any[]
    histogram_observation_counts::Vector{Int} = Int[]
    histogram_entropy_bits::Vector{Float64} = Float64[]
    histogram_offsets::Vector{Int} = Int[1]
    histogram_magnitude_symbols::Vector{Int} = Int[]
    histogram_counts::Vector{Int} = Int[]
    for grid_symbol::Int in sort!(collect(keys(cell_magnitude_counts)))
        symbol_counts::Dict{Int,Int} = cell_magnitude_counts[grid_symbol]
        symbols::Vector{Int} = sort!(collect(keys(symbol_counts)))
        counts::Vector{Int} = Int[symbol_counts[symbol] for symbol::Int in symbols]
        cell_observation_count::Int = sum(counts)
        append!(histogram_magnitude_symbols, symbols)
        append!(histogram_counts, counts)
        push!(histogram_grid_symbols, grid_symbol)
        push!(histogram_grid_labels, cell_labels[grid_symbol])
        push!(histogram_observation_counts, cell_observation_count)
        push!(
            histogram_entropy_bits,
            _discrete_entropy_from_counts(symbol_counts, cell_observation_count),
        )
        push!(histogram_offsets, length(histogram_magnitude_symbols) + 1)
    end
    cell_histograms::NamedTuple = (
        storage=:compressed_sparse_rows,
        offset_convention=:one_based_half_open,
        grid_symbols=histogram_grid_symbols,
        grid_labels=histogram_grid_labels,
        observation_counts=histogram_observation_counts,
        entropy_bits=histogram_entropy_bits,
        offsets=histogram_offsets,
        magnitude_symbols=histogram_magnitude_symbols,
        counts=histogram_counts,
    )
    cell_weighted_conditional_entropy_bits::Float64 = sum(
        histogram_observation_counts[cell_index] * histogram_entropy_bits[cell_index]
        for cell_index::Int in eachindex(histogram_grid_symbols)
    ) / valid_observation_count
    magnitude_conditional_entropy_bits::Float64 = clamp(
        cell_weighted_conditional_entropy_bits,
        0.0,
        magnitude_entropy_bits,
    )
    entropy_identity_tolerance::Float64 = max(
        1.0e-12,
        16.0 * valid_observation_count * eps(max(
            magnitude_entropy_bits,
            grid_entropy_bits,
            joint_entropy_bits,
            1.0,
        )),
    )
    cell_entropy_matches_joint_definition::Bool = isapprox(
        magnitude_conditional_entropy_bits,
        joint_difference_magnitude_conditional_entropy_bits;
        atol=entropy_identity_tolerance,
        rtol=16.0 * eps(Float64),
    )
    cell_entropy_matches_joint_definition || throw(ArgumentError(
        "Sparse cell histograms do not reproduce magnitude conditional entropy!",
    ))
    mutual_information_bits::Float64 = clamp(
        magnitude_entropy_bits - magnitude_conditional_entropy_bits,
        0.0,
        min(magnitude_entropy_bits, grid_entropy_bits),
    )
    normalized_magnitude_heterogeneity::Float64 = magnitude_entropy_bits > 0.0 ?
        clamp(magnitude_conditional_entropy_bits / magnitude_entropy_bits, 0.0, 1.0) : NaN
    magnitude_retention_fraction::Float64 = magnitude_entropy_bits > 0.0 ?
        clamp(mutual_information_bits / magnitude_entropy_bits, 0.0, 1.0) : NaN
    minimum_magnitude_symbol::Int = minimum(keys(magnitude_counts))
    maximum_magnitude_symbol::Int = maximum(keys(magnitude_counts))
    magnitude_symbol_span::BigInt =
        BigInt(maximum_magnitude_symbol) - BigInt(minimum_magnitude_symbol) + 1
    source_fixed_bits_per_symbol::Int = magnitude_symbol_span > 1 ?
        ndigits(magnitude_symbol_span - 1; base=2) : 0

    reconstruction_information::Vector{NamedTuple} = NamedTuple[]
    for method::Symbol in reconstruction_methods
        push!(
            reconstruction_information,
            _magnitude_reconstruction_information(
                magnitude_values,
                magnitude_symbols,
                label_symbols,
                cell_values,
                method,
                residual_coding,
                precision_value,
                origin_value,
                source_fixed_bits_per_symbol,
                magnitude_conditional_entropy_bits,
            ),
        )
    end
    occupied_grid_cell_count::Int = length(grid_counts)
    singleton_grid_cell_count::Int = count(==(1), values(grid_counts))
    return (
        estimator=:empirical_plugin,
        interpretation=:magnitude_mark_aggregation,
        observation_count=observation_count,
        valid_observation_count=valid_observation_count,
        excluded_observation_count=observation_count - valid_observation_count,
        occupied_grid_cell_count=occupied_grid_cell_count,
        grid_state_count=occupied_grid_cell_count,
        singleton_grid_cell_count=singleton_grid_cell_count,
        singleton_grid_cell_fraction=singleton_grid_cell_count / occupied_grid_cell_count,
        singleton_observation_count=singleton_grid_cell_count,
        singleton_observation_fraction=singleton_grid_cell_count / valid_observation_count,
        grid_feature_names=grid_label_data.feature_names,
        grid_label_representation=grid_label_data.representation,
        magnitude_precision=precision_value,
        magnitude_origin=origin_value,
        reconstructions=Tuple(reconstruction_methods),
        residual_coding=residual_coding,
        magnitude_quantization=(
            method=:origin_anchored_floor_lattice,
            precision=precision_value,
            origin=origin_value,
            interval_convention=:left_closed_right_open,
        ),
        magnitude_symbol_count=length(magnitude_counts),
        magnitude_symbol_minimum=minimum_magnitude_symbol,
        magnitude_symbol_maximum=maximum_magnitude_symbol,
        magnitude_symbol_span=magnitude_symbol_span,
        fixed_bits_per_magnitude_symbol=source_fixed_bits_per_symbol,
        magnitude_entropy_bits=magnitude_entropy_bits,
        grid_entropy_bits=grid_entropy_bits,
        joint_entropy_bits=joint_entropy_bits,
        mutual_information_bits=mutual_information_bits,
        magnitude_conditional_entropy_bits=magnitude_conditional_entropy_bits,
        joint_difference_magnitude_conditional_entropy_bits=
            joint_difference_magnitude_conditional_entropy_bits,
        conditional_entropy_diagnostic_difference_bits=
            joint_difference_magnitude_conditional_entropy_bits -
            magnitude_conditional_entropy_bits,
        entropy_identity_tolerance=entropy_identity_tolerance,
        magnitude_heterogeneity_bits=magnitude_conditional_entropy_bits,
        ideal_conditional_mark_coding_bits=
            valid_observation_count * magnitude_conditional_entropy_bits,
        cell_weighted_conditional_entropy_bits=
            cell_weighted_conditional_entropy_bits,
        cell_entropy_matches_joint_definition=cell_entropy_matches_joint_definition,
        normalized_magnitude_heterogeneity=normalized_magnitude_heterogeneity,
        magnitude_location_dependence_fraction=magnitude_retention_fraction,
        magnitude_retention_fraction=magnitude_retention_fraction,
        magnitude_aggregation_loss_fraction=normalized_magnitude_heterogeneity,
        effective_magnitude_states=exp2(magnitude_entropy_bits),
        effective_magnitudes_per_grid_cell=exp2(magnitude_conditional_entropy_bits),
        fraction_undefined_when_constant_magnitude=iszero(magnitude_entropy_bits),
        magnitude_scale_note=
            :physical_residuals_use_supplied_magnitude_units_which_may_be_logarithmic,
        magnitude_location_dependence_interpretation=
            :fraction_of_quantized_magnitude_entropy_associated_with_grid_labels,
        magnitude_retention_fraction_interpretation=
            :compatibility_alias_not_reconstruction_retention,
        magnitude_aggregation_loss_fraction_interpretation=
            :normalized_quantized_heterogeneity_not_physical_or_total_catalog_loss,
        magnitude_aggregation_loss_reconstruction_independent=true,
        empirical_estimator_caveat=
            :plugin_entropy_can_be_optimistic_for_sparse_singleton_heavy_samples,
        conditional_mark_coding_overhead_included=false,
        cell_histograms=cell_histograms,
        reconstruction_information=reconstruction_information,
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
    y_label="Grid resolution 2", annotation=:percent, title_extra="")

Plot the fraction of empirical raw-state distinguishability retained or merged
across a two-dimensional resolution sweep. Matrix rows correspond to `y_steps`
and columns correspond to `x_steps`. `quantity=:lost` uses green for low merging
loss and red for high merging loss. `annotation=:comparison` makes the raw/grid
comparison explicit in every cell by showing the selected percentage, its bits
out of the raw baseline, and the effective ambiguity caused by merging.
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
    annotation::Symbol=:percent,
    title_extra::AbstractString="",
)::Gadfly.Plot
    size(comparisons, 2) == length(x_steps) ||
        throw(DimensionMismatch("Heatmap columns must match the x-axis resolution labels!"))
    size(comparisons, 1) == length(y_steps) ||
        throw(DimensionMismatch("Heatmap rows must match the y-axis resolution labels!"))
    isempty(comparisons) && throw(ArgumentError("Raw-data grid heatmap comparisons must not be empty!"))
    quantity in (:retained, :lost) ||
        throw(ArgumentError("Raw-data heatmap quantity must be :retained or :lost!"))
    annotation in (:percent, :comparison) ||
        throw(ArgumentError("Raw-data heatmap annotation must be :percent or :comparison!"))
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
            raw_bits::Float64 = baseline == :states ?
                Float64(comparison.raw_entropy_bits) : Float64(comparison.record_entropy_bits)
            retained_bits::Float64 = baseline == :states ?
                Float64(comparison.retained_information_bits) :
                Float64(comparison.record_retained_information_bits)
            lost_bits::Float64 = baseline == :states ?
                Float64(comparison.lost_information_bits) :
                Float64(comparison.record_lost_information_bits)
            effective_ambiguity::Float64 = baseline == :states ?
                Float64(comparison.effective_ambiguity) :
                Float64(comparison.effective_record_ambiguity)
            plotted_fraction::Float64 = quantity == :retained ? retained_fraction : lost_fraction
            plotted_bits::Float64 = quantity == :retained ? retained_bits : lost_bits
            mapping_is_deterministic::Bool = Bool(comparison.mapping_is_deterministic)
            mapping_warning_present = mapping_warning_present || !mapping_is_deterministic
            mapping_suffix::String = mapping_is_deterministic ? "" : "*"
            percentage_label::String =
                "$(round(100.0 * plotted_fraction; digits=1))%$(mapping_suffix)"
            ambiguity_label::String = effective_ambiguity < 1000.0 ?
                string(round(effective_ambiguity; digits=2)) :
                string(round(effective_ambiguity; sigdigits=3))
            comparison_label::String = quantity == :lost ?
                "$(percentage_label) lost\n$(round(plotted_bits; digits=2)) of $(round(raw_bits; digits=2)) bits\n$(ambiguity_label)x ambiguity" :
                "$(percentage_label) retained\n$(round(plotted_bits; digits=2)) of $(round(raw_bits; digits=2)) bits\n$(ambiguity_label)x ambiguity"
            push!(x_values, x_labels[x_index])
            push!(y_values, y_labels[y_index])
            push!(fraction_values, plotted_fraction)
            push!(
                fraction_labels,
                annotation == :comparison ? comparison_label : percentage_label,
            )
        end
    end
    baseline_description::String = baseline == :states ? "raw states" : "raw records"
    baseline_symbol::String = baseline == :states ? "X" : "R"
    distinguishability_description::String = baseline == :states ? "raw-state" : "record"
    quantity_description::String = quantity == :retained ?
        "retained by grid labels" : "lost by grid merging"
    color_key_title::String = quantity == :retained ?
        "I($(baseline_symbol); G) / H($(baseline_symbol))" :
        "H($(baseline_symbol) | G) / H($(baseline_symbol))"
    color_map::Function = quantity == :retained ?
        Gadfly.Scale.lab_gradient("#e74c3c", "#f1c40f", "#16a085") :
        Gadfly.Scale.lab_gradient("#16a085", "#f1c40f", "#e74c3c")
    mapping_warning::String = mapping_warning_present ?
        " (* observed-state mapping conflict)" : ""
    title::String =
        "Observed $(distinguishability_description) distinguishability $(quantity_description): " *
        "$(baseline_description)$(mapping_warning)$(title_extra)"
    if annotation == :comparison
        reference_comparison::NamedTuple = first(flattened_comparisons)
        raw_baseline_bits::Float64 = baseline == :states ?
            Float64(reference_comparison.raw_entropy_bits) :
            Float64(reference_comparison.record_entropy_bits)
        draw_description::String = Bool(reference_comparison.weighted) ?
            "weighted draw" : "observation"
        title *=
            "\nRaw baseline H($(baseline_symbol))=$(round(raw_baseline_bits; digits=2)) bits per $(draw_description); " *
            "each cell compares the grid with that same baseline"
    end
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
        Gadfly.Guide.xticks(; orientation=:horizontal),
        Gadfly.Guide.xlabel(x_label),
        Gadfly.Guide.ylabel(y_label),
        Gadfly.Guide.title(title),
        Gadfly.Theme(; key_position=:right, background_color="white"),
    )
    if filename != ""
        @info("Saving raw-data grid-information heatmap to file: $filename")
        if annotation == :comparison
            Mads.plotfileformat(rawdata_heatmap, filename, 12Gadfly.inch, 7Gadfly.inch)
        else
            Mads.plotfileformat(rawdata_heatmap, filename, 10Gadfly.inch, 6Gadfly.inch)
        end
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
    include_depth::Bool = all(
        haskey(information, :dimension_metadata) &&
        :depth in information.dimension_metadata.roles
        for information::NamedTuple in information_steps
    )
    metric_labels::Vector{String} = [
        "Value entropy",
        "Spatial dependence",
        "Spatial coherence",
        "Temporal dependence",
        "Temporal coherence",
        "Residual coding savings"
    ]
    metric_colors::Vector{String} = ["#1f77b4", "#ff7f0e", "#2ca02c", "#8c564b", "#d62728", "#17becf"]
    if include_depth
        insert!(metric_labels, 4, "Depth dependence")
        insert!(metric_labels, 5, "Depth coherence")
        insert!(metric_colors, 4, "#bcbd22")
        insert!(metric_colors, 5, "#7f7f7f")
    end
    if include_spectral
        spectral_index::Int = include_depth ? 8 : 6
        insert!(metric_labels, spectral_index, "Spectral compactness")
        insert!(metric_colors, spectral_index, "#9467bd")
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
        if include_depth
            insert!(metric_values, 4, clamp(Float64(information.depth_dependence), 0.0, 1.0))
            insert!(metric_values, 5, clamp(1.0 - Float64(information.depth_variation), 0.0, 1.0))
        end
        if include_spectral
            current_spectral_index::Int = include_depth ? 8 : 6
            insert!(metric_values, current_spectral_index, _spectral_compactness(information))
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
