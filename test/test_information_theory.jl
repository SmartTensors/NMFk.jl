import Test
import Random
import Gadfly
import Dates
import Statistics

Test.@testset "structure-aware tensor information" begin
    random_generator::Random.MersenneTwister = Random.MersenneTwister(41)
    structured_frame::Matrix{Float64} = repeat(reshape(collect(0.0:1.0:7.0), 8, 1), 1, 8)
    random_frame::Matrix{Float64} = reshape(Random.randperm(random_generator, 64), 8, 8) ./ 8.0
    structured::Array{Float64, 3} = cat(structured_frame, structured_frame, structured_frame; dims=3)
    randomized::Array{Float64, 3} = cat(random_frame, Random.rand(random_generator, 8, 8) .* 8.0, Random.rand(random_generator, 8, 8) .* 8.0; dims=3)

    structured_flat::NamedTuple = NMFk.tensor_information(structured)
    permutation::Vector{Int} = Random.randperm(random_generator, length(structured))
    permuted_tensor::Array{Float64, 3} = reshape(structured[permutation], size(structured))
    permuted_flat::NamedTuple = NMFk.tensor_information(permuted_tensor)
    Test.@test structured_flat.entropy_bits ≈ permuted_flat.entropy_bits

    structured_info::NamedTuple = NMFk.structure_information(structured; bins=8, temporal_dim=3)
    randomized_info::NamedTuple = NMFk.structure_information(randomized; bins=8, temporal_dim=3)
    Test.@test structured_info.spatial_dependence > randomized_info.spatial_dependence
    Test.@test structured_info.spatial_variation < randomized_info.spatial_variation
    Test.@test structured_info.temporal_variation == 0.0
    Test.@test structured_info.temporal_dependence == 1.0
    Test.@test structured_info.temporal_predictive_gain_bits > randomized_info.temporal_predictive_gain_bits
    Test.@test structured_info.spectral_information[1].effective_rank < randomized_info.spectral_information[1].effective_rank
    Test.@test structured_info.valid_cell_count == length(structured)

    uncoded_info::NamedTuple = NMFk.structure_information(randomized; bins=8, temporal_dim=3, residual_coding=:none)
    shannon_info::NamedTuple = NMFk.structure_information(randomized; bins=8, temporal_dim=3, residual_coding=:shannon)
    huffman_info::NamedTuple = NMFk.structure_information(randomized; bins=8, temporal_dim=3, residual_coding=:huffman)
    uncoded_temporal::NamedTuple = uncoded_info.axis_information[3].residual_coding
    shannon_temporal::NamedTuple = shannon_info.axis_information[3].residual_coding
    huffman_temporal::NamedTuple = huffman_info.axis_information[3].residual_coding
    Test.@test uncoded_temporal.encoded_bits >= huffman_temporal.encoded_bits
    Test.@test huffman_temporal.encoded_bits >= shannon_temporal.encoded_bits
    Test.@test huffman_temporal.compression_ratio >= 1.0
    Test.@test huffman_temporal.coding_efficiency <= 1.0 + eps(Float64)
    no_spectral_info::NamedTuple = NMFk.structure_information(randomized; bins=8, temporal_dim=3, residual_coding=:huffman, compute_spectral=false)
    Test.@test isempty(no_spectral_info.spectral_information)
    Test.@test !no_spectral_info.spectral_computed

    structure_plot::Gadfly.Plot = NMFk.plot_structure_information([structured_info, randomized_info], ["structured", "randomized"])
    structure_range_plot::Gadfly.Plot = NMFk.plot_structure_information([structured_info, randomized_info], ["structured", "randomized"]; normalize=:range)
    structure_plot_without_spectral::Gadfly.Plot = NMFk.plot_structure_information([no_spectral_info, no_spectral_info], ["fine", "coarse"])
    coding_plot::Gadfly.Plot = NMFk.plot_residual_coding([huffman_info, huffman_info], ["fine", "coarse"])
    coding_fixed_plot::Gadfly.Plot = NMFk.plot_residual_coding([huffman_info, randomized_info], ["fine", "coarse"]; normalize=:fixed)
    coding_range_plot::Gadfly.Plot = NMFk.plot_residual_coding([huffman_info, randomized_info], ["fine", "coarse"]; normalize=:range)
    Test.@test structure_plot isa Gadfly.Plot
    Test.@test structure_range_plot isa Gadfly.Plot
    Test.@test structure_plot_without_spectral isa Gadfly.Plot
    Test.@test coding_plot isa Gadfly.Plot
    Test.@test coding_fixed_plot isa Gadfly.Plot
    Test.@test coding_range_plot isa Gadfly.Plot
end

Test.@testset "multidimensional lag information" begin
    default_tensor::Array{Float64,3} = reshape(Float64.(1:24), 3, 4, 2)
    default_information::NamedTuple = NMFk.structure_information(
        default_tensor;
        bins=4,
        temporal_dim=3,
        compute_spectral=false,
    )
    Test.@test default_information.lag_offsets == [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    Test.@test default_information.lag_information == default_information.axis_information
    Test.@test default_information.lag_pair_scope == :both_endpoints_valid
    Test.@test all(
        metric.pair_scope == :both_endpoints_valid
        for metric::NamedTuple in default_information.lag_information
    )
    Test.@test default_information.axis_information[3].role == :temporal
    Test.@test default_information.spatial_dependence == Statistics.mean(
        metric.normalized_mutual_information for metric::NamedTuple in default_information.axis_information[1:2]
    )

    stripe_pattern::Vector{Float64} = Float64[0, 0, 1, 1]
    diagonal_stripes::Matrix{Float64} = [
        stripe_pattern[mod1(row_index - column_index, 4)]
        for row_index::Int = 1:32, column_index::Int = 1:32
    ]
    diagonal_information::NamedTuple = NMFk.structure_information(
        diagonal_stripes;
        bins=2,
        temporal_dim=nothing,
        lag_offsets=[(1, 0), (0, 1), (1, 1), (1, -1)],
        compute_spectral=false,
    )
    positive_diagonal::NamedTuple = only(filter(
        metric::NamedTuple -> metric.offset == (1, 1),
        diagonal_information.lag_information,
    ))
    negative_diagonal::NamedTuple = only(filter(
        metric::NamedTuple -> metric.offset == (1, -1),
        diagonal_information.lag_information,
    ))
    axial_metric::NamedTuple = only(filter(
        metric::NamedTuple -> metric.offset == (1, 0),
        diagonal_information.lag_information,
    ))
    Test.@test positive_diagonal.axis === nothing
    Test.@test positive_diagonal.normalized_mutual_information ≈ 1.0
    Test.@test positive_diagonal.mean_normalized_difference == 0.0
    Test.@test negative_diagonal.normalized_mutual_information ≈ 1.0
    Test.@test negative_diagonal.mean_normalized_difference == 1.0
    Test.@test axial_metric.normalized_mutual_information < 0.01

    sparse_counts::Vector{Float64} = Float64[0, 1, 6, 7, 100]
    sparse_count_mask::BitVector = trues(length(sparse_counts))
    linear_symbols::Vector{Int}, linear_mask::BitVector = NMFk._quantize_tensor(
        sparse_counts,
        sparse_count_mask,
        16,
        :linear,
    )
    zero_preserving_symbols::Vector{Int}, zero_preserving_mask::BitVector =
        NMFk._quantize_tensor(
            sparse_counts,
            sparse_count_mask,
            16,
            :zero_preserving,
        )
    Test.@test linear_mask == zero_preserving_mask
    Test.@test linear_symbols[1] == linear_symbols[2]
    Test.@test zero_preserving_symbols[1] == 1
    Test.@test all(symbol::Int -> symbol >= 2, zero_preserving_symbols[2:end])
    all_zero_symbols::Vector{Int}, all_zero_mask::BitVector = NMFk._quantize_tensor(
        zeros(Float64, 4),
        trues(4),
        16,
        :zero_preserving,
    )
    Test.@test all_zero_mask == trues(4)
    Test.@test all(==(1), all_zero_symbols)
    constant_positive_symbols::Vector{Int}, constant_positive_mask::BitVector =
        NMFk._quantize_tensor(
            Float64[0, 5, 5],
            trues(3),
            16,
            :zero_preserving,
        )
    Test.@test constant_positive_mask == trues(3)
    Test.@test constant_positive_symbols == Int[1, 2, 2]
    masked_signed_symbols::Vector{Int}, masked_signed_mask::BitVector = NMFk._quantize_tensor(
        Float64[0, -1, 10],
        BitVector([true, false, true]),
        16,
        :zero_preserving,
    )
    Test.@test masked_signed_mask == BitVector([true, false, true])
    Test.@test masked_signed_symbols == Int[1, 0, 2]
    Test.@test_throws ArgumentError NMFk._quantize_tensor(
        Float64[-1, 0, 10],
        trues(3),
        16,
        :zero_preserving,
    )
    zero_preserving_information::NamedTuple = NMFk.structure_information(
        sparse_counts;
        temporal_dim=1,
        quantization=:zero_preserving,
        compute_spectral=false,
    )
    Test.@test zero_preserving_information.quantization == :zero_preserving

    canonical_information::NamedTuple = NMFk.structure_information(
        diagonal_stripes;
        bins=2,
        temporal_dim=nothing,
        lag_offsets=[(-1, -1)],
        compute_spectral=false,
    )
    Test.@test canonical_information.lag_offsets == [(1, 1)]
    Test.@test_throws ArgumentError NMFk.structure_information(
        diagonal_stripes;
        lag_offsets=[(1, 1), (-1, -1)],
        compute_spectral=false,
    )

    directed_series::Vector{Float64} = Float64[0, 0, 1, 2, 2, 2, 1]
    directed_information::NamedTuple = NMFk.structure_information(
        directed_series;
        bins=3,
        temporal_dim=1,
        lag_offsets=[(1,), (-1,)],
        lag_sign=:directed,
        compute_spectral=false,
    )
    forward_metric::NamedTuple = directed_information.lag_information[1]
    reverse_metric::NamedTuple = directed_information.lag_information[2]
    Test.@test forward_metric.mutual_information_bits ≈ reverse_metric.mutual_information_bits
    Test.@test forward_metric.left_entropy_bits ≈ reverse_metric.right_entropy_bits
    Test.@test forward_metric.right_entropy_bits ≈ reverse_metric.left_entropy_bits
    Test.@test forward_metric.conditional_entropy_bits ≈ reverse_metric.reverse_conditional_entropy_bits
    Test.@test forward_metric.residual_entropy_bits ≈ reverse_metric.residual_entropy_bits
    Test.@test forward_metric.residual_coding.encoded_bits ≈ reverse_metric.residual_coding.encoded_bits
    integer_temporal_dimension_information::NamedTuple = NMFk.structure_information(
        directed_series;
        temporal_dim=Int32(1),
        compute_spectral=false,
    )
    Test.@test integer_temporal_dimension_information.axis_information[1].role == :temporal

    four_dimensional_tensor::Array{Float64,4} = reshape(Float64.(mod.(0:239, 7)), 4, 5, 3, 4)
    four_dimensional_information::NamedTuple = NMFk.structure_information(
        four_dimensional_tensor;
        bins=7,
        temporal_dim=4,
        lag_offsets=[(0, 0, 1, 0), (0, 0, 0, -2), (1, -1, 0, 1), (-1, 0, 0, 1), (1, 0, 1, 0)],
        dimension_names=(:latitude, :longitude, :depth, :time),
        dimension_steps=(0.1, 0.2, 2.0, Dates.Day(1)),
        dimension_units=("deg", "deg", "km", "day"),
        dimension_roles=(:horizontal, :horizontal, :depth, :temporal),
        compute_spectral=false,
    )
    Test.@test four_dimensional_information.lag_offsets[2] == (0, 0, 0, 2)
    Test.@test four_dimensional_information.lag_information[1].role == :depth
    Test.@test four_dimensional_information.lag_information[2].role == :temporal
    Test.@test four_dimensional_information.lag_information[3].role == :spatiotemporal
    Test.@test four_dimensional_information.lag_information[4].role == :spatiotemporal
    Test.@test four_dimensional_information.lag_information[5].role == :spatial_depth
    Test.@test four_dimensional_information.lag_information[1].active_dimension_roles == (:depth,)
    Test.@test four_dimensional_information.lag_information[3].coordinate_offset ==
        (0.1, -0.2, 0.0, Dates.Day(1))
    Test.@test four_dimensional_information.lag_information[3].grid_index_norm ≈ sqrt(3.0)
    Test.@test occursin("latitude=0.1 deg", four_dimensional_information.lag_information[3].display_label)
    Test.@test four_dimensional_information.lag_information[2].display_label == "time=2 days"
    Test.@test four_dimensional_information.dimension_metadata.roles ==
        (:horizontal, :horizontal, :depth, :temporal)
    Test.@test four_dimensional_information.depth_dependence ==
        four_dimensional_information.axis_information[3].normalized_mutual_information
    Test.@test four_dimensional_information.spatial_dependence == Statistics.mean(
        metric.normalized_mutual_information
        for metric::NamedTuple in four_dimensional_information.axis_information[1:2]
    )
    lag_plot::Gadfly.Plot = NMFk.plot_lag_information(four_dimensional_information)
    lag_range_plot::Gadfly.Plot = NMFk.plot_lag_information(
        four_dimensional_information;
        normalize=:range,
    )
    Test.@test lag_plot isa Gadfly.Plot
    Test.@test lag_range_plot isa Gadfly.Plot
    depth_structure_plot::Gadfly.Plot = NMFk.plot_structure_information(
        [four_dimensional_information, four_dimensional_information],
        ["fine", "coarse"],
    )
    Test.@test depth_structure_plot isa Gadfly.Plot
    Test.@test_throws ArgumentError NMFk.plot_lag_information(
        four_dimensional_information;
        normalize=:invalid,
    )

    masked_tensor::Matrix{Float64} = reshape(Float64.(1:12), 3, 4)
    masked_cells::BitArray{2} = trues(size(masked_tensor))
    masked_cells[1, 3] = false
    masked_information::NamedTuple = NMFk.structure_information(
        masked_tensor;
        valid_mask=masked_cells,
        bins=4,
        temporal_dim=nothing,
        lag_offsets=[(1, -2), (3, 0)],
        compute_spectral=false,
    )
    masked_metric::NamedTuple = masked_information.lag_information[1]
    oversized_metric::NamedTuple = masked_information.lag_information[2]
    Test.@test masked_metric.candidate_pair_count == 4
    Test.@test masked_metric.pair_count == 3
    Test.@test masked_metric.valid_pair_fraction == 0.75
    Test.@test masked_metric.histogram_method == :dense
    Test.@test oversized_metric.candidate_pair_count == 0
    Test.@test oversized_metric.pair_count == 0
    Test.@test !oversized_metric.applicable

    sparse_tensor::Matrix{Float64} = zeros(Float64, 20, 20)
    sparse_tensor[2, 3] = 2.0
    sparse_tensor[17, 18] = 4.0
    sparse_information::NamedTuple = NMFk.structure_information(
        sparse_tensor;
        bins=4,
        temporal_dim=nothing,
        lag_offsets=[(2, -1)],
        residual_coding=:huffman,
        compute_spectral=false,
    )
    sparse_metric::NamedTuple = only(sparse_information.lag_information)
    sparse_quantized::Array{Int,2}, sparse_mask::BitArray{2} =
        NMFk._quantize_tensor(sparse_tensor, trues(size(sparse_tensor)), 4)
    dense_reference::NamedTuple = NMFk._lag_information(
        sparse_quantized,
        sparse_mask,
        (2, -1),
        :spatial,
        4,
        :huffman,
        (:dim1, :dim2),
        (1, 1),
        ("bins", "bins"),
        (:spatial, :spatial),
        nothing,
        nothing,
    )
    Test.@test sparse_metric.histogram_method == :sparse_baseline
    Test.@test sparse_metric.pair_count == dense_reference.pair_count
    Test.@test sparse_metric.joint_entropy_bits ≈ dense_reference.joint_entropy_bits
    Test.@test sparse_metric.residual_entropy_bits ≈ dense_reference.residual_entropy_bits
    Test.@test sparse_metric.residual_coding.encoded_bits == dense_reference.residual_coding.encoded_bits
    Test.@test sparse_metric.mean_normalized_difference ≈ dense_reference.mean_normalized_difference
    Test.@test_throws ArgumentError NMFk.plot_lag_information(
        sparse_information;
        normalize=:range,
    )

    unavailable_information::NamedTuple = NMFk.structure_information(
        masked_tensor;
        bins=4,
        temporal_dim=nothing,
        lag_offsets=[(3, 0)],
        compute_spectral=false,
    )
    Test.@test_throws ArgumentError NMFk.plot_lag_information(unavailable_information)

    Test.@test_throws DimensionMismatch NMFk.structure_information(
        diagonal_stripes;
        lag_offsets=[(1, 0, 0)],
        compute_spectral=false,
    )
    Test.@test_throws ArgumentError NMFk.structure_information(
        diagonal_stripes;
        lag_offsets=[(0, 0)],
        compute_spectral=false,
    )
    Test.@test_throws ArgumentError NMFk.structure_information(
        diagonal_stripes;
        lag_offsets=[(1.0, 0.0)],
        compute_spectral=false,
    )
    Test.@test_throws ArgumentError NMFk.structure_information(
        diagonal_stripes;
        lag_offsets=[(1, 0)],
        lag_sign=:invalid,
        compute_spectral=false,
    )
    Test.@test_throws ArgumentError NMFk.structure_information(
        diagonal_stripes;
        quantization=:invalid,
        compute_spectral=false,
    )
    Test.@test_throws ArgumentError NMFk.structure_information(
        diagonal_stripes;
        lag_offsets=Tuple[],
        compute_spectral=false,
    )
    Test.@test_throws DimensionMismatch NMFk.structure_information(
        diagonal_stripes;
        dimension_names=(:latitude,),
        compute_spectral=false,
    )
    Test.@test_throws ArgumentError NMFk.structure_information(
        diagonal_stripes;
        dimension_steps=(0.1, -0.2),
        compute_spectral=false,
    )
    Test.@test_throws DimensionMismatch NMFk.structure_information(
        diagonal_stripes;
        dimension_roles=(:horizontal,),
        compute_spectral=false,
    )
    Test.@test_throws ArgumentError NMFk.structure_information(
        diagonal_stripes;
        temporal_dim=2,
        dimension_roles=(:horizontal, :depth),
        compute_spectral=false,
    )
    Test.@test_throws ArgumentError NMFk.structure_information(
        diagonal_stripes;
        temporal_dim=nothing,
        dimension_roles=(:horizontal, :channel),
        compute_spectral=false,
    )
    extreme_offset_information::NamedTuple = NMFk.structure_information(
        diagonal_stripes;
        lag_offsets=[(-typemax(Int), 0)],
        lag_sign=:directed,
        compute_spectral=false,
    )
    Test.@test extreme_offset_information.lag_information[1].candidate_pair_count == 0
    Test.@test !extreme_offset_information.lag_information[1].applicable
end

Test.@testset "magnitude aggregation information" begin
    magnitudes::Vector{Float64} = Float64[1.0, 2.0, 1.0, 2.0]
    grid_labels::Vector{Int} = Int[1, 1, 2, 2]
    information::NamedTuple = NMFk.magnitude_aggregation_information(
        magnitudes,
        grid_labels;
        magnitude_precision=1.0,
    )
    Test.@test information.magnitude_entropy_bits ≈ 1.0
    Test.@test information.grid_entropy_bits ≈ 1.0
    Test.@test information.joint_entropy_bits ≈ 2.0
    Test.@test information.mutual_information_bits ≈ 0.0 atol=eps(Float64)
    Test.@test information.magnitude_conditional_entropy_bits ≈ 1.0
    Test.@test information.magnitude_heterogeneity_bits ≈ 1.0
    Test.@test information.normalized_magnitude_heterogeneity ≈ 1.0
    Test.@test information.magnitude_location_dependence_fraction ≈ 0.0 atol=eps(Float64)
    Test.@test information.magnitude_retention_fraction ≈ 0.0 atol=eps(Float64)
    Test.@test information.magnitude_location_dependence_fraction ===
        information.magnitude_retention_fraction
    Test.@test information.magnitude_retention_fraction_interpretation ==
        :compatibility_alias_not_reconstruction_retention
    Test.@test information.magnitude_aggregation_loss_fraction_interpretation ==
        :normalized_quantized_heterogeneity_not_physical_or_total_catalog_loss
    Test.@test information.empirical_estimator_caveat ==
        :plugin_entropy_can_be_optimistic_for_sparse_singleton_heavy_samples
    Test.@test information.normalized_magnitude_heterogeneity +
        information.magnitude_retention_fraction ≈ 1.0
    Test.@test information.effective_magnitudes_per_grid_cell ≈ 2.0

    histograms::NamedTuple = information.cell_histograms
    Test.@test histograms.storage == :compressed_sparse_rows
    Test.@test histograms.offsets == Int[1, 3, 5]
    Test.@test histograms.magnitude_symbols == Int[1, 2, 1, 2]
    Test.@test histograms.counts == ones(Int, 4)
    weighted_cell_entropy::Float64 = sum(
        histograms.observation_counts[cell_index] * histograms.entropy_bits[cell_index]
        for cell_index::Int in eachindex(histograms.grid_symbols)
    ) / information.valid_observation_count
    Test.@test weighted_cell_entropy ≈ information.magnitude_conditional_entropy_bits

    maximum_information::NamedTuple = only(filter(
        result::NamedTuple -> result.method == :maximum,
        information.reconstruction_information,
    ))
    Test.@test maximum_information.residual_transform_exact
    Test.@test !maximum_information.ordered_residual_stream_retained
    Test.@test maximum_information.residual_symbols == Int[-1, 0]
    Test.@test maximum_information.residual_counts == Int[2, 2]
    Test.@test maximum_information.residual_entropy_bits ≈ 1.0
    Test.@test maximum_information.conditional_residual_entropy_bits ≈
        information.magnitude_conditional_entropy_bits
    Test.@test maximum_information.conditional_entropy_matches_magnitude_heterogeneity
    Test.@test maximum_information.pooled_residual_entropy_bits + eps(Float64) >=
        maximum_information.conditional_residual_entropy_bits
    Test.@test maximum_information.shannon_encoded_bits ≈ 4.0
    Test.@test maximum_information.huffman_encoded_bits == 4
    Test.@test maximum_information.encoded_bits ≈ 4.0
    Test.@test maximum_information.fixed_width_bits == 4
    Test.@test maximum_information.fixed_bits_per_residual ≈ 1.0
    Test.@test maximum_information.mae ≈ 0.5
    Test.@test maximum_information.rmse ≈ sqrt(0.5)
    Test.@test maximum_information.bias ≈ -0.5
    Test.@test maximum_information.cell_predictors.predictor_symbols == Int[2, 2]
    Test.@test !maximum_information.predictor_overhead_included
    Test.@test !maximum_information.grid_label_overhead_included
    Test.@test !maximum_information.codebook_overhead_included

    asymmetric_information::NamedTuple = NMFk.magnitude_aggregation_information(
        Float64[1.0, 2.0, 9.0],
        ones(Int, 3);
        magnitude_precision=1.0,
        residual_coding=:shannon,
    )
    asymmetric_by_method::Dict{Symbol,NamedTuple} = Dict(
        result.method => result
        for result::NamedTuple in asymmetric_information.reconstruction_information
    )
    Test.@test asymmetric_by_method[:maximum].mae ≈ 5.0
    Test.@test asymmetric_by_method[:mean].mae ≈ 10 / 3
    Test.@test asymmetric_by_method[:median].mae ≈ 8 / 3
    Test.@test asymmetric_by_method[:mean].bias ≈ 0.0 atol=eps(Float64)
    Test.@test asymmetric_by_method[:mean].selected_coding == :shannon
    Test.@test asymmetric_by_method[:mean].encoded_bits ≈
        asymmetric_by_method[:mean].shannon_encoded_bits

    half_lattice_information::NamedTuple = NMFk.magnitude_aggregation_information(
        Float64[1.1, 1.2],
        ones(Int, 2);
        magnitude_precision=0.1,
        reconstructions=(:mean,),
    )
    half_lattice_mean::NamedTuple = only(half_lattice_information.reconstruction_information)
    Test.@test half_lattice_mean.cell_predictors.predictor_symbols == Int[12]
    reconstructed_symbols::Vector{Int} = Int[]
    predictor_symbol::Int = only(half_lattice_mean.cell_predictors.predictor_symbols)
    for residual_index::Int in eachindex(half_lattice_mean.residual_symbols)
        residual_symbol::Int = half_lattice_mean.residual_symbols[residual_index]
        residual_count::Int = half_lattice_mean.residual_counts[residual_index]
        append!(reconstructed_symbols, fill(predictor_symbol + residual_symbol, residual_count))
    end
    Test.@test sort!(reconstructed_symbols) == Int[11, 12]

    translation_magnitudes::Vector{Float64} =
        Float64[-0.2, 0.0, 0.1, 0.4, 0.4, 1.2]
    translation_shift::Float64 = 7.3
    translation_grid_labels::Vector{Int} = Int[1, 1, 1, 2, 2, 2]
    origin_zero_information::NamedTuple = NMFk.magnitude_aggregation_information(
        translation_magnitudes,
        translation_grid_labels;
        magnitude_precision=0.1,
        magnitude_origin=0.0,
    )
    shifted_origin_information::NamedTuple = NMFk.magnitude_aggregation_information(
        translation_magnitudes .+ translation_shift,
        translation_grid_labels;
        magnitude_precision=0.1,
        magnitude_origin=translation_shift,
    )
    Test.@test shifted_origin_information.cell_histograms.magnitude_symbols ==
        origin_zero_information.cell_histograms.magnitude_symbols
    Test.@test shifted_origin_information.cell_histograms.counts ==
        origin_zero_information.cell_histograms.counts
    Test.@test shifted_origin_information.magnitude_entropy_bits ==
        origin_zero_information.magnitude_entropy_bits
    Test.@test shifted_origin_information.magnitude_conditional_entropy_bits ==
        origin_zero_information.magnitude_conditional_entropy_bits
    for reconstruction_index::Int in eachindex(
        origin_zero_information.reconstruction_information,
    )
        origin_reconstruction::NamedTuple =
            origin_zero_information.reconstruction_information[reconstruction_index]
        shifted_reconstruction::NamedTuple =
            shifted_origin_information.reconstruction_information[reconstruction_index]
        Test.@test shifted_reconstruction.method == origin_reconstruction.method
        Test.@test shifted_reconstruction.residual_symbols ==
            origin_reconstruction.residual_symbols
        Test.@test shifted_reconstruction.residual_counts == origin_reconstruction.residual_counts
        Test.@test shifted_reconstruction.cell_predictors.predictor_symbols ==
            origin_reconstruction.cell_predictors.predictor_symbols
        Test.@test shifted_reconstruction.mae ≈ origin_reconstruction.mae atol=1.0e-14
        Test.@test shifted_reconstruction.rmse ≈ origin_reconstruction.rmse atol=1.0e-14
        Test.@test shifted_reconstruction.bias ≈ origin_reconstruction.bias atol=1.0e-14
    end

    shifted_half_lattice_information::NamedTuple =
        NMFk.magnitude_aggregation_information(
            Float64[0.1, 0.2] .+ translation_shift,
            ones(Int, 2);
            magnitude_precision=0.1,
            magnitude_origin=translation_shift,
            reconstructions=(:mean,),
        )
    shifted_half_lattice_mean::NamedTuple =
        only(shifted_half_lattice_information.reconstruction_information)
    Test.@test shifted_half_lattice_mean.cell_predictors.predictor_symbols == Int[2]
    Test.@test shifted_half_lattice_mean.residual_symbols == half_lattice_mean.residual_symbols
    Test.@test shifted_half_lattice_mean.residual_counts == half_lattice_mean.residual_counts

    coordinate_information::NamedTuple = NMFk.magnitude_aggregation_information(
        Float64[1.0, 1.2, NaN, 2.0],
        (
            latitude=Int[1, 1, 2, 2],
            longitude=Int[3, 3, 4, 4],
            time=Int[5, 5, 6, 6],
        );
        magnitude_precision=0.1,
        valid_mask=BitVector([true, true, true, false]),
        reconstructions=(:median,),
    )
    Test.@test coordinate_information.valid_observation_count == 2
    Test.@test coordinate_information.excluded_observation_count == 2
    Test.@test coordinate_information.occupied_grid_cell_count == 1
    Test.@test coordinate_information.singleton_grid_cell_count == 0
    Test.@test coordinate_information.grid_feature_names == (:latitude, :longitude, :time)
    Test.@test coordinate_information.grid_label_representation == :joint_feature_labels
    Test.@test length(coordinate_information.reconstruction_information) == 1

    constant_information::NamedTuple = NMFk.magnitude_aggregation_information(
        fill(3.0, 3),
        Int[1, 1, 2];
        magnitude_precision=0.1,
    )
    Test.@test constant_information.magnitude_entropy_bits == 0.0
    Test.@test isnan(constant_information.normalized_magnitude_heterogeneity)
    Test.@test isnan(constant_information.magnitude_location_dependence_fraction)
    Test.@test isnan(constant_information.magnitude_retention_fraction)
    Test.@test constant_information.fraction_undefined_when_constant_magnitude
    Test.@test constant_information.singleton_grid_cell_count == 1
    Test.@test constant_information.singleton_observation_fraction ≈ 1 / 3

    stress_generator::Random.MersenneTwister = Random.MersenneTwister(1907)
    stress_observation_count::Int = 10_000
    stress_magnitudes::Vector{Float64} =
        5.0 .* Random.rand(stress_generator, stress_observation_count)
    stress_grid_assignments::NamedTuple = (
        latitude=Random.rand(stress_generator, 1:80, stress_observation_count),
        longitude=Random.rand(stress_generator, 1:80, stress_observation_count),
    )
    stress_information::NamedTuple = NMFk.magnitude_aggregation_information(
        stress_magnitudes,
        stress_grid_assignments;
        magnitude_precision=0.1,
        reconstructions=(:mean,),
    )
    Test.@test stress_information.valid_observation_count == stress_observation_count
    Test.@test sum(stress_information.cell_histograms.counts) == stress_observation_count
    Test.@test length(stress_information.cell_histograms.offsets) ==
        stress_information.occupied_grid_cell_count + 1
    Test.@test stress_information.cell_entropy_matches_joint_definition
    Test.@test abs(stress_information.conditional_entropy_diagnostic_difference_bits) <=
        stress_information.entropy_identity_tolerance
    stress_mean_information::NamedTuple = only(stress_information.reconstruction_information)
    Test.@test stress_mean_information.conditional_entropy_matches_magnitude_heterogeneity
    Test.@test stress_mean_information.conditional_residual_entropy_bits ≈
        stress_information.magnitude_conditional_entropy_bits atol=
        stress_mean_information.entropy_identity_tolerance

    Test.@test_throws DimensionMismatch NMFk.magnitude_aggregation_information(
        Float64[1.0, 2.0],
        Int[1];
        magnitude_precision=0.1,
    )
    Test.@test_throws ArgumentError NMFk.magnitude_aggregation_information(
        Float64[1.0],
        Int[1];
        magnitude_precision=0.0,
    )
    Test.@test_throws ArgumentError NMFk.magnitude_aggregation_information(
        Float64[1.0],
        Int[1];
        magnitude_precision=0.1,
        reconstructions=(),
    )
    Test.@test_throws ArgumentError NMFk.magnitude_aggregation_information(
        Float64[1.0],
        Int[1];
        magnitude_precision=0.1,
        reconstructions=(:average,),
    )
    Test.@test_throws ArgumentError NMFk.magnitude_aggregation_information(
        Float64[1.0],
        Int[1];
        magnitude_precision=0.1,
        residual_coding=:none,
    )
end

Test.@testset "raw-data information" begin
    independent_data::NamedTuple = (
        x=Float64[0.0, 0.0, 1.0, 1.0],
        y=Float64[0.0, 1.0, 0.0, 1.0],
    )
    independent_information::NamedTuple = NMFk.rawdata_information(
        independent_data;
        precisions=(x=1.0, y=1.0),
    )
    Test.@test independent_information.state_entropy_bits ≈ 2.0
    Test.@test independent_information.state_count == 4
    Test.@test independent_information.effective_state_count ≈ 4.0
    Test.@test all(
        feature.entropy_bits ≈ 1.0 for feature::NamedTuple in independent_information.feature_information
    )

    categorical_information_a::NamedTuple = NMFk.rawdata_information(
        ["A", "B", "A", "B"];
        precision=nothing,
    )
    categorical_information_b::NamedTuple = NMFk.rawdata_information(
        ["X", "Y", "X", "Y"];
        precision=nothing,
    )
    Test.@test categorical_information_a.state_entropy_bits ≈ categorical_information_b.state_entropy_bits
    Test.@test categorical_information_a.state_fingerprint != categorical_information_b.state_fingerprint
    precision_information_a::NamedTuple = NMFk.rawdata_information(
        Float64[0.0, 1.0];
        precision=1.0,
    )
    precision_information_b::NamedTuple = NMFk.rawdata_information(
        Float64[0.0, 2.0];
        precision=2.0,
    )
    Test.@test precision_information_a.state_symbols == precision_information_b.state_symbols
    Test.@test precision_information_a.state_fingerprint != precision_information_b.state_fingerprint

    dependent_information::NamedTuple = NMFk.rawdata_information(
        (x=Float64[0, 0, 1, 1], y=Float64[0, 0, 1, 1], constant=ones(Float64, 4));
        precisions=(x=1.0, y=1.0, constant=1.0),
    )
    Test.@test dependent_information.state_entropy_bits ≈ 1.0
    Test.@test dependent_information.feature_information[3].entropy_bits == 0.0

    coarse_information::NamedTuple = NMFk.rawdata_information(
        (x=Float64[1, 2, 11, 12], y=Float64[101, 102, 101, 102]);
        precisions=(x=10.0, y=10.0),
        origins=(x=0.0, y=0.0),
    )
    Test.@test coarse_information.state_entropy_bits ≈ 1.0
    Test.@test coarse_information.state_count == 2

    decimal_boundary_information::NamedTuple = NMFk.rawdata_information(
        Float64[-0.3, 0.0, 0.3];
        precision=0.1,
    )
    Test.@test decimal_boundary_information.state_count == 3
    Test.@test NMFk._raw_floor_symbol(0.3, 0.1, 0.0) == 3
    Test.@test NMFk._raw_floor_symbol(prevfloat(0.3), 0.1, 0.0) == 2
    Test.@test NMFk._raw_floor_symbol(7.4, 0.1, 7.3) == 1
    Test.@test NMFk._raw_floor_symbol(prevfloat(prevfloat(7.4)), 0.1, 7.3) == 0
    Test.@test NMFk._raw_floor_symbol(1.0000000000000019e15, 2.0, 0.0) == 500000000000000
    raw_translation_values::Vector{Float64} =
        Float64[-0.2, 0.0, 0.1, 0.4, 0.4, 1.2]
    raw_translation_origin::Float64 = 7.3
    raw_origin_zero_information::NamedTuple = NMFk.rawdata_information(
        raw_translation_values;
        precision=0.1,
        origin=0.0,
    )
    raw_shifted_origin_information::NamedTuple = NMFk.rawdata_information(
        raw_translation_values .+ raw_translation_origin;
        precision=0.1,
        origin=raw_translation_origin,
    )
    Test.@test raw_shifted_origin_information.state_symbols ==
        raw_origin_zero_information.state_symbols
    Test.@test raw_shifted_origin_information.state_entropy_bits ==
        raw_origin_zero_information.state_entropy_bits
    large_integer_information::NamedTuple = NMFk.rawdata_information(
        Int64[2^53, 2^53 + 1];
        precision=1,
    )
    Test.@test large_integer_information.state_count == 2
    extreme_integer_information::NamedTuple = NMFk.rawdata_information(
        Int[typemin(Int), typemax(Int)];
        precision=1,
    )
    Test.@test extreme_integer_information.state_entropy_bits ≈ 1.0
    Test.@test extreme_integer_information.feature_information[1].residual_information.pair_count == 1
    Test.@test_throws ArgumentError NMFk.rawdata_information(
        Int[typemin(Int), typemax(Int)];
        precision=1,
        origin=typemax(Int),
    )
    Test.@test_throws ArgumentError NMFk.rawdata_information(
        Float64[1.0e15 + 0.75, 1.0e15 + 0.875];
        precision=1.0,
    )

    date_information::NamedTuple = NMFk.rawdata_information(
        (time=Dates.DateTime[
            Dates.DateTime(2020, 1, 1),
            Dates.DateTime(2020, 1, 1, 0, 0, 1),
            Dates.DateTime(2020, 1, 1, 0, 0, 2),
        ],);
        precisions=(time=Dates.Second(1),),
    )
    Test.@test date_information.state_count == 3
    Test.@test_throws ArgumentError NMFk.rawdata_information(
        (time=Dates.DateTime[Dates.DateTime(2020, 1, 1)],);
        precisions=(time=Dates.Microsecond(1),),
    )
    Test.@test_throws ArgumentError NMFk.rawdata_information(
        (time=Dates.DateTime[Dates.DateTime(2020, 1, 1)],);
        precisions=(time=Dates.Month(1),),
    )

    residual_information::NamedTuple = NMFk.rawdata_information(
        Float64[0, 0, 0, 1, 1, 1];
        precision=1.0,
        residual_coding=:huffman,
    ).feature_information[1].residual_information
    Test.@test residual_information.pair_count == 5
    Test.@test residual_information.residual_entropy_bits ≈ 0.7219280948873623
    Test.@test residual_information.encoded_bits == 5.0
    Test.@test residual_information.fixed_width_bits == 10
    Test.@test residual_information.coding_savings ≈ 0.5

    matrix_data::Matrix{Float64} = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]
    matrix_information::NamedTuple = NMFk.rawdata_information(
        matrix_data;
        precisions=Float64[1.0, 1.0],
        feature_names=Symbol[:x, :y],
    )
    transposed_information::NamedTuple = NMFk.rawdata_information(
        permutedims(matrix_data);
        precisions=Float64[1.0, 1.0],
        feature_names=Symbol[:x, :y],
        observation_dim=2,
    )
    Test.@test matrix_information.state_entropy_bits ≈ independent_information.state_entropy_bits
    Test.@test transposed_information.state_entropy_bits ≈ matrix_information.state_entropy_bits

    masked_information::NamedTuple = NMFk.rawdata_information(
        (x=Float64[0.0, NaN, 1.0],);
        precisions=(x=1.0,),
    )
    Test.@test masked_information.valid_observation_count == 2
    Test.@test masked_information.excluded_observation_count == 1
    Test.@test_throws ArgumentError NMFk.rawdata_information(
        independent_data;
        precisions=(x=0.0, y=1.0),
    )
    Test.@test_throws DimensionMismatch NMFk.rawdata_information(
        independent_data;
        precisions=(x=1.0, y=1.0),
        valid_mask=trues(3),
    )
    Test.@test_throws ArgumentError NMFk.rawdata_information(
        independent_data;
        precisions=(x=1.0, y=1.0),
        sequence_order=Int[1, 1, 3, 4],
    )

    zero_weight_information::NamedTuple = NMFk.rawdata_information(
        Int[1, 2, 100];
        precision=1,
        weights=Float64[1, 1, 0],
    )
    Test.@test zero_weight_information.valid_observation_count == 2
    Test.@test zero_weight_information.state_count == 2
    Test.@test zero_weight_information.feature_information[1].residual_information.pair_count == 1
    overflow_weight_information::NamedTuple = NMFk.rawdata_information(
        Int[1, 2];
        precision=1,
        weights=fill(floatmax(Float64), 2),
    )
    Test.@test overflow_weight_information.state_entropy_bits ≈ 1.0
    Test.@test isfinite(overflow_weight_information.scaled_total_weight)
    masked_weight_information::NamedTuple = NMFk.rawdata_information(
        Int[0, 1];
        precision=1,
        valid_mask=BitVector([false, true]),
        weights=Float64[floatmax(Float64), floatmin(Float64)],
    )
    Test.@test masked_weight_information.valid_observation_count == 1
    Test.@test isnan(masked_weight_information.normalized_state_entropy)
    Test.@test_throws ArgumentError NMFk.rawdata_information(
        Int[1, 2];
        precision=1,
        weights=zeros(Float64, 2),
    )
end

Test.@testset "raw-data versus grid information" begin
    raw_information::NamedTuple = NMFk.rawdata_information(
        Float64.(1:8);
        precision=1.0,
    )
    grid_assignments::Vector{Vector{Int}} = [
        collect(1:8),
        Int[1, 1, 2, 2, 3, 3, 4, 4],
        Int[1, 1, 1, 1, 2, 2, 2, 2],
        ones(Int, 8),
    ]
    expected_retention::Vector{Float64} = Float64[1.0, 2 / 3, 1 / 3, 0.0]
    comparisons::Vector{NamedTuple} = NamedTuple[]
    for comparison_index::Int in eachindex(grid_assignments)
        assignment::Vector{Int} = grid_assignments[comparison_index]
        comparison::NamedTuple = NMFk.compare_rawdata_grid(
            raw_information,
            assignment;
            grid_cell_count=maximum(assignment),
        )
        push!(comparisons, comparison)
        Test.@test comparison.retention_fraction ≈ expected_retention[comparison_index]
        Test.@test comparison.loss_fraction ≈ 1.0 - expected_retention[comparison_index]
        Test.@test comparison.retained_information_bits + comparison.lost_information_bits ≈ 3.0
        Test.@test comparison.mapping_is_deterministic
    end
    Test.@test comparisons[2].lost_information_bits ≈ 1.0
    Test.@test comparisons[3].lost_information_bits ≈ 2.0

    count_information::NamedTuple = NMFk.tensor_information(Float64[2, 2, 2, 2])
    Test.@test comparisons[2].grid_entropy_bits ≈ count_information.entropy_bits

    independent_raw::NamedTuple = NMFk.rawdata_information(
        ["A", "A", "B", "B"];
        precision=nothing,
    )
    independent_grid::NamedTuple = NMFk.compare_rawdata_grid(
        independent_raw,
        Int[1, 2, 1, 2];
        grid_cell_count=2,
    )
    Test.@test independent_grid.raw_entropy_bits ≈ 1.0
    Test.@test independent_grid.grid_entropy_bits ≈ 1.0
    Test.@test independent_grid.retained_information_bits ≈ 0.0 atol=eps(Float64)
    Test.@test independent_grid.retention_fraction ≈ 0.0 atol=eps(Float64)
    Test.@test !independent_grid.mapping_is_deterministic
    Test.@test independent_grid.mapping_uncertainty_bits ≈ 1.0

    constant_raw::NamedTuple = NMFk.rawdata_information(
        ones(Int, 3);
        precision=1,
    )
    constant_comparison::NamedTuple = NMFk.compare_rawdata_grid(
        constant_raw,
        Int[1, 2, 3],
    )
    Test.@test isnan(constant_comparison.retention_fraction)
    Test.@test isnan(constant_comparison.loss_fraction)
    Test.@test constant_comparison.mapping_uncertainty_bits ≈ log2(3)
    Test.@test_throws ArgumentError NMFk.plot_rawdata_grid_information(
        [constant_comparison],
        ["constant"],
    )

    weighted_raw::NamedTuple = NMFk.rawdata_information(
        Int[1, 2, 3];
        precision=1,
        weights=Float64[1, 1, 2],
    )
    weighted_comparison::NamedTuple = NMFk.compare_rawdata_grid(
        weighted_raw,
        Int[1, 1, 2];
        grid_cell_count=2,
    )
    Test.@test weighted_comparison.grid_entropy_bits ≈ 1.0
    Test.@test weighted_comparison.weighted
    weighted_conflict_raw::NamedTuple = NMFk.rawdata_information(
        Int[1, 1];
        precision=1,
        weights=Float64[1.0, 1.0e-18],
    )
    weighted_conflict_comparison::NamedTuple = NMFk.compare_rawdata_grid(
        weighted_conflict_raw,
        Int[1, 2];
        grid_cell_count=2,
    )
    Test.@test weighted_conflict_comparison.mapping_uncertainty_bits > 0.0
    Test.@test !weighted_conflict_comparison.mapping_is_deterministic
    Test.@test weighted_conflict_comparison.mapping_determinism_scope == :observed_support
    Test.@test weighted_conflict_comparison.conflicting_raw_state_count == 1
    Test.@test weighted_conflict_comparison.mapping_conflict_observation_count == 2
    Test.@test weighted_conflict_comparison.mapping_conflict_weight_fraction ≈ 1.0

    inferred_cell_count_comparison::NamedTuple = NMFk.compare_rawdata_grid(
        raw_information,
        grid_assignments[2],
    )
    Test.@test !inferred_cell_count_comparison.grid_cell_count_supplied
    Test.@test inferred_cell_count_comparison.grid_cell_count == 4
    Test.@test_throws ArgumentError NMFk.compare_rawdata_grid(
        raw_information,
        grid_assignments[2];
        grid_cell_count=3,
    )

    plot_values::NamedTuple = NMFk._rawdata_grid_plot_values(
        comparisons,
        ["fine", "pairs", "halves", "one"];
        xaxis=:steps,
        normalize=:fraction,
        baseline=:states,
    )
    Test.@test all(
        plot_values.bar_y_values[index] + plot_values.bar_y_values[index + 1] ≈ 1.0
        for index::Int in 1:2:length(plot_values.bar_y_values)
    )
    raw_grid_plot::Gadfly.Plot = NMFk.plot_rawdata_grid_information(
        comparisons,
        ["fine", "pairs", "halves", "one"],
    )
    raw_grid_bits_plot::Gadfly.Plot = NMFk.plot_rawdata_grid_information(
        comparisons,
        ["fine", "pairs", "halves", "one"];
        xaxis=:cells,
        normalize=:bits,
        baseline=:records,
    )
    comparison_matrix::Matrix{NamedTuple} = reshape(copy(comparisons), 2, 2)
    raw_grid_heatmap::Gadfly.Plot = NMFk.plot_rawdata_grid_heatmap(
        comparison_matrix,
        ["fine", "coarse"],
        ["short", "long"],
    )
    raw_grid_loss_heatmap::Gadfly.Plot = NMFk.plot_rawdata_grid_heatmap(
        comparison_matrix,
        ["fine", "coarse"],
        ["short", "long"];
        quantity=:lost,
    )
    raw_grid_comparison_heatmap::Gadfly.Plot = NMFk.plot_rawdata_grid_heatmap(
        comparison_matrix,
        ["fine", "coarse"],
        ["short", "long"];
        quantity=:lost,
        annotation=:comparison,
    )
    raw_grid_record_heatmap::Gadfly.Plot = NMFk.plot_rawdata_grid_heatmap(
        comparison_matrix,
        ["fine", "coarse"],
        ["short", "long"];
        baseline=:records,
        quantity=:lost,
        annotation=:comparison,
    )
    record_color_key::Gadfly.Guide.ColorKey = only(
        guide::Gadfly.GuideElement for guide::Gadfly.GuideElement in raw_grid_record_heatmap.guides
        if guide isa Gadfly.Guide.ColorKey
    )
    record_title::Gadfly.Guide.Title = only(
        guide::Gadfly.GuideElement for guide::Gadfly.GuideElement in raw_grid_record_heatmap.guides
        if guide isa Gadfly.Guide.Title
    )
    heatmap_x_ticks::Gadfly.Guide.XTicks = only(
        guide::Gadfly.GuideElement for guide::Gadfly.GuideElement in raw_grid_heatmap.guides
        if guide isa Gadfly.Guide.XTicks
    )
    Test.@test raw_grid_plot isa Gadfly.Plot
    Test.@test raw_grid_bits_plot isa Gadfly.Plot
    Test.@test raw_grid_heatmap isa Gadfly.Plot
    Test.@test raw_grid_loss_heatmap isa Gadfly.Plot
    Test.@test raw_grid_comparison_heatmap isa Gadfly.Plot
    Test.@test raw_grid_record_heatmap isa Gadfly.Plot
    Test.@test record_color_key.title == "H(R | G) / H(R)"
    Test.@test occursin("H(R)=", record_title.label)
    Test.@test heatmap_x_ticks.orientation == :horizontal
    Test.@test raw_grid_plot.layers[1].geom isa Gadfly.Geom.LabelGeometry
    Test.@test raw_grid_plot.layers[2].geom isa Gadfly.Geom.LineGeometry
    Test.@test raw_grid_plot.layers[3].geom isa Gadfly.Geom.BarGeometry
    Test.@test raw_grid_loss_heatmap.layers[1].geom isa Gadfly.Geom.LabelGeometry
    Test.@test raw_grid_loss_heatmap.layers[2].geom isa Gadfly.Geom.RectangularBinGeometry
    Test.@test_throws ArgumentError NMFk.compare_rawdata_grid((foo=1,), Int[1])
    Test.@test_throws DimensionMismatch NMFk.compare_rawdata_grid(raw_information, Int[1, 2])
    Test.@test_throws ArgumentError NMFk.plot_rawdata_grid_information(
        comparisons,
        ["fine", "pairs", "halves", "one"];
        normalize=:range,
    )
    Test.@test_throws ArgumentError NMFk.plot_rawdata_grid_heatmap(
        comparison_matrix,
        ["fine", "coarse"],
        ["short", "long"];
        quantity=:range,
    )
    Test.@test_throws ArgumentError NMFk.plot_rawdata_grid_heatmap(
        comparison_matrix,
        ["fine", "coarse"],
        ["short", "long"];
        annotation=:invalid,
    )
    Test.@test_throws DimensionMismatch NMFk.plot_rawdata_grid_heatmap(
        comparison_matrix,
        ["one"],
        ["short", "long"],
    )
    Test.@test_throws ArgumentError NMFk.plot_rawdata_grid_heatmap(
        comparison_matrix,
        Any[1, "1"],
        ["short", "long"],
    )
    masked_comparison::NamedTuple = NMFk.compare_rawdata_grid(
        raw_information,
        grid_assignments[2];
        valid_mask=BitVector([true, true, true, true, true, true, true, false]),
        grid_cell_count=4,
    )
    Test.@test_throws ArgumentError NMFk.plot_rawdata_grid_information(
        [comparisons[2], masked_comparison],
        ["all", "masked"],
    )
    plot_categorical_information_a::NamedTuple = NMFk.rawdata_information(
        ["A", "B", "A", "B"];
        precision=nothing,
    )
    plot_categorical_information_b::NamedTuple = NMFk.rawdata_information(
        ["X", "Y", "X", "Y"];
        precision=nothing,
    )
    categorical_comparison_a::NamedTuple = NMFk.compare_rawdata_grid(
        plot_categorical_information_a,
        Int[1, 1, 2, 2];
        grid_cell_count=2,
    )
    categorical_comparison_b::NamedTuple = NMFk.compare_rawdata_grid(
        plot_categorical_information_b,
        Int[1, 1, 2, 2];
        grid_cell_count=2,
    )
    Test.@test_throws ArgumentError NMFk.plot_rawdata_grid_information(
        [categorical_comparison_a, categorical_comparison_b],
        ["A/B", "X/Y"],
    )
end

Test.@testset "structure information validation and masking" begin
    tensor::Array{Float64, 3} = ones(Float64, 2, 2, 2)
    mask::BitArray{3} = trues(size(tensor))
    mask[1, 1, 1] = false
    information::NamedTuple = NMFk.structure_information(tensor; valid_mask=mask, bins=4, temporal_dim=nothing)
    Test.@test information.valid_cell_count == 7
    Test.@test information.normalized_value_entropy == 0.0
    Test.@test_throws DimensionMismatch NMFk.structure_information(tensor; valid_mask=trues(2, 2))
    Test.@test_throws ArgumentError NMFk.structure_information(tensor; bins=1)
    Test.@test_throws ArgumentError NMFk.structure_information(tensor; temporal_dim=4)
    Test.@test_throws ArgumentError NMFk.structure_information(tensor; residual_coding=:arithmetic)
    valid_information::NamedTuple = NMFk.structure_information(tensor; bins=4)
    Test.@test_throws ArgumentError NMFk.plot_structure_information([valid_information], ["one"]; normalize=:invalid)
    Test.@test_throws ArgumentError NMFk.plot_residual_coding([valid_information], ["one"]; normalize=:invalid)
end
