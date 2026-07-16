import Test
import Random
import Gadfly
import Dates

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
    Test.@test NMFk._raw_floor_symbol(1.0000000000000019e15, 2.0, 0.0) == 500000000000000
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
    Test.@test raw_grid_plot isa Gadfly.Plot
    Test.@test raw_grid_bits_plot isa Gadfly.Plot
    Test.@test raw_grid_heatmap isa Gadfly.Plot
    Test.@test raw_grid_loss_heatmap isa Gadfly.Plot
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
