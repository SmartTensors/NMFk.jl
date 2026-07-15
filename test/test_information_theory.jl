import Test
import Random
import Gadfly

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
