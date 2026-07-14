import Test
import Random

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
    Test.@test structured_info.temporal_predictive_gain_bits > randomized_info.temporal_predictive_gain_bits
    Test.@test structured_info.spectral_information[1].effective_rank < randomized_info.spectral_information[1].effective_rank
    Test.@test structured_info.valid_cell_count == length(structured)
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
end
