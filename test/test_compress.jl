import Test
import NMFk
import Random

Test.@testset "Row compression" begin
	Test.@testset "compress_rows/decompress_rows round-trip" begin
		X = [
			1.0 2.0
			1.0 2.0
			10.0 20.0
			NaN NaN
		]
		rng = Random.MersenneTwister(123)
		res = NMFk.compress_rows(X; k_range=2:2, n_restarts=1, max_iter=10, rng=rng, quiet=true)

		Test.@test size(res.compressed_matrix, 2) == size(X, 2)
		Test.@test length(res.original_to_group) == size(X, 1)
		Test.@test length(res.group_members) == size(res.compressed_matrix, 1)

		Xrec = NMFk.decompress_rows(res.compressed_matrix, res; mode=:representative)
		Test.@test size(Xrec) == size(X)
		Test.@test all(isnan, Xrec[end, :])
		Test.@test isapprox(Xrec[1:3, :], X[1:3, :]; rtol=0, atol=0)

		metrics = NMFk.evaluate_compression(X, Xrec)
		Test.@test metrics.n == 6
		Test.@test metrics.mae == 0.0
		Test.@test metrics.rmse == 0.0
		Test.@test metrics.max_abs == 0.0

		Xmean = NMFk.decompress_rows(res.group_means, res; mode=:mean, add_noise=false)
		Test.@test size(Xmean) == size(X)
		metrics_mean = NMFk.evaluate_compression(X, Xmean)
		Test.@test metrics_mean.n == 6
	end

	Test.@testset "evaluate_compression errors" begin
		X = [NaN NaN]
		Test.@test_throws ArgumentError NMFk.evaluate_compression(X, X)
		Test.@test_throws ArgumentError NMFk.evaluate_compression(ones(2,2), ones(3,2))
	end
end
