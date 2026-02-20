import Test
import NMFk
import Suppressor

Test.@testset "Tensor + flatten helpers" begin

	Test.@testset "flatten(dim)" begin
		X = reshape(Float64.(1:24), 2, 3, 4)  # size (2,3,4)

		A1 = NMFk.flatten(X, 1)
		Test.@test size(A1) == (3 * 4, 2)
		Test.@test A1[:, 1] == vec(X[1, :, :])
		Test.@test A1[:, 2] == vec(X[2, :, :])

		A2 = NMFk.flatten(X, 2)
		Test.@test size(A2) == (2 * 4, 3)
		Test.@test A2[:, 1] == vec(X[:, 1, :])
		Test.@test A2[:, 3] == vec(X[:, 3, :])

		A3 = NMFk.flatten(X, 3)
		Test.@test size(A3) == (2 * 3, 4)
		Test.@test A3[:, 1] == vec(X[:, :, 1])
		Test.@test A3[:, 4] == vec(X[:, :, 4])
	end

	Test.@testset "flatten(mask)" begin
		X = reshape(Float64.(1:12), 2, 2, 3)  # size (2,2,3)
		mask = falses(2, 2)
		mask[1, 2] = true

		A = NMFk.flatten(X, mask)
		Test.@test size(A) == (3, 3)  # 4 total - 1 masked = 3 rows, 3 slices

		# For each slice, we expect the unmasked entries of X[:,:,i] in linear order
		for i in 1:3
			slice = X[:, :, i]
			Test.@test A[:, i] == slice[.!mask]
		end
	end

	Test.@testset "flattenindex" begin
		X = reshape(Float64.(1:24), 2, 3, 4)

		I12 = NMFk.flattenindex(X, 3; order=[1, 2])
		Test.@test I12 == repeat(1:2, 3)

		I21 = NMFk.flattenindex(X, 3; order=[2, 1])
		Test.@test I21 == sort(repeat(1:3, 2))

		Test.@test_throws ErrorException NMFk.flattenindex(X, 3; order=[1, 1])
	end

	Test.@testset "tensorfactorization (loadonly)" begin
		X = reshape(Float64.(1:8), 2, 2, 2)
		tmp = mktempdir()

		# Use loadonly=true to avoid running any NMF; this should still exercise
		# the tensor wrapper, flattening, and result file naming.
		R = Suppressor.@suppress NMFk.tensorfactorization(
			X,
			1,          # nk
			1:3,        # dims
			1;          # nNMF
			casefilename="tf",
			resultdir=tmp,
			loadonly=true,
			load=true,
			save=false,
			method=:simple,
			quiet=true,
		)

		Test.@test length(R) == 3
		for d in 1:3
			Test.@test R[d] isa Tuple
			Test.@test length(R[d]) == 5
			W, H, fit, sil, aic = R[d]
			Test.@test size(W) == (0, 0)
			Test.@test size(H) == (0, 0)
			Test.@test fit == Inf
			Test.@test sil == -1
			Test.@test aic == -Inf
		end

		# execute(...) on tensors should delegate to tensorfactorization
		R2 = Suppressor.@suppress NMFk.execute(
			X,
			1,
			1;
			casefilename="tfexec",
			resultdir=tmp,
			loadonly=true,
			load=true,
			save=false,
			method=:simple,
			quiet=true,
		)
		Test.@test length(R2) == 3
	end

end
