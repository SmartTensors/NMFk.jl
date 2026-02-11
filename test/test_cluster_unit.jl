import Test
import NMFk
import Random

Test.@testset "Clustering utilities" begin
	Test.@testset "robustkmeans(k) returns valid assignments" begin
		Random.seed!(42)
		# columns are samples; first two columns near each other, last two near each other
		X = [
			1.0 1.1 10.0 10.1
			1.0 0.9 10.0 9.9
		]
		result = NMFk.robustkmeans(X, 2, 5; maxiter=50, tol=1e-8)
		assignments = result.assignments
		Test.@test length(assignments) == size(X, 2)
		Test.@test sort(unique(assignments)) == [1, 2]
		Test.@test size(result.centers, 2) == 2
	end

	Test.@testset "robustbgmm selects a reasonable k" begin
		Random.seed!(1)
		# 2D, 3 well-separated clusters; columns are samples
		x1 = randn(2, 80) .+ [5.0; 0.0]
		x2 = randn(2, 80) .+ [-5.0; 0.0]
		x3 = randn(2, 80) .+ [0.0; 5.0]
		X = hcat(x1, x2, x3)
		res = NMFk.robustbgmm(X, 1:6, 2; criterion=:bic, kind=:diag, nInit=1, nIter=10, nFinal=2)
		Test.@test !isnothing(res)
		Test.@test res.k in 1:6
		Test.@test length(res.assignments) == size(X, 2)
		Test.@test length(unique(res.assignments)) == res.k
		# BIC should generally prefer ~3 here (allow small deviations for tiny EM budgets)
		Test.@test abs(res.k - 3) <= 1
	end

	Test.@testset "clustersolutions assigns each column exactly once" begin
		f1 = [
			1.0 0.0
			0.0 1.0
			1.0 0.0
			0.0 1.0
		]
		f2 = [
			0.0 1.0
			1.0 0.0
			0.0 1.0
			1.0 0.0
		]
		labels, centers = NMFk.clustersolutions([f1, f2], true)
		Test.@test size(labels) == (2, 2)
		Test.@test labels[:, 1] == [1, 2]
		Test.@test sort(labels[:, 2]) == [1, 2]
		Test.@test size(centers) == (2, 4)
	end

	Test.@testset "finduniquesignals terminates on zero rows" begin
		# If a row has no non-zero entries, there is no valid one-to-one assignment.
		# The function should terminate (not hang) and return an incomplete map.
		X = [
			0.0 0.0
			0.0 -1.0
		]
		o, signalmap = NMFk.finduniquesignals(X; quiet=true)
		Test.@test signalmap[2] == 2
		Test.@test signalmap[1] == 0
		Test.@test o == 0
	end
end
