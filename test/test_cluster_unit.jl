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

	Test.@testset "finduniquesignals succeeds on permutation matrix" begin
		# Unique mapping exists and is unambiguous: one non-zero per row/col.
		p = [3, 1, 4, 2]
		k = length(p)
		X = zeros(Float64, k, k)
		for i in 1:k
			X[i, p[i]] = 1.0
		end
		o, signalmap = NMFk.finduniquesignals(X; quiet=true)
		Test.@test signalmap == p
		Test.@test o == k
	end

	Test.@testset "finduniquesignals ignores NaNs" begin
		X = [
			1.0  NaN  0.0
			0.0  2.0  NaN
			NaN  0.0  3.0
		]
		o, signalmap = NMFk.finduniquesignals(X; quiet=true)
		Test.@test signalmap == [1, 2, 3]
		Test.@test o == 6.0
	end

	Test.@testset "finduniquesignals greedy can fail despite existing perfect matching" begin
		# Counterexample: a perfect one-to-one assignment exists (row1->2, row2->1)
		# but the greedy global-max strategy grabs (1,1) first and never backtracks.
		X = [
			10.0 9.0
			 8.0 0.0
		]
		res = NMFk._finduniquesignals_impl(X; quiet=true)
		Test.@test res.failed
		Test.@test res.signalmap == [1, 0]
		Test.@test res.o == 0
		# The "best" variant should not crash on an incomplete greedy map.
		Test.@test NMFk.finduniquesignalsbest(X) == [2, 1]
	end

	Test.@testset "finduniquesignals fails on provided matrix but best recovers" begin
		X = [
			0.0 0.02241742268547143
			0.018754193314799186 0.05344343641937143
		]
		res = NMFk._finduniquesignals_impl(X; quiet=true)
		Test.@test res.failed
		Test.@test res.signalmap == [0, 2]
		Test.@test NMFk.finduniquesignalsbest(X) == [2, 1]
	end

	Test.@testset "finduniquesignals iteration count is small for diagonal-dominant" begin
		# Performance proxy: for a strictly diagonal-dominant case, we should assign
		# one row per iteration with no conflicts, so iters == k.
		k = 40
		X = zeros(Float64, k, k)
		for i in 1:k
			X[i, i] = 1000.0 - i
		end
		res = NMFk._finduniquesignals_impl(X; quiet=true)
		Test.@test !res.failed
		Test.@test res.signalmap == collect(1:k)
		Test.@test res.iters == k
	end
end
