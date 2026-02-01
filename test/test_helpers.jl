"""
Unit tests for lightweight helper utilities in NMFkHelpers.jl.

These tests are intentionally deterministic and fast. They target pure / low-level
functions that are easy to validate without running full factorization workflows.
"""

import Test
import NMFk
import Statistics
import Random

Test.@testset "NMFk.Helpers Unit Tests" begin

	Test.@testset "r2" begin
		x = [1.0, 2.0, 3.0, 4.0]
		y = 2.0 .* x
		Test.@test NMFk.r2(x, y) ≈ 1.0

		x2 = [1.0, 2.0, NaN, 4.0]
		y2 = [3.0, 6.0, 9.0, 12.0]
		Test.@test NMFk.r2(x2, y2) ≈ 1.0
	end

	Test.@testset "findfirst" begin
		v = [NaN, -1.0, 0.0, 2.0]
		Test.@test NMFk.findfirst(copy(v); zerod=false) == 4
		Test.@test NMFk.findfirst(copy(v); zerod=true) == 4
	end

	Test.@testset "nan-aware reductions" begin
		v = [1.0, NaN, 3.0, 2.0]
		Test.@test NMFk.maximumnan(copy(v)) == 3.0
		Test.@test NMFk.minimumnan(copy(v)) == 1.0
		Test.@test NMFk.sumnan(copy(v)) == 6.0
		Test.@test NMFk.meannan(copy(v)) == 2.0

		v_all_nan = [NaN, NaN]
		Test.@test isnan(NMFk.maximumnan(copy(v_all_nan)))
		Test.@test isnan(NMFk.minimumnan(copy(v_all_nan)))
		Test.@test isnan(NMFk.sumnan(copy(v_all_nan)))
		Test.@test isnan(NMFk.meannan(copy(v_all_nan)))

		A = Float64.([1 NaN 3; 4 5 NaN])
		Test.@test all(NMFk.maximumnan(copy(A); dims=1) .== [4.0 5.0 3.0])
		Test.@test all(NMFk.minimumnan(copy(A); dims=1) .== [1.0 5.0 3.0])
		Test.@test all(NMFk.sumnan(copy(A); dims=1) .== [5.0 5.0 3.0])
		Test.@test all(NMFk.meannan(copy(A); dims=1) .== [2.5 5.0 3.0])

		Test.@test NMFk.cumsumnan(copy([1.0, NaN, 2.0])) == [1.0, 1.0, 3.0]
	end

	Test.@testset "varnan/stdnan" begin
		v = [1.0, 2.0, NaN, 4.0]
		# variance of [1,2,4] is 7/3 with corrected=true
		Test.@test NMFk.varnan(copy(v)) ≈ Statistics.var([1.0, 2.0, 4.0])
		Test.@test NMFk.stdnan(copy(v)) ≈ Statistics.std([1.0, 2.0, 4.0])
	end

	Test.@testset "distance-like metrics ignoring NaNs" begin
		t = [1.0, NaN, 3.0]
		o = [2.0, 10.0, NaN]
		# Only first entry contributes
		Test.@test NMFk.rmsenan(t, o) == 1.0
		Test.@test NMFk.l1nan(t, o) == 1.0
		Test.@test NMFk.ssqrnan(t, o) == 1.0
	end

	Test.@testset "sortnan/sortpermnan" begin
		v = [2.0, NaN, 1.0]
		perm_fwd = NMFk.sortpermnan(v; rev=false)
		Test.@test isnan(v[perm_fwd[1]])
		perm_rev = NMFk.sortpermnan(v; rev=true)
		Test.@test isnan(v[perm_rev[end]])

		v_sorted = NMFk.sortnan(v)
		Test.@test v_sorted[1:2] == [1.0, 2.0]
		Test.@test isnan(v_sorted[3])
	end

	Test.@testset "covnan/cornan" begin
		x = [1.0, 2.0, NaN, 4.0]
		y = [2.0, 4.0, 6.0, NaN]
		Test.@test NMFk.covnan(x, y) ≈ Statistics.cov([1.0, 2.0], [2.0, 4.0])
		Test.@test NMFk.cornan(x, y) ≈ Statistics.cor([1.0, 2.0], [2.0, 4.0])

		Test.@test isnan(NMFk.covnan([NaN], [1.0]))
		Test.@test isnan(NMFk.cornan([NaN], [1.0]))
	end

	Test.@testset "hardencode/harddecode" begin
		x = [1.0, 2.0, 1.0]
		Test.@test NMFk.hardencodelength(x) == 2
		H = NMFk.hardencode(x)
		Test.@test size(H) == (3, 2)
		Test.@test all(sum(H; dims=2) .== 1.0)

		X = [1.0 10.0; 2.0 10.0; 1.0 20.0]
		H2 = NMFk.hardencode(X)
		s = NMFk.harddecode(X, H2)
		Test.@test size(s) == size(X)
		Test.@test all(s .== 1.0)
	end

	Test.@testset "movingwindow" begin
		A = [1.0, 2.0, 3.0]
		B = NMFk.movingwindow(A, 1)
		Test.@test B ≈ [1.5, 2.0, 2.5]
		Test.@test NMFk.movingwindow(A, 0) === A
	end

	Test.@testset "nanmask!/remask" begin
		X = Float64[0 1; 2 3]
		NMFk.nanmask!(X, 1)
		Test.@test isnan(X[1, 1])
		Test.@test isnan(X[1, 2])
		Test.@test X[2, 1] == 2.0
		Test.@test X[2, 2] == 3.0

		X3 = ones(2, 2, 3)
		mask = falses(2, 2)
		mask[1, 1] = true
		NMFk.nanmask!(X3, mask, 3)
		Test.@test all(isnan.(X3[1, 1, :]))
		Test.@test all(.!isnan.(X3[2, 2, :]))

		sm = BitArray([true false])
		rm = NMFk.remask(sm, 3)
		Test.@test size(rm) == (1, 2, 3)
	end

	Test.@testset "bincount/flip" begin
		x = [1, 1, 2, 3, 3, 3]
		bc = NMFk.bincount(x)
		Test.@test bc[1, 1] == 3
		Test.@test bc[1, 2] == 3
		Test.@test bc[2, 1] == 1
		Test.@test bc[2, 2] == 2

		v = [1.0, 2.0, 3.0]
		Test.@test NMFk.flip(v) == [3.0, 2.0, 1.0]
		M = Float64.([1 2; 3 4])
		Test.@test NMFk.flip(M) == Float64.([4 3; 2 1])
	end

	Test.@testset "aisnan/aisnan!" begin
		x = [1.0, NaN, 3.0]
		y = NMFk.aisnan(x)
		Test.@test y == [1.0, 1.0, 3.0]
		Test.@test x[2] !== 1.0  # aisnan makes a copy

		x2 = [NaN, 0.0]
		NMFk.aisnan!(x2)
		Test.@test x2 == [1.0, 0.0]

		x3 = [NaN, NaN]
		NMFk.aisnan!(x3, 7)
		Test.@test x3 == [7.0, 7.0]
	end

	Test.@testset "uniform_points/random_points" begin
		u = NMFk.uniform_points(4, 10, 1)
		Test.@test length(u) == 4
		Test.@test all(1 .<= u .<= 10)
		Test.@test eltype(u) <: Integer
		Test.@test issorted(u)

		Random.seed!(1234)
		r = NMFk.random_points(5, 100, 1)
		Test.@test length(r) == 5
		Test.@test all(1 .<= r .<= 100)
		Test.@test eltype(r) <: Integer
	end

	Test.@testset "stringproduct" begin
		a = ["a", "b"]
		b = ["1", "2", "3"]
		M = NMFk.stringproduct(a, b)
		Test.@test size(M) == (2, 3)
		Test.@test M[1, 1] == "a:1"
		Test.@test M[2, 3] == "b:3"
	end

end
