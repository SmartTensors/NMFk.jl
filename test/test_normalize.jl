import Test
import NMFk

Test.@testset "Normalization utilities" begin
	Test.@testset "normalize/denormalize (vector)" begin
		x = [1.0, 2.0, 3.0]
		xn, xmin, xmax = NMFk.normalize(x)
		Test.@test isapprox(xmin, 1.0)
		Test.@test isapprox(xmax, 3.0)
		Test.@test xn == [0.0, 0.5, 1.0]
		Test.@test NMFk.denormalize(xn, xmin, xmax) == x

		xr = [1.0, 2.0, 3.0]
		xn_rev, xmax_rev, xmin_rev = NMFk.normalize!(copy(xr); rev=true)
		Test.@test isapprox(xmin_rev, 1.0)
		Test.@test isapprox(xmax_rev, 3.0)
		Test.@test xn_rev == [1.0, 0.5, 0.0]
	end

	Test.@testset "normalize keeps NaNs" begin
		x = [1.0, NaN, 3.0]
		xn, xmin, xmax = NMFk.normalize(x)
		Test.@test isnan(xn[2])
		Test.@test xn[1] == 0.0
		Test.@test xn[3] == 1.0
		x_back = NMFk.denormalize(xn, xmin, xmax)
		Test.@test x_back[1] == 1.0
		Test.@test isnan(x_back[2])
		Test.@test x_back[3] == 3.0
	end

	Test.@testset "normalizematrix_col + denormalizematrix_col" begin
		X = [1.0 10.0; 2.0 20.0; 3.0 30.0]
		Xn, xmin, xmax, X_logtransform_type = NMFk.normalizematrix_col(X)
		Test.@test size(Xn) == size(X)
		Test.@test size(xmin) == (1, size(X, 2))
		Test.@test size(xmax) == (1, size(X, 2))
		Test.@test length(X_logtransform_type) == size(X, 2)

		Xback = NMFk.denormalizematrix_col!(copy(Xn), xmin, xmax; logv=falses(size(X, 2)), logtransform_type=X_logtransform_type)
		Test.@test isapprox(Xback, X; atol=0, rtol=0)
	end

	Test.@testset "zerostoepsilon" begin
		x = [0.0, -1.0, 1e-20, 1.0]
		y = NMFk.zerostoepsilon(x)
		e = eps(Float64)^2
		Test.@test all(y .>= e)
		Test.@test isapprox(y[3], 1e-20; atol=0, rtol=0)
		Test.@test isapprox(y[4], 1.0; atol=0, rtol=0)

		x2 = copy(x)
		NMFk.zerostoepsilon!(x2)
		Test.@test x2 == y
	end
end
