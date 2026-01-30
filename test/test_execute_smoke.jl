import Test
import NMFk
import Random

Test.@testset "Execute smoke tests" begin
	Test.@testset "execute_singlerun (simple) returns sane shapes" begin
		Random.seed!(123)
		X = abs.(randn(5, 4))
		W, H, obj = NMFk.execute_singlerun(X, 2; quiet=true, method=:simple, maxiter=50, tol=1e-8)
		Test.@test size(W) == (size(X, 1), 2)
		Test.@test size(H) == (2, size(X, 2))
		Test.@test isfinite(obj)
		Test.@test all(isfinite, W)
		Test.@test all(isfinite, H)
		Test.@test all(W .>= 0)
		Test.@test all(H .>= 0)
		# default normalization in execute_singlerun_compute (clusterWmatrix=false) normalizes H rows
		Test.@test isapprox(sum(@view(H[1, :])), 1.0; atol=1e-4)
		Test.@test isapprox(sum(@view(H[2, :])), 1.0; atol=1e-4)
	end

	Test.@testset "execute_run (nk=1) stays lightweight" begin
		Random.seed!(321)
		X = abs.(randn(6, 5))
		Wa, Ha, phi, minsilhouette, aic =
			NMFk.execute_run(X, 1, 2; serial=true, veryquiet=true, best=true, maxiter=40, tol=1e-8)
		Test.@test size(Wa) == (size(X, 1), 1)
		Test.@test size(Ha) == (1, size(X, 2))
		Test.@test isfinite(phi)
		Test.@test isfinite(aic)
		Test.@test minsilhouette == 1
	end

	Test.@testset "execute(loadonly=true) missing file returns empties" begin
		mktempdir() do d
			X = ones(3, 3)
			W, H, fit, robustness, aic = NMFk.execute(X, 2, 1; loadonly=true, load=true, save=false, casefilename="case", resultdir=d, quiet=true)
			Test.@test size(W) == (0, 0)
			Test.@test size(H) == (0, 0)
			Test.@test fit == Inf
			Test.@test robustness == -1
			Test.@test aic == -Inf
		end
	end
end
