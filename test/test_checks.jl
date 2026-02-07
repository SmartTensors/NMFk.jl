import NMFk
import Test

Test.@testset "checks utilities" begin
	Test.@testset "maskvector" begin
		v = Any[missing, 1.0, NaN, nothing, 2.0]
		m = NMFk.maskvector(v)
		Test.@test m == Bool[0, 1, 0, 0, 1]

		v2 = Any[missing, 1, nothing, 2]
		m2 = NMFk.maskvector(v2)
		Test.@test m2 == Bool[0, 1, 0, 1]
	end

	Test.@testset "checkarrayentries" begin
		X = [1.0 NaN; 2.0 3.0]

		bad = NMFk.checkarrayentries(X, .!isnan; quiet=true, cutoff=1)
		Test.@test bad[1] == [1]
		Test.@test bad[2] == [2]

		counts = NMFk.checkarrayentries(X, .!isnan; quiet=true, ecount=true)
		Test.@test counts[1] == [1, 2]
		Test.@test counts[2] == [2, 1]
	end

	Test.@testset "checkmatrix equivalence/correlation does not error" begin
		X = [1.0 1.0; 2.0 2.0; 3.0 3.0]
		result = NMFk.checkmatrix(X; names=["a", "b"], quiet=true, correlation_test=true)
		Test.@test result.same == Bool[0, 1]
		Test.@test result.remove == Bool[0, 1]
	end

	Test.@testset "checkmatrix_robust + recoupmatrix_rows" begin
		X = [1.0 2.0 0.0;
			NaN NaN NaN;
			3.0 4.0 0.0]
		names = ["c1", "c2", "c3"]
		Xf, namesf, row_mask, col_mask, info = NMFk.checkmatrix_robust(X, names; correlation_test=false)

		Test.@test size(Xf) == (2, 2)
		Test.@test namesf == ["c1", "c2"]
		Test.@test row_mask == Bool[0, 1, 0]
		Test.@test col_mask == Bool[0, 0, 1]
		Test.@test length(info.nan_rows) == size(Xf, 1)
		Test.@test all(.!info.nan_rows)

		Xfull = NMFk.recoupmatrix_rows(Xf, row_mask)
		Test.@test size(Xfull) == (3, 2)
		Test.@test all(isnan, Xfull[2, :])
		Test.@test Xfull[1, :] == Xf[1, :]
		Test.@test Xfull[3, :] == Xf[2, :]

		Test.@test_throws ArgumentError NMFk.recoupmatrix_rows(Xf, Int[0, 1, 0])
		Test.@test_throws ArgumentError NMFk.recoupmatrix_rows(Xf, Bool[0, 1])
	end

	Test.@testset "recoupmatrix_cols" begin
		X = reshape(collect(1:12), 3, 4)
		col_mask = Bool[false, true, false, true]  # remove columns 2 and 4
		Xf = X[:, .!col_mask]
		Xr = NMFk.recoupmatrix_cols(Xf, col_mask; fillvalue=-1)
		Test.@test size(Xr) == size(X)
		Test.@test Xr[:, 1] == X[:, 1]
		Test.@test Xr[:, 3] == X[:, 3]
		Test.@test Xr[:, 2] == fill(-1, size(X, 1))
		Test.@test Xr[:, 4] == fill(-1, size(X, 1))

		# Default fillvalue should promote element type when needed
		Xf_float = Float64.(Xf)
		Xr_nan = NMFk.recoupmatrix_cols(Xf_float, col_mask)
		Test.@test eltype(Xr_nan) == Float64
		Test.@test all(isnan, Xr_nan[:, 2])
		Test.@test all(isnan, Xr_nan[:, 4])

		# Dimension mismatch should throw
		Test.@test_throws ArgumentError NMFk.recoupmatrix_cols(Xf[:, 1:1], col_mask)

		# Mask type should throw
		Test.@test_throws ArgumentError NMFk.recoupmatrix_cols(Xf, [0, 1, 0, 1])
	end
end
