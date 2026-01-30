"""\
Unit tests for `NMFk.griddata()` and its key dependency functions.

Notes on the API (current behavior):
- `griddata(x, y; ...)` returns two 1D ranges (x grid, y grid) and requires either `nbins` or both `xstepvalue` and `ystepvalue`.
- `griddata(x, y, z; ...)` returns a gridded tensor `T` with size `(xbins, ybins, nfeatures)`.
"""

import Test
import NMFk
import DataFrames
import Dates

Test.@testset "NMFk.griddata() Unit Tests" begin

	Test.@testset "Helper Function: minimumnan" begin
		Test.@test NMFk.minimumnan([1, 2, 3, 4, 5]) == 1
		Test.@test NMFk.minimumnan([5, 4, 3, 2, 1]) == 1
		Test.@test NMFk.minimumnan([-2, -1, 0, 1, 2]) == -2
		Test.@test NMFk.minimumnan([1, NaN, 3, 4, 5]) == 1
		Test.@test NMFk.minimumnan([NaN, 2, 3, 4, 5]) == 2
		Test.@test NMFk.minimumnan([1, 2, 3, 4, NaN]) == 1
		Test.@test isnan(NMFk.minimumnan([NaN, NaN, NaN]))
		Test.@test isnan(NMFk.minimumnan(Float64[]))

		A = Float64.([1 2 3; 4 5 6; 7 8 9])
		Test.@test NMFk.minimumnan(A) == 1
		Test.@test all(NMFk.minimumnan(A; dims=1) .== [1. 2. 3.])
		Test.@test all(NMFk.minimumnan(A; dims=2) .== [1.; 4.; 7.])

		B = Float64.([1 NaN 3; 4 5 6; NaN 8 9])
		Test.@test NMFk.minimumnan(B) == 1
		Test.@test all(NMFk.minimumnan(B; dims=1) .== [1. 5. 3.])
		Test.@test all(NMFk.minimumnan(B; dims=2) .== [1.; 4.; 8.])
	end

	Test.@testset "Helper Function: processdata" begin
		data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
		result1 = NMFk.processdata(data1, Float64)
		Test.@test all(result1 .== data1)
		Test.@test eltype(result1) == Float64

		# Mixed types (Any) uses tryparse on string entries; nothing becomes `missing` under current defaults.
		data2 = Any[1.0, "2", 3.5, nothing]
		result2 = NMFk.processdata(data2, Float64)
		Test.@test result2[1] == 1.0
		Test.@test result2[2] == 2.0
		Test.@test result2[3] == 3.5
		Test.@test result2[4] === missing

		df = DataFrames.DataFrame(x=[1.0, 2.0, 3.0], y=Any["4", "5", "6"], z=[7.0, 8.0, 9.0])
		result_df = NMFk.processdata(df, Float64)
		Test.@test all(result_df.x .== [1.0, 2.0, 3.0])
		Test.@test all(result_df.y .== [4.0, 5.0, 6.0])
		Test.@test all(result_df.z .== [7.0, 8.0, 9.0])

		data3 = [-1.0, 2.0, -3.0, 4.0]
		result3 = NMFk.processdata(data3, Float64; negative_ok=false)
		Test.@test all(result3 .== [0.0, 2.0, 0.0, 4.0])

		# Invalid strings are handled when the container eltype is `Any` (string conversion uses tryparse).
		data4 = Any["1", "2", "invalid", "4"]
		result4 = NMFk.processdata(data4, Float64; string_ok=false)
		Test.@test result4[1] == 1.0
		Test.@test result4[2] == 2.0
		Test.@test isnan(result4[3])
		Test.@test result4[4] == 4.0
	end

	Test.@testset "Helper Function: indicize" begin
		x = [1.0, 2.0, 3.0, 4.0, 5.0]
		ix, xbins, xmin, xmax = NMFk.indicize(x; nbins=3)
		Test.@test length(ix) == length(x)
		Test.@test all(ix .>= 1)
		Test.@test all(ix .<= xbins)
		Test.@test xbins == 3
		Test.@test xmin <= minimum(x)
		Test.@test xmax >= maximum(x)

		ix2, xbins2, xmin2, xmax2 = NMFk.indicize(x; minvalue=0.0, maxvalue=6.0, nbins=6)
		Test.@test xmin2 == 0.0
		Test.@test xmax2 == 6.0
		Test.@test xbins2 == 6

		ix3, xbins3, xmin3, xmax3 = NMFk.indicize(x; stepvalue=1.0)
		Test.@test xbins3 > 0
		Test.@test xmin3 <= minimum(x)
		Test.@test xmax3 >= maximum(x)

		ix4, xbins4, _, _ = NMFk.indicize(x; rev=true, nbins=3)
		Test.@test length(ix4) == length(x)
		Test.@test all(ix4 .>= 1)
		Test.@test all(ix4 .<= xbins4)

		# NaNs in the data vector currently raise an InexactError during binning.
		x_nan = [1.0, NaN, 3.0, NaN, 5.0]
		Test.@test_throws InexactError NMFk.indicize(x_nan; nbins=3)

		# indicize is currently defined for numeric vectors; Date vectors are not supported here.
		dts = [Dates.Date(2020, 1, 1), Dates.Date(2020, 1, 15), Dates.Date(2020, 2, 1)]
		Test.@test_throws MethodError NMFk.indicize(dts; nbins=2)
	end

	Test.@testset "griddata(x, y): 2D grid coordinates" begin
		x = [1.0, 2.0, 3.0, 4.0, 5.0]
		y = [1.0, 1.5, 2.0, 2.5, 3.0]

		gx, gy = NMFk.griddata(x, y; nbins=5)
		Test.@test length(gx) == 5
		Test.@test length(gy) == 5
		Test.@test gx[1] <= gx[end]
		Test.@test gy[1] <= gy[end]

		gx2, gy2 = NMFk.griddata(x, y; stepvalue=0.5, xstepvalue=0.5, ystepvalue=0.5)
		Test.@test length(gx2) > 1
		Test.@test length(gy2) > 1
	end

	Test.@testset "griddata(x, y, z): gridding (vector z)" begin
		x = [1.0, 2.0, 3.0, 4.0, 5.0]
		y = [1.0, 1.5, 2.0, 2.5, 3.0]
		z = [10.0, 20.0, 30.0, 40.0, 50.0]

		T = NMFk.griddata(x, y, z; xnbins=3, ynbins=3, type=Float64)
		Test.@test ndims(T) == 3
		Test.@test size(T, 3) == 1
		Test.@test eltype(T) == Float64

		z_nan = [10.0, NaN, 30.0, NaN, 50.0]
		T2 = NMFk.griddata(x, y, z_nan; xnbins=3, ynbins=3)
		Test.@test ndims(T2) == 3
		Test.@test size(T2, 3) == 1
	end

	Test.@testset "griddata(x, y, Z): gridding (matrix Z)" begin
		x = [1.0, 2.0, 3.0, 4.0, 5.0]
		y = [1.0, 1.5, 2.0, 2.5, 3.0]
		Z = [10.0 100.0; 20.0 200.0; 30.0 300.0; 40.0 400.0; 50.0 500.0]

		# indicize() can be strict about the last bin boundary with granulate=true; use granulate=false here.
		T = NMFk.griddata(x, y, Z; xnbins=3, ynbins=3, granulate=false)
		Test.@test ndims(T) == 3
		Test.@test size(T, 3) == 2

		x_dup = [1.0, 1.0, 2.0, 2.0, 3.0]
		y_dup = [1.0, 1.0, 2.0, 2.0, 3.0]
		z_dup = [10.0, 20.0, 30.0, 40.0, 50.0]
		Tdup = NMFk.griddata(x_dup, y_dup, z_dup; xnbins=2, ynbins=2, granulate=false)
		Test.@test ndims(Tdup) == 3
		Test.@test size(Tdup, 3) == 1
	end

	Test.@testset "Edge Cases and Error Conditions" begin
		Test.@test_throws ArgumentError NMFk.griddata(Float64[], Float64[]; nbins=1)

		x = [1.0, 2.0, 3.0]
		y = [1.0, 2.0]
		z = [10.0, 20.0, 30.0]
		Test.@test_throws AssertionError NMFk.griddata(x, y, z)

		x_single = [1.0]
		y_single = [1.0]
		z_single = [10.0]
		# Degenerate ranges (min == max) currently raise during indicize().
		Test.@test_throws InexactError NMFk.griddata(x_single, y_single, z_single; xnbins=1, ynbins=1)

		x_same = [1.0, 1.0, 1.0]
		y_same = [1.0, 1.0, 1.0]
		z_same = [10.0, 20.0, 30.0]
		Test.@test_throws InexactError NMFk.griddata(x_same, y_same, z_same; xnbins=1, ynbins=1)

		x_large = [1e-10, 1e10]
		y_large = [1e-10, 1e10]
		z_large = [1.0, 2.0]
		Tlarge = NMFk.griddata(x_large, y_large, z_large; xnbins=2, ynbins=2)
		Test.@test size(Tlarge, 3) == 1
	end

	Test.@testset "Parameter Precedence and Logic" begin
		x = [1.0, 2.0, 3.0, 4.0, 5.0]
		y = [1.0, 2.0, 3.0, 4.0, 5.0]
		z = [10.0, 20.0, 30.0, 40.0, 50.0]

		T1 = NMFk.griddata(x, y, z; xstepvalue=1.0, xnbins=10, ynbins=5)
		T2 = NMFk.griddata(x, y, z; xstepvalue=1.0, ynbins=5)
		Test.@test size(T1) == size(T2)
	end

	Test.@testset "Interpolation Setup Validation" begin
		x = [0.0, 1.0, 2.0, 3.0, 4.0]
		y = [0.0, 1.0, 2.0, 3.0, 4.0]
		gx, gy = NMFk.griddata(x, y; nbins=5)
		Test.@test issorted(gx)
		Test.@test issorted(gy)
	end
end

println("All NMFk.griddata() unit tests completed!")
