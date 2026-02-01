import NMFk
import Test

Test.@testset "preprocess utilities" begin
	Test.@testset "log10s/log10s!" begin
		Test.@test NMFk.log10s(10.0) == 1.0
		Test.@test NMFk.log10s(0.0) <= 0

		x = [1.0, 10.0, 0.0, -1.0]
		y = NMFk.log10s!(copy(x))
		Test.@test y[1] == 0.0
		Test.@test y[2] == 1.0
		# zeros/negatives are replaced with minimum(log10(positive)) - offset
		Test.@test y[3] == -1.0
		Test.@test y[4] == -1.0

		x2 = [0.0, -1.0]
		y2 = NMFk.log10s!(copy(x2))
		Test.@test isinf(y2[1])
		Test.@test y2[2] == 0.0
	end

	Test.@testset "indicize" begin
		v = Float64[1, 2, 3, 4, 5]

		iv, nbins, minv, maxv = NMFk.indicize(v; stepvalue=1.0, granulate=true, quiet=true)
		Test.@test nbins == 4
		Test.@test minv == 1.0
		Test.@test maxv == 5.0
		Test.@test iv == [1, 1, 2, 3, 4]

		ivrev, nbinsrev, _, _ = NMFk.indicize(v; stepvalue=1.0, granulate=true, rev=true, quiet=true)
		Test.@test nbinsrev == 4
		Test.@test ivrev == [4, 4, 3, 2, 1]

		iv2, nbins2, _, _ = NMFk.indicize(v; nbins=4, minvalue=1.0, maxvalue=5.0, granulate=false, quiet=true)
		Test.@test nbins2 == 5
		Test.@test iv2 == [1, 1, 2, 3, 4]
	end

	Test.@testset "processdata/processdata!" begin
		Test.@testset "Float32, enforce_nan=true, string_ok=false" begin
			M = Any["1", "", "NaN", "-2", "bad", missing, nothing]
			out = NMFk.processdata!(copy(M), Float32; enforce_nan=true, string_ok=false, negative_ok=false)
			# parsed
			Test.@test out[1] == 1.0f0
			# blanks/nanstring -> NaN when enforce_nan=true
			Test.@test isnan(out[2])
			Test.@test isnan(out[3])
			# negative -> 0 when negative_ok=false
			Test.@test out[4] == 0.0f0
			# unparseable strings -> NaN when string_ok=false
			Test.@test isnan(out[5])
			# missing/nothing -> missing under current defaults
			Test.@test out[6] === missing
			Test.@test out[7] === missing
		end

		Test.@testset "Float32, enforce_nan=false keeps missings" begin
			M = Any["1", "", "bad", missing, nothing]
			out = NMFk.processdata!(copy(M), Float32; enforce_nan=false, string_ok=true)
			Test.@test out[1] == 1.0f0
			Test.@test out[2] === missing
			Test.@test out[3] == "bad"
			Test.@test out[4] === missing
			Test.@test out[5] === missing
		end

		Test.@testset "String output converts numbers" begin
			M = Any[1, 2.5, missing]
			out = NMFk.processdata!(copy(M), String)
			Test.@test out[1] == "1"
			Test.@test out[2] == "2.5"
			Test.@test out[3] === missing
		end

		Test.@testset "Matrix Any input handles placeholders" begin
			M = Any["1" ""; nothing "NaN"]
			out = NMFk.processdata!(copy(M), Float32; enforce_nan=true, string_ok=false)
			Test.@test out[1, 1] == 1.0f0
			Test.@test isnan(out[1, 2])
			Test.@test out[2, 1] === missing
			Test.@test isnan(out[2, 2])
		end

		Test.@testset "String array widens when targeting numbers" begin
			M = ["1", "", "2"]
			out = NMFk.processdata!(copy(M), Float32; enforce_nan=false)
			Test.@test out[1] == 1.0f0
			Test.@test out[2] === missing
			Test.@test out[3] == 2.0f0
		end
	end

	Test.@testset "remap (vector/matrix)" begin
		v = Float64[10, 20]
		mapping = Any[1, nothing, 2]
		o = NMFk.remap(v, mapping)
		Test.@test length(o) == 3
		Test.@test o[1] == 10.0
		Test.@test isnan(o[2])
		Test.@test o[3] == 20.0

		vi = Int[10, 20]
		oi = NMFk.remap(vi, mapping)
		Test.@test oi == Int[10, 0, 20]

		V = Float64[1 2; 10 20]
		O = NMFk.remap(V, mapping)
		Test.@test size(O) == (3, 2)
		Test.@test O[1, :] == Float64[1, 2]
		Test.@test all(isnan.(O[2, :]))
		Test.@test O[3, :] == Float64[10, 20]
	end

	Test.@testset "slopes" begin
		v = Float64[1, 2, 4]
		s = NMFk.slopes(v)
		Test.@test s == Float64[1, 1.5, 2]
	end

	Test.@testset "bincoordinates" begin
		v = Float64[0, 1, 2]
		b = NMFk.bincoordinates(v; nbins=2, minvalue=0.0, maxvalue=2.0)
		Test.@test length(b) == 2
		Test.@test b == Float64[0.5, 1.5]

		brev = NMFk.bincoordinates(v; nbins=2, minvalue=0.0, maxvalue=2.0, rev=true)
		Test.@test brev == reverse(b)
	end
end
