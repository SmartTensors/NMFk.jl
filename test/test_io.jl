import NMFk
import Test
import JLD

Test.@testset "io utilities" begin
	Test.@testset "load() missing file is non-throwing" begin
		tmp = mktempdir()
		W, H, fitquality, robustness, aic = NMFk.load(2, 1; resultdir=tmp, casefilename="definitely_missing", quiet=true, ordersignals=false)
		Test.@test size(W) == (0, 0)
		Test.@test size(H) == (0, 0)
		Test.@test isnan(fitquality)
		Test.@test isnan(robustness)
		Test.@test isnan(aic)
	end

	Test.@testset "load() finds size-encoded convention" begin
		tmp = mktempdir()
		nk = 2
		nNMF = 1

		W0 = reshape(collect(1.0:6.0), 3, 2)
		H0 = reshape(collect(11.0:18.0), 2, 4)
		fit0 = 0.123
		rob0 = 0.456
		aic0 = 7.89

		filename = joinpath(tmp, "case_3_4_$(nk)_$(nNMF).jld")
		JLD.save(filename, "W", W0, "H", H0, "fit", fit0, "robustness", rob0, "aic", aic0)

		W, H, fitquality, robustness, aic = NMFk.load(nk, nNMF; resultdir=tmp, casefilename="case", quiet=true, ordersignals=false)
		Test.@test W == W0
		Test.@test H == H0
		Test.@test fitquality == fit0
		Test.@test robustness == rob0
		Test.@test aic == aic0
	end

	Test.@testset "save() uses W/H sizes in filename" begin
		tmp = mktempdir()
		nk = 2
		nNMF = 1

		W0 = reshape(collect(1.0:6.0), 3, 2)
		H0 = reshape(collect(11.0:18.0), 2, 4)
		fit0 = 0.123
		rob0 = 0.456
		aic0 = 7.89

		NMFk.save(W0, H0, fit0, rob0, aic0, nk, nNMF; resultdir=tmp, casefilename="case")
		Test.@test isfile(joinpath(tmp, "case_3_4_$(nk)_$(nNMF).jld"))
	end
end
