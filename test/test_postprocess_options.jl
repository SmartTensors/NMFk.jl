import Test
import NMFk
import Suppressor

Test.@testset "PostprocessOptions" begin
	# Minimal, deterministic inputs: one-signal factorization.
	krange = [1]
	W = Vector{Matrix{Float64}}(undef, 1)
	H = Vector{Matrix{Float64}}(undef, 1)
	W[1] = [1.0; 2.0; 3.0] * reshape([1.0], 1, 1)  # 3x1
	H[1] = reshape([1.0, 2.0, 3.0, 4.0], 1, 4)     # 1x4
	X = W[1] * H[1]

	resultdir = mktempdir()
	figuredir = mktempdir()

	opts = NMFk.PostprocessOptions(
		; createdendrogramsonly=true,
		createplots=false,
		createbiplots=false,
		creatematrixplotsall=false,
		movies=false,
		plotmaps=false,
		plottimeseries=:none,
		clusterW=false,
		clusterH=false,
		loadassignements=false,
		resultdir=resultdir,
		figuredir=figuredir,
		quiet=true,
		veryquiet=true,
	)

	Sorder, Wclusters, Hclusters = Suppressor.@suppress NMFk.postprocess(opts, krange, W, H, X)
	Test.@test length(Sorder) == 1
	Test.@test length(Wclusters) == 1
	Test.@test length(Hclusters) == 1
end
