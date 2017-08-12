import NMFk
using Base.Test

srand(2015)
a = rand(20)
b = rand(20)
X = [a a*10 b b*5 a+b*2]
W, H, p, s = NMFk.execute(X, 2, 20)
@test isapprox(p, 0, atol=1e-3)
@test isapprox(s, 1, rtol=1e-1)
@test isapprox(H[1,2] / H[1,1], 10, rtol=1e-3)
@test isapprox(H[2,2] / H[2,1], 10, rtol=1e-3)
@test isapprox(H[2,4] / H[2,3], 5, rtol=1e-3)
@test isapprox(H[1,4] / H[1,3], 5, rtol=1e-3)

srand(2015)
a = exp.(-(0:.5:10))*100
b = 100 + sin.(0:20)*10
X = [a a*10 b b*5 a+b*2]
W, H, p, s = NMFk.execute(X, 2, 20)
@test isapprox(p, 0, atol=1e-3)
@test isapprox(s, 1, rtol=1e-1)
@test isapprox(H[1,2] / H[1,1], 10, rtol=1e-3)
@test isapprox(H[2,3] / H[1,3], 2.28109, rtol=1e-3)
@test isapprox(H[2,4] / H[2,3], 5, rtol=1e-3)
@test isapprox(H[1,4] / H[1,3], 5, rtol=1e-3)

function runtest(concs::Matrix, buckets::Matrix, ratios=nothing; concmatches=collect(1:size(concs, 2)), ratiomatches=Int[])
	numbuckets = size(buckets, 1)
	idxnan = isnan.(concs)
	mixerestimate, bucketestimate, objfuncval = NMFk.mixmatchdata(convert(Array{Float32, 2}, concs), numbuckets; ratios=ratios, regularizationweight=convert(Float32, 1e-3), verbosity=0)
	concs[idxnan] = 0
	predictedconcs = mixerestimate * bucketestimate
	predictedconcs[idxnan] = 0
	if length(concmatches) > 0
		@test norm(predictedconcs[:, concmatches] - concs[:, concmatches], 2) / norm(concs[:, concmatches], 2) < 1e-2 # fit the data within 1%
		for j = 1:size(buckets, 1)
			@test minimum(map(i->vecnorm(buckets[i, concmatches] - bucketestimate[j, concmatches]) / vecnorm(buckets[i, concmatches], 2), 1:size(buckets, 1))) < 3e-1 # reproduce the buckets within 30%
		end
	end
	checkratios(mixerestimate, bucketestimate, ratios, ratiomatches)
end

function checkratios(mixerestimate::Matrix, bucketestimate::Matrix, ratios::Void, ratiomatches)
	# if the ratios are nothing, do nothing
end
function checkratios(mixerestimate::Matrix, bucketestimate::Matrix, ratios, ratiomatches)
	# ratioestimate = similar(ratios)
	concs = mixerestimate * bucketestimate
	for i = 1:size(mixerestimate, 1)
		for j = 1:size(ratiomatches, 2)
			ratioratio = concs[i, ratiomatches[1, j]] / concs[i, ratiomatches[2, j]] / ratios[i, ratiomatches[1, j], ratiomatches[2, j]]
			@test ratioratio > .5 # get the ratio within a factor of 2
			@test ratioratio < 2.
		end
	end
end

function firsttest()
	nummixtures = 20
	numbuckets = 2
	numconstituents = 3
	for iternum = 1:10
		mixer = rand(nummixtures, numbuckets)
		for i = 1:nummixtures
			mixer[i, :] /= sum(mixer[i, :])
		end
		buckets = convert(Array{Float32,2}, [100 0 3; 5 10 20])
		data = convert(Array{Float32, 2}, mixer * buckets)
		data[1, 1] = NaN
		runtest(data, buckets)
	end
end

function nmfktest()
	M = convert(Array{Float64,2}, [1. 10. 0. 0. 1.; 0. 0. 1. 5. 2.])
	for iternum = 1:10
		a = rand(20)
		b = rand(20)
		S = [a b]
		for i = 1:size(S, 1)
			S[i, :] /= sum(S[i, :])
		end
		X = convert(Array{Float32, 2}, S * M)
		runtest(X, M)
	end
end

function ratiotest()
	nummixtures = 20
	numbuckets = 2
	numconstituents = 6
	for iternum = 1:10
		mixer = rand(nummixtures, numbuckets)
		for i = 1:nummixtures
			mixer[i, :] /= sum(mixer[i, :])
		end
		buckets = convert(Array{Float32, 2}, [100 0.001 .15 1 1. 3; 5 1 1 .2 .33 20])
		truedata = mixer * buckets
		data = fill(NaN, size(truedata))
		data[:, 1] = truedata[:, 1] # we only observe concentrations for the first constituent
		data[:, end] = truedata[:, end] # we only observe concentrations for the last constituent
		ratios = fill(NaN, nummixtures, numconstituents, numconstituents)
		ratiocomponents = Int[1 3; 2 4]
		for i = 1:nummixtures
			for j = 1:size(ratiocomponents, 2)
				a = ratiocomponents[1, j]
				b = ratiocomponents[2, j]
				ratios[i, a, b] = truedata[i, a] / truedata[i, b]
			end
		end
		runtest(convert(Array{Float32, 2}, data), buckets, ratios; concmatches=Int[1, 6], ratiomatches=ratiocomponents)
	end
end

function pureratiotest()
	nummixtures = 20
	numbuckets = 2
	numconstituents = 4
	for iternum = 1:10
		mixer = rand(nummixtures, numbuckets)
		for i = 1:nummixtures
			mixer[i, :] /= sum(mixer[i, :])
		end
		buckets = convert(Array{Float32, 2}, [0.001 1. .03 1.; .01 1. .0001 1.])
		truedata = mixer * buckets
		data = fill(NaN, size(truedata))
		ratios = fill(NaN, nummixtures, numconstituents, numconstituents)
		ratiocomponents = Int[1 3; 2 4]
		for i = 1:nummixtures
			for j = 1:size(ratiocomponents, 2)
				a = ratiocomponents[1, j]
				b = ratiocomponents[2, j]
				ratios[i, a, b] = truedata[i, a] / truedata[i, b]
			end
		end
		runtest(convert(Array{Float32, 2}, data), buckets, ratios; concmatches=Int[], ratiomatches=ratiocomponents)
	end
end

srand(2015)
# ratiotest()
firsttest() ## WARNING this needs fix for julia v0.6
nmfktest()
# pureratiotest()

a0 = Float64[[20,10,1] [5,1,1]]
b = NMFk.getisotopeconcentration(a0, [0.001,0.002], [[100,10,1] [500,50,5]])
a = NMFk.getisotopedelta(b, [0.001,0.002], [[100,10,1] [500,50,5]])
@test a0 ≈ a

a0 = Float64[20,10,1]
b = NMFk.getisotopeconcentration(a0, 0.001, [100,10,1])
a = NMFk.getisotopedelta(b, 0.001, [100,10,1])
@test a0 ≈ a

a0 = 20
b = NMFk.getisotopeconcentration(a0, 0.001, 100)
a = NMFk.getisotopedelta(b, 0.001, 100)
@test a0 ≈ a

srand(2015)
a = rand(15)
b = rand(15)
c = rand(15)
X = [a+c*3 a*10 b b*5+c a+b*2+c*5]
W, H, p, s = NMFk.execute(X, 2:4, 100; method=:ipopt);
W, H, p, s = NMFk.execute(X, 2:4, 100; method=:simple);

