import NMFk
import Base.Test

@NMFk.stderrcapture function runtest(concs::Matrix, buckets::Matrix, ratios::Array{Float32, 2}=Array{Float32}(0, 0), ratioindices::Union{Array{Int, 1},Array{Int, 2}}=Array{Int}(0, 0); conccomponents=collect(1:size(concs, 2)), ratiocomponents=Int[])
	numbuckets = size(buckets, 1)
	idxnan = isnan.(concs)
	mixerestimate, bucketestimate, objfuncval = NMFk.mixmatchdata(convert(Array{Float32, 2}, concs), numbuckets; random=false, ratios=ratios, ratioindices=ratiocomponents, regularizationweight=convert(Float32, 1e-3), maxiter=100, verbosity=0, tol=10., method=:ipopt)
	concs[idxnan] = 0
	predictedconcs = mixerestimate * bucketestimate
	predictedconcs[idxnan] = 0
	if length(conccomponents) > 0
		@Base.Test.test norm(predictedconcs[:, conccomponents] - concs[:, conccomponents], 2) / norm(concs[:, conccomponents], 2) < 1 # fit the data within 1%
		for j = 1:size(buckets, 1)
			@Base.Test.test minimum(map(i->vecnorm(buckets[i, conccomponents] - bucketestimate[j, conccomponents]) / vecnorm(buckets[i, conccomponents], 2), 1:size(buckets, 1))) < 1 # reproduce the buckets within 30%
		end
	end
	checkratios(mixerestimate, bucketestimate, ratios, ratiocomponents)
end

@NMFk.stderrcapture function checkratios(mixerestimate::Matrix, bucketestimate::Matrix, ratios::Array{Float32, 2}=Array{Float32}(0, 0), ratiocomponents::Union{Array{Int, 1},Array{Int, 2}}=Array{Int}(0, 0))
	if sizeof(ratios) == 0
		return
	end
	predictedconcs = mixerestimate * bucketestimate
	numberofratios = size(ratiocomponents, 2)
	for i = 1:size(mixerestimate, 1)
		for j = 1:numberofratios
			ratioratio = predictedconcs[i, ratiocomponents[j, 1]] / predictedconcs[i, ratiocomponents[j, 2]] / ratios[i, j]
			@Base.Test.test ratioratio > .4 # get the ratio within a factor of 2
			@Base.Test.test ratioratio < 4.
		end
	end
end

@NMFk.stderrcapture function buckettest()
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

@NMFk.stderrcapture function nmfktest()
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

@NMFk.stderrcapture function pureratiotest()
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
		ratiocomponents = Int[1 3; 2 4]'
		numberofratios = size(ratiocomponents, 1)
		ratiomatrix = Array{Float32, 2}(nummixtures, numberofratios)
		for i = 1:nummixtures
			for j = 1:numberofratios
				ratiomatrix[i, j] = truedata[i, ratiocomponents[j, 1]] / truedata[i, ratiocomponents[j, 2]]
			end
		end
		runtest(convert(Array{Float32, 2}, data), buckets, ratiomatrix; conccomponents=Int[], ratiocomponents=ratiocomponents)
	end
end

@NMFk.stderrcapture function ratiotest()
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
		ratiocomponents = Int[1 3; 2 6]'
		numberofratios = size(ratiocomponents, 1)
		ratiomatrix = Array{Float32, 2}(nummixtures, numberofratios)
		for i = 1:nummixtures
			for j = 1:numberofratios
				ratiomatrix[i, j] = truedata[i, ratiocomponents[j, 1]] / truedata[i, ratiocomponents[j, 2]]
			end
		end
		runtest(convert(Array{Float32, 2}, data), buckets, ratiomatrix; conccomponents=Int[1, 6], ratiocomponents=ratiocomponents)
	end
end

srand(2015)
info("NMFk: pure ratio test ...")
@NMFk.stdouterrcapture pureratiotest()
info("NMFk: ratio test ...")
@NMFk.stdouterrcapture ratiotest()
info("NMFk: bucket test ...")
@NMFk.stdouterrcapture buckettest()
info("NMFk: nmfk test ...")
@NMFk.stdouterrcapture nmfktest()

info("NMFk ipopt: 2 sources, 5 sensors, 20 transients")
srand(2015)
a = rand(20)
b = rand(20)
W = [a b]
H = [.1 1 0 0 .1; 0 0 .1 .5 .2]
X = W * H
X = [a a*10 b b*5 a+b*2]
@NMFk.stdouterrcapture We, He, p, s = NMFk.execute(X, 2, 10; method=:ipopt, tolX=1e-3, tol=1e-12)
@Base.Test.test isapprox(p, 0, atol=1e-3)
@Base.Test.test isapprox(s, 1, rtol=1e-1)
@Base.Test.test isapprox(He[1,2] / He[1,1], 10, rtol=1e-3)
@Base.Test.test isapprox(He[2,2] / He[2,1], 10, rtol=1e-3)
@Base.Test.test isapprox(He[2,4] / He[2,3], 5, rtol=1e-3)
@Base.Test.test isapprox(He[1,4] / He[1,3], 5, rtol=1e-3)

info("NMFk nlopt: 2 sources, 5 sensors, 20 transients")
srand(2015)
a = rand(20)
b = rand(20)
W = [a b]
H = [.1 1 0 0 .1; 0 0 .1 .5 .2]
X = W * H
X = [a a*10 b b*5 a+b*2]
@NMFk.stdouterrcapture We, He, p, s = NMFk.execute(X, 2, 10; method=:nlopt, tolX=1e-6, tol=1e-19)
@Base.Test.test isapprox(p, 0, atol=1e-3)
@Base.Test.test isapprox(s, 1, rtol=1e-1)
@Base.Test.test isapprox(He[1,2] / He[1,1], 10, rtol=1e-3)
@Base.Test.test isapprox(He[2,2] / He[2,1], 10, rtol=1e-3)
@Base.Test.test isapprox(He[2,4] / He[2,3], 5, rtol=1e-3)
@Base.Test.test isapprox(He[1,4] / He[1,3], 5, rtol=1e-3)


info("NMFk ipopt: 2 sources, 5 sensors, 100 transients")
srand(2015)
a = exp.(-(0:.5:10))*100
b = 100 + sin.(0:20)*10
X = [a a*10 b b*5 a+b*2]
@NMFk.stdouterrcapture W, H, p, s = NMFk.execute(X, 2, 10; method=:ipopt, tolX=1e-3, tol=1e-7)
@Base.Test.test isapprox(p, 0, atol=1e-3)
@Base.Test.test isapprox(s, 1, rtol=1e-1)
@Base.Test.test isapprox(H[1,2] / H[1,1], 10, rtol=1e-3)
@Base.Test.test isapprox(H[2,3] / H[1,3], 2.57, rtol=1e-1)
@Base.Test.test isapprox(H[2,4] / H[2,3], 5, rtol=1e-3)
@Base.Test.test isapprox(H[1,4] / H[1,3], 5, rtol=1e-3)

info("NMFk nlopt: 2 sources, 5 sensors, 100 transients")
srand(2015)
a = exp.(-(0:.5:10))*100
b = 100 + sin.(0:20)*10
X = [a a*10 b b*5 a+b*2]
@NMFk.stdouterrcapture W, H, p, s = NMFk.execute(X, 2, 10; method=:nlopt, tolX=1e-6, tol=1e-19)
@Base.Test.test isapprox(p, 0, atol=1e-3)
@Base.Test.test isapprox(s, 1, rtol=1e-1)
@Base.Test.test isapprox(H[1,2] / H[1,1], 10, rtol=1e-3)
@Base.Test.test isapprox(H[2,3] / H[1,3], 2.68, rtol=1e-1)
@Base.Test.test isapprox(H[2,4] / H[2,3], 5, rtol=1e-3)
@Base.Test.test isapprox(H[1,4] / H[1,3], 5, rtol=1e-3)

info("NMFk: 3 sources, 5 sensors, 15 transients")
srand(2015)
a = rand(15)
b = rand(15)
c = rand(15)
X = [a+c*3 a*10 b b*5+c a+b*2+c*5]
info("NMFk: ipopt ...")
@NMFk.stdouterrcapture W, H, p, s = NMFk.execute(X, 2:4, 10; maxiter=100, tol=1e-2, tolX=1e-2, method=:ipopt)
info("NMFk: nlopt ...")
@NMFk.stdouterrcapture W, H, p, s = NMFk.execute(X, 2:4, 10; maxiter=100, tol=1e-2, tolX=1e-2, method=:nlopt)
info("NMFk: simple ...")
@NMFk.stdouterrcapture W, H, p, s = NMFk.execute(X, 2:4, 10; maxiter=100, tol=1e-2, method=:simple)
info("NMFk: nmf ...")
@NMFk.stdouterrcapture W, H, p, s = NMFk.execute(X, 2:4, 10; maxiter=100, tol=1e-2, method=:nmf)
info("NMFk: sparse ...")
@NMFk.stdouterrcapture W, H, p, s = NMFk.execute(X, 2:4, 10; maxiter=100, tol=1e-2, method=:sparse)

info("NMFk: concentrantions/delta tests ...")
a0 = Float64[[20,10,1] [5,1,1]]
b = NMFk.getisotopeconcentration(a0, [0.001,0.002], [[100,10,1] [500,50,5]])
a = NMFk.getisotopedelta(b, [0.001,0.002], [[100,10,1] [500,50,5]])
@Base.Test.test a0 ≈ a

a0 = Float64[20,10,1]
b = NMFk.getisotopeconcentration(a0, 0.001, [100,10,1])
a = NMFk.getisotopedelta(b, 0.001, [100,10,1])
@Base.Test.test a0 ≈ a

a0 = 20.0
b = NMFk.getisotopeconcentration(a0, 0.001, 100)
a = NMFk.getisotopedelta(b, 0.001, 100)[1]
@Base.Test.test a0 ≈ a
:passed