import NMFk

@info("Reconstruction of sin/rand disturbance signal ...")
Random.seed!(2015)
nTests = 10 # number of tests
noise = [0, 0.1, 0.2, 0.5, 1] # noise levels
xsize = 1000 # signal size
suc = Array{Float64}(undef, length(noise))
for n in 1:length(noise)
	suc[n] = 0
	for i = 1:nTests
		s1 = (sin.(30/xsize:30/xsize:30) .+ 1) ./ 2 # signal 1
		s2 = rand(xsize) .* 0.5 # signal 2
		s3 = rand(xsize) # signal 3
		s3[1:convert(Int, ceil(xsize/2))] .= 0
		s3[convert(Int, ceil(3*xsize/4)):end] .= 0
		S = [s1 s2 s3] # matrix with 3 signal
		M = [[1,1,1] [0,2,1] [0,2,1] [1,0,2] [2,0,1] [1,2,0] [2,1,0]] # mixing matrix
		X = S * M + rand(xsize, 7) .* noise[n] # data matrix to process with noise
		W, H, fitquality, robustness, aic = NMFk.execute(X, 2:4, 10) # run for 2, 3 and 4 sources
		if robustness[2] > 0.9 && robustness[3] > 0.9 && robustness[4] < 0.9 # success if the number of signals is 3
			suc[n] += 1
			@info("NMFk Success!")
		else
			@warn("NMFk Failure!")
		end
	end
	suc[n] /= nTests
end
[noise suc]

s1 = (sin.(30/xsize:30/xsize:30) .+ 1) ./ 2 # signal 1
s2 = rand(xsize) * 0.5 # signal 2
s3 = rand(xsize) # signal 3
s3[1:convert(Int, ceil(xsize/2))] .= 0
s3[convert(Int, ceil(3*xsize/4)):end] .= 0
S = [s1 s2 s3] # matrix with 3 signal
M = [[1,1,1] [0,2,1] [0,2,1] [1,0,2] [2,0,1] [1,2,0] [2,1,0]] # mixing matrix
X = S * M + rand(xsize, 7)
NMFk.execute(X, 3, 10; maxiter=10, quiet=false)
