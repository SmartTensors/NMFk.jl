import NMFk

@info("Reconstruction of sin/rand disturbance signal ...")
Random.seed!(2015)
noise = [0, 0.1, 0.2, 0.5, 1]
suc = Array{Int}(undef, length(noise))
for n in 1:length(noise)
	suc[n] = 0
	for i = 1:1000
		s1 = (sin.(0.3:0.3:30) .+ 1) ./ 2
		s2 = rand(100) .* 0.5
		s3 = rand(100)
		s3[1:50] .= 0
		s3[70:end] .= 0
		S = [s1 s2 s3]
		M = [[1,1,1] [0,2,1] [0,2,1] [1,0,2] [2,0,1] [1,2,0] [2,1,0]]
		X = S * M + rand(100, 7) .* noise[n]
		W, H, fitquality, robustness, aic = NMFk.execute(X, 2:4, 100)
		if robustness[2] > 0.9 && robustness[3] > 0.9 && robustness[4] < 0.9
			suc[n] += 1
			@info("Success!")
		end
	end
end
[noise suc]