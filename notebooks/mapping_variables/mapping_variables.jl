import NMFk

import Mads

import Statistics

A = permutedims([0.168427        0.049914        0.031383        0.020747        0.007634        0.004797        0.003955
	   0.959030        0.203276        0.095674        0.043699        0.000000        0.000000        0.000000
	   0.208403        0.064995        0.039014        0.019713        0.002357        0.000000        0.000000
	   0.948621        0.217649        0.101904        0.049093        0.024234        0.012169        0.008160])

B = permutedims([0.654060        0.142989        0.043485        0.000000        0.000000        0.000000        0.000000
	   1.000000        0.090943        0.048150        0.018898        0.006329        0.001725        0.000258
	   0.076188        0.020636        0.011489        0.006166        0.002998        0.000000        0.000000
	   0.378206        0.098391        0.041083        0.009261        0.000000        0.000000        0.000000
	   0.055413        0.021730        0.010460        0.004788        0.001719        0.000000        0.000000])

X = permutedims([0.500        0.002        0.667        0.40
	   0.800        0.200        0.667        0.76
	   0.800        0.100        0.400        0.80
	   0.600        0.010        1.000        0.40])

Y = permutedims([1.000        0.600        0.267        1.00
	   0.700        0.020        0.333        0.60
	   1.000        0.020        0.200        0.72
	   0.700        1.000        0.233        0.60
	   1.000        0.060        0.133        0.80])

Z = permutedims([0.800        0.400        0.100        0.60]);

Mads.plotseries(A; name="Well", logy=true, title="Well Group A (matrix A)")

Mads.plotseries(B; name="Well", logy=true, title="Well Group B (matrix B)")

NMFk.plotmatrix(Y; title="Attribute matrix Y (Well Group A)", xticks=["W$i" for i=1:5], yticks=["Attribute $i" for i=1:4])

NMFk.plotmatrix(X; title="Attribute matrix X (Well Group B)", xticks=["W$i" for i=1:4], yticks=["Attribute $i" for i=1:4])

W, H, of, sil, aic = NMFk.mapping(X, Y, A, B; method=:ipopt, save=false);

NMFk.normnan(B .- (A * H))

Mads.plotseries(A * H; logy=true, name="Well", linestyle=:dash)

Mads.plotseries(A * H; linestyle=:dash, name="Well (est.)", logy=true, gl=Mads.plotseries(B; name="Well (true)", code=true, quiet=true))
