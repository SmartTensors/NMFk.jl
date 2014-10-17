using Gadfly # Gadly demages Clustering
using DataFrames
using Compose
using NMF
using Clustering
using MultivariateStats
using Wells
using Optim
using Calculus

include("nmfk-test-20141013.jl")
#include("nmfk-test-20141012.jl")
#include("nmfk-test-20141005.jl")
intermediate_figs = false
nNMF=1
dd = Wells.solve( WellsD, WellsQ, Points, time, T, S )
println(sort(collect(keys(dd))))
nP = numrows = size(collect(keys(dd)))[1]
nT = numcols = size(dd[collect(keys(dd))[1]])[1]
X = Array(Float64, nP, nT)
W = Array(Float64, nP, nk)
H = Array(Float64, nk, nT)
Hcheat = Array(Float64, nk, nT)
df = Array(Any, numrows)
pl = Array(Plot, numrows)
dW = Wells.solve( "R-28", WellsD, WellsQ, Points, time, T, S )
i = 0
for w in sort(collect(keys(WellsD)))
	i += 1
	pl[i] = plot( x=time, y=dW[w], Guide.XLabel("Time [d]"), Guide.title("Well $w"), Geom.line)
	Hcheat[i,:] = dW[w]
end
nW= i
p = vstack( pl[1:nW] )
draw(PNG(string("nmfk-test-$testproblem-r28-dd.png"), 18inch, 12inch), p)
dW = Wells.solve( 0.1, WellsD, WellsQ, time, T, S )
i = 0
for w in sort(collect(keys(WellsD)))
	i += 1
	pl[i] = plot( x=time, y=dW[w], Guide.XLabel("Time [d]"), Guide.title("Well $w"), Geom.line)
end
nW = i
p = vstack( pl[1:nW] )
draw(PNG(string("nmfk-test-$testproblem-wdd.png"), 18inch, 12inch), p)
i = 0
for k in sort(collect(keys(dd)))
	i += 1
	println(k)
	X[i,:] = dd[k]
	pl[i] = plot(x=time, y=dd[k], Guide.XLabel("Time [d]"), Guide.YLabel("Drawdown [m]"), Guide.title(k), Geom.line )
end
numfigrows = 3
remainder = numrows % numfigrows
if remainder == 0 
	cs = reshape([Context[render(pl[i]) for i in 1:numrows]],numfigrows,iround(numrows/numfigrows));
else
	cs = reshape([Context[render(pl[i]) for i in 1:numrows],[context() for i in remainder+1:numfigrows]],numfigrows,iceil(numrows/numfigrows));
end
p = gridstack(cs)
draw(PNG(string("nmfk-test-$testproblem-input.png"), 18inch, 12inch), p)
println("Size of the matrix to solve ",size(X))
writecsv("nmfk-test-$testproblem-well-names.csv",collect(keys(dd))')
writecsv("nmfk-test-$testproblem.csv",X')

# RANDOM test
# X = rand(5, 1000)

Wt = llsq(Hcheat',X'; bias=false)
W = Wt'
# println("Size of W = ", size(W) )
# println("Size of H = ", size(H) )

WBig = Array(Float64, numrows, 0)
HBig = Array(Float64, 0, numcols)
P = Array(Float64, numrows, numcols)
phi = Array(Float64, nNMF)
for n = 1:nNMF
	# initialize W & H matrices randomly
	# W, H = NMF.randinit(X, nk, normalize=true)
	W, H = NMF.randinit(X, nk)

	# initialize W & H using Non-Negative Double Singular Value Decomposition (NNDSVD) algorithm
	# Reference: C. Boutsidis, and E. Gallopoulos. SVD based initialization: A head start for nonnegative matrix factorization. Pattern Recognition, 2007.
	# W, H = NMF.nndsvd(X, nk)
	# H = Hcheat

	# println("Size of W = ", size(W) )
	# println("Size of H = ", size(H) )

	# Solve NMF
	NMF.solve!(NMF.MultUpdate(obj=:div,maxiter=200000,tol=1.0e-6,lambda=1,lambda=1.0e-9), X, W, H)
	# NMF.solve!(NMF.ProjectedALS(maxiter=100), X, W, H)
	# NMF.solve!(NMF.ALSPGrad(maxiter=100, tolg=1.0e-6), X, W, H)
	P = W * H
	E = X - P
	phi[n] = sum( E' * E )
	println("NMF ", n, " Objective function = ", phi[n], " Max error = ", maximum(E), " Min error = ", minimum(E) )
	if intermediate_figs
		i = 0
		for k in sort(collect(keys(dd)))
			i += 1
			#df1 = DataFrame(x=time, y=dd[k], label="data")
			#df2 = DataFrame(x=time, y=dd[k], label="model")
			#df[i] = vcat(df1, df2)
			#pl[i] = plot(df, x="x", y="y", color="label", Guide.XLabel("Time [d]"), Guide.YLabel("Drawdown [m]"), Guide.title(k), Geom.line, Scale.discrete_color_manual("blue","red") )
			pl[i] = plot(
			layer(x=time, y=P[i,:], Geom.point, Theme(default_color=color("white"), default_point_size=1pt)),
			layer(x=time, y=X[i,:], Geom.line, Theme(default_color=color("red"))),
			Guide.XLabel("Time [d]"), Guide.YLabel("Drawdown [m]"), Guide.title(k) )
		end
		if remainder == 0 
			cs = reshape([Context[render(pl[i]) for i in 1:numrows]],numfigrows,iround(numrows/numfigrows));
		else
			cs = reshape([Context[render(pl[i]) for i in 1:numrows],[context() for i in remainder+1:numfigrows]],numfigrows,iceil(numrows/numfigrows));
		end
		# p = vstack( pl )
		p = gridstack(cs)
		draw(PNG(string("nmfk-test-$testproblem-output-",n,".png"), 18inch, 12inch), p)
		for i in 1:nk
			pl[i] = plot( x=time, y=H[i,:], Guide.XLabel("Time [d]"), Guide.title("Source $i"), Geom.line)
		end
		p = vstack( pl[1:nk] )
		draw(PNG(string("nmfk-test-$testproblem-sources-",n,".png"), 18inch, 12inch), p)
	end
	# println("Size of W = ", size(W) )
	# println("Size of WBig = ", size(WBig) )
	# println("Size of HBig = ", size(HBig) )
	WBig=[WBig W]
	HBig=[HBig, H]
	# println(WBig)
	# println(W)
end

if nNMF > 1 
	println("NMFk done.")
	println("Size of WBig = ", size(WBig) )
	println("Size of HBig = ", size(HBig) )
	# performs K-means over W, trying to group them into nk clusters
	# set maximum number of iterations to 200
	# set display to :iter, so it shows progressive info at each iteration
	R = kmeans(HBig', nk; maxiter=200, display=:iter)

	# the number of resultant clusters should be nk
	@assert nclusters(R) == nk

	# Cluster assignments
	# a[i] indicates which cluster the i-th sample is assigned to
	clusterassignments = assignments(R)
	# println(a)

	# Number of samples in each cluster
	# c[k] is the number of samples assigned to the k-th cluster
	clustercounts = counts(R)
	println("Number of samples in eash cluster = ", clustercounts)

	# Cluster centers (i.e. mean vectors)
	# M is a matrix of size (numcols, nk)
	# M[:,nk] is the mean vector of the k-th cluster
	M = R.centers
	println("Size of matrix containing cluster centers = ", size(M))
	# println("Cluser centers = ", M)
	Ha = M'
	Wt = llsq(Ha',X'; bias=false)
	Wa = Wt'
	println("Size of the new weigth matrix = ", size(Wa))
	P = Wa * Ha
	E = X - P
	phi_final = sum( E' * E )
	println("Objective function = ", phi_final, " Max error = ", maximum(E), " Min error = ", minimum(E) )

	# Silhouettes
	s = silhouettes(clusterassignments, clustercounts, HBig*HBig')
	println("Silhouettes vector size = ", size(s))
	println("Silhouettes Avg = ", sum(s)/size(s)[1], " Max = ", maximum(s), " Min = ", minimum(s) )
	# println("Silhouettes vector = ", s)

	# K-medoids is a clustering algorithm that seeks a subset of points out of a given set such that
	# the total costs or distances between each point to the closest point in the chosen subset is minimal.
	# This chosen subset of points are called medoids.
	# Q = kmedoids(W, nk; maxiter=200, display=:iter)
else
	Ha = HBig
	Wa = WBig
end

writecsv(string("nmfk-test-$testproblem-sources-NMFk=",nk,"-",nNMF,".csv"),Ha)
for i in 1:nk
	pl[i] = plot( x=time, y=Ha[i,:], Guide.XLabel("Time [d]"), Guide.title("Source $i"), Geom.line)
end
p = vstack( pl[1:nk] )
draw(PNG(string("nmfk-test-$testproblem-sources-NMFk=",nk,"-",nNMF,".png"), 18inch, 12inch), p)
writecsv(string("nmfk-test-$testproblem-weights-NMFk=",nk,"-",nNMF,".csv"),Wa)
i = 0
for k in sort(collect(keys(dd)))
	i += 1
	pl[i] = plot(
	layer(x=time, y=P[i,:], Geom.point, Theme(default_color=color("white"), default_point_size=1pt)),
	layer(x=time, y=X[i,:], Geom.line, Theme(default_color=color("red"))),
	Guide.XLabel("Time [d]"), Guide.YLabel("Drawdown [m]"), Guide.title(k) )
end
if remainder == 0 
	cs = reshape([Context[render(pl[i]) for i in 1:numrows]],numfigrows,iround(numrows/numfigrows));
else
	cs = reshape([Context[render(pl[i]) for i in 1:numrows],[context() for i in remainder+1:numfigrows]],numfigrows,iceil(numrows/numfigrows));
end
p = gridstack(cs)
draw(PNG(string("nmfk-test-$testproblem-output-NMFk=",nk,"-",nNMF,".png"), 18inch, 12inch), p)

# println(Wa)
pname = sort(collect(keys(Points)))
nP = length(pname)
px = Array(Float64,nP)
py = Array(Float64,nP)
pz = Array(Float64,nk,nP)
pl = Array(String,nk,nP)
i = 0
for k in sort(collect(keys(Points)))
	i += 1
	px[i] = Points[k][1]
	py[i] = Points[k][2]
	for j in 1:nk
		pz[j,i] = Wa[i,j]
		pl[j,i] = @sprintf( "%.3f", Wa[i,j])
	end
end

nW = length(collect(keys(WellsD)))
wname = Array(String,nW)
wx = Array(Float64,nW)
wy = Array(Float64,nW)
i = 0
for k in sort(collect(keys(WellsD)))
	i += 1
	wx[i] = WellsD[k][1]
	wy[i] = WellsD[k][2]
	wname[i] = k
end
dfw = DataFrame(x=wx, y=wy, label=wname, label2=wname, category="wells")

function r1( x::Vector )	
	d = Array(Float64,nP) 
	println( x )
	for k in 1:nP
		d[k] = ( x[1] / sqrt( ( px[k] - x[2] )^2 + ( py[k] - x[3] )^2 ) ) - target[k]
		println( target[k], "->", x[1] / sqrt( ( px[k] - x[2] )^2 + ( py[k] - x[3] )^2 ) )
	end
	return d
end

function r1g( x::Vector )
	l = length(x)
	d = Array(Float64,nP,l)
	for k in 1:nP
		d[k,1] = - 1 / sqrt((px[k] - x[2])^2 + (py[k] - x[3])^2)
		d[k,2] = ((-((-2 * (px[k] - x[2])) * (0.5 / sqrt((px[k] - x[2])^2 + (py[k] - x[3])^2))) * x[1]) / sqrt((px[k] - x[2])^2 + (py[k] - x[3])^2)^2)
		d[k,3] = ((-((-2 * (py[k] - x[3])) * (0.5 / sqrt((px[k] - x[2])^2 + (py[k] - x[3])^2))) * x[1]) / sqrt((px[k] - x[2])^2 + (py[k] - x[3])^2)^2)
	end
	return d
end

function r2( x::Vector )	
	d = Array(Float64,nP)
	for k in 1:nP
		d[k] = ( x[1] / ( ( px[k] - x[2] )^2 + ( py[k] - x[3] )^2 ) ) - target[k]
	end
	return d
end

function r2g( x::Vector )
	l = length(x)
	d = Array(Float64,nP,l)
	for k in 1:nP
		d[k,1] = -1 / ((px[k] - x[2])^2 + (py[k] - x[3])^2)
		d[k,2] = -(-(-2 * (px[k] - x[2])) * x[1]) / ((px[k] - x[2])^2 + (py[k] - x[3])^2)^2
		d[k,3] = -(-(-2 * (py[k] - x[3])) * x[1]) / ((px[k] - x[2])^2 + (py[k] - x[3])^2)^2
	end
	return d
end

function rn( x::Vector )	
	d = Array(Float64,nP)
	for k in 1:nP
		d[k] = ( x[1] / ( ( px[k] - x[2] )^2 + ( py[k] - x[3] )^2 )^x[4] ) - target[k]
	end
	return d
end

function rng( x::Vector )
	l = length(x)
	d = Array(Float64,nP,l)
	for k in 1:nP
		d[k,1] = (1 / ((px[k] - x[2])^2 + (py[k] - x[3])^2)^x[4])
		d[k,2] = ((-(x[4] * (-2 * (px[k] - x[2])) * ((px[k] - x[2])^2 + (py[k] - x[3])^2)^(x[4] - 1)) * x[1]) / (((px[k] - x[2])^2 + (py[k] - x[3])^2)^x[4])^2)
		d[k,3] = ((-(x[4] * (-2 * (py[k] - x[3])) * ((px[k] - x[2])^2 + (py[k] - x[3])^2)^(x[4] - 1)) * x[1]) / (((px[k] - x[2])^2 + (py[k] - x[3])^2)^x[4])^2)
		d[k,4] = ((-(((px[k] - x[2])^2 + (py[k] - x[3])^2)^x[4] * log((px[k] - x[2])^2 + (py[k] - x[3])^2)) * x[1]) / (((px[k] - x[2])^2 + (py[k] - x[3])^2)^x[4])^2)
	end
	return d
end

function logr2( x::Vector )	
	d = Array(Float64,nP)
	for k in 1:nP
		d[k] = ( x[1] * log ( x[2] / ( ( px[k] - x[3] )^2 + ( py[k] - x[4] )^2 ) ) ) - target[k]
	end
	return d
end

function logr2g( x::Vector )
	l = length(x)
	d = Array(Float64,nP,l)
	for k in 1:nP
		d[k,1] = (log(x[2] / ((px[k] - x[2])^2 + (py[k] - x[3])^2)))
		d[k,2] = (x[1] * ((1 / ((px[k] - x[3])^2 + (py[k] - x[4])^2)) * (1 / (x[2] / ((px[k] - x[3])^2 + (py[k] - x[4])^2)))))
		d[k,3] = (x[1] * (((-(-2 * (px[k] - x[3])) * x[2]) / ((px[k] - x[3])^2 + (py[k] - x[4])^2)^2) * (1 / (x[2] / ((px[k] - x[3])^2 + (py[k] - x[4])^2)))))
		d[k,4] = (x[1] * (((-(-2 * (py[k] - x[4])) * x[2]) / ((px[k] - x[3])^2 + (py[k] - x[4])^2)^2) * (1 / (x[2] / ((px[k] - x[3])^2 + (py[k] - x[4])^2)))))
	end
	return d
end

target = Array(Float64, nP)
for i in 1:nk
	target = collect(pz[i,:])
	# results = Optim.levenberg_marquardt(r2, r2g, [1.0,499100.0,539100.0], show_trace=true, maxIter=500)
	results = Optim.levenberg_marquardt(rn, rng, [1.0,499100.0,539100.0,0.2], maxIter=1000, tolG=1e-19)
	# results = Optim.levenberg_marquardt(logr2, logr2g, [1.0,1000000,499100.0,539100.0], maxIter=1000, tolG=1e-19)
	# results = Optim.levenberg_marquardt(r2, r2g, [10000.0,499100.0,539100.0], maxIter=1000, tolG=1e-19)
	println(results)
	pred = rn( results.minimum )
	for j in 1:nP
		pl[i,j] = @sprintf( "%.2f-%.2f=%.2f", pred[j]-target[j],target[j],pred[j])
	end
	df = DataFrame(x=px, y=py, label=pname, label2=collect(pl[i,:]), category="points" )
	dfr = DataFrame(x=results.minimum[2], y=results.minimum[3], label="E", label2="E", category="result" )
	p = plot(vcat(df,dfw,dfr), x="x", y="y", label=4, color="category", Geom.point, Geom.label, 
	Guide.XLabel("x [m]"), Guide.YLabel("y [m]"), Guide.title("Source $i") )
	draw(PNG(string("nmfk-test-$testproblem-output-NMFk=",nk,"-",nNMF,"-S$i.png"), 8inch, 6inch), p)
end
