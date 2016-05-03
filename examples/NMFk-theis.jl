using Gadfly # Gadly demages Clustering
using DataFrames
using Compose
using NMF
using Clustering
using MultivariateStats
using Optim
using Wells
pwd()
dirname(Base.source_path())
push!(LOAD_PATH, dirname(Base.source_path()))
cd(dirname(Base.source_path()))
using NMFk

# read the problem setup
include("nmfk-test-20150721.jl")
# include("nmfk-test-20141013.jl")
#include("nmfk-test-20141012.jl")
#include("nmfk-test-20141005.jl")
if !isdir("nmfk-test-$testproblem")
	mkdir("nmfk-test-$testproblem")
end
# flags
intermediate_figs = false
flag_kmeans = false # true = buildin kmeans; false = use clustering in NMFk
source_location_identification = true # identify spatial location of the sources using LM and dummy radial functions
nNMF=10 # number of NMFk's

# solve the Theis problem for all the wells
dd = Wells.solve( WellsD, WellsQ, Points, time, T, S )
println("Observation points: ", sort(collect(keys(dd))))

nP = numrows = size(collect(keys(dd)))[1] # number of observation points (number of observation records)
nT = numcols = size(dd[collect(keys(dd))[1]])[1] # number of observations for each point
X = Array(Float64, nP, nT) # input matrix of observations
W = Array(Float64, nP, nk) # estimated weight matrix
H = Array(Float64, nk, nT) # estimated source matrix
P = Array(Float64, nP, nT) # model prediction matrix
phi = Array(Float64, nNMF) # vector for the computed objective functions
WBig = Array(Float64, nP, 0) # estimated weight matrix collected over a series of NMFk's (initialized emptu)
HBig = Array(Float64, 0, nT) # estimated source matrix collected over a series of NMFk's (initialized emptu)
Hcheat = Array(Float64, nk, nT) # initial guess for the source matrix
df = Array(Any, nP) # DataFrames matrix needed for ploting
pl = Array(Plot, nP) # Plot matrix

# solve the Theis problem for R-28 to comute initial H guess (Hcheat)
dW = Wells.solve( "R3", WellsD, WellsQ, Points, time, T, S )
i = 0
for w in sort(collect(keys(WellsD)))
	i += 1
	pl[i] = plot( x=time, y=dW[w], Guide.XLabel("Time [d]"), Guide.title("Well $w"), Geom.line)
	Hcheat[i,:] = dW[w]
end
nW = i # number of true sources (number of pumping wells)
p = vstack( pl[1:nW] )
draw(PNG(string("nmfk-test-$testproblem/nmfk-test-$testproblem-r28-dd.png"), 18inch, 12inch), p)

# solve the Theis problem for a pumping well (radius set to 0.1 which is the borehole diameter)
dW = Wells.solve( 0.1, WellsD, WellsQ, time, T, S )
i = 0
for w in sort(collect(keys(WellsD)))
	i += 1
	pl[i] = plot( x=time, y=dW[w], Guide.XLabel("Time [d]"), Guide.title("Well $w"), Geom.line)
end
nW = i # number of true sources (number of pumping wells)
p = vstack( pl[1:nW] )
draw(PNG(string("nmfk-test-$testproblem/nmfk-test-$testproblem-wdd.png"), 18inch, 12inch), p)

# plot the drawdowns for all the wells
i = 0
for k in sort(collect(keys(dd)))
	i += 1
	# println("Plotting ",k)
	X[i,:] = dd[k] # setup the input NMFk matrix
	# generate the plot for each well
	pl[i] = plot(x=time, y=dd[k], Guide.XLabel("Time [d]"), Guide.YLabel("Drawdown [m]"), Guide.title(k), Geom.line, Theme(default_color=color("red"),line_width=2pt) )
end
# form a matrx plot
numfigrows = 3
remainder = numrows % numfigrows
if remainder == 0
	cs = reshape([Context[render(pl[i]) for i in 1:numrows]],numfigrows,iround(numrows/numfigrows));
else
	cs = reshape([Context[render(pl[i]) for i in 1:numrows],[context() for i in remainder+1:numfigrows]],numfigrows,iceil(numrows/numfigrows));
end
p = gridstack(cs)
draw(PNG(string("nmfk-test-$testproblem/nmfk-test-$testproblem-input.png"), 18inch, 12inch), p)
println("Size of the X matrix to solve ",size(X))
writecsv("nmfk-test-$testproblem/nmfk-test-$testproblem-well-names.csv",collect(keys(dd))')
writecsv("nmfk-test-$testproblem/nmfk-test-$testproblem.csv",X')

# prepare the data for the DataFrames
nP = length(collect(keys(Points)))
pname = Array(String,nP)
px = Array(Float64,nP)
py = Array(Float64,nP)
i = 0
for k in sort(collect(keys(Points)))
i += 1
px[i] = Points[k][1]
py[i] = Points[k][2]
pname[i] = k
end
# create a data frame for the observation points
dfp = DataFrame(x=px, y=py, label=pname, info=pname, category="points")

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
# create a data frame for the pumping wells
dfw = DataFrame(x=wx, y=wy, label=wname, info=wname, category="wells")

# RANDOM test
# X = rand(5, 1000)

W = [Hcheat' \ X']'
# println("Size of W = ", size(W) )
# println("Size of H = ", size(H) )

info("NMF runs ... ($nNMF)")
for n = 1:nNMF
	# initialize W & H matrices randomly
	# W, H = NMF.randinit(X, nk, normalize=true)
	W, H = NMF.randinit(X, nk)

	# initialize W & H using Non-Negative Double Singular Value Decomposition (NNDSVD) algorithm
	# Reference: C. Boutsidis, and E. Gallopoulos. SVD based initialization: A head start for nonnegative matrix factorization. Pattern Recognition, 2007.
	# W, H = NMF.nndsvd(X, nk)

	# use a good guess for H (Hcheat) for testing
	H = Hcheat

	# println("Size of W = ", size(W) )
	# println("Size of H = ", size(H) )

	# Solve NMF
	NMF.solve!(NMF.MultUpdate(obj=:div,maxiter=200000,tol=1.0e-6,lambda=1,lambda=1.0e-9), X, W, H)
	# NMF.solve!(NMF.ProjectedALS(maxiter=100), X, W, H)
	# NMF.solve!(NMF.ALSPGrad(maxiter=100, tolg=1.0e-6), X, W, H)
	P = W * H
	E = X - P
	phi[n] = sum( E' * E )
	println("NMF #", n, " Objective function = ", phi[n], " Max error = ", maximum(E), " Min error = ", minimum(E) )
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
		draw(PNG(string("nmfk-test-$testproblem/nmfk-test-$testproblem-output-",n,".png"), 18inch, 12inch), p)
		for i in 1:nk
			pl[i] = plot( x=time, y=H[i,:], Guide.XLabel("Time [d]"), Guide.title("Source $i"), Geom.line)
		end
		p = vstack( pl[1:nk] )
		draw(PNG(string("nmfk-test-$testproblem/nmfk-test-$testproblem-sources-",n,".png"), 18inch, 12inch), p)
	end
	# println("Size of W = ", size(W) )
	# println("Size of WBig = ", size(WBig) )
	# println("Size of HBig = ", size(HBig) )
	WBig=[WBig W]
	HBig=[HBig, H]
	# println(WBig)
	# println(W)
end
info("NMF runs done.")

if nNMF > 1
	println("Size of WBig = ", size(WBig) )
	println("Size of HBig = ", size(HBig) )
	if flag_kmeans
		info("NMFk analysis of the NMF runs using Julia kmeans algorithm")
		# performs K-means over W, trying to group them into nk clusters
		# set maximum number of iterations to 200
		# set display to :iter, so it shows progressive info at each iteration
		R = kmeans(HBig', nk; maxiter=200, display=:iter)

		# the number of resultant clusters should be nk
		@assert nclusters(R) == nk

		# Cluster assignments
		# clusterassignments[i] indicates which cluster the i-th sample is assigned to
		clusterassignments = assignments(R)
		# println(clusterassignments)

		# Number of samples in each cluster
		# clustercounts[k] is the number of samples assigned to the k-th cluster
		clustercounts = counts(R)
		println("Number of samples in eash cluster = ", clustercounts)

		# Cluster centers (i.e. mean vectors)
		# M is a matrix of size (numcols, nk)
		# M[:,nk] is the mean vector of the k-th cluster
		M = R.centers
		println("Size of the R matrix containing cluster centers = ", size(M))
		# println("Cluser centers = ", M)
		Ha = M'
		Wa = [Ha' \ X']'
		println("Size of the new weigth matrix (Wa) = ", size(Wa))
		P = Wa * Ha
		E = X - P
		phi_final = sum( E' * E )
		println("Objective function = ", phi_final, " Max error = ", maximum(E), " Min error = ", minimum(E) )

		# Silhouettes
		WaDist = Distances.pairwise(Distances.CosineDist(), WBig')
		s = Clustering.silhouettes(clusterassignments, clustercounts, WaDist)
		println("Silhouettes vector size = ", size(s))
		println("Silhouettes Avg = ", sum(s)/size(s)[1], " Max = ", maximum(s), " Min = ", minimum(s) )
		# println("Silhouettes vector = ", s)

		# K-medoids is a clustering algorithm that seeks a subset of points out of a given set such that
		# the total costs or distances between each point to the closest point in the chosen subset is minimal.
		# This chosen subset of points are called medoids.
		# Q = kmedoids(W, nk; maxiter=200, display=:iter)
	else
		info("NMFk analysis of the NMF runs using NMFk kmeans algorithm")
		# use imrpoved kmeans clustering accounting for the expected number of samples in each cluster
		clusterassignments, M = NMFk.clustersolutions(HBig', nNMF)
		# println("clusterassignments ", clusterassignments )
		# println("centroids ", M )
		Wa, Ha, avgStabilityProcesses = NMFk.finalize(WBig, HBig, nNMF, clusterassignments);
		println("Size of Ha = ", size(Ha) )
		println("Size of Wa = ", size(Wa) )
		println("Silhouettes Avg = ", avgStabilityProcesses )
		P = Wa * Ha;
		E = X - P;
		phi_final = sum( E' * E )
		println("Objective function = ", phi_final, " Max error = ", maximum(E), " Min error = ", minimum(E) )
		Pcorr = zeros(nT, 1);
		for i = 1:nT
				Pcorr[i] = cor( X[:,i], P[:, i] );
		end
	end
else
	Ha = HBig
	Wa = WBig
end

writecsv(string("nmfk-test-$testproblem/nmfk-test-$testproblem-sources-NMFk=",nk,"-",nNMF,".csv"),Ha)
for i in 1:nk
	pl[i] = plot( x=time, y=Ha[i,:], Guide.XLabel("Time [d]"), Guide.title("Source $i"), Geom.line)
end
p = vstack( pl[1:nk] )
draw(PNG(string("nmfk-test-$testproblem/nmfk-test-$testproblem-sources-NMFk=",nk,"-",nNMF,".png"), 18inch, 12inch), p)
writecsv(string("nmfk-test-$testproblem/nmfk-test-$testproblem-weights-NMFk=",nk,"-",nNMF,".csv"),Wa)
i = 0
for k in sort(collect(keys(dd)))
	i += 1
	pl[i] = plot(
	layer(x=time, y=P[i,:], Geom.point, Theme(default_color=color("white"), default_point_size=1pt)),
	layer(x=time, y=X[i,:], Geom.line, Theme(default_color=color("red"),line_width=2pt)),
	Guide.XLabel("Time [d]"), Guide.YLabel("Drawdown [m]"), Guide.title(k) )
end
if remainder == 0
	cs = reshape([Context[render(pl[i]) for i in 1:numrows]],numfigrows,iround(numrows/numfigrows));
else
	cs = reshape([Context[render(pl[i]) for i in 1:numrows],[context() for i in remainder+1:numfigrows]],numfigrows,iceil(numrows/numfigrows));
end
p = gridstack(cs)
draw(PNG(string("nmfk-test-$testproblem/nmfk-test-$testproblem-output-NMFk=",nk,"-",nNMF,".png"), 18inch, 12inch), p)

if source_location_identification
	info("Identification of the spatial location of the sources ...")
	# println(Wa)
	p = plot(vcat(dfp,dfw), x="x", y="y", label=3, color="category", Geom.point, Geom.label,
	Guide.XLabel("x [m]"), Guide.YLabel("y [m]"), Guide.yticks(orientation=:vertical), Scale.x_continuous(labels=x -> @sprintf("%.0f", x)))
	draw(SVG(string("nmfk-test-$testproblem/nmfk-test-$testproblem.svg"), 8inch, 6inch), p)
	# draw(PNG(string("nmfk-test-$testproblem/nmfk-test-$testproblem.png"), 8inch, 6inch), p)
	target = Array(Float64, nP)
	dfr = DataFrame(x = Float64[], y = Float64[], label = String[], info = String[], category =  String[])
	include("radial-functions.jl")
	println("Number of source locations to be analyzed = ", size(WBig)[2])
	idxs = reshape(clusterassignments, 1, size(WBig)[2])
	for i in 1:size(WBig)[2]
		println("Location $i ... labeled as Source #$(idxs[i]) ...")
		target = collect(WBig[:,i])
		# results = Optim.levenberg_marquardt(r2, r2g, [1.0, 499100.0, 539100.0], show_trace=true, maxIter=500)
		# results = Optim.levenberg_marquardt(logr2, logr2g, [1.0, 1000000, 499100.0, 539100.0], maxIter=1000, tolG=1e-19)
		# results = Optim.levenberg_marquardt(r2, r2g, [10000.0, 499100.0, 539100.0], maxIter=1000, tolG=1e-19)
		results = Optim.levenberg_marquardt(rn, rng, [1.0, 499100.0, 539100.0, 0.2], maxIter=1000, tolG=1e-19)
		# println(results)
		push!(dfr, (results.minimum[2], results.minimum[3], "$(idxs[i])", "", "results"))
		#pred = rn( results.minimum )
		#for j in 1:nP
		#	dfp[:info][j] = @sprintf( "%.2f-%.2f=%.2f", target[j],target[j]-pred[j],pred[j])
		#end
	end
	println("Estimated source locations $dfr")
	for i in 1:nk
		l = array(dfr[(dfr[:label].=="$i"),1:2])
		# println("Locations for Source # $i \n $l")
		println("Source location $i: mean ", mean(l[:,1]), " ", mean(l[:,2]), " variance ", var(l[:,1]), " ", var(l[:,2]) )
	end

	p = plot(vcat(dfp,dfw,dfr), x="x", y="y", label=3, color="category", Geom.point, Geom.label,
	Guide.XLabel("x [m]"), Guide.YLabel("y [m]"), Guide.yticks(orientation=:vertical), Scale.x_continuous(labels=x -> @sprintf("%.0f", x)))
	draw(SVG(string("nmfk-test-$testproblem/nmfk-test-$testproblem-output-NMFk=",nk,"-",nNMF,"-sources.svg"), 8inch, 6inch), p)
	# draw(PNG(string("nmfk-test-$testproblem/nmfk-test-$testproblem-output-NMFk=",nk,"-",nNMF,"-sources.png"), 8inch, 6inch), p)
end
