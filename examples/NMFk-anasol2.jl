using Gadfly # Gadly demages Clustering
using DataFrames
using DataStructures
using Compose
using NMF
using Clustering
using MultivariateStats
using Optim
using Anasol
using Mads
pwd()
dirname(Base.source_path())
push!(LOAD_PATH, dirname(Base.source_path()))
cd(dirname(Base.source_path()))
using NMFk

# read the problem setup
md = Mads.loadyamlmadsfile("NMFk-anasol-20150721.mads")
include("NMFk-anasol2-20150721.jl")
i = 1
conc = OrderedDict()
for key in sort(collect(keys(Points)))
	wells = OrderedDict()
	welldata = Dict()
	welldata["x"] = Points[key][1]
	welldata["y"] = Points[key][2]
	welldata["z0"] = Points[key][3]
	welldata["z1"] = Points[key][3]
	welldata["obs"] = Array(Any,time[end])
	for t = time
		obs = Dict()
		obs["t"] = t
		obs["c"] = 0
		wobs = Dict()
		wobs[t] = obs
		welldata["obs"][t] = wobs
	end
	if i % 2 == 1
		md["Parameters"]["ax"]["init"] = 10
		xshift = 500
	else
		md["Parameters"]["ax"]["init"] = 100
		xshift = 1000
	end
	ox = md["Sources"][1]["box"]["x"]["init"]
	ot0 = md["Sources"][1]["box"]["t0"]["init"]
	ot1 = md["Sources"][1]["box"]["t1"]["init"]
	md["Sources"][1]["box"]["x"]["init"] = welldata["x"] - xshift
	md["Sources"][1]["box"]["t0"]["init"] = ( welldata["x"] - xshift ) / md["Parameters"]["vx"]["init"]
	md["Sources"][1]["box"]["t1"]["init"] = ( welldata["x"] - xshift ) / md["Parameters"]["vx"]["init"] + 1
	wells[key] = welldata
	md["Wells"] = wells
	paramkeys = Mads.getparamkeys(md)
	paramdict = OrderedDict(paramkeys, map(key->md["Parameters"][key]["init"], paramkeys))
	computeconcentrations = Mads.makecomputeconcentrations(md)
	forward_preds = computeconcentrations(paramdict)
	conc[key] = collect(values(forward_preds))
	md["Sources"][1]["box"]["x"]["init"] = ox
	md["Sources"][1]["box"]["t0"]["init"] = ot0
	md["Sources"][1]["box"]["t1"]["init"] = ot1
	i = i + 1
end
nP = length(keys(Points)) # number of observation points
nT = length(conc["R1"]) # number of observation times at each point
X = Array(Float64, nP, nT) # input matrix of observations
i = 1
for key in sort(collect(keys(Points)))
	X[i,:] = conc[key]
	i = i + 1
end

if !isdir("nmfk-$testproblem")
	mkdir("nmfk-$testproblem")
end
# flags
intermediate_figs = false
flag_kmeans = true # true = buildin kmeans; false = use clustering in NMFk
source_location_identification = true # identify spatial location of the sources using LM and dummy radial functions
nNMF=10 # number of NMFk's

numrows = nP
numcols = nT
W = Array(Float64, nP, nk) # estimated weight matrix
H = Array(Float64, nk, nT) # estimated source matrix
P = Array(Float64, nP, nT) # model prediction matrix
phi = Array(Float64, nNMF) # vector for the computed objective functions
WBig = Array(Float64, nP, 0) # estimated weight matrix collected over a series of NMFk's (initialized emptu)
HBig = Array(Float64, 0, nT) # estimated source matrix collected over a series of NMFk's (initialized emptu)
df = Array(Any, nP) # DataFrames matrix needed for ploting
pl = Array(Plot, nP) # Plot matrix

# plot the concentration for all the observation points
i = 0
for k in sort(collect(keys(conc)))
	i += 1
	# println("Plotting ",k)
	# generate the plot for each observation point
	pl[i] = plot(x=time, y=conc[k], Guide.XLabel("Time [d]"), Guide.YLabel("Concentration [ppb]"), Guide.title(k), Geom.line, Theme(default_color=color("red"),line_width=2pt) )
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
draw(PNG(string("nmfk-$testproblem/nmfk-$testproblem-input.png"), 18inch, 12inch), p)
writecsv("nmfk-$testproblem/nmfk-$testproblem-well-names.csv",collect(keys(conc))')
writecsv("nmfk-$testproblem/nmfk-$testproblem.csv",X')

# prepare the data for the DataFrames
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

nSources = length(md["Sources"])
sname = Array(String,nSources)
sx = Array(Float64,nSources)
sy = Array(Float64,nSources)
for i = 1:nSources
	key = collect(keys(md["Sources"][i]))
	sourcedata = md["Sources"][1][key[1]]
	sx[i] = sourcedata["x"]["init"]
	sy[i] = sourcedata["y"]["init"]
	sname[i] = "S"*string(i)
end
# create a data frame for the sources
dfs = DataFrame(x=sx, y=sy, label=sname, info=sname, category="sources")

info("NMF runs ... ($nNMF)")
for n = 1:nNMF
	# initialize W & H matrices randomly
	# W, H = NMF.randinit(X, nk, normalize=true)
	W, H = NMF.randinit(X, nk)

	# initialize W & H using Non-Negative Double Singular Value Decomposition (NNDSVD) algorithm
	# Reference: C. Boutsidis, and E. Gallopoulos. SVD based initialization: A head start for nonnegative matrix factorization. Pattern Recognition, 2007.
	# W, H = NMF.nndsvd(X, nk)

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
		for k in sort(collect(keys(conc)))
			i += 1
			#df1 = DataFrame(x=time, y=conc[k], label="data")
			#df2 = DataFrame(x=time, y=conc[k], label="model")
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
		draw(PNG(string("nmfk-$testproblem/nmfk-$testproblem-output-",n,".png"), 18inch, 12inch), p)
		for i in 1:nk
			pl[i] = plot( x=time, y=H[i,:], Guide.XLabel("Time [d]"), Guide.title("Source $i"), Geom.line)
		end
		p = vstack( pl[1:nk] )
		draw(PNG(string("nmfk-$testproblem/nmfk-$testproblem-sources-",n,".png"), 18inch, 12inch), p)
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
		s = silhouettes(clusterassignments, clustercounts, HBig*HBig')
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
		clusterassignments, M = NMFk.cluster_NMF_solutions(HBig', nNMF);
		println("clusterassignments ", clusterassignments )
		# println("centroids ", M )
		Ht, Wt, avgStabilityProcesses = NMFk.final_processes_and_mixtures(HBig', WBig', nNMF, clusterassignments);
		println("Size of Ha = ", size(Ht) )
		println("Size of Wa = ", size(Wt) )
		println("Silhouettes Avg = ", avgStabilityProcesses )
		Wa = Wt';
		Ha = Ht';
		P = Wa * Ha;
		E = X - P;
		phi_final = sum( E' * E )
		println("Objective function = ", phi_final, " Max error = ", maximum(E), " Min error = ", minimum(E) )
		Pcorr = zeros(nT, 1);
		for i = 1 : nT
				Pcorr[i] = cor( X[:,i], P[:, i] );
		end
	end
else
	Ha = HBig
	Wa = WBig
end

writecsv(string("nmfk-$testproblem/nmfk-$testproblem-sources-NMFk=",nk,"-",nNMF,".csv"),Ha)
for i in 1:nk
	pl[i] = plot( x=time, y=Ha[i,:], Guide.XLabel("Time [d]"), Guide.title("Source $i"), Geom.line)
end
p = vstack( pl[1:nk] )
draw(PNG(string("nmfk-$testproblem/nmfk-$testproblem-sources-NMFk=",nk,"-",nNMF,".png"), 18inch, 12inch), p)
writecsv(string("nmfk-$testproblem/nmfk-$testproblem-weights-NMFk=",nk,"-",nNMF,".csv"),Wa)
i = 0
for k in sort(collect(keys(conc)))
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
draw(PNG(string("nmfk-$testproblem/nmfk-$testproblem-output-NMFk=",nk,"-",nNMF,".png"), 18inch, 12inch), p)

if source_location_identification
	info("Identification of the spatial location of the sources ...")
	# println(Wa)
	p = plot(vcat(dfp,dfs), x="x", y="y", label=3, color="category", Geom.point, Geom.label,
	Guide.XLabel("x [m]"), Guide.YLabel("y [m]"), Guide.yticks(orientation=:vertical), Scale.x_continuous(labels=x -> @sprintf("%.0f", x)))
	draw(SVG(string("nmfk-$testproblem/nmfk-$testproblem.svg"), 8inch, 6inch), p)
	# draw(PNG(string("nmfk-$testproblem/nmfk-$testproblem.png"), 8inch, 6inch), p)
	target = Array(Float64, nP)
	dfr = DataFrame(x = Float64[], y = Float64[], label = String[], info = String[], category =  String[])
	include("radial-functions.jl")
	println("Number of source locations to be analyzed = ", size(WBig)[2])
	idxs = reshape(clusterassignments,1,size(WBig)[2])
	for i in 1:size(WBig)[2]
		println("Location $i ... labeled as Source #$(idxs[i]) ...")
		target = collect(WBig[:,i])
		# results = Optim.levenberg_marquardt(r2, r2g, [1.0,499100.0,539100.0], show_trace=true, maxIter=500)
		# results = Optim.levenberg_marquardt(logr2, logr2g, [1.0,1000000,499100.0,539100.0], maxIter=1000, tolG=1e-19)
		# results = Optim.levenberg_marquardt(r2, r2g, [10000.0,499100.0,539100.0], maxIter=1000, tolG=1e-19)
		results = Optim.levenberg_marquardt(rn, rng, [1.0,499100.0,539100.0,0.2], maxIter=1000, tolG=1e-19)
		# println(results)
		push!(dfr,(results.minimum[2],results.minimum[3],"$(idxs[i])","","results"))
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

	p = plot(vcat(dfp,dfs,dfr), x="x", y="y", label=3, color="category", Geom.point, Geom.label,
	Guide.XLabel("x [m]"), Guide.YLabel("y [m]"), Guide.yticks(orientation=:vertical), Scale.x_continuous(labels=x -> @sprintf("%.0f", x)))
	draw(SVG(string("nmfk-$testproblem/nmfk-$testproblem-output-NMFk=",nk,"-",nNMF,"-sources.svg"), 8inch, 6inch), p)
	# draw(PNG(string("nmfk-$testproblem/nmfk-$testproblem-output-NMFk=",nk,"-",nNMF,"-sources.png"), 8inch, 6inch), p)
end
