module NMFk

using NMF
using Clustering
using Distances
using Stats

export NMF_single_iter, cluster_NMF_solutions, final_processes_and_mixtures;

# Function definitions
function NMF_single_iter(inputMatrix, numberOfProcesses, nmfIter)
	processes, mixtures = NMF.randinit( inputMatrix, numberOfProcesses, normalize = true);
	NMF.solve!(NMF.MultUpdate( obj = :mse, maxiter = nmfIter ), inputMatrix, processes, mixtures);

	for j = 1 : numberOfProcesses
		total = sum( processes[:, j] );
		processes[:, j] = processes[:, j] ./ total;
		mixtures[j, :]  = mixtures[j, :] .* total;
	end	

	return processes, mixtures;
end

function cluster_NMF_solutions(HBigT, clusterRepeatMax)

	nNMF = clusterRepeatMax;
	println( size(HBigT) );
	nT = size(HBigT, 1);
	nW = size(HBigT, 2);
	nk = convert(Int, nW / nNMF );
	# numberOfPoints = size(allProcesses, 1); # nT
	# numberOfProcesses = size(allProcesses, 2); # nk
	# globalIter =  size(allProcesses, 3); # nNMF
	numberOfPoints = nT;
	numberOfProcesses = nk;
	globalIter = nNMF;

	centroids = HBigT[:, 1:nk];
	idx = zeros(Int, numberOfProcesses, globalIter);
	# idx_old = zeros(Int, numberOfProcesses, globalIter);

	for clusterIt = 1 : clusterRepeatMax

		for globalIterID = 1 : globalIter

			processesTaken = zeros(numberOfProcesses , 1);
			centroidsTaken = zeros(numberOfProcesses , 1);

			for currentProcessID = 1 : numberOfProcesses
				distMatrix = ones(numberOfProcesses, numberOfProcesses) * 100; 

				for processID = 1 : numberOfProcesses
					for centroidID = 1 : numberOfProcesses
						if ( (centroidsTaken[centroidID] == 0) && ( processesTaken[processID] == 0) )
							distMatrix[processID, centroidID] = cosine_dist(HBigT[:, processID +  ( globalIterID - 1 ) * nk], centroids[:,centroidID]);
							# distMatrix[processID, centroidID] = cosine_dist(HBigT[:, ( processID - 1 ) * nNMF +  globalIterID], centroids[:,centroidID]);
						end
					end
				end
				minProcess,minCentroid = ind2sub(size(distMatrix), indmin(distMatrix));
				processesTaken[minProcess] = 1;
				centroidsTaken[minCentroid] = 1;
				idx[minProcess, globalIterID] = minCentroid;
			end

		end

		centroids = zeros( numberOfPoints, numberOfProcesses );
		for centroidID = 1 : numberOfProcesses
			for globalIterID = 1 : globalIter
				centroids[:, centroidID] = centroids[:, centroidID] + HBigT[:, findin(idx[:, globalIterID], centroidID) + ( globalIterID - 1 ) * nk];
			# 	centroids[:, centroidID] = centroids[:, centroidID] + HBigT[:, ( findin(idx[:, globalIterID], centroidID) - 1 ) * nNMF  + globalIterID];
			end
		end
		centroids = centroids ./ globalIter;

		# if ( sum(abs(idx_old - idx)) == 0 )
		#    break;
		# else
		#    idx_old = idx;
		# end

	end

	return idx, centroids;
end

function final_processes_and_mixtures(allProcesses, allMixtures, nNMF, idx)

	println( size(allProcesses) );
	println( size(allMixtures) );
	numberOfPoints = size(allProcesses, 1); # nT
	println("numberOfPoints (nT) ", numberOfPoints)
	nW = size(allProcesses, 2); # nW
	println("nW ", nW)
	# globalIter =  size(allProcesses, 3);
	# println("globalIter ", globalIter)
	globalIter = nNMF;
	nk = numberOfProcesses = convert(Int, nW / nNMF );
	println("nk ", nk)
	nW = size(allMixtures, 1); # nW
	numberOfSamples = size(allMixtures, 2); # nP
	println("nW ", nW)
	println("numberOfSamples (nP) ", numberOfSamples)

	idx_r = vec(reshape(idx, nW, 1));

	allProcesses_r = reshape(allProcesses, numberOfPoints, numberOfProcesses * globalIter);
	println( size(allProcesses_r) );
	allMixtures_r = reshape(allMixtures, numberOfProcesses * globalIter, numberOfSamples);
	println( size(allMixtures_r) );
	allProcessesDist = pairwise(CosineDist(), allProcesses_r);
	println( size(allProcessesDist) );
	stabilityProcesses = silhouettes( idx_r, vec(repmat([globalIter], numberOfProcesses, 1)), allProcessesDist);

	avgStabilityProcesses = zeros(numberOfProcesses, 1);
	processes = zeros(numberOfPoints, numberOfProcesses);
	mixtures = zeros( numberOfProcesses, numberOfSamples);

	for i = 1 : numberOfProcesses
		avgStabilityProcesses[i] = mean(stabilityProcesses[ findin(idx_r,i) ]);
		processes[:, i] = mean( allProcesses_r[ :, findin(idx_r,i) ] ,2 );
		mixtures[i, :] = mean( allMixtures_r[ findin(idx_r,i),: ] ,1);
	end

	return processes, mixtures, avgStabilityProcesses
end

end
