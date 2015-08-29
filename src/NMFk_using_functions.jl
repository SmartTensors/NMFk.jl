# loading librarires
using NMF
using Clustering
using Distances
using Stats

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

function cluster_NMF_solutions(allProcesses, clusterRepeatMax)
	numberOfProcesses = size(allProcesses, 2);
	globalIter =  size(allProcesses, 3);
	centroids = allProcesses[:, :, 1];
	idx = zeros(Int, numberOfProcesses, globalIter);

	for clusterIt = 1 : clusterRepeatMax

		for globalIterID = 1 : globalIter

			processesTaken = zeros(numberOfProcesses , 1);
			centroidsTaken = zeros(numberOfProcesses , 1);

			for currentProcessID = 1 : numberOfProcesses
				distMatrix = ones(numberOfProcesses, numberOfProcesses) * 100; 

				for processID = 1 : numberOfProcesses
					for centroidID = 1 : numberOfProcesses
						if ( (centroidsTaken[centroidID] == 0) && ( processesTaken[processID] == 0) )
							distMatrix[processID, centroidID] = cosine_dist(allProcesses[:, processID, globalIterID], centroids[:,centroidID]);
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
				centroids[:, centroidID] = centroids[:, centroidID] + allProcesses[:, findin(idx[:, globalIterID], centroidID), globalIterID];
			end
		end
		centroids = centroids ./ globalIter;

	end
	return idx;
end

function final_processes_and_mixtures(allProcesses, allMixtures, idx)
	numberOfPoints = size(allProcesses, 1);
	numberOfProcesses = size(allProcesses, 2);
	globalIter =  size(allProcesses, 3);

	idx_r = vec(reshape(idx, numberOfProcesses * globalIter, 1));
	allProcesses_r = reshape(allProcesses, numberOfPoints, numberOfProcesses * globalIter);

	#allMixtures_r = reshape(allMixtures, numberOfProcesses * globalIter, numberOfSamples);		# does not stack as expected

	allMixtures_r = Array(Float64, numberOfProcesses * globalIter, numberOfSamples);
	for i=1:size(allMixtures,3)
       if i==1
       allMixtures_r = allMixtures[:,:,1];
       else
       allMixtures_r = [allMixtures_r ; allMixtures[:,:,i]];
       end
    end


	allProcessesDist = pairwise(CosineDist(), allProcesses_r);
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

# setting directory and adjusting params
#cd("/home/boian/Desktop/NMF_2014/Julia/");

globalIter = 10;
nmfIter = 100000;
numberOfProcesses = 2;

# reading input file and initilizing arrays
#inputMatrix = readcsv("Obs_noDelay.csv");
#inputMatrix=inputMatrix';
inputMatrix = readdlm("input/input.txt", '\t');
numberOfPoints   = size(inputMatrix, 1);
numberOfSamples = size(inputMatrix, 2);

allProcesses = zeros( numberOfPoints, numberOfProcesses, globalIter );
allMixtures  = zeros( numberOfProcesses, numberOfSamples, globalIter );

# matrix factorization over multiple iterations
for curentIteration = 1 : globalIter
	allProcesses[:, :, curentIteration], allMixtures[:, :, curentIteration] = NMF_single_iter(inputMatrix, numberOfProcesses, nmfIter)
	println("Iteration $(curentIteration)/$(globalIter) has completed!");
end

# clustering extracted processes
idx = cluster_NMF_solutions(allProcesses, 10000);

# calculating stability and final processes and mixtures
processes, mixtures, avgStabilityProcesses = final_processes_and_mixtures(allProcesses, allMixtures, idx);

dataRecon = processes * mixtures;

dataReconCorr = zeros(numberOfSamples, 1);

for i = 1 : numberOfSamples
	dataReconCorr[i] = cor( inputMatrix[:,i], dataRecon[:, i] );
end
