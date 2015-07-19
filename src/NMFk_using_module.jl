# loading librarires
@everywhere begin
	cd("/home/boian/Desktop/NMF_2014/Julia/");
	include("NMFk.jl");

	# setting directory and adjusting params
	inputFile = "input/input.txt";
	totalNodes = nprocs();
	iterPerNode = 10;
	nmfIter = 100000;
	numberOfProcesses = 2;


	# reading input file and initilizing arrays
	inputMatrix = readdlm(inputFile, '\t');

	numberOfPoints   = size(inputMatrix, 1);
	numberOfSamples = size(inputMatrix, 2);
	allProcessesPerNode = zeros( numberOfPoints, numberOfProcesses, iterPerNode );
	allMixturesPerNode  = zeros( numberOfProcesses, numberOfSamples, iterPerNode );

	# matrix factorization over multiple iterations	
	for currentIteration = 1 : iterPerNode
		allProcessesPerNode[:, :, currentIteration], allMixturesPerNode[:, :, currentIteration] = NMFk.NMF_single_iter(inputMatrix, numberOfProcesses, nmfIter);
		println("Iteration $(currentIteration)/$(iterPerNode) has completed!");
	end
end

globalIter = totalNodes * iterPerNode;
allProcesses = zeros( numberOfPoints, numberOfProcesses, globalIter );
allMixtures  = zeros( numberOfProcesses, numberOfSamples, globalIter );

for my_id_i = 1 : totalNodes
	allProcesses[:,:, ((my_id_i-1)*iterPerNode+1):(my_id_i*iterPerNode) ] = remotecall_fetch(my_id_i, ()->allProcessesPerNode);
	allMixtures[:,:, ((my_id_i-1)*iterPerNode+1):(my_id_i*iterPerNode) ] = remotecall_fetch(my_id_i, ()->allMixturesPerNode);	
end

# clustering extracted processes
idx = NMFk.cluster_NMF_solutions(allProcesses, 10000);

# calculating stability and final processes and mixtures
processes, mixtures, avgStabilityProcesses = NMFk.final_processes_and_mixtures(allProcesses, allMixtures, idx);

dataRecon = processes * mixtures;

dataReconCorr = zeros(numberOfSamples, 1);

for i = 1 : numberOfSamples
	dataReconCorr[i] = cor( inputMatrix[:,i], dataRecon[:, i] );
end
