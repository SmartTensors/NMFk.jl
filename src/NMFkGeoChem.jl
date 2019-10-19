"Convert stable isotope deltas to concentrations"
function getisotopeconcentration(delta::Union{Number,Vector,Matrix}, deltastandard::Union{Number,Vector}, concentration_species::Union{Number,Vector,Matrix}, scalefactor::Union{Number,Vector}=ones(length(deltastandard)))
	lsd = length(size(delta))
	lds = length(deltastandard)
	if lsd == 1 || (lsd == 2 && size(delta)[2] == 1)
		@assert size(delta)[1] == length(concentration_species)
		@assert lds == 1
	elseif lsd == 2
		@assert size(delta) == size(concentration_species)
		@assert size(delta)[2] == lds
	end
	if lds > 1
		Adeltastandard = permutedims(repeat(collect(deltastandard), outer=[1,size(delta)[1]]))
		Ascalefactor = permutedims(repeat(collect(scalefactor), outer=[1,size(delta)[1]]))
	else
		Adeltastandard = deltastandard
		Ascalefactor = scalefactor
	end
	ratio = (delta / 1000 .+ 1) .* Adeltastandard
	concentration_isotope  = concentration_species .* ratio ./ (ratio .+ 1) .* Ascalefactor
end

"Convert stable isotope concentrations to deltas"
function getisotopedelta(concentration_isotope::Union{Number,Vector,Matrix}, deltastandard::Union{Number,Vector}, concentration_species::Union{Number,Vector,Matrix}, scalefactor::Union{Number,Vector}=ones(length(deltastandard)))
	lsd = length(size(concentration_isotope))
	lds = length(deltastandard)
	if lsd == 1 || (lsd == 2 && size(concentration_isotope)[2] == 1)
		@assert size(concentration_isotope)[1] == length(concentration_species)
		@assert lds == 1
	elseif lsd == 2
		@assert size(concentration_isotope) == size(concentration_species)
		@assert size(concentration_isotope)[2] == lds
	end
	if lds > 1
		Adeltastandard = permutedims(repeat(collect(deltastandard), outer=[1,size(concentration_isotope)[1]]))
		Ascalefactor = permutedims(repeat(collect(scalefactor), outer=[1,size(concentration_isotope)[1]]))
	else
		Adeltastandard = deltastandard
		Ascalefactor = scalefactor
	end
	ratio = (concentration_isotope .* Ascalefactor ) ./ (concentration_species .- concentration_isotope)
	delta_isotope = (ratio .- Adeltastandard) ./ Adeltastandard * 1000
end

"Compute deltas of mixtures (`compute_contributions` requires external normalization)"
function computedeltas(mixer::Matrix, buckets::Matrix, bucketdeltas::Matrix, deltaindices::Vector; compute_contributions::Bool=false)
	numwells = size(mixer, 1)
	numdeltas = length(deltaindices)
	deltas = Array{Float64}(undef, numwells, numdeltas)
	for i = 1:numwells
		for j = 1:numdeltas
			v = vec(mixer[i, :]) .* vec(buckets[:, deltaindices[j]])
			if compute_contributions
				deltas[i, j] = LinearAlgebra.dot(v, bucketdeltas[:, j])
			else
				deltas[i, j] = LinearAlgebra.dot(v, bucketdeltas[:, j]) / sum(v)
			end
		end
	end
	return deltas
end