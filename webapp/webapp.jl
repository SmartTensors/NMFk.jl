import NMFk
using Escher

function inputstring2data(s::AbstractString)
	rows = split(s, ";")
	numcols = length(split(rows[1], " "; keep=false))
	data = Array{Float64}(undef, length(rows), numcols)
	for i = 1:length(rows)
		row = split(rows[i], " "; keep=false)
		if length(row) > numcols
			return "too many columns in the $(i)th row"
		elseif length(row) < numcols
			return "not enough columns in the $(i)th row"
		end
		data[i, :] = map(x->parse(Float64, x), row)
	end
	return data
end

function processdata(data::Matrix, n::Int, components::Array{Any, 1})
	mixer, buckets, objectiveeval = NMFk.mixmatchdata(data, n)
	push!(components, plaintext("Fit quality (lower is better): $objectiveeval"))
	for i = 1:n
		push!(components, plaintext("Bucket $i: $(buckets[i, :])"))
	end
	for i = 1:size(data, 1)
		push!(components, plaintext("Mixture $i: $(mixer[i, :])"))
	end
	return vbox(components)
end

function processdata(s::AbstractString, n::Int, components::Array{Any, 1})
	push!(components, plaintext(s))
	return vbox(components)
end

function main(window)
	println(now())
	push!(window.assets, "widgets")
	initialstring = "0. 1.; 1. 0.; 0.25 0.75; 0.5 NaN"
	initialbuckets = 2
	numbucketsinput = Input(initialbuckets)
	datainput = Input(initialstring)
	connected_textinput = subscribe(textinput(initialstring), datainput)
	connected_slider = subscribe(slider(1:10, value=initialbuckets), numbucketsinput)
	lift(numbucketsinput) do n
		datastring = value(datainput)
		data = inputstring2data(datastring)
		components = Any[plaintext("Data:"), connected_textinput, plaintext("Number of buckets:"), connected_slider, plaintext("You have to click the slider for the analysis to update.")]
		processdata(data, n, components)
	end
end
