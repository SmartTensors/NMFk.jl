import Gadfly
import Mads
import Measures
import Compose

function inferperm(A, target)
	sz = size(A)
	dims = length(sz)
	@assert dims == length(target) "A and target must have the same number of dimensions"
	perm = collect(1:dims)
	used = falses(dims)
	for k in 1:dims
		lenk = target[k]
		idx = Base.findfirst(i -> !used[i] && sz[i] == lenk, 1:dims)
		perm[k] = idx === nothing ? k : idx
		used[perm[k]] = true
	end
	return perm
end

function plotmatches(X::AbstractArray, Xest::AbstractArray, timebins::AbstractVector, attributes::AbstractVector, locations::AbstractVector; figuredir::AbstractString=".", filename::AbstractString="", combine::Bool=false, dpi::Int64=300, background_color=nothing, hsize::Measures.AbsoluteLength=8Compose.inch, vsize::Measures.AbsoluteLength=4Compose.inch, quiet::Bool=false)
	ndims(X) == 3 || throw(ArgumentError("X must be a 3-D array"))
	ndims(Xest) == 3 || throw(ArgumentError("Xest must be a 3-D array"))
	@assert size(X) == size(Xest) "X $(size(X)) and Xest $(size(Xest)) must have the same dimensions"

	nw = length(locations)
	nt = length(timebins)
	ns = length(attributes)
	target = (nw, nt, ns)
	permX = inferperm(X, target)
	permE = inferperm(Xest, target)

	Xv1 = @view PermutedDimsArray(X, permX)[:, :, :]
	Xv2 = @view PermutedDimsArray(Xest, permE)[:, :, :]
	for i in axes(Xv1, 1)
		p1 = Mads.plotseries(Xv1[i, :, :]; title=locations[i], xaxis=timebins, name = "", names=combine ? fill("", ns) : attributes, key_position=combine ? :none : :right, hsize=hsize, vsize=vsize, plotline=false, quiet=true, code=combine, returnplot=!combine)
		p2 = Mads.plotseries(Xv2[i, :, :]; title="", xaxis=timebins, names=attributes, hsize=hsize, vsize=vsize, quiet=true, code=combine, returnplot=!combine)
		if combine
			p = Gadfly.plot(p1..., p2...)
		else
			p = Gadfly.vstack(p1, p2)
			vsize = vsize * 2
		end
		display(p)
		if filename != ""
			s = splitext(filename)
			fn = joinpath(figuredir,"$(s[1])_$(locations[i])$(s[2])")
			Mads.plotfileformat(p, fn, hsize, vsize; dpi=dpi)
		end
	end
end
