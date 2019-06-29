import DocumentFunction

"""
Set image dpi

$(DocumentFunction.documentfunction(setdpi))
"""
function setdpi(dpi::Integer)
	global imagedpi = dpi;
end

toupper(x::String, i=1) = x[1:i-1] * uppercase(x[i:i]) * x[i+1:end]

function maximumnan(X, c...; functionname="isnan", kw...)
	i = Core.eval(NMFk, Meta.parse(functionname)).(X)
	maximum(X[.!i], c...; kw...)
end

function minimumnan(X, c...; functionname="isnan", kw...)
	i = Core.eval(NMFk, Meta.parse(functionname)).(X)
	minimum(X[.!i], c...; kw...)
end

function sumnan(X, c=nothing; kw...)
	if c == nothing
		return sum(X[.!isnan.(X)]; kw...)
	else
		count = .*(size(X)[vec(collect(c))]...)
		I = isnan.(X)
		X[I] .= 0
		sX = sum(X; dims=c, kw...)
		X[I] .= NaN
		sI = sum(I; dims=c, kw...)
		sX[sI.==count] .= NaN
		return sX
	end
end

function ssqrnan(X)
	sum(X[.!isnan.(X)].^2)
end

function normnan(X)
	norm(X[.!isnan.(X)])
end

function cornan(x, y)
	isn = .!(isnan.(x) .| isnan.(y))
	if length(x) > 0 && length(y) > 0 && sum(isn) > 1
		return cov(x[isn], y[isn])
	else
		return NaN
	end
end

function hardencodelength(x::Vector{T}) where {T}
	u = unique(x)
	i = indexin(x, u)
	inan = indexin(true, isnan.(u))[1]
	d = inan != nothing ? length(u) - 1 : d = length(u)
	return d
end

function hardencode(x::Vector{T}) where {T}
	u = unique(x)
	i = indexin(x, u)
	inan = indexin(true, isnan.(u))[1]
	d = inan != nothing ? length(u) - 1 : d = length(u)
	m = zeros(length(x), d)
	for (j, k) in enumerate(i)
		if inan != nothing
			if k == inan
				m[j, :] .= NaN
			elseif k < inan
				m[j, k] = 1
			else
				m[j, k-1] = 1
			end
		else
			m[j, k] = 1
		end
	end
	return m
end


function hardencode(x::Matrix{T}) where {T}
	hcat([hardencode(x[:,i]) for i = 1:size(x, 2)]...)
end

function gettypes(x::Matrix{T}, levels=[0.05,0.35]) where {T}
	nlevels = length(levels)
	nw = size(x, 1)
	ns = size(x, 2)
	s = zeros(T, nw, ns)
	for i = 1:ns
		for l = 1:nlevels
			if l == nlevels
				s[:,i] .+= (x[:,i] .>= levels[l]) .* l
			else
				s[:,i] .+= ((x[:,i] .>= levels[l]) .& (x[:,i] .< levels[l+1])) .* l
			end
		end
	end
	wcode = sum(hcat(map(j->s[:,j].*10^j, 1:ns)...); dims=2)
	uniquewcode = unique(wcode)
	nt = length(uniquewcode)
	@info("Number of unique types is $nt)")
	types = nt < 31 ? collect(range('A'; length=nt)) : ["T$i" for i=1:nt]
	iwcode = indexin(wcode, uniquewcode)
	map(i->types[i], iwcode)
end

function harddecode(x::Matrix, h::Matrix{T}) where {T}
	na = size(x, 2)
	d = [hardencodelength(x[:,i]) for i = 1:na]
	ns = size(h, 1)
	s = Matrix{T}(undef, ns, na)
	c = 1
	for i = 1:na
		ce = c + d[i] - 1
		s[:,i] = sum(h[:,c:ce]; dims=2)
		c = ce + 1
	end
	return s
end