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