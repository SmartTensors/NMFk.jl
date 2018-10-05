function sumnan(X, c...; kw...)
	sum(X[.!isnan.(X)], c...; kw...)
end

function ssqrnan(X)
	sum(X[.!isnan.(X)].^2)
end

function vecnormnan(X)
	vecnorm(X[.!isnan.(X)])
end

function cornan(x, y)
	isn = isnan.(x) .| isnan.(y)
	cov(x[.!isn], y[.!isn])
end