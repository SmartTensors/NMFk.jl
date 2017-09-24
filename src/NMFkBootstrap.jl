"""
Randomize matrix by assigning a random vector to each column based on a multinomial distribution (with the original column vector as the mean)
"""
function bootstrap(X::Array, scale::Number=0.1, epsilon::Number=sqrt(eps()))
	N = similar(X)
	for i in 1:size(X, 2)
		r = round(Int64, X[:, i])
		n = sum(r)
		p = r ./ n
		v = Distributions.Multinomial(n, p)
		N[:, i] = float(max.(rand(v) .* scale, epsilon))
	end
	return N
end
