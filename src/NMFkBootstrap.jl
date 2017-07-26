"""
Randomize matrix by assigning a random vector to each column based on a multinomial distribution (with the original column vector as the mean)
"""
function bootstrap(H::Array, scale::Number=0.1)
	G = similar(H)
	for i in 1:size(H, 2)
		r = round(Int64, H[:, i])
		n = sum(r)
		p = r ./ n
		X = Distributions.Multinomial(n, p)
		G[:, i] = rand(X) .* scale
	end
	G
end
