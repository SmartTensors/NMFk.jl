# nP, px, py, and target are global variables
# nP = number of microphones (wells)
# px, py = microphone coordinates
# target = the NMF weights (contributions) estimated for each microphone

function r1(x::Vector)	
	d = Array(Float64, nP) 
	for k in 1:nP
		d[k] = (x[1] / sqrt((px[k] - x[2])^2 + (py[k] - x[3] )^2)) - target[k]
	end
	return d
end

function r1g(x::Vector)
	l = length(x)
	d = Array(Float64, nP, l)
	for k in 1:nP
		d[k,1] = - 1 / sqrt((px[k] - x[2])^2 + (py[k] - x[3])^2)
		d[k,2] = ((-((-2 * (px[k] - x[2])) * (0.5 / sqrt((px[k] - x[2])^2 + (py[k] - x[3])^2))) * x[1]) / sqrt((px[k] - x[2])^2 + (py[k] - x[3])^2)^2)
		d[k,3] = ((-((-2 * (py[k] - x[3])) * (0.5 / sqrt((px[k] - x[2])^2 + (py[k] - x[3])^2))) * x[1]) / sqrt((px[k] - x[2])^2 + (py[k] - x[3])^2)^2)
	end
	return d
end

function r2(x::Vector)	
	d = Array(Float64, nP)
	for k in 1:nP
		d[k] = (x[1] / ((px[k] - x[2])^2 + (py[k] - x[3])^2)) - target[k]
	end
	return d
end

function r2g(x::Vector)
	l = length(x)
	d = Array(Float64, nP, l)
	for k in 1:nP
		d[k,1] = -1 / ((px[k] - x[2])^2 + (py[k] - x[3])^2)
		d[k,2] = -(-(-2 * (px[k] - x[2])) * x[1]) / ((px[k] - x[2])^2 + (py[k] - x[3])^2)^2
		d[k,3] = -(-(-2 * (py[k] - x[3])) * x[1]) / ((px[k] - x[2])^2 + (py[k] - x[3])^2)^2
	end
	return d
end

function rn(x::Vector)	
	d = Array(Float64, nP)
	for k in 1:nP
		d[k] = (x[1]/((px[k] - x[2])^2 + (py[k] - x[3])^2)^x[4]) - target[k]
	end
	return d
end

function rng( x::Vector )
	l = length(x)
	d = Array(Float64, nP, l)
	for k in 1:nP
		d[k,1] = (1 / ((px[k] - x[2])^2 + (py[k] - x[3])^2)^x[4])
		d[k,2] = ((-(x[4] * (-2 * (px[k] - x[2])) * ((px[k] - x[2])^2 + (py[k] - x[3])^2)^(x[4] - 1)) * x[1]) / (((px[k] - x[2])^2 + (py[k] - x[3])^2)^x[4])^2)
		d[k,3] = ((-(x[4] * (-2 * (py[k] - x[3])) * ((px[k] - x[2])^2 + (py[k] - x[3])^2)^(x[4] - 1)) * x[1]) / (((px[k] - x[2])^2 + (py[k] - x[3])^2)^x[4])^2)
		d[k,4] = ((-(((px[k] - x[2])^2 + (py[k] - x[3])^2)^x[4] * log((px[k] - x[2])^2 + (py[k] - x[3])^2)) * x[1]) / (((px[k] - x[2])^2 + (py[k] - x[3])^2)^x[4])^2)
	end
	return d
end

function logr2(x::Vector)	
	d = Array(Float64, nP)
	for k in 1:nP
		d[k] = (x[1] * log(x[2] / ((px[k] - x[3])^2 + (py[k] - x[4])^2))) - target[k]
	end
	return d
end

function logr2g(x::Vector)
	l = length(x)
	d = Array(Float64, nP, l)
	for k in 1:nP
		d[k,1] = (log(x[2] / ((px[k] - x[2])^2 + (py[k] - x[3])^2)))
		d[k,2] = (x[1] * ((1 / ((px[k] - x[3])^2 + (py[k] - x[4])^2)) * (1 / (x[2] / ((px[k] - x[3])^2 + (py[k] - x[4])^2)))))
		d[k,3] = (x[1] * (((-(-2 * (px[k] - x[3])) * x[2]) / ((px[k] - x[3])^2 + (py[k] - x[4])^2)^2) * (1 / (x[2] / ((px[k] - x[3])^2 + (py[k] - x[4])^2)))))
		d[k,4] = (x[1] * (((-(-2 * (py[k] - x[4])) * x[2]) / ((px[k] - x[3])^2 + (py[k] - x[4])^2)^2) * (1 / (x[2] / ((px[k] - x[3])^2 + (py[k] - x[4])^2)))))
	end
	return d
end
