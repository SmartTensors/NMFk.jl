import Statistics

function firstjump(y; lag=min(length(y), 30), threshold=5, influence=0)
	isn = .!isnan.(y)
	if sum(y[isn]) == 0
		return nothing
	end
	if sum(isn) == 0
		return nothing
	end
	y[.!isn] .= 0
	r = smoothedzscore(y; lag=lag, threshold=threshold, influence=influence)
	if1 = subset([0., 1.], r[:signals])
	if if1 != nothing
		if2 = if1 + findfirst(i->i > 0, y[if1:end]) - 1
	else
		if2 = findfirst(i->i > 0, y)
	end
	y[.!isn] .= NaN
	if2
end

function smoothedzscore(y; lag=30, threshold=5, influence=0)
	n = length(y)
	if lag > n
		lag = n
	end
	signals = zeros(n)
	yfiltered = copy(y)
	avgfilter = zeros(n)
	stdfilter = zeros(n)
	avgfilter[lag - 1] = Statistics.mean(y[1:lag])
	stdfilter[lag - 1] = Statistics.std(y[1:lag])

	for i in range(lag, stop=n-1)
		if abs(y[i] - avgfilter[i-1]) > threshold * stdfilter[i-1]
			if y[i] > avgfilter[i-1]
				signals[i] += 1
			else
				signals[i] += -1
			end
			yfiltered[i] = influence * y[i] + (1 - influence) * yfiltered[i - 1]
		else
			signals[i] = 0
			yfiltered[i] = y[i]
		end
		avgfilter[i] = Statistics.mean(yfiltered[i-lag+1:i])
		stdfilter[i] = Statistics.std(yfiltered[i-lag+1:i])
	end
	return (signals = signals, avgfilter = avgfilter, stdfilter = stdfilter)
end

function subset(x, y)
	lenx = length(x)
	first = x[1]
	if lenx == 1
		return findnext(i->i==first, y, 1)
	end
	leny = length(y)
	lim = length(y) - length(x) + 1
	cur = 1
	while (cur = findnext(i->i==first, y, cur)) != nothing
		cur > lim && break
		beg = cur
		@inbounds for i = 2:lenx
			y[beg += 1] != x[i] && (beg = 0; break)
		end
		beg != 0 && return cur + 1
		cur += 1
	end
	return nothing
end