import Gadfly
import Measures
import Colors
import Compose
import Dates

function progressbar_regular(i::Number, timescale::Bool=false, timestep::Number=1, datestart=nothing, dateend=nothing, dateincrement::AbstractString="Dates.Day")
	s = timescale ? sprintf("%6.4f", i * timestep) : sprintf("%6d", i)
	if !isnothing(datestart)
		if !isnothing(dateend)
			s = datestart + ((dateend .- datestart) * (i-1) * timestep)
			if typeof(datestart) <: Integer
				try
					s = convert(Int64, s)
				catch
				end
			end
		else
			s = datestart + Core.eval(Main, Meta.parse(dateincrement))(i-1)
		end
	end
	return Compose.compose(Compose.context(0, 0, 1Compose.w, 0.05Compose.h),
		(Compose.context(), Compose.fill("gray"), Compose.fontsize(10Compose.pt), Compose.text(0.01, 0.0, s, Compose.hleft, Compose.vtop)),
		(Compose.context(), Compose.fill("tomato"), Compose.rectangle(0.75, 0.0, i * timestep * 0.2, 5)),
		(Compose.context(), Compose.fill("gray"), Compose.rectangle(0.75, 0.0, 0.2, 5)))
end

function make_progressbar_2d(s; vlinecolor="gray", vlinesize=2Gadfly.pt)
	function progressbar_2d(i::Number, timescale::Bool=false, timestep::Number=1, datestart=nothing, dateend=nothing, dateincrement::AbstractString="Dates.Day")
		if i > 0
			xi = timescale ? i * timestep : i
			if !isnothing(datestart)
				if !isnothing(dateend)
					if typeof(datestart) <: Integer
						try
							timestep = convert(Int64, timestep)
						catch
						end
					end
					xi = datestart + ((dateend .- datestart) * (i-1) * timestep)
					if typeof(datestart) <: Integer
						try
							xi = convert(Int64, xi)
						catch
						end
					end
				else
					xi = datestart + Core.eval(Main, Meta.parse(dateincrement))(i-1)
				end
			end
			return Gadfly.plot(s..., Gadfly.layer(xintercept=[xi], Gadfly.Geom.vline(color=[vlinecolor], size=[vlinesize])))
		else
			return Gadfly.plot(s...)
		end
	end
	return progressbar_2d
end