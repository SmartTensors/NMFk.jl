import Plotly
import PlotlyJS

function geomap(filename::AbstractString, x::AbstractVector, y::AbstractVector, s::AbstractVector=ones(length(x)) * 10; figuredir::AbstractString=".", title::AbstractString="")
	layout = PlotlyJS.Layout(;title=title, showlegend=false, geo=geo)
	trace = PlotlyJS.scattergeo(;locationmode="USA-states",
				lat=x,
				lon=y,
				hoverinfo="text",
				text=["loc" for x = 1:length(x)],
				marker_size=s,
				marker_line_color="black",
				marker_line_width=2)
	p = PlotlyJS.plot(trace, layout)
	j = joinpathcheck(figuredir, filename)
	recursivemkdir(j)
	PlotlyJS.savefig(p, j; format="html")
	return p
end