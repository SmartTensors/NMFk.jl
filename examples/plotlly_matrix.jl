import PlotlyJS

PlotlyJS.plot(
	PlotlyJS.heatmap(;
		x=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
		y=["Morning", "Afternoon", "Evening"],
		z=[1 25 30 50 1; 20 1 60 80 30; 30 60 1 5 20],
		colorscale = "Portland",
	),
	PlotlyJS.Layout(; xaxis_side="top", yaxis_scaleanchor="x")
)