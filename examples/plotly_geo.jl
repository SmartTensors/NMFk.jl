import PlotlyJS
import PlotlyBase

mapbox_token = "pk.eyJ1IjoibW9udHl2IiwiYSI6ImNsMDhvNTJwMzA1OHgzY256N2c2aDdzdXoifQ.cGUz0Wuc3rYRqGNwm9v5iQ"

PlotlyJS.plot(
	PlotlyJS.heatmap(;
		lon=collect(-105.8:0.1:-105.6),
		lat=collect(35.4:0.1:35.8),
		z=[[1 25 30 50 1], [20 1 60 80 30], [30 60 1 5 20]],
		colorscale = "Portland",
	),
	PlotlyJS.Layout(
		margin = PlotlyJS.attr(r=0, t=0, b=0, l=0),
		mapbox = PlotlyJS.attr(accesstoken=mapbox_token, style="mapbox://styles/mapbox/satellite-streets-v12")
		)
)

df = CSV.read(download("https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv"), DataFrames.DataFrame)
PlotlyJS.plot(
	PlotlyJS.densitymapbox(df, lat =:Latitude, lon = :Longitude, z = :Magnitude, zoom = 6),
	PlotlyJS.Layout(
		margin = PlotlyJS.attr(r=0, t=0, b=0, l=0),
		mapbox = PlotlyJS.attr(accesstoken=mapbox_token, style="mapbox://styles/mapbox/satellite-streets-v12")
	)
)
