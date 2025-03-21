import Geodesy
import Statistics

"""
Convert LAT/LON to Cartesian coordinates (x,y)

$(DocumentFunction.documentfunction(latlon_to_xy))
"""
function latlon_to_xy(lat, lon; zone_isnorth::Tuple=Geodesy.utm_zone(Statistics.median(lat),  Statistics.median(lon)), zone::Integer=zone_isnorth[1], isnorth::Bool=zone_isnorth[2], datum=Geodesy.nad83, utm_map=Geodesy.UTMfromLLA(zone, isnorth, datum))
	@assert length(lat) == length(lon)
	utm = utm_map.([Geodesy.LLA([lat lon][i,:]...) for i=eachindex(lat)])
	println("Zone = $zone")
	x = [utm[i].x for i=eachindex(utm)]
	y = [utm[i].y for i=eachindex(utm)]
	if length(lat) == 1
		return x[1], y[1]
	else
		return x, y
	end
end

"""
Convert Cartesian coordinates (x,y) to LAT/LON

$(DocumentFunction.documentfunction(xy_to_latlon))
"""
function xy_to_latlon(x, y; zone_isnorth::Tuple=Geodesy.utm_zone(lat[1], lon[1]), zone::Integer=zone_isnorth[1], isnorth::Bool=zone_isnorth[2], datum=Geodesy.nad83, utm_map=Geodesy.LLAfromUTM(zone, isnorth, datum))
	@assert length(x) == length(y)
	utm = utm_map.([Geodesy.UTM([x y][i,:]...) for i=eachindex(x)])
	lat = [utm[i].lat for i=eachindex(utm)]
	lon = [utm[i].lon for i=eachindex(utm)]
	if length(x) == 1
		return lat[1], lon[1]
	else
		return lat, lon
	end
end

"""
Compute the haversine distance between two points on a sphere of radius `r`,
where the points are given by the latitude/longitude pairs `lat1/lon1` and
`lat2/lon2` (in degrees).

$(DocumentFunction.documentfunction(haversine))
"""
function haversine(lat1, lon1, lat2, lon2; r = 6372.8)
	lat1, lon1 = deg2rad(lat1), deg2rad(lon1)
	lat2, lon2 = deg2rad(lat2), deg2rad(lon2)
	hav(a, b) = sin((b - a) / 2)^2
	inner_term = hav(lat1, lat2) + cos(lat1) * cos(lat2) * hav(lon1, lon2)
	d = 2 * r * asin(sqrt(inner_term))
	return d
end