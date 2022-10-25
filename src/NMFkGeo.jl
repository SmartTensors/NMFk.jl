import Geodesy

"""
Convert LAT/LON to Cartesian coordinates (x,y)

$(DocumentFunction.documentfunction(latlon_to_xy))
"""
function latlon_to_xy(lat, lon; zone=14, isnorth=true, datum=Geodesy.nad83, utm_map=Geodesy.UTMfromLLA(zone, isnorth, datum))
	l = length(lat)
	@assert l == length(lon)
	utm = utm_map.([Geodesy.LLA([lat lon][i,:]...) for i=1:l])
	x = [utm[i].x for i=1:l]
	y = [utm[i].y for i=1:l]
	if l == 1
		return x[1], y[1]
	else
		return x, y
	end
end

"""
Convert Cartesian coordinates (x,y) to LAT/LON

$(DocumentFunction.documentfunction(xy_to_latlon))
"""
function xy_to_latlon(x, y; zone=14, isnorth=true, datum=Geodesy.nad83, utm_map=Geodesy.LLAfromUTM(zone, isnorth, datum))
	l = length(x)
	@assert l == length(y)
	utm = utm_map.([Geodesy.UTM([x y][i,:]...) for i=1:l])
	lat = [utm[i].lat for i=1:l]
	lon = [utm[i].lon for i=1:l]
	if l == 1
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