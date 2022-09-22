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
	lat = [utm[i].x for i=1:l]
	lon = [utm[i].y for i=1:l]
	if l == 1
		return lat[1], lon[1]
	else
		return lat, lon
	end
end