import Geodesy

"""
Convert LAT/LON to Cartesian coordinates (x,y)

$(DocumentFunction.documentfunction(latlon_to_xy))
"""
function latlon_to_xy(lat, lon; zone=14, north=true, datum=Geodesy.nad83, utm_map=Geodesy.UTMfromLLA(zone, north, datum))
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