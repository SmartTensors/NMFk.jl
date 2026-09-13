import Geodesy
import Statistics

"""
    latlon_to_xy(lat, lon;
                 zone_isnorth=Geodesy.utm_zone(Statistics.median(lat), Statistics.median(lon)),
                 zone=zone_isnorth[1], isnorth=zone_isnorth[2], datum=Geodesy.nad83,
                 utm_map=Geodesy.UTMfromLLA(zone, isnorth, datum))

Convert paired latitude and longitude coordinates in degrees to UTM Cartesian
coordinates.
The selected UTM zone is printed to `stdout`.

# Arguments

- `lat`: Collection of latitudes in degrees.
- `lon`: Equally sized collection of longitudes in degrees.

# Keywords

- `zone_isnorth`: Tuple `(zone, is_north)` used to select the UTM zone and
  hemisphere; by default it is inferred from the median input coordinate.
- `zone=zone_isnorth[1]`: Integer UTM zone passed to the default transform.
- `isnorth=zone_isnorth[2]`: Whether the selected UTM zone is in the northern
  hemisphere.
- `datum=Geodesy.nad83`: Geodetic datum used to construct the default
  transform.
- `utm_map=Geodesy.UTMfromLLA(zone, isnorth, datum)`: Callable transform from
  `Geodesy.LLA` coordinates to UTM coordinates.

# Returns

A tuple `(x, y)`.
Each component is a vector unless the inputs contain one coordinate, in which
case each component is a scalar.

# Examples

```julia
x, y = NMFk.latlon_to_xy([35.0844], [-106.6504])
@assert isfinite(x) && isfinite(y)
```
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
    xy_to_latlon(x, y;
                 zone_isnorth::Tuple=Geodesy.utm_zone(lat[1], lon[1]),
                 zone=zone_isnorth[1], isnorth=zone_isnorth[2], datum=Geodesy.nad83,
                 utm_map=Geodesy.LLAfromUTM(zone, isnorth, datum))

Convert paired UTM Cartesian coordinates to latitude and longitude in degrees.
Pass `zone_isnorth` explicitly because Cartesian coordinates do not identify
their UTM zone or hemisphere.

# Arguments

- `x`: Collection of UTM easting coordinates.
- `y`: Equally sized collection of UTM northing coordinates.

# Keywords

- `zone_isnorth`: Tuple `(zone, is_north)` identifying the UTM zone and
  hemisphere of the input coordinates.
- `zone=zone_isnorth[1]`: Integer UTM zone passed to the default transform.
- `isnorth=zone_isnorth[2]`: Whether the selected UTM zone is in the northern
  hemisphere.
- `datum=Geodesy.nad83`: Geodetic datum used to construct the default
  transform.
- `utm_map=Geodesy.LLAfromUTM(zone, isnorth, datum)`: Callable transform from
  `Geodesy.UTM` coordinates to latitude and longitude.

# Returns

A tuple `(lat, lon)` in degrees.
Each component is a vector unless the inputs contain one coordinate, in which
case each component is a scalar.

# Examples

```julia
lat, lon = NMFk.xy_to_latlon([349000.0], [3884000.0]; zone_isnorth=(13, true))
@assert isfinite(lat) && isfinite(lon)
```
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
    haversine(lat1, lon1, lat2, lon2; r=6372.8)

Compute the haversine distance between two points on a sphere of radius `r`,
where the points are given by the latitude/longitude pairs `lat1/lon1` and
`lat2/lon2` (in degrees).

# Arguments

- `lat1`: Latitude of the first point in degrees.
- `lon1`: Longitude of the first point in degrees.
- `lat2`: Latitude of the second point in degrees.
- `lon2`: Longitude of the second point in degrees.

# Keywords

- `r=6372.8`: Sphere radius; the returned distance uses the same units as `r`.

# Returns

The great-circle distance between the two points.

# Examples

```julia
@assert NMFk.haversine(35.0, -106.0, 35.0, -106.0) == 0.0
```
"""
function haversine(lat1, lon1, lat2, lon2; r = 6372.8)
	lat1, lon1 = deg2rad(lat1), deg2rad(lon1)
	lat2, lon2 = deg2rad(lat2), deg2rad(lon2)
	hav(a, b) = sin((b - a) / 2)^2
	inner_term = hav(lat1, lat2) + cos(lat1) * cos(lat2) * hav(lon1, lon2)
	d = 2 * r * asin(sqrt(inner_term))
	return d
end
