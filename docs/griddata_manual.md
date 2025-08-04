# NMFk.griddata() Function Manual

## Overview

The `NMFk.griddata()` function is a versatile tool for creating regular grids from scattered spatial data points. It's particularly useful for spatial data processing, interpolation setup, and creating uniform coordinate grids for geospatial analysis. This function supports both 2D coordinate grid generation and 3D data gridding operations.

## Function Signatures

### 1. 2D Grid Generation (x, y coordinates only)
```julia
NMFk.griddata(x::AbstractVector, y::AbstractVector;
              stepvalue=nothing,
              nbins=nothing,
              xrev::Bool=false,
              xnbins::Integer=length(x),
              xminvalue=minimumnan(x),
              xmaxvalue=maximumnan(x),
              xstepvalue=stepvalue,
              yrev::Bool=false,
              ynbins=length(y),
              yminvalue=minimumnan(y),
              ymaxvalue=maximumnan(y),
              ystepvalue=stepvalue,
              granulate::Bool=true,
              quiet::Bool=true)
```

### 2. 3D Data Gridding (x, y coordinates with z values)
```julia
NMFk.griddata(x::AbstractVector, y::AbstractVector, z::AbstractVector; kw...)
NMFk.griddata(x::AbstractVector, y::AbstractVector, z::AbstractMatrix;
              type::DataType=eltype(z),
              xrev::Bool=false,
              xnbins::Integer=length(x),
              xminvalue=minimum(x),
              xmaxvalue=maximum(x),
              xstepvalue=nothing,
              yrev::Bool=false,
              ynbins=length(y),
              yminvalue=minimum(y),
              ymaxvalue=maximum(y),
              ystepvalue=nothing,
              granulate::Bool=true,
              quiet::Bool=false)
```

## Parameters

### Core Parameters

- **`x`**: Vector of x-coordinates (e.g., longitude values)
- **`y`**: Vector of y-coordinates (e.g., latitude values)
- **`z`**: Vector or matrix of data values at (x,y) locations (optional)

### Grid Control Parameters

- **`stepvalue`**: Single step size for both x and y dimensions (takes precedence over individual step values)
- **`xstepvalue`**, **`ystepvalue`**: Individual step sizes for x and y dimensions
- **`nbins`**: Number of bins for both dimensions (uniform grid)
- **`xnbins`**, **`ynbins`**: Number of bins for each dimension individually

### Range Control Parameters

- **`xminvalue`**, **`xmaxvalue`**: Minimum and maximum values for x-axis
- **`yminvalue`**, **`ymaxvalue`**: Minimum and maximum values for y-axis

### Options

- **`xrev`**, **`yrev`**: Reverse the order of bins for x or y axis
- **`granulate`**: Adjust min/max values to align with step sizes
- **`quiet`**: Suppress informational output
- **`type`**: Data type for output arrays (for 3D gridding)

## Return Values

### For 2D Grid Generation
Returns a tuple `(xgrid, ygrid)` where:
- **`xgrid`**: Range object for x-coordinates
- **`ygrid`**: Range object for y-coordinates

### For 3D Data Gridding
Returns a 3D array `T` where:
- **`T[i, j, k]`**: Averaged values at grid point (i,j) for attribute k
- Dimensions: `(xbins, ybins, number_of_attributes)`

## Usage Examples

### Example 1: Basic Grid Generation with Step Size
```julia
using NMFk
using DataFrames

# Sample coordinate data (e.g., longitude/latitude)
longitude = [-120.5, -120.3, -119.8, -119.2, -118.9, -118.5]
latitude = [35.2, 35.8, 36.1, 36.5, 36.8, 37.1]

# Create grid with 0.02-degree step size
lon_grid, lat_grid = NMFk.griddata(longitude, latitude; stepvalue=0.02, quiet=false)

println("Longitude grid: ", lon_grid)
println("Latitude grid: ", lat_grid)
```

### Example 2: Grid Generation with Different Step Sizes for X and Y
```julia
# Create grid with different step sizes for x and y
lon_grid, lat_grid = NMFk.griddata(longitude, latitude;
                                   xstepvalue=0.04,
                                   ystepvalue=0.03,
                                   quiet=false)

println("X-grid points: ", length(lon_grid))
println("Y-grid points: ", length(lat_grid))
```

### Example 3: Grid Generation with Fixed Number of Bins
```julia
# Create uniform grid with 51 bins in each dimension
xgrid, ygrid = NMFk.griddata(-20000:20000, -20000:20000; nbins=51)

println("Grid dimensions: $(length(xgrid)) x $(length(ygrid))")
```

### Example 4: 3D Data Gridding for Spatial Interpolation Setup
```julia
# Sample geospatial data
locations_x = [1.0, 2.5, 4.2, 3.8, 5.1]
locations_y = [2.3, 4.1, 1.8, 5.2, 3.7]
measurements = [15.2, 23.8, 12.4, 28.1, 19.5]

# Grid the data for interpolation
gridded_data = NMFk.griddata(locations_x, locations_y, measurements;
                            xstepvalue=0.5,
                            ystepvalue=0.5,
                            quiet=false)

println("Gridded data dimensions: ", size(gridded_data))
```

### Example 5: Multi-attribute Gridding
```julia
# Multiple attributes at same locations
temperature = [25.1, 27.3, 23.8, 29.2, 26.5]
humidity = [45.2, 52.1, 38.9, 58.3, 48.7]
pressure = [1013.2, 1015.8, 1011.4, 1017.3, 1014.1]

# Combine into matrix (each row is a location, each column is an attribute)
multi_data = hcat(temperature, humidity, pressure)

# Grid multiple attributes simultaneously
gridded_multi = NMFk.griddata(locations_x, locations_y, multi_data;
                             xstepvalue=0.5,
                             ystepvalue=0.5)

println("Multi-attribute grid dimensions: ", size(gridded_multi))
println("Temperature at grid point [1,1,1]: ", gridded_multi[1,1,1])
println("Humidity at grid point [1,1,2]: ", gridded_multi[1,1,2])
println("Pressure at grid point [1,1,3]: ", gridded_multi[1,1,3])
```

## Practical Applications

### 1. Preparing Data for Kriging Interpolation
```julia
# Typical workflow from the GeoDAWN examples
df_magnetics = DataFrame(Lon=longitude, Lat=latitude, Magnetics=measurements)

# Create regular grid
lon_grid, lat_grid = NMFk.griddata(df_magnetics.Lon, df_magnetics.Lat; stepvalue=0.02, quiet=true)

# Get grid points for interpolation
using Kriging  # Assuming Kriging package is available
lon_lat_grid = Kriging.getgridpoints(lon_grid, lat_grid)

# Now use for interpolation
interpolated_values = Kriging.interpolate_neighborhood(
    lon_lat_grid,
    permutedims([df_magnetics.Lon df_magnetics.Lat]),
    df_magnetics.Magnetics;
    neighborsearch=50,
    numobsneighbors=10,
    interpolate=Kriging.inversedistance,
    pow=2,
    cutoff=0.02
)
```

### 2. Creating Coordinate Meshes for Analysis
```julia
# Create coordinate system for analysis
side_length = 40000  # 40 km
xgrid, ygrid = NMFk.griddata(-side_length:side_length, -side_length:side_length; nbins=51)

# Use for spatial analysis
coord_pairs = [(x, y) for x in xgrid, y in ygrid]
println("Created $(length(coord_pairs)) coordinate pairs")
```

### 3. Data Quality Assessment
```julia
# Check data coverage and overlap
lon_data = [-119.5, -119.3, -119.3, -119.1]  # Some duplicate locations
lat_data = [35.2, 35.4, 35.4, 35.6]
values = [10.1, 12.3, 11.8, 15.2]

# Grid the data to see overlaps
gridded = NMFk.griddata(lon_data, lat_data, values; xstepvalue=0.1, ystepvalue=0.1, quiet=false)

# The function will report "Maximum number of data overlaps"
# helping identify data density issues
```

## Important Notes

### 1. Data Handling
- **NaN handling**: The function automatically handles NaN values in the data
- **Data averaging**: When multiple data points fall into the same grid cell, they are automatically averaged
- **Empty cells**: Grid cells with no data points will contain zeros

### 2. Memory Considerations
- For large datasets, consider the memory requirements of the output grid
- The 3D version creates arrays of size `(xbins, ybins, number_of_attributes)`

### 3. Coordinate Systems
- The function works with any coordinate system (geographic, projected, etc.)
- Ensure consistent units for step sizes and coordinates

### 4. Grid Alignment
- When `granulate=true`, the function adjusts min/max values to align with step sizes
- This ensures clean grid boundaries but may slightly expand the data range

## Common Patterns from Examples

### Pattern 1: Geographic Data Processing
```julia
# Common pattern for processing geographic data
lon_grid, lat_grid = NMFk.griddata(df.Lon, df.Lat; quiet=true, stepvalue=0.02)
```

### Pattern 2: Regular Mesh Creation
```julia
# Common pattern for creating regular meshes
xgrid, ygrid = NMFk.griddata(-half_size:half_size, -half_size:half_size; nbins=grid_resolution)
```

### Pattern 3: Multi-Resolution Grids
```julia
# Different resolutions for different dimensions
xgrid, ygrid = NMFk.griddata(x_coords, y_coords; xstepvalue=fine_resolution, ystepvalue=coarse_resolution)
```

## Error Handling

The function includes several built-in checks:
- Validates that x and y vectors have the same length
- Ensures z data matches x,y dimensions
- Reports empty bins when `quiet=false`
- Provides information about data overlaps

## See Also

- `NMFk.indicize()`: The underlying binning function
- `NMFk.plotscatter()`: For visualizing gridded results
- `NMFk.mapbox()`: For creating interactive maps with gridded data
- Kriging interpolation functions for spatial analysis

This function is commonly used in the NMFk ecosystem for preprocessing spatial data before NMF analysis, setting up interpolation grids, and creating regular coordinate systems for geospatial machine learning applications.
