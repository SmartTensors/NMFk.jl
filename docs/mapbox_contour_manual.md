# Mapbox Contour Heatmaps with IDW Interpolation

The NMFk package now includes functionality to create GeoJSON-based continuous contour heatmaps using Inverse Distance Weighting (IDW) interpolation. This feature is particularly useful for visualizing spatial data with smooth, continuous surfaces overlaid on interactive Mapbox maps.

## Overview

The `mapbox_contour` function creates smooth, interpolated heatmaps from scattered geographic data points. It uses IDW interpolation to create a dense grid of interpolated values, which are then displayed as a continuous surface on an interactive Mapbox map.

## Key Features

- **IDW Interpolation**: Uses Inverse Distance Weighting for smooth interpolation between data points
- **Resolution Control**: Adjustable grid resolution for performance vs. quality trade-offs
- **Interactive Maps**: Built on Mapbox for pan, zoom, and hover interactions
- **Flexible Input**: Accepts both coordinate arrays and DataFrames
- **Customizable Visualization**: Multiple color scales, opacity controls, and styling options
- **Publication Ready**: High-resolution output options for figures

## Functions

### mapbox_contour(lon, lat, values; kw...)

Creates a continuous contour heatmap from longitude, latitude, and value vectors.

**Arguments:**
- `lon::AbstractVector`: Longitude coordinates
- `lat::AbstractVector`: Latitude coordinates
- `values::AbstractVector`: Values to interpolate
- `resolution::Int=50`: Grid resolution (higher = smoother but slower)
- `power::Real=2`: IDW power parameter (higher = more localized)
- `smoothing::Real=0.0`: Smoothing parameter for interpolation
- `show_points::Bool=false`: Whether to show original data points
- `colorscale::Symbol=:turbo`: Color scheme
- `opacity::Real=0.7`: Transparency of contour layer
- `filename::AbstractString=""`: Output filename

### mapbox_contour(df, column; kw...)

Creates a contour heatmap from a DataFrame.

**Arguments:**
- `df::DataFrames.DataFrame`: DataFrame with longitude/latitude columns
- `column::Union{Symbol, AbstractString}`: Column name containing values
- Additional keyword arguments as above

### idw_interpolate(x_data, y_data, values, x_interp, y_interp; kw...)

Performs IDW interpolation for a single point (utility function).

## Usage Examples

### Basic Usage

```julia
import NMFk

# Simple temperature data
lon = [-105.0, -104.5, -104.0, -103.5]
lat = [35.5, 36.0, 36.5, 37.0]
temperature = [15.0, 18.0, 22.0, 25.0]

# Create contour heatmap
p = NMFk.mapbox_contour(lon, lat, temperature;
    resolution=80,
    title="Temperature Distribution",
    colorscale=:hot,
    filename="temperature_map.html"
)
```

### DataFrame Usage

**Important**: For DataFrame usage, longitude and latitude columns must be named according to these patterns:
- **Longitude**: `lon`, `Lon`, `LON`, `longitude`, `Longitude`, `LONGITUDE`, or single `x`/`X`
- **Latitude**: `lat`, `Lat`, `LAT`, `latitude`, `Latitude`, `LATITUDE`, or single `y`/`Y`

```julia
import DataFrames

df = DataFrames.DataFrame(
    lon=[-105.0, -104.5, -104.0],
    lat=[35.5, 36.0, 36.5],
    elevation=[1200, 1500, 1800]
)

p = NMFk.mapbox_contour(df, :elevation;
    resolution=100,
    colorscale=:terrain,
    show_points=true
)
```

### Advanced Customization

```julia
# High-resolution publication figure
p = NMFk.mapbox_contour(lon, lat, values;
    resolution=150,         # High resolution
    power=2,               # Standard IDW power
    opacity=0.8,           # Semi-transparent
    show_points=true,      # Show data points
    point_size=8,          # Point size
    colorscale=:plasma,    # Color scheme
    width=3200,            # High-res output
    height=2400,
    filename="publication_figure.html"
)
```

## Parameter Guidelines

### Resolution
- **Low (20-40)**: Fast rendering, coarse interpolation
- **Medium (50-80)**: Good balance of speed and quality
- **High (100-200)**: Smooth interpolation, slower rendering

### IDW Power Parameter
- **Low (1-1.5)**: Smoother, more global interpolation
- **Medium (2-3)**: Standard interpolation (recommended)
- **High (4-8)**: More localized, preserves data point values

### Color Scales
Popular options include:
- `:turbo` - Perceptually uniform, good for scientific data
- `:hot` - Traditional heat map colors
- `:plasma` - High contrast, good for presentations
- `:terrain` - Natural colors for elevation/geological data
- `:turbo` - Full spectrum (use cautiously)

## Performance Considerations

- **Resolution**: Higher values create smoother maps but increase computation time
- **Data density**: More data points increase interpolation accuracy but computation time
- **Grid optimization**: The function automatically subsamples dense grids for display performance

## Integration with Existing NMFk Workflow

The contour functionality integrates seamlessly with existing NMFk plotting functions:

```julia
# Can be combined with existing mapbox functions
traces = [existing_mapbox_trace]
p = NMFk.mapbox_contour(lon, lat, values;
    traces=traces,  # Add to existing plot
    opacity=0.6     # Semi-transparent overlay
)
```## Output Formats

The function supports the same output formats as other NMFk plotting functions:
- HTML (interactive)
- PNG (static, high-resolution)
- PDF (vector graphics)
- SVG (vector graphics)

Set the format via the `format` parameter or file extension in `filename`.

## Best Practices

1. **Data quality**: Ensure longitude/latitude coordinates are accurate
2. **Coordinate systems**: Use WGS84 (standard GPS coordinates)
3. **Data distribution**: More evenly distributed points give better interpolation
4. **Resolution tuning**: Start with medium resolution, adjust based on needs
5. **Color selection**: Choose perceptually uniform color scales for scientific data
6. **Transparency**: Use opacity < 1.0 when overlaying on satellite imagery

## Troubleshooting

**Common issues:**
- Ensure at least 3 data points for interpolation
- Check for NaN values in coordinates or data
- Verify longitude/latitude column names match regex patterns
- For large datasets, consider data preprocessing or sampling

See the demo file `examples/mapbox_contour_demo.jl` for comprehensive examples.
