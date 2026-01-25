# NMFk.mapbox() Function Manual

## Overview
`NMFk.mapbox()` is a Julia function that creates interactive Mapbox-based geographic visualizations for spatial data. It's commonly used for plotting geospatial datasets with longitude, latitude, and associated values.

## Basic Syntax

### Method 1: DataFrame Input
```julia
NMFk.mapbox(dataframe; keyword_arguments...)
```

### Method 2: Separate Coordinate Arrays
```julia
NMFk.mapbox(longitude_array, latitude_array, values_array; keyword_arguments...)
```

### Method 3: Specific DataFrame Columns
```julia
NMFk.mapbox(df[!, [:LONGITUDE, :LATITUDE, :VALUE_COLUMN]]; keyword_arguments...)
```

## Common Parameters

### Essential Parameters
- **`filename`**: Output filename (without path)
  ```julia
  filename = "my_map.png"
  ```

- **`figuredir`**: Directory path for saving the figure
  ```julia
  figuredir = "/path/to/output/directory"
  ```

- **`title`**: Map title
  ```julia
  title = "Temperature Distribution"
  ```

### Display Control Parameters
- **`width`**: Map width in pixels (common: `1200`, `2800`)
- **`height`**: Map height in pixels (common: `800`, `1800`)
- **`zoom`**: Map zoom level (common: `4-8`)
- **`dot_size`**: Size of data points (common: `5-20`)

### Value Range Parameters
- **`zmin`**: Minimum value for color scale
- **`zmax`**: Maximum value for color scale
  ```julia
  zmin = 0, zmax = 100
  ```

### Figure Export Parameters
- **`zoom_fig`**: Zoom level for exported figure (often higher than display zoom)
- **`dot_size_fig`**: Point size in exported figure
- **`font_size_fig`**: Font size in exported figure
- **`title_length`**: Maximum characters in title

### Additional Parameters
- **`showcount`**: Boolean to show/hide point count
- **`traces`**: Additional map traces/overlays
- **`marker_color`**: Custom marker colors

## Usage Examples

### Example 1: Basic DataFrame Plotting
```julia
# Simple plotting with a DataFrame containing Lon, Lat, Value columns
NMFk.mapbox(my_dataframe)
```

### Example 2: DataFrame with Custom Parameters
```julia
map_kw = Dict(
    :zoom => 6,
    :showcount => false,
    :dot_size => 8,
    :width => 2800,
    :height => 1800,
    :zoom_fig => 7.8,
    :dot_size_fig => 16,
    :title_length => 22
)

NMFk.mapbox(df[!, [:LONGITUDE, :LATITUDE, :TDS]];
    map_kw...,
    filename = "produced_water_map.png",
    figuredir = "output/figures"
)
```

### Example 3: Separate Arrays with Value Ranges
```julia
NMFk.mapbox(lon_array, lat_array, temperature_values;
    title = "Temperature Distribution",
    zmin = 0.0,
    zmax = 50.0,
    width = 1200,
    height = 800,
    filename = "temp_map.png",
    figuredir = "maps/"
)
```

### Example 4: High-Resolution Export
```julia
NMFk.mapbox(df.Lon, df.Lat, df.Values;
    title = "Geophysical Survey Results",
    zoom = 6,
    dot_size = 5,
    width = 2800,
    height = 1800,
    zoom_fig = 7.8,
    dot_size_fig = 12,
    font_size_fig = 28,
    title_length = 22,
    zmin = 0,
    zmax = 1,
    filename = "survey_results.png",
    figuredir = "results/maps"
)
```

### Example 5: Log-Transformed Data
```julia
NMFk.mapbox(X_lon, X_lat, log10.(data_values);
    title = "Log " * variable_name,
    filename = "log_$(variable_name).png",
    figuredir = "predictions/figures"
)
```

## Common Workflows

### 1. Produced Water Analysis
```julia
map_kw = Dict(:zoom => 4, :showcount => false, :dot_size => 5,
              :width => 14, :height => 9,  :zoom_fig => 5,
              :dot_size_fig => 20, :title_length => 22)

NMFk.mapbox(X_df; map_kw...,
    filename = joinpath(output_dir, "water_data_map.png"))
```

### 2. Geophysical Data Visualization
```julia
for (i, attribute) in enumerate(attribute_names)
    NMFk.mapbox(df.Lon, df.Lat, predictions[:, i];
        title = "$(attribute) Prediction",
        filename = "$(attribute)_pred.png",
        figuredir = "results/predictions",
        zmin = 0, zmax = 1,
        zoom = 6, dot_size = 5,
        width = 2800, height = 1800
    )
end
```

### 3. Interpolation Results
```julia
NMFk.mapbox(grid_df[!, ["lon", "lat", "interpolated_values"]];
    zmin = minimum(original_values),
    zmax = maximum(original_values),
    filename = "interpolation_results.png",
    figuredir = data_directory,
    width = 1200, height = 800
)
```

## Tips and Best Practices

1. **Data Preparation**: Ensure your DataFrame has longitude as the first column, latitude as the second, and values as the third.

2. **Performance**: For large datasets, consider subsampling (e.g., `df[1:10:end, :]`) for initial visualization.

3. **File Organization**: Use consistent `figuredir` paths and meaningful `filename` conventions.

4. **Value Ranges**: Set appropriate `zmin` and `zmax` values to highlight data patterns effectively.

5. **Export Quality**: Use higher `zoom_fig`, `dot_size_fig`, and `font_size_fig` values for publication-quality figures.

6. **Coordinate Order**: The function expects longitude first, then latitude (standard GIS convention).

This manual covers the most common usage patterns found in your workspace. The function is quite flexible and can handle various data formats and visualization requirements for geospatial analysis.