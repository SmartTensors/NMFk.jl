# Demo: Creating GeoJSON-based continuous contour heatmaps using IDW interpolation
# This example demonstrates how to use the new mapbox_contour function

import NMFk
import DataFrames

# Example 1: Simple synthetic data
println("Example 1: Simple synthetic temperature data")

# Create some sample temperature data
lon = [-105.5, -105.0, -104.5, -104.0, -103.5, -105.2, -104.8, -104.2, -103.8]
lat = [35.5, 36.0, 36.5, 37.0, 37.5, 35.8, 36.3, 36.8, 37.3]
temperature = [15.2, 18.5, 22.1, 25.8, 28.3, 16.7, 20.4, 24.2, 27.1]

# Create contour heatmap with high resolution
# Using mapbox_contour_simple for guaranteed visibility
p1 = NMFk.mapbox_contour_simple(
    lon, lat, temperature;
    resolution=30,           # Medium resolution for good performance
    power=2,                 # Standard IDW power
    marker_scale=3.0,        # Larger markers for better visibility
    title="Temperature Distribution",
    title_colorbar="Temperature (°C)",
    colorscale=:hot,         # Use hot colorscale for temperature
    opacity=0.8,
    show_points=false,       # Hide original data points initially
    filename="temperature_contour.html"
)

# Example 2: Using DataFrame approach
println("\nExample 2: Using DataFrame with elevation data")

# Create DataFrame with elevation data
df = DataFrames.DataFrame(
    lon=[-105.3, -105.0, -104.7, -104.4, -104.1, -105.1, -104.8, -104.5, -104.2],
    lat=[35.6, 36.1, 36.6, 37.1, 37.6, 35.9, 36.4, 36.9, 37.4],
    elevation=[1200, 1450, 1680, 1920, 2150, 1320, 1580, 1840, 2080]
)

# Create contour heatmap from DataFrame
p2 = NMFk.mapbox_contour(
    df, :elevation;
    resolution=60,
    power=1.5,              # Lower power for smoother interpolation
    title="Elevation Contours",
    title_colorbar="Elevation (m)",
    colorscale=:terrain,
    opacity=0.7,
    show_points=true,
    point_size=6,
    filename="elevation_contour.html"
)

# Example 3: Different interpolation parameters
println("\nExample 3: Comparing different interpolation parameters")

# Same data, different parameters
concentration = [0.5, 1.2, 2.1, 3.8, 5.2, 0.8, 1.7, 2.9, 4.1]

# High power (more localized)
p3a = NMFk.mapbox_contour(
    lon, lat, concentration;
    resolution=50,
    power=5,                # High power = more localized
    title="High Power (p=5) - Localized",
    title_colorbar="Concentration",
    colorscale=:viridis,
    filename="concentration_high_power.html"
)

# Low power (smoother)
p3b = NMFk.mapbox_contour(
    lon, lat, concentration;
    resolution=50,
    power=1,                # Low power = smoother
    title="Low Power (p=1) - Smooth",
    title_colorbar="Concentration",
    colorscale=:viridis,
    filename="concentration_low_power.html"
)

# Example 4: High resolution for publication-quality figures
println("\nExample 4: High resolution for publication")

p4 = NMFk.mapbox_contour(
    lon, lat, temperature;
    resolution=150,         # Very high resolution
    power=2,
    title="High Resolution Temperature Map",
    title_colorbar="Temperature (°C)",
    colorscale=:plasma,
    opacity=0.75,
    show_points=true,
    point_size=10,
    width=3200,            # High resolution output
    height=2400,
    filename="high_res_temperature.html"
)

println("\nDemo completed! Check the generated HTML files:")
println("- temperature_contour.html")
println("- elevation_contour.html")
println("- concentration_high_power.html")
println("- concentration_low_power.html")
println("- high_res_temperature.html")

println("\nKey parameters for mapbox_contour:")
println("- resolution: Controls smoothness (higher = smoother but slower)")
println("- power: IDW power parameter (higher = more localized)")
println("- opacity: Transparency of contour layer (0-1)")
println("- show_points: Whether to display original data points")
println("- colorscale: Color scheme (:viridis, :hot, :plasma, :terrain, etc.)")

println("\nAvailable functions:")
println("- mapbox_contour: Advanced version with density-based rendering")
println("- mapbox_contour_simple: Simpler version with guaranteed visibility")
println("- Use mapbox_contour_simple if the heatmap is not visible with mapbox_contour")
