# Test script to verify heatmap visibility
# This script demonstrates the difference between the two approaches

import NMFk
import DataFrames

println("Testing heatmap visibility...")

# Create test data
lon = [-105.5, -105.0, -104.5, -104.0, -103.5]
lat = [35.5, 36.0, 36.5, 37.0, 37.5]
values = [10.0, 15.0, 20.0, 25.0, 30.0]

println("\nTest 1: Using mapbox_contour_simple (guaranteed visibility)")
try
    p1 = NMFk.mapbox_contour_simple(
        lon, lat, values;
        resolution=25,
        marker_scale=3.0,  # Large markers for visibility
        title="Simple Heatmap Test",
        colorscale=:hot,
        opacity=0.8,
        show_points=true,
        filename="test_simple.html"
    )
    println("✓ Simple heatmap created successfully")
catch e
    println("✗ Error creating simple heatmap: $e")
end

println("\nTest 2: Using standard mapbox_contour")
try
    p2 = NMFk.mapbox_contour(
        lon, lat, values;
        resolution=50,
        title="Standard Heatmap Test",
        colorscale=:hot,
        opacity=0.8,
        show_points=true,
        filename="test_standard.html"
    )
    println("✓ Standard heatmap created successfully")
catch e
    println("✗ Error creating standard heatmap: $e")
end

println("\nTest 3: DataFrame usage with simple version")
try
    df = DataFrames.DataFrame(
        longitude=lon,
        latitude=lat,
        temperature=values
    )

    p3 = NMFk.mapbox_contour_simple(
        df, :temperature;
        resolution=30,
        title="DataFrame Simple Test",
        colorscale=:plasma,
        filename="test_dataframe_simple.html"
    )
    println("✓ DataFrame simple heatmap created successfully")
catch e
    println("✗ Error creating DataFrame simple heatmap: $e")
end

println("\nTest completed!")
println("Check the generated HTML files to verify visibility:")
println("- test_simple.html (should be clearly visible)")
println("- test_standard.html (may have visibility issues)")
println("- test_dataframe_simple.html (DataFrame version)")

println("\nRecommendation: Use mapbox_contour_simple for guaranteed visibility")
