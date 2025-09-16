# Minimal working example for mapbox_contour_simple
import NMFk

println("Testing mapbox_contour_simple with minimal parameters...")

# Simple test data - 5 points
lon = [-105.0, -104.5, -104.0, -103.5, -103.0]
lat = [35.5, 36.0, 36.5, 37.0, 37.5]
temperature = [10.0, 15.0, 20.0, 25.0, 30.0]

println("Data: $(length(lon)) points")
println("Lon range: $(minimum(lon)) to $(maximum(lon))")
println("Lat range: $(minimum(lat)) to $(maximum(lat))")
println("Temp range: $(minimum(temperature)) to $(maximum(temperature))")

try
    # Very minimal call
    println("\nAttempting minimal function call...")
    p = NMFk.mapbox_contour_simple(
        lon, lat, temperature;
        resolution=15,         # Very low resolution to start
        marker_scale=3.0,      # Large markers
        title="Simple Test",
        filename="minimal_test.html"
    )

    println("✓ SUCCESS: Function executed without errors!")
    println("Plot object type: $(typeof(p))")

    # Test with more parameters
    println("\nAttempting with more parameters...")
    p2 = NMFk.mapbox_contour_simple(
        lon, lat, temperature;
        resolution=20,
        power=2,
        marker_scale=2.5,
        title="Advanced Test",
        title_colorbar="Temperature (°C)",
        colorscale=:hot,
        opacity=0.8,
        show_points=true,
        filename="advanced_test.html"
    )

    println("✓ SUCCESS: Advanced function call worked!")

catch e
    println("✗ ERROR: Function failed")
    println("Error type: $(typeof(e))")
    println("Error message: $e")

    # Print detailed error info
    if isa(e, MethodError)
        println("\nMethod Error Details:")
        println("Function: $(e.f)")
        println("Arguments: $(e.args)")
    end

    # Show stack trace
    println("\nStack Trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\nTest completed.")
