# Simple test script to debug the mapbox_contour_simple function

import NMFk

println("Testing mapbox_contour_simple function...")

# Simple test data
lon = [-105.0, -104.5, -104.0]
lat = [35.5, 36.0, 36.5]
temperature = [10.0, 15.0, 20.0]

println("Test data:")
println("lon: $lon")
println("lat: $lat")
println("temperature: $temperature")

try
    println("\nCalling mapbox_contour_simple...")
    p1 = NMFk.mapbox_contour_simple(
        lon, lat, temperature;
        resolution=10,           # Very low resolution for testing
        power=2,
        marker_scale=2.0,
        title="Test",
        colorscale=:hot,
        opacity=0.8,
        filename="debug_test.html"
    )
    println("✓ Function executed successfully!")
    println("Result type: $(typeof(p1))")

catch e
    println("✗ Error occurred:")
    println("Error type: $(typeof(e))")
    println("Error message: $e")
    if isa(e, MethodError)
        println("Method error details:")
        println("Function: $(e.f)")
        println("Arguments: $(e.args)")
    end

    # Print stack trace for debugging
    println("\nStack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
