"""
Aprs' relationship
"""
function arps(t::Union{Number,AbstractVector}, q0::Number, D::Number, b::Number)
	q = q0 .* (1 .+ b * D .* t) .^ (1. / b)
	return q
end

"""
Aprs' exponential relationship (b=0)
"""
function arps_exponential(t::Union{Number,AbstractVector}, q0::Number, D::Number)
	q = q0 .* exp.(-D .* t)
	return q
end

"""
Aprs' harmonic relationship (b=1)
"""
function arps_harmonic(t::Union{Number,AbstractVector}, q0::Number, D::Number)
	q = q0 .* (1 .+ D .* t)
	return q
end