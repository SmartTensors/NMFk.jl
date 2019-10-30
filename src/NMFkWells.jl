"""
Arp's relationship
"""
function arp(t::Union{Number,AbstractVector}, q0::Number, D::Number, b::Number)
	q = q0 .* (1 .+ b * D .* t) .^ (1. / b)
	return q
end

"""
Arp's exponential relationship (b=0)
"""
function arp_exponential(t::Union{Number,AbstractVector}, q0::Number, D::Number)
	q = q0 .* exp.(-D .* t)
	return q
end

"""
Arp's harmonic relationship (b=1)
"""
function arp_harmonic(t::Union{Number,AbstractVector}, q0::Number, D::Number)
	q = q0 .* (1 .+ D .* t)
	return q
end