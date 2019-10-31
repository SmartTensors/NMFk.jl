"""
Arp's relationship (hyperbolic decline; 0 < b < 1)
"""
function arp(t::Union{Number,AbstractVector}, q0::Number, D::Number, b::Number)
	q = q0 .* (1 .+ b * D .* t) .^ (1. / b)
	return q
end

"""
Arp's relationship (exponential decline; b = 0)
"""
function arp_exponential(t::Union{Number,AbstractVector}, q0::Number, D::Number)
	q = q0 .* exp.(-D .* t)
	return q
end

"""
Arp's relationship (harmonic decline; b = 1)
"""
function arp_harmonic(t::Union{Number,AbstractVector}, q0::Number, D::Number)
	q = q0 .* (1 .+ D .* t)
	return q
end

"""
Arp's EUR estimate (hyperbolic decline; 0 < b < 1)
"""
function arp_eur(q0::Number, D::Number, b::Number, qE::Number=q0/1000, Qi::Number=0)
	eur = Qi + q0^b / ((1. - b) * D) * (q0^(1. - b) - qE^(1. - b))
	return eur
end

"""
Arp's EUR estimate (exponential decline; b = 0)
"""
function arp_eur_exponential(q0::Number, D::Number, Qi::Number=0)
	eur = Qi + q0 / D
	return eur
end

"""
Arp's EUR estimate (harmonic decline; b = 1)
"""
function arp_eur_harmonic(q0::Number, D::Number, qE::Number=q0/10000, Qi::Number=0)
	eur = Qi + q0 / D * log(q0 / qE)
	return eur
end