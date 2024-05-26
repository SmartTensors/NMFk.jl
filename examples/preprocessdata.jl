import DataFrames
import NMFk

cd(@__DIR__)

df = DataFrames.DataFrame(A=rand(10), B="a", C=nothing, D=missing)
df[!, :E] = repeat([true, false], 5)
df[!, :F] = rand(10)
df[!, :G] = rand(10)
df[1, 1] = 0
df[2, 1] *= -1

df2 = DataFrames.DataFrame(Matrix(df), names(df))
df2[4, 2] = "1.1"
df2[5, 3] = "1.2"
df2[6, 4] = "1.3"

df3 = NMFk.processdata(df2)

m = rand(7,4)
m = convert(Array{Any}, m)
m[1, 1] = "a"
m[2, 2] = nothing
m[3, 3] = missing
m[4, 4] = "1.1"

NMFk.processdata(m)

NMFk.processdata(m; enforce_nan=true, string_ok=false)
NMFk.processdata(m; enforce_nan=false, string_ok=false)
NMFk.processdata(m; enforce_nan=false, string_ok=true)
