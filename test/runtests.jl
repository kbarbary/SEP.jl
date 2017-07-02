using SEP
using Base.Test
using FITSIO


# use test image in deps directory
datadir = joinpath(dirname(@__FILE__), "..", "deps", "src", "sep-1.0.0",
                   "data")

data = read(FITS(joinpath(datadir, "image.fits"))[1])
back_sextractor = read(FITS(joinpath(datadir, "back.fits"))[1])
rms_sextractor = read(FITS(joinpath(datadir, "rms.fits"))[1])

# test background
bkg = background(data)
println(bkg)

back = collect(bkg)
println(back ≈ back_sextractor)

for T in (Float32, Float64)
    A = zeros(T, size(data))
    back = collect(T, bkg)
    println(back ≈ back_sextractor)
end

A = zeros(Float32, size(data))
A .-= bkg

B = zeros(Float32, size(data))
B -= bkg
@test A ≈ B

