using SEP
using Base.Test


using FITSIO

# use test image in deps directory
image_name = joinpath(dirname(@__FILE__), "..", "deps", "src", "sep-1.0.0",
                      "data", "image.fits")
f = FITS(image_name)
data = read(f[1])

bkg = background(data)
println(bkg)
#subtract!(data, bkg)
#print(data)
