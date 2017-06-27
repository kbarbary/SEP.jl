using SourceExtract
using Base.Test

# write your own tests here
using FITSIO
f = FITS("image.fits")
data = read(f[1])

bkg = BkgMap(data)
subtract!(data, bkg)
print(data)
