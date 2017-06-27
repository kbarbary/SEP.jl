module SEP

import Base: collect, -
export Background, background

if isfile(joinpath(dirname(@__FILE__),"..","deps","deps.jl"))
    include("../deps/deps.jl")
else
    error("SEP not properly installed. Please run Pkg.build(\"SEP\")")
end

# Julia Type -> numeric type code from sep.h
const SEPImageType = Union{Bool, UInt8, Cint, Cfloat, Cdouble}
sep_typecode(::Type{Bool}) = Cint(11)
sep_typecode(::Type{UInt8}) = Cint(11)
sep_typecode(::Type{Cint}) = Cint(31)
sep_typecode(::Type{Cfloat}) = Cint(42)
sep_typecode(::Type{Cdouble}) = Cint(82)


# definitions from sep.h
const SEP_NOISE_NONE = Cint(0)
const SEP_NOISE_STDDEV = Cint(1)
const SEP_NOISE_VAR = Cint(2)

function sep_assert_ok(status::Cint)
    if status != 0
        msg = Vector{UInt8}(61)
        ccall((:sep_get_errmsg, libsep), Void, (Int32,Ptr{UInt8}), status, msg)
        msg[end] = 0  # ensure NULL-termination, just in case.
        error(unsafe_string(pointer(msg)))
    end
end

# internal use only.
struct sep_image
data::Ptr{Void}
noise::Ptr{Void}
mask::Ptr{Void}
dtype::Cint
ndtype::Cint
mdtype::Cint
w::Cint
h::Cint
noiseval::Cdouble
noise_type::Cshort
gain::Cdouble
maskthresh::Cdouble
end



# ---------------------------------------------------------------------------
# Background functions

mutable struct Background
    ptr::Ptr{Void}
    data_size::Tuple{Int, Int}
end

free!(bkg::Background) = ccall((:sep_bkg_free, libsep), Void, (Ptr{Void},),
                               bkg.ptr)


function background(data::Array{T, 2}; mask=nothing, boxsize=(64, 64),
                    filtersize=(3, 3), filterthresh=0.0, maskthresh=0) where {T<:SEPImageType}
    sz = size(data)
    dtype = sep_typecode(T)

    # mask options
    mdtype = Cint(0)
    mask_ptr = C_NULL
    if mask !== nothing
        isa(mask, Matrix) || error("mask must be a 2-d array")
        size(mask) == sz || error("data and mask must be same size")
        mdtype = sep_typecode(T)
        mask_ptr = Ptr{Void}(pointer(mask))
    end

    im = sep_image(Ptr{Void}(pointer(data)),
                   C_NULL,
                   mask_ptr,
                   dtype,
                   Cint(0),
                   mdtype,
                   sz[1],
                   sz[2],
                   0.0,
                   SEP_NOISE_NONE,  # SEP_NOISE_NONE
                   0.0,
                   Cdouble(maskthresh))

    result = Ref{Ptr{Void}}(C_NULL)
    status = ccall((:sep_background, libsep), Cint,
                   (Ptr{sep_image}, Cint, Cint, Cint, Cint, Cdouble,
                    Ref{Ptr{Void}}),
                   &im, boxsize[1], boxsize[2],
                   filtersize[1], filtersize[2], filterthresh, result)
    sep_assert_ok(status)

    bkg = Background(result[], sz)
    finalizer(bkg, free!)
    return bkg
end

#=
function collect{T}(bkgmap::Background{T})
    result = Array(T, bkgmap.data_size)
    status = ccall((:sep_backarray, libsep), Cint, (Ptr{Void}, Ptr{Void}, Cint),
                   bkgmap.ptr, result, sep_typecode(T))
    sep_assert_ok(status)
    return result
end

"""
subtract!(bkgmap, A)

Subtract a background from an array in-place.
"""
function subtract!{T}(bkgmap::Background, A::Array{T, 2})
    status = ccall((:sep_subbackarray, libsep), Cint,
                   (Ptr{Void}, Ptr{Void}, Cint),
                   bkgmap.ptr, A, sep_typecode(T))
    sep_assert_ok(status)
end

function (-){T}(A::Array{T, 2}, bkgmap::Background)
    B = copy(A)
    subtract!(bkgmap, B)
    return B
end

=#

# ---------------------------------------------------------------------------
# Source Extraction

#=
# Mirror of C struct
immutable sepobj
    thresh::Cfloat
    npix::Cint
    tnpix::Cint
    xmin::Cint
    xmax::Cint
    ymin::Cint
    ymax::Cint
    mx::Cdouble
    my::Cdouble
    mx2::Cdouble
    my2::Cdouble
    mxy::Cdouble
    a::Cfloat
    b::Cfloat
    theta::Cfloat
    abcor::Cfloat
    cxx::Cfloat
    cyy::Cfloat
    cxy::Cfloat
    cflux::Cfloat
    flux::Cfloat
    cpeak::Cfloat
    peak::Cfloat
    flag::Cshort
    pix::Ptr{Cint}
end

type AstroSource
    thresh::Cfloat
    npix::Cint
    tnpix::Cint
    xmin::Cint
    xmax::Cint
    ymin::Cint
    ymax::Cint
    mx::Cdouble
    my::Cdouble
    mx2::Cdouble
    my2::Cdouble
    mxy::Cdouble
    a::Cfloat
    b::Cfloat
    theta::Cfloat
    abcor::Cfloat
    cxx::Cfloat
    cyy::Cfloat
    cxy::Cfloat
    cflux::Cfloat
    flux::Cfloat
    cpeak::Cfloat
    peak::Cfloat
    flag::Cshort
    pix::Array{Cint, 1}
end

function extractobjs(im::Array{Float32, 2}, thresh::Real)
    nx, ny = size(im)
    conv = Float32[1 2 1; 2 4 2; 1 2 1]
    nobj = Array(Int32, 1)
    ptr = Array(Ptr{sepobj}, 1)
    status = ccall((:extractobj, libsep), Cint,
                   (Ptr{Cfloat}, Ptr{Cfloat}, Cint, Cint, Cfloat, Cint,
                    Ptr{Cfloat}, Cint, Cint,
                    Cint, Cdouble, Cint, Cdouble,
                    Ptr{Cint}, Ptr{Ptr{sepobj}}),
                   im, C_NULL, nx, ny, thresh, 5,
                   conv, size(conv, 1), size(conv, 2),
                   32, 0.005, 1, 1.0,
                   nobj, ptr)

    n = int64(nobj[1])

    # not taking ownership... need to free at the end.
    cresult = pointer_to_array(ptr[1], n)
    
    result = Array(AstroSource, n)
    for i=1:n
        result[i] = AstroSource(cresult[i].thresh,
                                cresult[i].npix,
                                cresult[i].tnpix,
                                cresult[i].xmin,
                                cresult[i].xmax,
                                cresult[i].ymin,
                                cresult[i].ymax,
                                cresult[i].mx,
                                cresult[i].my,
                                cresult[i].mx2,
                                cresult[i].my2,
                                cresult[i].mxy,
                                cresult[i].a,
                                cresult[i].b,
                                cresult[i].theta,
                                cresult[i].abcor,
                                cresult[i].cxx,
                                cresult[i].cyy,
                                cresult[i].cxy,
                                cresult[i].cflux,
                                cresult[i].flux,
                                cresult[i].cpeak,
                                cresult[i].peak,
                                cresult[i].flag,
                                pointer_to_array(cresult[i].pix,
                                                 (int(cresult[i].npix),),
                                                 true))
    end

    c_free(ptr[1])  # Cannot access cresult after this!
    
    result
end

=#


end # module
