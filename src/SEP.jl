module SourceExtract

import Base: collect, -
export BkgMap, subtract!

f isfile(joinpath(dirname(@__FILE__),"..","deps","deps.jl"))
    include("../deps/deps.jl")
else
    error("SEP not properly installed. Please run Pkg.build(\"SEP\")")
end

# Julia Type -> numeric type code from sep.h
sep_typecode(::Type{Bool}) = Cint(11)
sep_typecode(::Type{UInt8}) = Cint(11)
sep_typecode(::Type{Cint}) = Cint(31)
sep_typecode(::Type{Cfloat}) = Cint(42)
sep_typecode(::Type{Cdouble}) = Cint(82)

function sep_assert_ok(status::Cint)
    if status != 0
        msg = Array(Uint8, 61)
        ccall((:sep_get_errmsg, libsep), Void, (Int32,Ptr{Uint8}), status, msg)
        error(bytestring(msg))
    end
end

# ---------------------------------------------------------------------------
# Background functions

# This could be made slightly more efficient by mirroring the C struct and
# changing the C library to not allocate the struct itself (only the
# internal pointers).

type BkgMap{T}
    ptr::Ptr{Void}
    data_size::Tuple{Int, Int}
end

function BkgMap{T}(data::Array{T, 2}; mask=nothing, boxsize=(64, 64),
                   filtersize=(3, 3), filterthresh=0.0, maskthresh=0)
    sz = size(data)
    dtype = sep_typecode(T)

    # mask options
    mdtype = Cint(0)
    mptr = C_NULL
    if !isa(mask, Void)
        isa(mask, Matrix) || error("mask must be a 2-d array")
        size(mask) == sz || error("data and mask must be same size")
        mdtype = sep_typecode(T)
        mptr = Ptr{Void}(pointer(mask))
    end

    result = Ref{Ptr{Void}}(C_NULL)
    status = ccall((:sep_makeback, libsep), Cint,
                   (Ptr{Void}, Ptr{Void}, Cint, Cint, Cint, Cint, Cint, Cint,
                    Cfloat, Cint, Cint, Cfloat, Ref{Ptr{Void}}),
                   data, mptr, dtype, mdtype, sz[1], sz[2],
                   boxsize[1], boxsize[2], maskthresh,
                   filtersize[1], filtersize[2], filterthresh, result)
    sep_assert_ok(status)

    bkgmap = BkgMap{T}(result[], sz)
    finalizer(bkgmap, free!)
    return bkgmap
end

"""
collect(bkgmap)

Materialize the background as a 2-d array with same type as original data.
"""
function collect{T}(bkgmap::BkgMap{T})
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
function subtract!{T}(bkgmap::BkgMap, A::Array{T, 2})
    status = ccall((:sep_subbackarray, libsep), Cint,
                   (Ptr{Void}, Ptr{Void}, Cint),
                   bkgmap.ptr, A, sep_typecode(T))
    sep_assert_ok(status)
end

function (-){T}(A::Array{T, 2}, bkgmap::BkgMap)
    B = copy(A)
    subtract!(bkgmap, B)
    return B
end

free!(bkgmap::BkgMap) = ccall((:sep_freeback, libsep), Void, (Ptr{Void},),
                              bkgmap.ptr)

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
