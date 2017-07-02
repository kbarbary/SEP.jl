module SEP

import Base: collect, broadcast!, -, show
export Background, background, rms

if isfile(joinpath(dirname(@__FILE__),"..","deps","deps.jl"))
    include("../deps/deps.jl")
else
    error("SEP not properly installed. Please run Pkg.build(\"SEP\")")
end

# Julia Type -> numeric type code from sep.h
const SEPMaskType = Union{Bool, UInt8, Cint, Cfloat, Cdouble}
const PixType = Union{UInt8, Cint, Cfloat, Cdouble}
sep_typecode(::Type{Bool}) = Cint(11)
sep_typecode(::Type{UInt8}) = Cint(11)
sep_typecode(::Type{Cint}) = Cint(31)
sep_typecode(::Type{Cfloat}) = Cint(42)
sep_typecode(::Type{Cdouble}) = Cint(82)


# definitions from sep.h
const SEP_NOISE_NONE = Cshort(0)
const SEP_NOISE_STDDEV = Cshort(1)
const SEP_NOISE_VAR = Cshort(2)

const SEP_THRESH_REL = Cint(0)
const SEP_THRESH_ABS = Cint(1)

const SEP_FILTER_CONV = Cint(0)
const SEP_FILTER_MATCHED = Cint(1)

function sep_assert_ok(status::Cint)
    if status != 0
        msg = Vector{UInt8}(61)
        ccall((:sep_get_errmsg, libsep), Void, (Int32,Ptr{UInt8}), status, msg)
        msg[end] = 0  # ensure NULL-termination, just in case.
        error(unsafe_string(pointer(msg)))
    end
end

# internal use only: mirrors `sep_image` struct in sep.h
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

function sep_image(data::Array{T, 2};
                   noise=nothing, mask=nothing, noise_type=:stddev,
                   gain=0.0, mask_thresh=0.0) where {T<:PixType}
    sz = size(data)

    # data is required
    data_ptr = Ptr{Void}(pointer(data))
    dtype = sep_typecode(T)

    # mask options
    mask_ptr = C_NULL
    mdtype = Cint(0)
    if mask !== nothing
        isa(mask, Matrix) || error("mask must be a 2-d array")
        size(mask) == sz || error("data and mask must be same size")
        mask_ptr = Ptr{Void}(pointer(mask))
        mdtype = sep_typecode(eltype(mask))
    end

    # noise options
    ndtype = Cint(0)
    noise_ptr = C_NULL
    noiseval = 0.0
    noise_typecode = SEP_NOISE_NONE
    if noise !== nothing
        if isa(noise, Matrix)
            size(noise) == sz || error("noise array must be same size as data")
            noise_ptr = sep_typecode(eltype(noise))
            ndtype = sep_typecode(eltype(ndtype))
        elseif isa(noise, Number)
            noiseval = Cdouble(noise)
        else
            error("noise must be array or number")
        end
        noise_typecode = ((noise_type == :stddev) ? SEP_NOISE_STDDEV :
                          (noise_type == :var) ? SEP_NOISE_VAR :
                          error("noise_type must be :stddev or :var"))
    end
            
    return sep_image(data_ptr, noise_ptr, mask_ptr,
                     dtype, ndtype, mdtype,
                     sz[1], sz[2],
                     noiseval, noise_typecode,
                     Cdouble(gain),
                     Cdouble(maskthresh))
end

# ---------------------------------------------------------------------------
# Background functions

"""
    Background


"""
mutable struct Background
    ptr::Ptr{Void}
    data_size::Tuple{Int, Int}
end

free!(bkg::Background) = ccall((:sep_bkg_free, libsep), Void, (Ptr{Void},),
                               bkg.ptr)

global_mean(bkg::Background) = ccall((:sep_bkg_global, libsep), Cfloat,
                                     (Ptr{Void},), bkg.ptr)

global_rms(bkg::Background) = ccall((:sep_bkg_globalrms, libsep), Cfloat,
                                    (Ptr{Void},), bkg.ptr)

function show(io::IO, bkg::Background)
    print(io, "Background $(bkg.data_size[1])×$(bkg.data_size[2])\n")
    print(io, " - global mean: $(global_mean(bkg))\n")
    print(io, " - global rms : $(global_rms(bkg))\n")
end

function background(data::Array{T, 2}; mask=nothing, boxsize=(64, 64),
                    filtersize=(3, 3), filterthresh=0.0, mask_thresh=0) where {T<:PixType}
    im = sep_image(data; mask=mask, mask_thresh=mask_thresh)
    result = Ref{Ptr{Void}}(C_NULL)
    status = ccall((:sep_background, libsep), Cint,
                   (Ptr{sep_image}, Cint, Cint, Cint, Cint, Cdouble,
                    Ref{Ptr{Void}}),
                   &im, boxsize[1], boxsize[2], filtersize[1], filtersize[2],
                   filterthresh, result)
    sep_assert_ok(status)
    bkg = Background(result[], sz)
    finalizer(bkg, free!)
    return bkg
end


function collect(::Type{T}, bkg::Background) where {T<:PixType}
    result = Array{T}(bkg.data_size)
    status = ccall((:sep_bkg_array, libsep), Cint, (Ptr{Void}, Ptr{Void}, Cint),
                   bkg.ptr, result, sep_typecode(T))
    sep_assert_ok(status)
    return result
end
# default collection type is Float32, because that's what's natively stored in
# background.
# TODO: make default the input array type in `background` instead?
collect(bkg::Background) = collect(Float32, bkg)


"""
    rms(bkg)
    rms(T, bkg)

Return an array of the standard deviation of the background. The result is
the size of the original image and is of type `T`, if given.
"""
function rms(::Type{T}, bkg::Background) where {T<:PixType}
    result = Array{T}(bkg.data_size)
    status = ccall((:sep_bkg_rmsarray, libsep), Cint,
                   (Ptr{Void}, Ptr{Void}, Cint),
                   bkg.ptr, result, sep_typecode(T))
    sep_assert_ok(status)
    return result
end
rms(bkg::Background) = rms(Float32, bkg)


# In-place background subtraction: A .-= bkg
function broadcast!(-, A::Array{T, 2},  ::Array{T, 2}, bkg::Background) where {T<:PixType}
    if size(A) != bkg.data_size
         throw(DimensionMismatch("dimensions must match"))
    end
    status = ccall((:sep_bkg_subarray, libsep), Cint,
                   (Ptr{Void}, Ptr{Void}, Cint),
                   bkg.ptr, A, sep_typecode(T))
    sep_assert_ok(status)
end

function (-)(A::Array{T, 2}, bkg::Background) where {T<:PixType}
    B = copy(A)
    B .-= bkg
    return B
end


# ---------------------------------------------------------------------------
# Source Extraction


# Mirror of C struct
struct sep_catalog
    nobj::Cint                 # number of objects (length of all arrays)
    thresh::Ptr{Cfloat}              # threshold (ADU)
    npix::Ptr{Cint}              # # pixels extracted (size of pix array)
    tnpix::Ptr{Cint}                # # pixels above thresh (unconvolved)
    xmin::Ptr{Cint}
    xmax::Ptr{Cint}
    ymin::Ptr{Cint}
    ymax::Ptr{Cint}
    x::Ptr{Cdouble}              # barycenter (first moments)
    y::Ptr{Cdouble}
    x2::Ptr{Cdouble}     # second moments
    y2::Ptr{Cdouble}
    xy::Ptr{Cdouble}
    errx2::Ptr{Cdouble}  # second moment errors
    erry2::Ptr{Cdouble}
    errxy::Ptr{Cdouble}
    a::Ptr{Cfloat}       # ellipse parameters
    b::Ptr{Cfloat}
    theta::Ptr{Cfloat}
    cxx::Ptr{Cfloat}        # alternative ellipse parameters
    cyy::Ptr{Cfloat}
    cxy::Ptr{Cfloat}
    cflux::Ptr{Cfloat}   # total flux of pixels (convolved)
    flux::Ptr{Cfloat}    # total flux of pixels (unconvolved)
    cpeak::Ptr{Cfloat}   # peak pixel flux (convolved)
    peak::Ptr{Cfloat}    # peak pixel flux (unconvolved)
    xcpeak::Ptr{Cint}    # x, y coords of peak (convolved) pixel
    ycpeak::Ptr{Cint}
    xpeak::Ptr{Cint}     # x, y coords of peak (unconvolved) pixel
    ypeak::Ptr{Cint}
    flag::Ptr{Cshort}    # extraction flags
    pix::Ptr{Ptr{Cint}}  # array giving indicies of object's pixels in
                         # image (linearly indexed). Length is `npix`.
                         # (pointer to within the `objectspix` buffer)
    objectspix::Ptr{Cint}  # buffer holding pixel indicies for all objects
end


function extract(data::Array{T, 2}, thresh::Real;
                 noise=nothing, mask=nothing, noise_type=:stddev,
                 thresh_type=:relative, minarea=5,
                 filter_kernel=Float32[1 2 1; 2 4 2; 1 2 1],
                 filter_type=:matched,
                 deblend_nthresh=32, deblend_cont=0.005,
                 clean=true, clean_param=1.0,
                 gain=0.0, mask_thresh=0) where {T}

    im = sep_image(data; noise=noise, mask=mask, noise_type=noise_type,
                   gain=gain, mask_thresh=mask_thresh)

    thresh_typecode = ((thresh_type == :relative)? SEP_THRESH_REL :
                       (thresh_type == :absolute)? SEP_THRESH_ABS :
                       error("thresh_type must be :relative or :absolute"))

    filter_typecode = ((filter_type == :matched)? SEP_FILTER_MATCHED :
                       (filter_type == :convolution)? SEP_FILTER_CONV :
                       error("filter_type must be :matched or :convolution"))

    # convert filter kernel to Cfloat array
    filter_kernel_cfloat = convert(Array{Cfloat, 2}, filter_kernel)
    filter_size = size(filter_kernel_cfloat)
    
    result = Ref{Ptr{sep_catalog}}(C_NULL)
    status = ccall((:sep_extract, libsep), Cint,
                   (Ptr{sep_image},
                    Cfloat, Cint, # thresh, thresh_type
                    Cint,  # minarea  
                    Ptr{Cfloat}, Cint, Cint,  # conv, convw, convh
                    Cint, # filter_type
                    Cint, Cdouble,  # deblend_nthresh, deblend_cont
                    Cint, Cdouble,  # clean_flag, clean_param
                    Ref{Ptr{sep_catalog}}),
                   &im,
                   thresh, thresh_typecode,
                   minarea,
                   filter_kernel_cfloat, filter_size[1], filter_size[2],
                   filter_typecode,
                   deblend_nthresh, deblend_cont,
                   clean_flag, clean_param,
                   result)
    sep_assert_ok(status)

    # parse result
    catalog_ptr = result[]
    # ...
    
    # free result allocated in sep_extract
    ccall((:sep_catalog_free, libsep), Void, (Ptr{sep_catalog},), catalog_ptr)

    # todo: return actual result
    nothing
end

end # module
