using Serialization

"""
Supertype for types encoding the family of perturbations allowed.
"""
abstract type PerturbationFamily end

struct UnrestrictedPerturbationFamily <: PerturbationFamily end
Base.show(io::IO, pp::UnrestrictedPerturbationFamily) = print(io, "unrestricted")

abstract type RestrictedPerturbationFamily <: PerturbationFamily end

"""
For blurring perturbations, we currently allow colors to "bleed" across color channels -
that is, the value of the output of channel 1 can depend on the input to all channels.
(This is something that is worth reconsidering if we are working on color input).
"""
struct BlurringPerturbationFamily <: RestrictedPerturbationFamily
    blur_kernel_size::NTuple{2}
end
Base.show(io::IO, pp::BlurringPerturbationFamily) =
    print(io, filter(x -> !isspace(x), "blur-$(pp.blur_kernel_size)"))

struct LInfNormBoundedPerturbationFamily <: RestrictedPerturbationFamily
    norm_bound::Real

    function LInfNormBoundedPerturbationFamily(norm_bound::Real)
        @assert(norm_bound > 0, "Norm bound $(norm_bound) should be positive")
        return new(norm_bound)
    end
end
Base.show(io::IO, pp::LInfNormBoundedPerturbationFamily) =
    print(io, "linf-norm-bounded-$(pp.norm_bound)")

struct BrightnessPerturbationFamily <: RestrictedPerturbationFamily
    bright_bound::Real

    function BrightnessPerturbationFamily(bright_bound::Real)
        @assert(bright_bound > 0, "Norm bound $(bright_bound) should be positive")
        return new(bright_bound)
    end
end
Base.show(io::IO, pp::BrightnessPerturbationFamily) =
    print(io, "brightness-bounded-$(pp.bright_bound)")

function get_model(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::PerturbationFamily,
    optimizer,
    tightening_options::Dict,
    tightening_algorithm::TighteningAlgorithm,
)::Dict{Symbol,Any}
    notice(
        MIPVerify.LOGGER,
        "Determining upper and lower bounds for the input to each non-linear unit.",
    )
    m = Model(optimizer_with_attributes(optimizer, tightening_options...))
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)

    d_common = Dict(
        :Model => m,
        :PerturbationFamily => pp,
        :TighteningApproach => string(tightening_algorithm),
    )

    return merge(d_common, get_perturbation_specific_keys(nn, input, pp, m))
end

function get_model(
    nn::NeuralNet,
    input_size,
    global_predicted,
    global_confidence,
    target_indexes,
    pp::PerturbationFamily,
    optimizer,
    tightening_options::Dict,
    tightening_algorithm::TighteningAlgorithm,
)::Dict{Symbol,Any}
    notice(
        MIPVerify.LOGGER,
        "Determining upper and lower bounds for the REGULAR input to each non-linear unit.",
    )
    m = Model(optimizer_with_attributes(optimizer, tightening_options...))
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)

    d_common = Dict(
        :Model => m,
        :PerturbationFamily => pp,
        :TighteningApproach => string(tightening_algorithm),
    )

    input_range = CartesianIndices(input_size)
    # input is the input without perturbation added
    input = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    d_common[:Input] = input

    v_output = input |> nn

    target_vars = v_output[Bool[i ∈ target_indexes for i in 1:length(v_output)]]
    @constraint(m, target_vars .<= v_output[global_predicted] - global_confidence)

    notice(
        MIPVerify.LOGGER,
        "Determining upper and lower bounds for the PERTUBATED input to each non-linear unit.",
    )

    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)

    return merge(d_common, get_perturbation_specific_keys(nn, input, pp, m))
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::UnrestrictedPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))

    # v_x0 is the input with the perturbation added
    v_x0 = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_x0 - input, :Output => v_output)
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::BlurringPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}

    input_size = size(input)
    num_channels = size(input)[4]
    filter_size = (pp.blur_kernel_size..., num_channels, num_channels)

    v_f = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), CartesianIndices(filter_size))
    @constraint(m, sum(v_f) == num_channels)
    v_x0 = map(_ -> @variable(m, lower_bound = 0, upper_bound = 1), CartesianIndices(input_size))
    @constraint(m, v_x0 .== input |> Conv2d(v_f))

    v_output = v_x0 |> nn

    return Dict(
        :PerturbedInput => v_x0,
        :Perturbation => v_x0 - input,
        :Output => v_output,
        :BlurKernel => v_f,
    )
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::LInfNormBoundedPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))
    # v_e is the perturbation added
    v_e = map(
        _ -> @variable(m, lower_bound = -pp.norm_bound, upper_bound = pp.norm_bound),
        input_range,
    )
    # v_x0 is the input with the perturbation added
    v_x0 = map(
        i -> @variable(
            m,
            lower_bound = max(0, input[i] - pp.norm_bound),
            upper_bound = min(1, input[i] + pp.norm_bound)
        ),
        input_range,
    )
    @constraint(m, v_x0 .== input + v_e)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_e, :Output => v_output)
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::Array{<:Real},
    pp::BrightnessPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))
    # v_e is the perturbation added
    v_tmp = @variable(m, lower_bound = -pp.bright_bound, upper_bound = pp.bright_bound)
    v_e = map(
        _ -> v_tmp,
        input_range,
    )
    # v_x0 is the input with the perturbation added
    v_x0 = map(
        i -> @variable(
            m,
            lower_bound = max(0, input[i] - pp.bright_bound),
            upper_bound = min(1, input[i] + pp.bright_bound)
        ),
        input_range,
    )
    @constraint(m, v_x0 .== input + v_tmp)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_e, :Output => v_output)
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::AbstractArray{<:JuMPLinearType},
    pp::LInfNormBoundedPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))
    # v_e is the perturbation added
    v_e = map(
        _ -> @variable(m, lower_bound = -pp.norm_bound, upper_bound = pp.norm_bound),
        input_range,
    )
    # v_x0 is the input with the perturbation added
    v_x0 = map(
        i -> @variable(
            m,
            lower_bound = 0,
            upper_bound = 1
        ),
        input_range,
    )
    @constraint(m, v_x0 .== input + v_e)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_e, :Output => v_output)
end

function get_perturbation_specific_keys(
    nn::NeuralNet,
    input::AbstractArray{<:JuMPLinearType},
    pp::BrightnessPerturbationFamily,
    m::Model,
)::Dict{Symbol,Any}

    input_range = CartesianIndices(size(input))
    # v_e is the perturbation added
    v_tmp = @variable(m, lower_bound = -pp.bright_bound, upper_bound = pp.bright_bound)
    v_e = [v_tmp for i in input_range]
    # v_x0 is the input with the perturbation added
    v_x0 = map(
        i -> @variable(
            m,
            lower_bound = 0,
            upper_bound = 1,
        ),
        input_range,
    )
    @constraint(m, v_x0 .== input + v_e)

    v_output = v_x0 |> nn

    return Dict(:PerturbedInput => v_x0, :Perturbation => v_e, :Output => v_output)
end

struct MIPVerifyExt
    tightening_algorithm::TighteningAlgorithm
end

