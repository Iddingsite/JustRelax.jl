## VISCO-ELASTIC STOKES SOLVER
"""
    solve_DYREL!(
        stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
        kwargs...,
    )

Solve the Stokes system with the self-tuned dynamic relaxation (DYREL) method.

# Arguments (in the following order)
- `stokes`: `JustRelax.StokesArrays` containing the simulation fields.
- `ρg`: buoyancy forces arrays.
- `dyrel`: DYREL-specific parameters and fields.
- `flow_bcs`: `AbstractFlowBoundaryConditions` defining velocity boundary conditions.
- `phase_ratios`: `JustPIC.PhaseRatios` for material phase tracking.
- `rheology`: Material properties and rheological laws.
- `args`: Tuple of additional arguments needed to update viscosity, stress, and buoyancy forces.
- `grid`: `Geometry` object carrying grid spacing and staggered-grid coordinates. A legacy
  2D spacing tuple or named tuple is also accepted and converted to a uniform `Geometry`.
- `dt`: Time step.
- `igg`: `IGG` object for global grid information (MPI).

# Keyword Arguments
- `viscosity_cutoff`: Limits for viscosity `(min, max)`. Default: `(-Inf, Inf)`.
- `viscosity_relaxation`: Relaxation factor for viscosity updates. Default: `1.0e-2`.
- `λ_relaxation_DR`: Relaxation factor for dynamic relaxation. Default: `1`.
- `λ_relaxation_PH`: Relaxation factor for Powell-Hestenes iterations. Default: `1`.
- `iterMax`: Maximum number of iterations for each dynamic-relaxation solve. Default: `50.0e3`.
- `total_iterMax`: Maximum number of total dynamic-relaxation iterations. Default: `50.0e3`.
- `nout`: Output frequency for residuals. Default: `100`.
- `rel_drop`: Relative residual drop tolerance. Default: `1.0e-2`.
- `verbose_PH`: Print Powell-Hestenes iteration info. Default: `true`.
- `verbose_DR`: Print Dynamic Relaxation iteration info. Default: `true`.
- `linear_viscosity`: Whether to use linear viscosity. Default: `false`.
- `algorithm`: Velocity accelerator, `:DYREL` (default) or `:ARDR`. See below.

# Algorithms

`:DYREL` is the self-tuned dynamic relaxation of Duretz et al. 2026: the damping
`c = 2·c_fact·√λmin` is rebuilt every `nout` iterations from a Rayleigh-quotient
estimate of `λmin`, giving the per-cell momentum `(2−c·dτ)/(2+c·dτ)` and step
`2dτ²/(2+c·dτ)`.

`:ARDR` (adaptive-restart dynamic relaxation) removes the `λmin` estimate entirely.
The momentum instead follows Nesterov's canonical ramp `β_k = (k−1)/(k+2)`
(see [`ardr_momentum`](@ref)), reset to zero whenever the O'Donoghue–Candès
alignment test `⟨r, ΔV⟩ < 0` fires at the `nout` cadence, i.e. whenever the step
just taken stopped being a descent direction (see [`ardr_restart_dot`](@ref)). The step keeps the same stability
relation with `λmin` removed, `dτ²(1+β_k)/2`. `c_fact` is unused on this path;
`CFL` and `nout` still apply, and the restart can only fire at the `nout` cadence,
so a smaller `nout` makes the method more reactive.

The returned NamedTuple always carries an `n_restarts` field (necessarily
`0` for `:DYREL`).
"""
function solve_DYREL!(stokes::JustRelax.StokesArrays, args...; kwargs)
    out = solve_DYREL!(backend(stokes), stokes, args...; kwargs)
    return out
end

# entry point for extensions
solve_DYREL!(::CPUBackendTrait, stokes, args...; kwargs) = _solve_DYREL!(stokes, args...; kwargs...)

function _solve_DYREL!(
        stokes::JustRelax.StokesArrays,
        ρg,
        dyrel,
        flow_bcs::AbstractFlowBoundaryConditions,
        phase_ratios::JustPIC.PhaseRatios,
        rheology,
        args,
        grid::Geometry{N},
        dt,
        igg::IGG;
        viscosity_cutoff = (-Inf, Inf),
        viscosity_relaxation = 1.0e-2,
        λ_relaxation_DR = 1,
        λ_relaxation_PH = 1,
        iterMax = 50.0e3,
        total_iterMax = 50.0e3,
        nout = 100,
        rel_drop = 1.0e-2,
        b_width = (4, 4, 0),
        verbose_PH = true,
        verbose_DR = true,
        linear_viscosity = false,
        algorithm = :DYREL,
        kwargs...,
    ) where {N}

    algorithm in (:DYREL, :ARDR) ||
        throw(ArgumentError("algorithm must be :DYREL or :ARDR, got :$algorithm"))
    use_ARDR = algorithm === :ARDR

    dim = Val(N)
    v_dofs = velocity_dofs(dim)
    p_dof = pressure_dof(dim)
    di = grid.di
    _di = grid._di
    di_center = di.center
    ni = size(stokes.P)

    residuals = @residuals(stokes.R)
    fields = dyrel_fields(dyrel, dim)

    # errors
    err = 1.0
    iter = 0

    # solver loop
    @copy stokes.P0 stokes.P
    residuals0 = fields.R0

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # reset plastic multiplier at the beginning of the time step
    stokes.λ .= 0.0
    stokes.λv .= 0.0

    # Iteration loop
    err_min = Inf
    err = 1.0
    errV0 = ntuple(_ -> 1.0, dim)
    errPt0 = 1.0
    errV00 = ntuple(_ -> 1.0, dim)
    iter = 0
    ϵ = dyrel.ϵ
    err = 2 * ϵ
    err_evo_tot = Float64[]
    err_evo_V = Float64[]
    err_evo_P = Float64[]
    err_evo_it = Float64[]
    itg = 0
    # AR-DR state. `k_ARDR` is the 1-based sweep index within the current restart cycle
    # and `β_ARDR` is the momentum it implies. Both are carried across Powell-Hestenes
    # iterations, like DYREL's own momentum
    k_ARDR = 1
    # match the array eltype so the scalar never promotes the kernel's arithmetic
    T_ARDR = eltype(fields.dVdτ[1])
    β_ARDR = zero(T_ARDR)
    n_restarts = 0
    # small pressure correction θc = P_num + ΔPψ = γ_eff·RP + ΔPψ, assembled by the stress kernel and
    # read (alongside the separately-differenced P) by the momentum kernel. Reuses the dyrel.P_num
    # scratch — P_num is no longer materialized separately.
    θc = dyrel.P_num

    # recompute all the DYREL variables
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    compute_ρg!(ρg[end], phase_ratios, rheology, args)
    DYREL!(dyrel, stokes, rheology, phase_ratios, grid.di, dt)

    # Powell-Hestenes iterations
    for itPH in 1:1000
        # update buoyancy forces
        update_ρg!(ρg, phase_ratios, rheology, args)

        # compute divergence, deviatoric strain rate and pressure residual in one pass
        # isone(itPH) &&
        compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, args, true)

        # compute deviatoric stress, refresh τII viscosity, and assemble θc = γ_eff·RP + ΔPψ in one pass
        compute_stress_viscosity_DRYEL!(stokes, θc, dyrel.γ_eff, rheology, phase_ratios, λ_relaxation_PH, dt, viscosity_relaxation, args, viscosity_cutoff, linear_viscosity)
        # update_halo!(stokes.λv)
        # update_halo!(stokes.τ.xx_v)
        # update_halo!(stokes.τ.yy_v)
        # update_halo!(stokes.τ.xy)

        # compute velocity residuals
        @parallel (@idx ni) compute_PH_residual_V!(
            residuals...,
            stokes.P,
            stokes.ΔPψ,
            @stress(stokes)...,
            ρg...,
            _di.center,
            _di.vertex,
        )

        # pressure residual stokes.R.RP already computed in compute_∇V_strain_rate_RP! above

        # Residual check
        errV = ntuple(d -> norm_mpi(residuals[d]) / √(v_dofs[d]), dim)
        errPt = norm_mpi(stokes.R.RP) / √(p_dof)
        if isone(itPH)
            errV0 = map(x -> x + eps(), errV)
            errPt0 = errPt + eps()
        end
        if itPH == 2
            errPt0 = errPt + eps()
        end
        errV_rel = ntuple(d -> min(errV[d] / errV0[d], errV[d]), dim)
        err = maximum((errV_rel..., min(errPt / errPt0, errPt)))

        if verbose_PH && igg.me == 0
            errV_msg = join(
                ntuple(d -> @sprintf("R%d=%1.3e %1.3e", d, errV[d], errV[d] / errV0[d]), dim),
                ", ",
            )
            @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e - norm[%s, Rp=%1.3e %1.3e] \n", itPH, iter, iter / ni[1], err, errV_msg, errPt, errPt / errPt0)
        end
        igg.me == 0 && isnan(err) && error("NaN detected in outer loop")
        igg.me == 0 && err > 1.0e10 && error("Kaboom! Error > 1e10 in outer loop")
        err < ϵ && break

        # Set tolerance of velocity solve proportional to residual
        if err > err_min * 1.05
            # rel_drop = max(rel_drop * 0.1, ϵ)
            rel_drop = max(rel_drop * 0.1, 1.0e-3)
        end
        if err_min > err
            err_min = err
        end

        ϵ_vel = err * rel_drop
        itPT = 0
        while (err > ϵ_vel && itPT ≤ iterMax)
            itPT += 1
            itg += 1
            iter += 1

            # Pseudo-old dudes (only needed by compute_λminV! on residual-check iterations;
            # AR-DR has no λmin estimate, so the copy is skipped there)
            !use_ARDR && iszero(iter % nout) && foreach(copyto!, residuals0, residuals)

            # compute divergence, deviatoric strain rate and pressure residual in one pass
            compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, args, true)

            # Deviatoric stress, τII viscosity refresh, and θc = γ_eff·RP + ΔPψ assembly in one pass
            compute_stress_viscosity_DRYEL!(stokes, θc, dyrel.γ_eff, rheology, phase_ratios, λ_relaxation_DR, dt, viscosity_relaxation, args, viscosity_cutoff, linear_viscosity)
            # update_halo!(stokes.λv)
            # batch the vertex-stress halos (+ vertex viscosity, refreshed above in the fused
            # kernel from pre-halo stress) into a single MPI exchange, so shared boundary vertices
            # stay consistent across ranks — matching the original stress→halo→viscosity ordering.
            if linear_viscosity
                update_halo!(stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy)
            else
                update_halo!(stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy, stokes.viscosity.ηv)
            end

            # Velocity residuals + damped pseudo-transient velocity update (fused; the small pressure
            # correction θc = γ_eff·RP + ΔPψ was assembled by the stress kernel above; P stays separate).
            # DYREL passes its per-cell (α, β) fields; AR-DR passes the scalar ramp momentum and
            # `nothing`, which makes the kernel derive the step from dτV as dτ²(1+β_k)/2.
            if use_ARDR
                dr_velocity_update!(
                    stokes, fields, residuals, θc, ρg, _di, ni,
                    ntuple(_ -> β_ARDR, dim), ntuple(_ -> nothing, dim),
                )
            else
                dr_velocity_update!(
                    stokes, fields, residuals, θc, ρg, _di, ni,
                    fields.αV, fields.βV,
                )
            end
            flow_bcs!(stokes, flow_bcs)
            update_halo!(@velocity(stokes)...)

            # Advance the ramp. β_0 = 0, so the first sweep of a solve (and the first after
            # a restart) is a stable zero-momentum step, matching DYREL's own warm start.
            if use_ARDR
                k_ARDR += 1
                β_ARDR = T_ARDR(ardr_momentum(k_ARDR))
            end

            # Residual check
            if iszero(iter % nout)

                errV = ntuple(d -> norm_mpi(fields.D[d] .* residuals[d]) / √(v_dofs[d]), dim)

                if iter == nout
                    errV_scale = maximum(errV) + eps()
                    errV00 = ntuple(_ -> errV_scale, dim)
                end

                errV_ratio = ntuple(d -> errV[d] / errV00[d], dim)
                err = maximum(errV_ratio)
                isnan(err) && igg.me == 0 && error("NaN detected in inner loop")

                push!(err_evo_tot, err)
                push!(err_evo_V, maximum(errV_ratio))
                push!(err_evo_P, errPt / errPt0)
                push!(err_evo_it, iter)

                # @printf("it = %d, iter = %d, ϵ_vel = %1.3e, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e] \n", itPT, iter, ϵ_vel, err, errVx, errVy)
                if verbose_DR && igg.me == 0
                    @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err)
                end
                if use_ARDR
                    # Adaptive restart (O'Donoghue & Candès 2015): discard the momentum
                    # once the step just taken stopped being a descent direction. Fused
                    # into the same cadence as DYREL's λmin refresh, and cheaper than it
                    # (one reduction per component instead of two).
                    if ardr_restart_dot(fields, residuals, dim) < 0
                        β_ARDR = zero(T_ARDR)
                        k_ARDR = 1
                        n_restarts += 1
                    end

                    # Optimal pseudo-time steps - can be replaced by AD
                    Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, grid.di, dt)

                    # Select dτ only: α and β are rebuilt per sweep from the ramp
                    update_dτV!(dyrel)
                else
                    λminV = compute_λminV!(fields, residuals, residuals0, ni, dim)
                    @parallel (@idx ni) update_cV!(fields.cV, 2 * √(λminV) * dyrel.c_fact)

                    # Optimal pseudo-time steps - can be replaced by AD
                    Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, grid.di, dt)

                    # Select dτ
                    update_dτV_α_β!(dyrel)
                end
            end
        end

        # update pressure
        compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, args, false)
        @. stokes.P += dyrel.γ_eff .* stokes.R.RP

        iter > total_iterMax && break
    end

    # absorb plastic pressure correction into P (mirrors APT: stokes.P .= θ = P + ΔPψ)
    @. stokes.P += stokes.ΔPψ

    # refresh the ∇V diagnostic from the converged velocity field (it is not stored inside the
    # DYREL/PH loop — see compute_∇V_strain_rate_RP!)
    @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di.vertex)

    # compute vorticity
    compute_vorticity!(stokes, _di, ni, dim)

    # Interpolate shear components to cell center arrays
    shear2center!(stokes.ε)
    shear2center!(stokes.ε_pl)
    shear2center!(stokes.Δε)

    # accumulate plastic strain tensor
    accumulate_tensor!(stokes.EII_pl, stokes.ε_pl, dt)
    accumulate_vol!(stokes.EVol_pl, stokes.ε_vol_pl, dt)

    @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))
    copy_stress_vertices!(stokes, dim)

    return (; err_evo_it, err_evo_V, err_evo_P, err_evo_tot, n_restarts)

end

# Fused velocity residual + damped update. `αV`/`βV` are either the DYREL per-cell
# coefficient fields or, on the AR-DR path, a tuple of the scalar ramp momentum and a
# tuple of `nothing`. Kept as a function so each call site is monomorphic.
function dr_velocity_update!(stokes, fields, residuals, θc, ρg, _di, ni, αV, βV)
    @parallel (@idx ni) compute_DR_residual_update_V!(
        residuals...,
        @velocity(stokes)...,
        fields.dVdτ...,
        stokes.P,
        θc,
        @stress(stokes)...,
        ρg...,
        fields.D...,
        αV...,
        βV...,
        fields.dτV...,
        _di.center,
        _di.vertex,
    )
    return nothing
end

"""
    ardr_restart_dot(fields, residuals, ::Val{N})

Adaptive-restart statistic for the AR-DR path (O'Donoghue & Candès 2015). Negative
means the step just taken stopped being a descent direction, so the caller resets the
momentum ramp.

The statistic is `⟨r, ΔV⟩`, the directional derivative along the displacement the
sweep actually applied: `ΔV = d·dτ²(1+β)/2`.

The test wants `⟨r_k, d_{k-1}⟩`, but at the check point the solver holds
`d_k = β·d_{k-1} + R_k`, not `d_{k-1}`. With `r = D·R` the unpreconditioned residual
(`residuals` store `R = M⁻¹r`), the identity

    ⟨r_k, d_k⟩ - ⟨r_k, R_k⟩ = β·⟨r_k, d_{k-1}⟩

recovers it up to the positive factor `β`, which the sign test does not care about. It
needs no stored copy of the previous direction, and when `β = 0` (first sweep, or
straight after a restart) `d_k = R_k` makes it identically zero, so a restart can never
immediately fire a second one.
"""
function ardr_restart_dot(fields, residuals, ::Val{N}) where {N}
    return sum(
        ntuple(Val(N)) do d
            @inline
            # `@.` fuses the whole expression into a single broadcast, so this
            # materializes one temporary rather than four.
            D, R, dV, dτ = fields.D[d], residuals[d], fields.dVdτ[d], fields.dτV[d]
            sum_mpi(@. D * R * (dV - R) * dτ * dτ)
        end
    )
end

function _solve_DYREL!(
        stokes::JustRelax.StokesArrays,
        ρg,
        dyrel,
        flow_bcs::AbstractFlowBoundaryConditions,
        phase_ratios::JustPIC.PhaseRatios,
        rheology,
        args,
        di::Union{NTuple{2, <:Real}, NamedTuple},
        dt,
        igg::IGG;
        kwargs...,
    )
    grid = JustRelax.legacy_uniform_grid(size(stokes.P), di)
    return _solve_DYREL!(stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg; kwargs...)
end

# Dimension-agnostic helpers for DYREL

@inline function dyrel_fields(dyrel::JustRelax.DYREL, ::Val{2})
    return (
        D = (dyrel.Dx, dyrel.Dy),
        λmaxV = (dyrel.λmaxVx, dyrel.λmaxVy),
        dVdτ = (dyrel.dVxdτ, dyrel.dVydτ),
        dτV = (dyrel.dτVx, dyrel.dτVy),
        dV = (dyrel.dVx, dyrel.dVy),
        βV = (dyrel.βVx, dyrel.βVy),
        cV = (dyrel.cVx, dyrel.cVy),
        αV = (dyrel.αVx, dyrel.αVy),
        R0 = (dyrel.Rx0, dyrel.Ry0),
    )
end

@inline function dyrel_fields(dyrel::JustRelax.DYREL, ::Val{3})
    return (
        D = (dyrel.Dx, dyrel.Dy, dyrel.Dz),
        λmaxV = (dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz),
        dVdτ = (dyrel.dVxdτ, dyrel.dVydτ, dyrel.dVzdτ),
        dτV = (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz),
        dV = (dyrel.dVx, dyrel.dVy, dyrel.dVz),
        βV = (dyrel.βVx, dyrel.βVy, dyrel.βVz),
        cV = (dyrel.cVx, dyrel.cVy, dyrel.cVz),
        αV = (dyrel.αVx, dyrel.αVy, dyrel.αVz),
        R0 = (dyrel.Rx0, dyrel.Ry0, dyrel.Rz0),
    )
end

@inline dyrel_fields(::JustRelax.DYREL, ::Val{N}) where {N} = error("Unsupported dimension $N")

@inline global_grid_size(::Val{2}) = nx_g(), ny_g()
@inline global_grid_size(::Val{3}) = nx_g(), ny_g(), nz_g()
@inline global_grid_size(::Val{N}) where {N} = error("Unsupported dimension $N")

@inline pressure_dof(N) = prod(global_grid_size(N))

function velocity_dofs(::Val{N}) where {N}
    global_size = global_grid_size(Val(N))
    return ntuple(Val(N)) do d
        @inline
        prod(i -> i == d ? global_size[i] - 2 : global_size[i] - 1, 1:N)
    end
end

function compute_λminV!(fields, residuals, residuals0, ni, ::Val{N}) where {N}
    @parallel (@idx ni) compute_dV!(fields.dV, fields.dVdτ, fields.βV, fields.dτV)

    numerator = sum(ntuple(d -> sum_mpi(fields.dV[d] .* (residuals[d] .- residuals0[d])), Val(N)))
    denominator = sum(ntuple(d -> sum_mpi(fields.dV[d] .^ 2), Val(N)))
    return abs(numerator) / denominator
end

function copy_stress_vertices!(stokes::JustRelax.StokesArrays, ::Val{2})
    stokes.τ_o.xx_v .= stokes.τ.xx_v
    return stokes.τ_o.yy_v .= stokes.τ.yy_v
end

function copy_stress_vertices!(stokes::JustRelax.StokesArrays, ::Val{3})
    stokes.τ_o.xx_v .= stokes.τ.xx_v
    stokes.τ_o.yy_v .= stokes.τ.yy_v
    return stokes.τ_o.zz_v .= stokes.τ.zz_v
end
