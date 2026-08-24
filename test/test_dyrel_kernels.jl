push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPU
end

const JR2K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax2D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax2D
else
    JustRelax.JustRelax2D
end

@parallel_indices (i, j) function _init_single_phase!(phases)
    @index phases[1, i, j] = 1.0
    return nothing
end

@testset "DYREL kernels" begin

    # ------------------------------------------------------------------ #
    # Pure, analytic helpers (no grid / rheology needed)
    # ------------------------------------------------------------------ #
    @testset "pure helpers" begin
        # _compute_RP!(P, P0, ∇V, Q, ηb, dt) = -∇V - (P - P0)/ηb + Q/dt
        P, P0, ∇V, Q, ηb, dt = 3.0, 1.0, 0.5, 0.2, 4.0, 0.25
        expected = -∇V - (P - P0) / ηb + Q / dt
        @test JustRelax2D._compute_RP!(P, P0, ∇V, Q, ηb, dt) ≈ expected

        # thermal variant: _compute_RP!(P, P0, ∇V, Q, ΔT, α, ηb, dt)
        ΔT, α = 10.0, 3.0e-5
        expected_T = -∇V - (P - P0) / ηb + α * (ΔT / dt) + Q / dt
        @test JustRelax2D._compute_RP!(P, P0, ∇V, Q, ΔT, α, ηb, dt) ≈ expected_T

        # damped_update_V(dVdτ, R, α, β, dτ) = (dVdτ_new, dVdτ_new*β*dτ)
        dVdτ, R, a, b, dτ = 2.0, 0.5, 0.9, 0.8, 0.3
        dVdτ_new, ΔV = JustRelax2D.damped_update_V(dVdτ, R, a, b, dτ)
        @test dVdτ_new ≈ a * dVdτ + R
        @test ΔV ≈ (a * dVdτ + R) * b * dτ

        # AR-DR overload: the momentum is the global ramp β_k and the step follows
        # the same stability relation with λmin removed, α = dτ²(1+β_k)/2.
        # Signalled by passing `nothing` in the per-cell step slot.
        β_k = 0.75
        dVdτ_new, ΔV = JustRelax2D.damped_update_V(dVdτ, R, β_k, nothing, dτ)
        @test dVdτ_new ≈ β_k * dVdτ + R
        @test ΔV ≈ (β_k * dVdτ + R) * dτ^2 * (1 + β_k) / 2

        # With dτ = 2·CFL/√λmax this is exactly the prototype's α = CFL²·2(1+β)/λmax
        CFL, λmax = 0.99, 7.0
        dτ_c = 2 * CFL / sqrt(λmax)
        _, ΔV_c = JustRelax2D.damped_update_V(dVdτ, R, β_k, nothing, dτ_c)
        @test ΔV_c ≈ (β_k * dVdτ + R) * CFL^2 * 2 * (1 + β_k) / λmax

        # β_k = 0 (first sweep, or straight after a restart) reproduces DYREL's
        # zero-momentum warm start, α = dτ²/2
        _, ΔV0 = JustRelax2D.damped_update_V(dVdτ, R, 0.0, nothing, dτ)
        @test ΔV0 ≈ R * dτ^2 / 2
    end

    # ------------------------------------------------------------------ #
    # AR-DR adaptive restart test (O'Donoghue & Candès 2015).
    #
    # The statistic is ⟨r, ΔV⟩ along the displacement actually applied,
    # ΔV = d·dτ²(1+β)/2, so the per-cell dτ² is carried (this solver's dτV is
    # per-cell, so it does not factor out of the sign).
    #
    # The test wants ⟨r_k, d_{k-1}⟩ < 0, but at the check point JustRelax holds
    # d_k = β·d_{k-1} + R_k, not d_{k-1}. Since r = D·R and β > 0,
    #     ⟨r_k, d_k⟩ - ⟨r_k, R_k⟩ = β·⟨r_k, d_{k-1}⟩
    # recovers it up to a positive factor, which the sign test does not care about.
    # ------------------------------------------------------------------ #
    @testset "ardr_restart_dot" begin
        # the reduction goes through sum_mpi, which needs a live communicator
        JustRelax.MPI.Initialized() || JustRelax.MPI.Init()

        nx, ny = 6, 5
        D = (
            PTArray(backend_JR)(rand(nx - 1, ny) .+ 0.5),
            PTArray(backend_JR)(rand(nx, ny - 1) .+ 0.5),
        )
        R = (
            PTArray(backend_JR)(randn(nx - 1, ny)),
            PTArray(backend_JR)(randn(nx, ny - 1)),
        )
        d_prev = (
            PTArray(backend_JR)(randn(nx - 1, ny)),
            PTArray(backend_JR)(randn(nx, ny - 1)),
        )

        # deliberately non-uniform dτ: a single global dτ would factor out of the
        # sign, a per-cell one does not, which is the whole reason it is carried
        dτV = (
            PTArray(backend_JR)(rand(nx - 1, ny) .+ 0.25),
            PTArray(backend_JR)(rand(nx, ny - 1) .+ 0.25),
        )

        β = 0.8
        dVdτ = ntuple(i -> β .* d_prev[i] .+ R[i], 2)
        fields = (; D = D, dVdτ = dVdτ, dτV = dτV)

        expected = β * sum(
            sum(Array(D[i]) .* Array(R[i]) .* Array(d_prev[i]) .* Array(dτV[i]) .^ 2)
                for i in 1:2
        )
        @test JR2K.ardr_restart_dot(fields, R, Val(2)) ≈ expected

        # β = 0 (fresh restart / first sweep): d_k = R_k exactly, so the test is
        # neutral and can never spuriously fire a second restart.
        @test JR2K.ardr_restart_dot((; D = D, dVdτ = R, dτV = dτV), R, Val(2)) == 0

        # Sign: momentum pointing against the residual must fire a restart.
        anti = ntuple(i -> .-abs.(R[i]) .- abs.(D[i]), 2)
        @test JR2K.ardr_restart_dot(
            (; D = D, dVdτ = anti, dτV = dτV), (abs.(R[1]), abs.(R[2])), Val(2)
        ) < 0

        # A per-cell dτ CAN in principle flip the sign an unweighted sum reports. Two
        # cells: contributions r·(d−r) are +3 and −1, so unweighted the sum is +2 (no
        # restart); but the negative cell takes a 10x longer pseudo-step and dominates
        # the displacement, 3·1² − 1·10² = −97 (restart fires).
        #
        # This is a constructed case, not one the solver reaches: λmaxV is a row-sum /
        # diagonal ratio, so contrast cancels out of it (it spans ~1.02x even at 1e6
        # viscosity contrast) and a 10x dτ spread does not arise. The weight is carried
        # because it is the correct statistic, not to fix an observed failure.
        m(v) = PTArray(backend_JR)(reshape(v, length(v), 1))
        D1 = (m([1.0, 1.0]), m([0.0]))
        R1 = (m([1.0, 1.0]), m([0.0]))
        d1 = (m([4.0, 0.0]), m([0.0]))     # d−R = +3 and −1
        dτ1 = (m([1.0, 10.0]), m([1.0]))

        unweighted = sum(Array(D1[1]) .* Array(R1[1]) .* (Array(d1[1]) .- Array(R1[1])))
        weighted = JR2K.ardr_restart_dot((; D = D1, dVdτ = d1, dτV = dτ1), R1, Val(2))
        @test unweighted ≈ 2.0        # unweighted: positive, no restart
        @test weighted ≈ -97.0        # dτ-weighted: negative, restart fires
    end

    # ------------------------------------------------------------------ #
    # Geometric divergence + deviatoric strain rate (2D), driven through
    # the public wrapper with an analytic pure-strain velocity field:
    #   Vx = a·x , Vy = b·y  ⇒  ∇V = a+b, εxx = a-(a+b)/3, εyy = b-(a+b)/3, εxy = 0
    # ------------------------------------------------------------------ #
    @testset "compute_∇V_strain_rate! 2D" begin
        nx, ny = 6, 5
        ni = nx, ny
        li = 1.0, 1.0
        grid = Geometry(ni, li; origin = (0.0, 0.0))
        (; xvi) = grid
        _di = grid._di

        stokes = StokesArrays(backend_JR, ni)
        a, b = 2.0, -0.7
        stokes.V.Vx .= PTArray(backend_JR)([a * x for x in xvi[1], _ in 1:(ny + 2)])
        stokes.V.Vy .= PTArray(backend_JR)([b * y for _ in 1:(nx + 2), y in xvi[2]])

        JR2K.compute_∇V_strain_rate!(stokes, _di, ni, Val(2))

        div = a + b
        @test all(Array(stokes.∇V) .≈ div)
        @test all(Array(stokes.ε.xx) .≈ a - div / 3)
        @test all(Array(stokes.ε.yy) .≈ b - div / 3)
        @test all(abs.(Array(stokes.ε.xy)) .< 1.0e-12)
    end

    # ------------------------------------------------------------------ #
    # Fused DYREL kernels (2D). A tiny single-phase, viscoelastic setup
    # drives the fused strain-rate+RP, stress+τII-viscosity (nonlinear
    # `linear_viscosity = false` branch), and the residual kernels.
    # ------------------------------------------------------------------ #
    @testset "fused kernels 2D" begin
        nx, ny = 6, 5
        ni = nx, ny
        li = 1.0, 1.0
        grid = Geometry(ni, li; origin = (0.0, 0.0))
        (; xvi) = grid
        di = grid.di
        _di = grid._di
        dt = 1.0

        el = ConstantElasticity(; G = 1.0, Kb = 5.0)
        rheology = (
            SetMaterialParams(;
                Phase = 1,
                Density = ConstantDensity(; ρ = 1.0),
                Gravity = ConstantGravity(; g = 1.0),
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el)),
                Elasticity = el,
            ),
        )

        phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
        @parallel (@idx ni) _init_single_phase!(phase_ratios.center)
        @parallel (@idx ni .+ 1) _init_single_phase!(phase_ratios.vertex)

        stokes = StokesArrays(backend_JR, ni)
        args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
        compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

        # analytic pure-strain velocity field ⇒ known divergence a+b
        a, b = 1.3, -0.4
        stokes.V.Vx .= PTArray(backend_JR)([a * x for x in xvi[1], _ in 1:(ny + 2)])
        stokes.V.Vy .= PTArray(backend_JR)([b * y for _ in 1:(nx + 2), y in xvi[2]])

        dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, di, dt; ϵ = 1.0e-6)

        # --- fused divergence + strain rate + pressure residual ---
        # P0 = P and Q = 0 ⇒ RP = -∇V = -(a+b), independent of ηb
        stokes.P0 .= stokes.P
        stokes.Q .= 0.0
        JR2K.compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, args)
        @test all(Array(stokes.R.RP) .≈ -(a + b))
        @test all(Array(stokes.ε.xx) .≈ a - (a + b) / 3)

        # --- fused stress + τII viscosity refresh (nonlinear branch) ---
        θc = copy(dyrel.P_num)
        η_before = copy(stokes.viscosity.η)
        JR2K.compute_stress_viscosity_DRYEL!(
            stokes, θc, dyrel.γ_eff, rheology, phase_ratios,
            1.0, dt, 1.0, args, (-Inf, Inf), false,
        )
        @test all(isfinite, Array(stokes.viscosity.η))
        @test all(>(0), Array(stokes.viscosity.η))
        @test all(isfinite, Array(stokes.viscosity.ηv))
        # θc assembles the small pressure correction γ_eff·RP + ΔPψ
        @test Array(θc) ≈ Array(dyrel.γ_eff) .* Array(stokes.R.RP) .+ Array(stokes.ΔPψ)

        # --- Powell-Hestenes velocity residual (no D division: safe) ---
        ρg = @zeros(ni...), @zeros(ni...)
        @parallel (@idx ni) JR2K.compute_PH_residual_V!(
            stokes.R.Rx, stokes.R.Ry, stokes.P, stokes.ΔPψ,
            stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, ρg...,
            _di.center, _di.vertex,
        )
        @test all(isfinite, Array(stokes.R.Rx))
        @test all(isfinite, Array(stokes.R.Ry))

        # --- fused DR residual + damped velocity update ---
        # D = 1, β = 0 ⇒ ΔV = 0 (velocity unchanged), residuals finite
        dyrel.Dx .= 1.0; dyrel.Dy .= 1.0
        dyrel.βVx .= 0.0; dyrel.βVy .= 0.0
        dyrel.αVx .= 0.0; dyrel.αVy .= 0.0
        dyrel.dτVx .= 1.0; dyrel.dτVy .= 1.0
        Vx_before = copy(stokes.V.Vx)
        Vy_before = copy(stokes.V.Vy)
        @parallel (@idx ni) JR2K.compute_DR_residual_update_V!(
            stokes.R.Rx, stokes.R.Ry,
            stokes.V.Vx, stokes.V.Vy,
            dyrel.dVxdτ, dyrel.dVydτ,
            stokes.P, θc,
            stokes.τ.xx, stokes.τ.yy, stokes.τ.xy,
            ρg...,
            dyrel.Dx, dyrel.Dy,
            dyrel.αVx, dyrel.αVy,
            dyrel.βVx, dyrel.βVy,
            dyrel.dτVx, dyrel.dτVy,
            _di.center, _di.vertex,
        )
        @test all(isfinite, Array(stokes.R.Rx))
        @test Array(stokes.V.Vx) == Array(Vx_before)
        @test Array(stokes.V.Vy) == Array(Vy_before)

        # --- same fused kernel, AR-DR coefficients ---
        # The per-cell momentum/step arrays are replaced by the global ramp β_k and
        # `nothing`, and the step is derived from dτV as dτ²(1+β_k)/2. Rather than
        # difference two O(1) velocity fields (the residual here is ~1e-15, so that
        # is pure roundoff), assert the scalar path is EQUIVALENT to the array path
        # fed exactly those coefficients — which is the whole claim.
        β_k = 0.6
        dτ = 0.4
        dyrel.dτVx .= dτ; dyrel.dτVy .= dτ
        dyrel.αVx .= β_k; dyrel.αVy .= β_k
        dyrel.βVx .= dτ * (1 + β_k) / 2      # so that βV·dτV = dτ²(1+β_k)/2
        dyrel.βVy .= dτ * (1 + β_k) / 2

        Vx0 = copy(stokes.V.Vx); Vy0 = copy(stokes.V.Vy)
        dVxdτ0 = PTArray(backend_JR)(fill(0.3, size(dyrel.dVxdτ)))
        dVydτ0 = PTArray(backend_JR)(fill(-0.2, size(dyrel.dVydτ)))

        dr_args = (
            stokes.R.Rx, stokes.R.Ry,
            stokes.V.Vx, stokes.V.Vy,
            dyrel.dVxdτ, dyrel.dVydτ,
            stokes.P, θc,
            stokes.τ.xx, stokes.τ.yy, stokes.τ.xy,
            ρg...,
            dyrel.Dx, dyrel.Dy,
        )
        tail = (dyrel.dτVx, dyrel.dτVy, _di.center, _di.vertex)

        # DYREL path, per-cell coefficient arrays
        copyto!(dyrel.dVxdτ, dVxdτ0); copyto!(dyrel.dVydτ, dVydτ0)
        @parallel (@idx ni) JR2K.compute_DR_residual_update_V!(
            dr_args..., dyrel.αVx, dyrel.αVy, dyrel.βVx, dyrel.βVy, tail...,
        )
        Vx_dyrel = Array(copy(stokes.V.Vx))
        Vy_dyrel = Array(copy(stokes.V.Vy))
        dVxdτ_dyrel = Array(copy(dyrel.dVxdτ))
        dVydτ_dyrel = Array(copy(dyrel.dVydτ))

        # AR-DR path, scalar momentum + `nothing` step
        copyto!(stokes.V.Vx, Vx0); copyto!(stokes.V.Vy, Vy0)
        copyto!(dyrel.dVxdτ, dVxdτ0); copyto!(dyrel.dVydτ, dVydτ0)
        @parallel (@idx ni) JR2K.compute_DR_residual_update_V!(
            dr_args..., β_k, β_k, nothing, nothing, tail...,
        )

        @test Array(dyrel.dVxdτ) ≈ dVxdτ_dyrel
        @test Array(dyrel.dVydτ) ≈ dVydτ_dyrel
        @test Array(stokes.V.Vx) ≈ Vx_dyrel
        @test Array(stokes.V.Vy) ≈ Vy_dyrel

        # and the momentum recursion itself is exact (no cancellation here)
        @test Array(dyrel.dVxdτ) ≈ β_k .* Array(dVxdτ0) .+ Array(stokes.R.Rx)
        @test Array(dyrel.dVydτ) ≈ β_k .* Array(dVydτ0) .+ Array(stokes.R.Ry)
    end
end
