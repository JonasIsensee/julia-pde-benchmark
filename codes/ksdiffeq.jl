using ApproxFun, OrdinaryDiffEq
using LinearAlgebra
using DiffEqOperators
using Plots


"""
ksintegrateDiffEq: integrate kuramoto-sivashinsky equation (Julia)
       u_t = -u*u_x - u_xx - u_xxxx, domain x in [0,Lx], periodic BCs
 inputs
          u = initial condition (vector of u(x) values on uniform gridpoints))
         Lx = domain length
         dt = time step
         Nt = number of integration timesteps
         nsave = save every n-th timestep
 outputs
          u = final state, vector of u(x, Nt*dt) at uniform x gridpoints
This an implementation using ApproxFun and OrdinaryDiffEq.
"""
function ksintegrateDiffEq(u, Lx, dt, Nt, nsave)
    n = length(u)                  # number of gridpoints

    S = Fourier(0..Lx)
    T = ApproxFun.plan_transform(S, n)
    Ti = ApproxFun.plan_itransform(S, n)

    #Linear Part
    D  = (Derivative(S) → S)[1:n,1:n]
    D2 = Derivative(S,2)[1:n,1:n]
    D4 = Derivative(S,4)[1:n,1:n]
    A = DiffEqArrayOperator(Diagonal(-D2-D4))

    #Nonlinear Part
    function ks_nonlin(du,u,p,t)
        D, T, Ti, tmp = p
        mul!(du,Ti,u)
        @. du = -1/2*du^2
        mul!(tmp, T, du)
        mul!(du, D, tmp)
    end

    params = (D, T, Ti, zeros(n))
    prob = SplitODEProblem(A,ks_nonlin, T*u, (0.0,Nt*dt), params)

    sol = solve(prob, ETDRK4(), dt=dt, saveat=0.0:dt*nsave:Nt*dt)
    return sol.t, map(u -> Ti*u, sol.u)
end

function make_demo_plot()
    Lx = 64*pi
    Nx = 1024
    dt = 1/16
    nsave = 8
    Nt = 3200

    x = Lx*(0:Nx-1)/Nx
    u = cos.(x) + 0.1*sin.(x/8) + 0.01*cos.((2*pi/Lx)*x);

    t,U = ksintegrateDiffEq(u, Lx, dt, Nt, nsave)
    Umat = hcat(U...)'

    heatmap(x,t,Umat, xlim=(x[1], x[end]), ylim=(t[1], t[end]), xlabel="x", ylabel="t",
        title="Kuramoto-Sivashinsky dynamics", fillcolor=:bluesreds)
end

make_demo_plot()
