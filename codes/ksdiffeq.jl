using ApproxFun, OrdinaryDiffEq
using LinearAlgebra
using DiffEqOperators
using PyPlot

function gen_prob(L,n)
    Lx = L
    Nx = n

    S = Fourier(0..L)
    D =  Derivative(S,1)[1:n,1:n]
    D2 = Derivative(S,2)[1:n,1:n]
    D4 = Derivative(S,4)[1:n,1:n]
    T = ApproxFun.plan_transform(S, n)
    Ti = ApproxFun.plan_itransform(S, n)

    A =  (D2 - D4)
    Ax = DiffEqArrayOperator(Diagonal(A))
    tmp1 = zeros(n)
    tmp2 = zeros(n)

    function ks_split(du,u,p,t)
        mul!(tmp1,D,u)
        mul!(tmp2,Ti,tmp1)
        @. tmp2 = -1/2 * tmp2^2
        mul!(tmp1, T, tmp2)
        mul!(du, D, tmp1)
    end

    ks_split_short(du, u, p, t) = du .= -D/2*(T*((Ti*(D*u)).^2))

    x = Lx*(0:Nx-1)/Nx
    u0 = T*(cos.(x) + 0.1*sin.(x/8) + 0.01*cos.((2*pi/Lx)*x));

    prob = SplitODEProblem(Ax,ks_split, u0, (0.0,10.0))
    return prob, Ti
end

prob, Ti = gen_prob(22, 64)

@time sol = solve(prob, ETDRK4(), dt=0.25)
@time sol = solve(prob, CNAB2(), dt=0.25) #Throws an error, possibly a bug

mat = map(u -> Ti*u, sol.u)
pcolormesh(mat); colorbar()
