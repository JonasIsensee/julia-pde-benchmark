using ApproxFun, OrdinaryDiffEq
using LinearAlgebra
using DiffEqOperators
using PyPlot


function deriv(N)
    coeffs = [n%2==1 ? 0 : n÷2 for n=1:N-1]
    Tridiagonal(coeffs, zeros(Int, N), -coeffs)
end

function gen_prob(L,n)
    Lx = L
    Nx = n

    S = Fourier(0..L)
    D =  2π/L*deriv(n)#Derivative(S,1)[1:n,1:n]
    D2 = Derivative(S,2)[1:n,1:n]
    D4 = Derivative(S,4)[1:n,1:n]
    T = ApproxFun.plan_transform(S, n)
    Ti = ApproxFun.plan_itransform(S, n)

    A =  Diagonal(-D2 -D4)
    Ax = DiffEqArrayOperator(A)
    tmp1 = zeros(n)
    tmp2 = zeros(n)

    function ks_split(du,u,p,t)
        mul!(tmp1,Ti,u)
        @. tmp1 = tmp1^2
        mul!(tmp2, T, tmp1)
        mul!(du, D, tmp2)
        @. du = -1/2*du
    end


    x = Lx*(0:Nx-1)/Nx
    u0 = T*(cos.(x).*(1 .+sin.(x)))
    prob = SplitODEProblem(Ax,ks_split, u0, (0.0,400.0))
    return prob, Ti
end


prob, Ti = gen_prob(100, 256)

@time sol = solve(prob, ETDRK4(), dt=1/4)
@time sol = solve(prob, CNAB2(), dt=0.25) #Throws an error, possibly a bug

mat = map(u -> Ti*u, sol.u)
pcolormesh(mat); colorbar()
