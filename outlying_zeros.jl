using LinearAlgebra

"""
zmax, zmin = outlying_zeros(p, v0, nIters)
Use power iteration to compute the largest and smallest magnitude zeros,
respectively, of the polynomial defined by the input coefficients
In:
− p vector of length n + 1 defining the polynomial P(x)=p[1]x^n+p[2]x^n−1+...+p[n]x+p[n+1] with p[1]!=0
Option:
− v0 vector of length n with initial guess of an eigenvector; default randn
− nIters number of power iterations to perform; default 100
Out:
− zmax zero of P(x) with largest magnitude
− zmin zero of P(x) with smallest magnitude """
function outlying_zeros(p ; v0::AbstractVector{<:Real}=randn(length(p)-1), nIters::Int=100)

    if length(p) == 2
            zmax = -p[2]/p[1]
            return zmax, zmax
        end

    a = -transpose(p[2:end]./p[1])
    pr = reverse(p)
    b = -transpose(pr[2:end]./pr[1])
    A = [a; [I zeros(length(a)-1)]]
    Ar = [b; [I zeros(length(b)-1)]]

    #For zmax
    x = A*v0/norm(A*v0)
    for i in 1:nIters
        xnew = A*x/norm(A*x)
        x = xnew
    end
    zmax = (x'*A)*x

    #For zmin
    if p[end] == 0
        zmin = 0
    else
        x = Ar*v0/norm(Ar*v0)
        for i in 1:nIters
            xnew = Ar*x/norm(Ar*x)
            x = xnew
        end
        zmin = 1/((x'*Ar)*x)
    end

    return zmax, zmin
end
