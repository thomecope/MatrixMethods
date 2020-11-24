using LinearAlgebra
"""
haveCommonRoot = common_root(p1, p2 ; atol)
Determine whether the polynomials described by input coefficient vectors
p1 and p2 share a common root, to within an absolute tolerance parameter atol .
Assume leading coefficients p1[1] and p2[1] are nonzero.
In:
− p1 isavectoroflength m+1 with p1[1]!=0
that defines an m th degree polynomial of the form:
P1(x) = p1[1] x^m + p1[2] x^(m − 1) + ... + p1[m] x + p1[m + 1]
− p2 isavectoroflength n+1 with p2[1]!=0
that defines an n th degree polynomial of the form:
P2(x) = p2[1] x^n + p2[2] x^(n − 1) + ... + p2[n] x + p2[n + 1]
Option:
− atol absolute tolerance for calling isapprox
Out:
− haveCommonRoot = true when P1 and P2 share a common root, else false
"""
function common_root(p1::AbstractVector, p2::AbstractVector ; atol::Real=1e-6)

    # compan = c -> [-transpose(reverse(c)); [I zeros(length(c)-1)]]
    # A = compan(p1)
    # B = compan(p2)
    a = -transpose(p1[2:end]./p1[1])
    b = -transpose(p2[2:end]./p2[1])
    A = [a; [I zeros(length(a)-1)]]
    B = [b; [I zeros(length(b)-1)]]
    return isapprox(det(kron(I(length(b)), A)+kron(-B, I(length(a)))), 0 , atol=atol)

end
