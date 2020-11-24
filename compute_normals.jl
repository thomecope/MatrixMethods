using LinearAlgebra

"""
`N = compute_normals(data, L)`
In:
− `data` `m × n × d` matrix whose `d` slices contain `m × n` images
of a common scene under different lighting conditions
− `L` `3 × d` matrix whose columns are the lighting direction vectors
for the images in data, with `d ≥ 3`

Out:
− `N` `m × n × 3` matrix containing the unit−norm surface normal vectors
for each pixel in the scene
"""
function compute_normals(data, L)
    # L \ data
    m, n, d = size(data)

    N = zeros(m, n, 3)


    for i in 1:m #rows
        for j in 1:n #columns
            N[i, j, :] = normalize(L' \ data[i,j,:])
        end
    end

    return N
end

# For each i:
#     There is one light source L[3, i]
#     There is one data image data[m, n, i]
#     -> at each point there is a  surface normal
#     -> so at a point (m,n)
#         there are d images showing response
#         there are d light inputs
#         so this will give us surface normal
#             light input [all the ds stacked] \ image response [all the responses at point m,n stacked]
