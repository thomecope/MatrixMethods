# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Julia 1.5.2
#     language: julia
#     name: julia-1.5
# ---

# ## Photometric stereo
# EECS 551, F20
# 2020-09-27 Julia 1.5.2

# This notebook derives and implements an algorithm from computer vision called *photometric stereo* that reconstructs a 3D object from 2D images of it under different lighting conditions.

using IterativeSolvers: lsqr # add this package if you do not have it yet
using Plots
using MIRT: jim

# Previous homework solution files used below
# (these must be in your directory for "include" here to work).
#
# dir = ENV["hw551test"] # replace with "./" or your path
include("compute_normals.jl")
include("first_diffs_2d_matrix.jl")

# ## Images to normals

# Suppose we are in a dark room, and we have an object, a camera fixed above it,
# and a moveable light source.
# We model the object as a surface $f$ in $\mathbb{R}^3$,
# and, without loss of generality,
# we choose a coordinate system so that our camera,
# which takes $m \times n$ photos,
# captures the point $f(x,y)$ on the object's surface into pixel $I(x,y)$.
#
# Let's load some example images of this kind and visualize them.

# copy files from web to local directory, if not here already
files = ("cat_xy_data.float32", "cat_xy_L.float32", "cat_xy_mask.uint8")
url = "http://web.eecs.umich.edu/~fessler/course/551/data/cat/"
for file in files
    if !isfile(file)
        download(url * file, file)
    end
end

# load data from files
dim = (430, 550, 12) # data size for 12 light positions
data = Array{Float32}(undef, dim)
read!(files[1], data)
L = Array{Float32}(undef, (3, dim[3]))
read!(files[2], L)
M = Array{Bool}(undef, dim[1:2])
read!(files[3], M)
@show m, n, d = dim

# show mask and p ≤ d of the input images
p = 2
# rowify = (I3) -> reshape(permutedims(I3, [1 3 2]), :, size(I3,2))
jim(cat(M, data[:,:,1:p], dims=3), "mask and first $p input images")


# Intuitively, the pixel intensity $I(x,y)$ in the above images is determined by the how much light reflects off the surface $f(x,y)$. Indeed, assuming our object is diffuse (also called matte or Lambertian), one can derive the relationship
#
# $$
# I(x,y) = \alpha(x,y) (\ell^T n(x,y))
# $$
#
# where $\ell \in \mathbb{R}^3$ is a unit vector describing the orientation of the indicent light rays on the surface, $n(x,y) \in \mathbb{R}^3$ is the unit-norm surface normal vector of $f$ at $(x,y)$, and $\alpha(x,y) > 0$ is a scaling constant called the surface albedo.
#
# Suppose that we have $d$ images $I_1,\ldots,I_d$ of our object, each with a different lighting direction $\ell_1,\ldots,\ell_d$. We can stack the data into an overdetermined system of the form
#
# $$
# \underbrace{\begin{bmatrix} I_1(x,y) \\ \vdots \\ I_d(x,y) \end{bmatrix}}_{=:I_{xy}} = \underbrace{\begin{bmatrix} \ell_1 & \ldots & \ell_d \end{bmatrix}^T}_{=:L^T} \underbrace{\left(\alpha(x,y) n(x,y)\right)}_{=:\rho(x,y)}.
# $$
#
# We can solve this equation for $n(x,y)$ when $d = 3$, but, in practice,
# when there is noise and our assumptions do not hold exactly,
# a more robust approach is to take $d > 3$ images
# and approximate $n(x,y)$ by solving the least squares problem
#
# $$
# \rho(x,y) = \arg \min_{\rho \in \mathbb{R}^3} \|I_{xy} - L^T \rho\|_2^2
# $$
#
# and taking
#
# $$
# n(x,y) = \frac{\rho(x,y)}{\|\rho(x,y)\|_2}.
# $$
#
# Now run your `compute_normals` code to do that!

# Compute unit-normals from images
N = compute_normals(data, L)
@show size(N)

# display the 3 components of the normal vectors as images
jim(N, title = "3 components of unit-normals")


# ## Normals to gradients

# Now that we have the normal vectors of our surface,
# we reconstruct the surface itself!
#
# Recall from vector calculus that we can express the surface normal vector as
#
# $$
# n(x,y) = \frac{\begin{bmatrix} -\displaystyle\frac{\partial f(x,y)}{\partial x}, & -\displaystyle\frac{\partial f(x,y)}{\partial y}, & 1 \end{bmatrix}^T}{\sqrt{1 + \displaystyle\frac{\partial f(x,y)}{\partial x}^2 + \displaystyle\frac{\partial f(x,y)}{\partial y}^2}},
# $$
#
# where $~\partial f(x,y) / \partial x~$ and $~\partial f(x,y) / \partial y~$ are the partial derivatives of $f$ with respect to $x$ and $y$. From the above equation, we can compute the partial derivatives as follows
#
# $$
# \frac{\partial f(x,y)}{\partial x} = -\frac{n_1(x,y)}{n_3(x,y)}
# $$
#
# $$
# \frac{\partial f(x,y)}{\partial y} = -\frac{n_2(x,y)}{n_3(x,y)},
# $$
#
# where
#
# $$
# n(x,y) = \begin{bmatrix} n_1(x,y), & n_2(x,y), & n_3(x,y) \end{bmatrix}^T.
# $$
#
# Now compute approximate the $x$ and $y$ gradients of our surface
# using the above strategy.

# Compute gradients from normals
DFDX = -N[:, :, 1] ./ N[:, :, 3]
DFDY = -N[:, :, 2] ./ N[:, :, 3]
DFDX[.!isfinite.(DFDX) .| .!M] .= 0 # Clean data and apply mask
DFDY[.!isfinite.(DFDY) .| .!M] .= 0 # Clean data and apply mask
@show size(DFDX)
@show size(DFDY)

# display the gradients
jim(cat(DFDX, DFDY, dims=3), # feline pun intended?
    color=:redsblues,
    title = "DFDX and DFDY")


# ## Gradients to surface

# To estimate the surface from its gradients, use the finite differences approximations:
#
# $$
# \begin{array}{rcl}
# \displaystyle\frac{\partial f(x,y)}{\partial x} &\approx& \displaystyle\frac{f(x + 1,y) - f(x,y)}{(x + 1) - x} \\
# &=& f(x + 1,y) - f(x,y)
# \end{array}
# $$
#
# and
#
# $$
# \begin{array}{rcl}
# \displaystyle\frac{\partial f(x,y)}{\partial y} &\approx& \displaystyle\frac{f(x,y + 1) - f(x,y)}{(y + 1) - y} \\
# &=& f(x,y + 1) - f(x,y),
# \end{array}
# $$
#
# These relationships give us an overdetermined linear system of $2mn$ equations that we can solve to estimate our surface $f(x,y)$ given its partial derivatives on the grid
# $x = 1,\ldots,m$ and
# $y = 1,\ldots,n$.
#
# To simplify notation,
# define the $m \times n$ matrices `FXY`, `DFDX`, and `DFDY` such that
#
# $$
# \begin{array}{rcl}
# \texttt{FXY(i,j)} &=& f(i,j) \\
# \texttt{DFDX(i,j)} &=& \frac{\partial f(i,j)}{\partial x} \\
# \texttt{DFDY(i,j)} &=& \frac{\partial f(i,j)}{\partial y}
# \end{array}
# $$
#
# Then, if we let $A$ denote the 2D first differences matrix returned by your
# `first_diffs_2d_matrix`  function, then we have the relationship
#
# $$ \texttt{b} \approx A ~ \texttt{fxy} $$
#
# where, in Julia:
#
# ```julia
# b = [vec(DFDX); vec(DFDY)]
# fxy = vec(FXY)
# ```
#
# Now construct this system!

# +
# Generate least squares problem
A = first_diffs_2d_matrix(m, n)
b = [vec(DFDX); vec(DFDY)] # stacking up the differences into a vector for LS

@show size(A)
@show size(b);
# -

# Now that we have `A` and `b`, what remains is to solve the least squares problem
#
# $$\texttt{fxy} = \arg\min_{f \in \mathbb{R}^{mn}} \| b - A f \|^2$$
#
# for our (vectorized) surface, `fxy`.
#
# Because `A` is a *gigantic* (sparse) matrix with over **100 billion** entries,
# computing $A^+ b$ directly is impossible!
#
# Instead, we use the built-in iterative solver `lsqr` to do the job.

# +
# Solve least squares problem (patience needed)
if !@isdefined(fxy)
	fxy = lsqr(A, b, maxiter=1000)
end

@show size(fxy);
# -

# All that remains is to reshape back into a surface and view the results.

# +
# Reshape into surface
FXY = reshape(fxy, m, n)

# Format surface
FXY .-= minimum(FXY[M]) # Anchor to z axis
FXY .*= M               # Zero data outside of mask
@show size(FXY);
# -

# Plot surface
#=
If you uncomment the following line you will be able to make a somewhat
prettier surface than the default `gr()` backend to the `Plots` package,
but you will also have to install the `PyPlot` package to use it and it
may take longer to render.  So this is definitely not required.
=#
# pyplot() # optional to uncomment this
surface(FXY, title = "Reconstructed surface (lsqr) by $(ENV["USER"]) ")

# also display surface as an "image"
jim(FXY, color=:redsblues,
    title = "depth map f(x,y) by $(ENV["USER"])")
