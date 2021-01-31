# Optimization algorithms implemented in Julia
Julia Code to represent and visualize all models developed during lectures of Prof. Frangioni 

Since we use Contour Plots, it is mandatory to install Plots and a backend plot library, 
which exhaustive list of Backend is on https://docs.juliaplots.org/latest/backends/#backends.

# Installation 
I have decided to use PyPlot as Backend for plotting, so I use the following commands on a Julia shell:
julia> using Pkg
julia> Pkg.add("Plots")
julia> Pkg.add("PyPlot")

#Execution steps
After installing the required packages, I have executed the file plot.jl (contain all subroutine for plotting)
with the command "julia -i plot.jl", also open a Julia shell and then include the file is accepted.
The important thing is that if i run the command julia without the -i option was not able Julia to represent the plot. 

# Models implemented
- Gradient Method
- Contour Plot
- Line Search 
-
