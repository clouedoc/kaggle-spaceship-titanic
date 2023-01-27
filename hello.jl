# imports
import Pkg;
Pkg.add("CSV");
Pkg.add("DataFrames")
Pkg.add("Plots")
Pkg.add("MLJ")


# usages

using CSV;
using DataFrames;
using Plots;

train = CSV.read("data/train.csv", DataFrame)

transported = train[train[!, :Transported].==1, :];
not_transported = train[train[!, :Transported].==0, :];


plot!(train[!, :Age], train[!, :Transported], xlabel="Age", ylabel="Transported")

histogram(transported[!, :Age], legend=false, xlabel="Age", ylabel="Count")
histogram(not_transported[!, :Age], legend=false, xlabel="Age", ylabel="Count")


# same thing but with hue with Transported

