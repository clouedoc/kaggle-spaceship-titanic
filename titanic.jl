using MLJ
using CSV
using DataFrames


titanic = CSV.read("data/train.csv", DataFrame)

y, X = unpack(titanic, ==(:Transported), rng=123)

first(X, 3)

#X = X[:, [:Age]]

y = convert(Vector{Bool}, y)
y = ifelse.(y, "Yes", "No")
# convert to Categorical array
y = categorical(y)

models(matching(X, y))

## trying to fit using XGBoost

# Tree = @load XGBoostClassifier pkg = XGBoost verbosity = 0
Tree = @load DecisionTreeClassifier pkg = BetaML verbosity = 0

tree = Tree()

evaluate(tree, X, y, resampling=CV(shuffle=true), measure=[log_loss, accuracy])