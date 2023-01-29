using MLJ
using CSV
using DataFrames
using Plots

include("./parse.jl")

# -------- Load training data
train_data = CSV.read("./data/train.csv", DataFrame)
dropmissing!(train_data)
y, X = unpack(train_data, ==(:Transported), rng=123)
parse_titanic_data(X)
y = coerce(y, OrderedFactor)

schema(X)
describe(y)



# -------- Pick and evaluate a model
EvoTreeClassifier = @load EvoTreeClassifier pkg = EvoTrees
model = EvoTreeClassifier()
mach = machine(model, X, y)


evaluate(model, X, y,
  resampling=CV(shuffle=true),
  measures=[log_loss, accuracy],
  verbosity=0)

# -------- Train the model
fit!(mach)

# -------- Load test data


test_data = CSV.read("./data/test.csv", DataFrame)
dropmissing!(test_data)
parse_titanic_data(test_data)
predictions = predict(mach, test_data)

predictions

classes(predictions)
DataFrame(predictions)

d = decoder(predictions)
int(predictions)
d(predictions)

convert(Bool, predictions)
predictions

pdf(predictions, levels(predictions))

submission = DataFrame()
submission[!, :PassengerId] = test_data[!, :PassengerId]