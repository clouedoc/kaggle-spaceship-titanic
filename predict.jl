using MLJ
using CSV
using DataFrames

include("data.jl")
include("machine.jl")

# -------- Load test data
test_data = titanic_testing_data()

# -------- Predict
predictions = predict(final_machine, test_data)

# -------- Convert into output data
predictions = convert.(Bool, predictions)
test_data = CSV.read("./data/test.csv", DataFrame)
submission = DataFrame(PassengerId=test_data[!, :PassengerId], Transported=predictions)

CSV.writecell(buf, pos, len, io, x::Bool, opts) = CSV.writecell(buf, pos, len, io, x ? "True" : "False", opts)
CSV.write("out.csv", submission)