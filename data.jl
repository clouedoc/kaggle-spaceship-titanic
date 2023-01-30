using DataFrames
using MLJ
using CategoricalArrays
using CSV

FillImputer = @load FillImputer pkg = MLJModels

"Parses Titanic input data. "
function parse_titanic_data(X::DataFrame)
  passenger_id_splits = DataFrame(
    [
    (passenger_id_a=first, passenger_id_b=last) for (first, last) in split.(X[!, :PassengerId], "_")
  ]
  )
  X[!, :passenger_id_a] = passenger_id_splits[!, :passenger_id_a]
  X[!, :passenger_id_b] = passenger_id_splits[!, :passenger_id_b]
  select!(X, Not([:Name]))

  coerce!(X,
    :PassengerId => Continuous,
    :HomePlanet => OrderedFactor,
    :CryoSleep => OrderedFactor,
    :Cabin => OrderedFactor,
    :Destination => OrderedFactor,
    :VIP => OrderedFactor,
    :RoomService => Continuous,
    :FoodCourt => Continuous,
    :ShoppingMall => Continuous,
    :Spa => Continuous,
    :VRDeck => Continuous,
    :passenger_id_a => Continuous,
    :passenger_id_b => OrderedFactor
  )
  X[!, :HomePlanet] = int(X[!, :HomePlanet])
  X[!, :CryoSleep] = int(X[!, :CryoSleep])
  X[!, :Cabin] = int(X[!, :Cabin])
  X[!, :Destination] = int(X[!, :Destination])
  X[!, :VIP] = int(X[!, :VIP])
  X[!, :passenger_id_b] = int(X[!, :passenger_id_b])

  # Replace missing data
  X = MLJ.transform(fit!(machine(FillImputer(), X)), X)
  return X
end

"Return Spaceship Titanic training data"
function titanic_training_data()::Tuple{CategoricalArray{Bool},DataFrame}
  train_data = CSV.read("./data/train.csv", DataFrame)
  y, X = unpack(train_data, ==(:Transported), rng=123)
  X = parse_titanic_data(X)
  y = coerce(y, OrderedFactor)
  return y, X
end

function titanic_testing_data()::DataFrame
  test_data = CSV.read("./data/test.csv", DataFrame)
  X = parse_titanic_data(test_data)
  return X
end