using DataFrames
using MLJ
using CategoricalArrays
using CSV

FillImputer = @load FillImputer pkg = MLJModels

"Parses Titanic input data. "
function parse_titanic_data(X::DataFrame)
  ## ----- Split :PassengerId
  passenger_id_splits = DataFrame(
    [
    (passenger_id_a=first, passenger_id_b=last) for (first, last) in split.(X[!, :PassengerId], "_")
  ]
  )
  X[!, :passenger_id_a] = passenger_id_splits[!, :passenger_id_a]
  X[!, :passenger_id_b] = passenger_id_splits[!, :passenger_id_b]

  ## ----- Split :Cabin
  coerce!(X,
    :Cabin => OrderedFactor
  )
  cabin_filler = FillImputer(features=[:Cabin])
  cabin_machine = machine(cabin_filler, X) |> fit!
  X = MLJ.transform(cabin_machine, X)
  cabin_splits = DataFrame(
    [
    (cabin_a=first, cabin_b=middle, cabin_c=last) for (first, middle, last) in split.(convert.(String, X[!, :Cabin]), "/", limit=3)
  ]
  )
  X[!, :cabin_a] = cabin_splits[!, :cabin_a]
  X[!, :cabin_b] = parse.(Int, cabin_splits[!, :cabin_b])
  X[!, :cabin_c] = cabin_splits[!, :cabin_c]
  coerce!(X,
    :cabin_a => OrderedFactor,
    :cabin_b => Continuous,
    :cabin_c => OrderedFactor
  )

  ## Final coercion

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
    :passenger_id_b => OrderedFactor,
    :cabin_a => OrderedFactor,
    :cabin_b => OrderedFactor,
    :cabin_c => OrderedFactor,
  )
  X[!, :HomePlanet] = int(X[!, :HomePlanet])
  X[!, :CryoSleep] = int(X[!, :CryoSleep])
  X[!, :Cabin] = int(X[!, :Cabin])
  X[!, :Destination] = int(X[!, :Destination])
  X[!, :VIP] = int(X[!, :VIP])
  X[!, :passenger_id_b] = int(X[!, :passenger_id_b])
  X[!, :cabin_a] = int(X[!, :cabin_a])
  X[!, :cabin_c] = int(X[!, :cabin_c])

  select!(X, Not([:Name]))
  select!(X, Not([:PassengerId]))

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