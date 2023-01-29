using DataFrames
using MLJ

"Parses Titanic input data. "
function parse_titanic_data(X::DataFrame)
  dropmissing!(X)
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
  return X
end