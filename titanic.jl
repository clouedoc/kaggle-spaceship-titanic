using MLJ
using CSV
using DataFrames
using Plots

# -------- data loading
titanic = CSV.read("data/train.csv", DataFrame)
# TODO: fill missing data
dropmissing!(titanic)

# -------- check data repartition
pie(
  ["Transported", "Alive"],
  [
    count(f -> convert(Bool, f) === true, titanic[!, :Transported]),
    count(f -> convert(Bool, f) === false, titanic[!, :Transported])
  ]
)
histogram(titanic[!, :Transported])

# good news: it's balanced!!!

# let's analyze the data to try and find correlations

describe(titanic)

#   1 │ PassengerId              0001_01              9280_02                  0  String7

# It looks like we can split the passenger id

passenger_id_splits = DataFrame(
  [
  (passenger_id_a=first, passenger_id_b=last) for (first, last) in split.(titanic[!, :PassengerId], "_")
]
)

titanic[!, :passenger_id_a] = passenger_id_splits[!, :passenger_id_a]
titanic[!, :passenger_id_b] = passenger_id_splits[!, :passenger_id_b]

titanic[!, :passenger_id_a] = categorical(titanic[!, :passenger_id_a])
titanic[!, :passenger_id_b] = categorical(titanic[!, :passenger_id_b])

titanic[!, :Transported] = categorical(titanic[!, :Transported])
describe(titanic)

schema(titanic)

#   2 │ HomePlanet               Earth                Mars                   201  Union{Missing, String7}

titanic[!, :Name] = coalesce.(titanic[!, :Name], "Unknown Unknown")
#   3 │ CryoSleep     0.358306   false        0.0     true                   217  Union{Missing, Bool}


#   4 │ Cabin                    A/0/P                T/3/P                  199  Union{Missing, String15}
#   5 │ Destination              55 Cancri e          TRAPPIST-1e            182  Union{Missing, String15}
#   6 │ Age           28.8279    0.0          27.0    79.0                   179  Union{Missing, Float64}
#   7 │ VIP           0.0234393  false        0.0     true                   203  Union{Missing, Bool}
#   8 │ RoomService   224.688    0.0          0.0     14327.0                181  Union{Missing, Float64}
#   9 │ FoodCourt     458.077    0.0          0.0     29813.0                183  Union{Missing, Float64}
#  10 │ ShoppingMall  173.729    0.0          0.0     23492.0                208  Union{Missing, Float64}
#  11 │ Spa           311.139    0.0          0.0     22408.0                183  Union{Missing, Float64}
#  12 │ VRDeck        304.855    0.0          0.0     24133.0                188  Union{Missing, Float64}
#  13 │ Name                     Aard Curle           Zubeneb Pasharne       200  Union{Missing, String31}
#  14 │ Transported   0.503624   false        1.0     true                     0  Bool


# -------- parsing and engineering

y, X = unpack(titanic, ==(:Transported), rng=123)

describe(X)

X = coerce(X,
  :PassengerId => Continuous,
  :HomePlanet => Multiclass,
  # HomePlanet
  :CryoSleep => Multiclass,
  # TODO: split this
  :Cabin => Multiclass,
  :Destination => Multiclass,
  # Age
  :VIP => Multiclass,
  :RoomService => Continuous,
  :FoodCourt => Continuous,
  :ShoppingMall => Continuous,
  :Spa => Continuous,
  :VRDeck => Continuous,
  # TODO: split this
  :Name => Multiclass,
  :passenger_id_a => Continuous,
  :passenger_id_b => Multiclass
)

describe(X)
schema(X)
# -------- evaluation