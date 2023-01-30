using MLJ
using CSV
using DataFrames
using Plots

include("./data.jl")
y, X = titanic_training_data()

#### Load models
EvoTreeClassifier = @load EvoTreeClassifier pkg = EvoTrees verbosity = 0

#### Pipeline
prob_predictor = EvoTreeClassifier()
point_predictor = BinaryThresholdPredictor(prob_predictor, threshold=0.5)

#### Machine

mach = machine(point_predictor, X, y) |> fit!

#y = convert.(Bool, int.(y) .- 1)

#evaluate(point_predictor, X, y,
#  resampling=CV(shuffle=true),
#  measures=[log_loss, accuracy],
#  verbosity=0)


balanced = BalancedAccuracy(adjusted=true)
e = evaluate!(mach, resampling=CV(nfolds=6), measures=[balanced, accuracy])
e.measurement[1] # 0.584 ± 0.0273

####
r = range(point_predictor, :threshold, lower=0.1, upper=0.9)
tuned_point_predictor = TunedModel(
  point_predictor,
  tuning=RandomSearch(rng=123),
  resampling=CV(nfolds=6),
  range=r,
  measure=balanced,
  n=30,
)
mach2 = machine(tuned_point_predictor, X, y) |> fit!
optimized_point_predictor = report(mach2).best_model
optimized_point_predictor.threshold # 0.260
predict(mach2, X)[1:3] # [1, 1, 0]


e = evaluate!(mach2, resampling=CV(nfolds=6), measure=[balanced, accuracy])
e.measurement[1] # 0.576 ± 0.0263

final_machine = machine(optimized_point_predictor, X, y) |> fit!
evaluate!(final_machine, measure=[balanced, accuracy])

# -------- Train the model
#fit!(mach)
