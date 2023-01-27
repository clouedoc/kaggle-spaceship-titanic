using MLJ;

iris = load_iris()

schema(iris)



import DataFrames
iris = DataFrames.DataFrame(iris);

y, X = unpack(iris, ==(:target), rng=123);

first(X, 3) |> pretty
first(y, 3) |> print

models(matching(X, y))

doc("DecisionTreeClassifier", pkg="DecisionTree")
Tree = @load DecisionTreeClassifier pkg = DecisionTree verbosity = 0
tree = Tree()

evaluate(tree, X, y, resampling=CV(shuffle=true), measure=[log_loss, accuracy])