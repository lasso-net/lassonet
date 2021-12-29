URL="https://raw.githubusercontent.com/warbelo/Lockout/main/Synthetic_Data2/dataset_a"

for name in X.csv Y.csv xtest.csv xtrain.csv xvalid.csv ytest.csv ytrain.csv yvalid.csv; do
    curl $URL/$name > $name
done