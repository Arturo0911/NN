Neuronal Network
=======

<img src="https://miro.medium.com/max/2310/1*Z5FdBMXzl5PGHt8AVfslcw.gif" width=350><br>
[![Go Reference](https://pkg.go.dev/badge/github.com/sjwhitworth/golearn.svg)](https://pkg.go.dev/github.com/sjwhitworth/golearn)


My neural network was built and is being built, using third party packages like GoLearn

twitter: [@Arturo0911](https://twitter.com/DevTuron)

Modules used
=======

See [here](https://github.com/sjwhitworth/golearn) for GoLearn.

Getting Started
=======

Data are loaded in as Instances. You can then perform matrix like operations on them, and pass them to estimators.
GoLearn implements the scikit-learn interface of Fit/Predict, so you can easily swap out estimators for trial and error.
GoLearn also includes helper functions for data, like cross validation, and train and test splitting.

```go
package main

import (
	"fmt"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	// Load in a dataset, with headers. Header attributes will be stored.
	// Think of instances as a Data Frame structure in R or Pandas.
	// You can also create instances from scratch.
	rawData, err := base.ParseCSVToInstances("datasets/iris.csv", false)
	if err != nil {
		panic(err)
	}

	// Print a pleasant summary of your data.
	fmt.Println(rawData)

	//Initialises a new KNN classifier
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)

	//Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
	cls.Fit(trainData)

	//Calculates the Euclidean distance and returns the most popular label
	predictions, err := cls.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))
}
```

```
Iris-virginica	28	2	  56	0.9333	0.9333  0.9333
Iris-setosa	    29	0	  59	1.0000  1.0000	1.0000
Iris-versicolor	27	2	  57	0.9310	0.9310  0.9310
Overall accuracy: 0.9545
```

Examples
========

GoLearn comes with practical examples. Dive in and see what is going on.

```bash
cd $GOPATH/src/github.com/sjwhitworth/golearn/examples/knnclassifier
go run knnclassifier_iris.go
```
```bash
cd $GOPATH/src/github.com/sjwhitworth/golearn/examples/instances
go run instances.go
```
```bash
cd $GOPATH/src/github.com/sjwhitworth/golearn/examples/trees
go run trees.go
```
