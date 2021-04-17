package main

/**
* @author: Arturo Negreiros
* @description: Machine Learning Training with Golang
 */

import (
	"time"

	/*"github.com/sjwhitworth/golearn/base"

	"github.com/sjwhitworth/golearn/evaluation"

	"github.com/sjwhitworth/golearn/knn"*/

	"github.com/Arturo0911/NN/nn"
)

type BehaviorClouds struct {
	TimeStart          time.Time
	TimeEnd            time.Time
	CloudDescription   string
	RelativityHumidity float32
	Clouds             int
	Precipitaion       float32
	Temperature        float32
	Icon               string
	Code               string
}

func main() {

	nn.LoadFile("../practices/netflix_titles.csv")

	/*// Load in a dataset, with headers. Header attributes will be stored.
	// Think of instances as a Data Frame structure in R or Pandas.
	// You can also create instances from scratch.
	rawData, err := base.ParseCSVToInstances("../datasets/iris.csv", false)
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
	fmt.Println(evaluation.GetSummary(confusionMat))*/
}
