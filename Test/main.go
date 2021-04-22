package main

/**
* @author: Arturo Negreiros
* @description: Machine Learning Training with Golang
 */

import (
	"fmt"
	"time"

	"github.com/sjwhitworth/golearn/base"

	"github.com/sjwhitworth/golearn/evaluation"

	"github.com/sjwhitworth/golearn/knn"
)

type FeaturesClouds struct {
	TimeStart          time.Time
	TimeEnd            time.Time
	CloudDescription   string
	RelativityHumidity float32
	Clouds             int
	Precipitation      float32
	Temperature        float32
	Icon               string
	Code               string
}

func LoadData(pathFile string) {

	rawData, err := base.ParseCSVToInstances(pathFile, false)

	if err != nil {
		panic(err)
	}

	// when the data is obtained.

	fmt.Println(rawData)

	classifier := knn.NewKnnClassifier("euclidean", "linear", 2)

	// making train and test data
	// to be ploted and analysed

	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)

	classifier.Fit(trainData)

	predictions, err := classifier.Predict(testData)

	if err != nil {
		panic(err)
	}

	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)

	if err != nil {
		panic(err)
	}

	fmt.Println(evaluation.GetSummary(confusionMat))

}

func main() {

	//nn.LoadFile("../practices/netflix_titles.csv")
	pathFile := "../datasets/iris.csv"

	LoadData(pathFile)

}
