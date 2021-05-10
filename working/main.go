package main

import (
	"bufio"
	"log"
	"os"

	"github.com/go-gota/gota/dataframe"
)

const PATH_FILE = "../datasets/iris_labeled.csv"

func readingDatasets() {
	file, err := os.Open(PATH_FILE)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// creating a reader for the dataframe
	irisDF := dataframe.ReadCSV(file)

	// Spliting the number of rows in portion of 4 and 1
	trainSet := (4 * irisDF.Nrow()) / 5
	trainSlice := make([]int, trainSet)
	testSet := irisDF.Nrow() / 5
	testSlice := make([]int, testSet)

	for i := 0; i < trainSet; i++ {
		trainSlice[i] = i
	}

	for j := 0; j < testSet; j++ {
		testSlice[j] = trainSet + j
	}

	// After that creating subsets
	trainSubset := irisDF.Subset(trainSlice)
	testSubset := irisDF.Subset(testSlice)

	// with maps, created the files
	setMap := map[int]dataframe.DataFrame{
		0: trainSubset,
		1: testSubset,
	}

	for idx, name := range []string{"train.csv", "test.csv"} {

		setFile, err := os.Create(name)
		if err != nil {
			log.Fatal(err)
		}

		w := bufio.NewWriter(setFile)

		if err := setMap[idx].WriteCSV(w); err != nil {
			log.Fatal(err)
		}
	}

}

func makingPredictions() {

}

func main() {
	readingDatasets()
	makingPredictions()
}
