package settings

import (
	"bufio"
	"os"

	"github.com/go-gota/gota/dataframe"
)

const directoryName = "sets"

func MakingFiles(pathFile string) error {
	// Making train, validation and test datasets files

	file, err := os.Open(pathFile)
	if err != nil {
		return err
	}
	defer file.Close()
	readerDF := dataframe.ReadCSV(file)

	trainNumber := (4 * readerDF.Nrow()) / 5
	testNumber := readerDF.Nrow() / 5

	if trainNumber+testNumber < readerDF.Nrow() {
		trainNumber++
	}

	trainSet := make([]int, trainNumber)
	testSet := make([]int, testNumber)
	for i := 0; i < trainNumber; i++ {
		trainSet[i] = i
	}
	for j := 0; j < testNumber; j++ {
		testSet[j] = j + trainNumber
	}

	trainDF := readerDF.Subset(trainSet)
	testDF := readerDF.Subset(testSet)

	setMap := map[int]dataframe.DataFrame{
		0: trainDF,
		1: testDF,
	}

	for idx, ColName := range []string{"train.csv", "test.csv"} {
		f, err := os.Create(directoryName + "/" + ColName)
		if err != nil {
			return err
		}
		defer f.Close()

		w := bufio.NewWriter(f)
		if err := setMap[idx].WriteCSV(w); err != nil {
			return err
		}
	}

	return nil
}
