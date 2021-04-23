package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/Arturo0911/NN/math_process"
	//"bufio"
)

func readCSVFile(pathFile string) {

	file, err := os.Open(pathFile)

	if err != nil {
		panic(err)
	}

	reader := csv.NewReader(file)

	records, _ := reader.ReadAll()
	newArray := make([]float64, 0)

	for i := 0; i < len(records); i++ {
		// get the ages to set context
		if i > 0 {

			value, err := strconv.ParseFloat(records[i][0], 64)

			if err != nil {
				break
			}
			newArray = append(newArray, value)
		}

	}

	statistics := math_process.MakeStatisticsMethods(newArray)

	fmt.Printf("variance %.2f standard desviation %.2f  and average %.2f\n", statistics.Variance, statistics.StandardDesviation, statistics.Average)

}

func main() {

	readCSVFile("../../practices/heart.csv")
}