package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/Arturo0911/NN/math_process"
)

/*
	age
	sex
	cp = Chest Pain type chest pain type
	trtbps=	resting blood pressure (in mm Hg)
	chol = cholestoral in mg/dl fetched via BMI sensor

	fbs = (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
	restecg = resting electrocardiographic results
	thalachh = maximum heart rate achieved

	exng = exercise induced angina (1 = yes; 0 = no)
	oldpeak = Previous peak
	slp
	caa
	thall
	output

*/

func readCSVFile(pathFile string) {

	file, err := os.Open(pathFile)

	if err != nil {
		panic(err)
	}

	reader := csv.NewReader(file)

	records, _ := reader.ReadAll()
	newArray := make([]float64, 0)

	chestPain := make([]float64, 0)
	cholesterol := make([]float64, 0)

	mens := 0
	womens := 0

	for i := 1; i < len(records); i++ {
		// get the ages to set context
		value, err := strconv.ParseFloat(records[i][0], 64)

		if err != nil {
			continue
		}
		newArray = append(newArray, value)

		gender, err := strconv.Atoi(records[i][1])
		cp, _ := strconv.ParseFloat(records[i][2], 64)
		cl, _ := strconv.ParseFloat(records[i][4], 64)

		chestPain = append(chestPain, cp)
		cholesterol = append(cholesterol, cl)

		if err != nil {
			continue
		}
		if gender == 1 {
			mens++
		} else {
			womens++
		}

	}

	statistics := math_process.MakeStatisticsMethods(newArray)
	/*var percent float64 = 0
	fmt.Println(statistics.Average)

	if mens >= womens {
		fmt.Println(mens, " mens are propensed to get a heart attack")
		percent = float64(mens * 100 / (mens + womens))
		fmt.Printf("%.2f percent\n", percent)
		//fmt.Printf("%.3f percent \n", float64((mens/(mens+womens))*100))
	} else {
		fmt.Println(womens, " womens are propensed to get a heart attack")
		percent = float64(womens * 100 / (mens + womens))
		fmt.Printf("%.2f percent\n", percent)
		//fmt.Printf("%.3f percent\n", float64((womens/(mens+womens))*100))
	}*/

	fmt.Println("==================================")

	statistics.MakeCovariance(chestPain, cholesterol)
	fmt.Println("==================================")

	fmt.Println(statistics.Variance)
	fmt.Println(statistics.BetaOne)
	fmt.Println(statistics.BetaZero)

	fmt.Println(statistics)
}

func main() {

	readCSVFile("../../practices/heart.csv")
}
