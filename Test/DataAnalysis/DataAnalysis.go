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
	ageArray := make([]float64, 0)

	chestPain := make([]float64, 0)
	//hearRate := make([]float64, 0)

	mens := 0
	womens := 0

	for _, element := range records {

		values, _ := strconv.Atoi(element[0])

		if values <= 45 {
			gender, _ := strconv.Atoi(element[1])
			ageArray = append(ageArray, float64(values))
			cp, _ := strconv.ParseFloat(element[2], 64)
			chestPain = append(chestPain, cp)

			if gender == 1 {
				mens++
			} else {
				womens++
			}
		}
	}

	// Values from age under 45 years old and chest pain

	statistics := math_process.MakeStatisticsMethods(ageArray)
	statistics.MakeCovariance(ageArray, chestPain)
	statistics.PresentingStatisticModel()
	fmt.Println("\nPEOPLE INDICATORS")
	fmt.Println("\nPeople under 45 years old in heart attack", (womens + mens))
	fmt.Println("Mens percent affected:", ((mens * 100) / (mens + womens)))
	fmt.Println("Womens percent affected: ", ((womens * 100) / (womens + mens)))

	prediction := math_process.InitTest(float64(28), statistics.BetaOne, statistics.BetaZero)
	fmt.Println(prediction)

	// still for make predictions taking another parameters
}
func main() {

	readCSVFile("../../practices/heart.csv")
}
