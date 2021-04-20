package math_process

import "math"

func MakeStatisticsMethods(values []float64) (float64, float64) {

	var sumTot float64 = 0
	var average float64 = 0
	var variance float64 = 0
	for _, value := range values {

		sumTot += value
	}

	average = float64(sumTot) / float64(len(values))

	for _, element := range values {

		variance += math.Pow((element - average), 2)

	}

}
