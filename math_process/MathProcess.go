package math_process

import (
	"errors"
	"math"
)

// Body of parameters

//
//.d8888b. 888           888   d8b        888   d8b                8888888b.                                     888
//d88P  Y88b888           888   Y8P        888   Y8P                888   Y88b                                    888
//Y88b.     888           888              888                      888    888                                    888
// "Y888b.  888888 8888b. 888888888.d8888b 888888888 .d8888b.d8888b 888   d88P 8888b. 888d88888888b.d88b.  .d88b. 888888 .d88b. 888d888.d8888b
//    "Y88b.888       "88b888   88888K     888   888d88P"   88K     8888888P"     "88b888P"  888 "888 "88bd8P  Y8b888   d8P  Y8b888P"  88K
//      "888888   .d888888888   888"Y8888b.888   888888     "Y8888b.888       .d888888888    888  888  88888888888888   88888888888    "Y8888b.
// Y88b  d88PY88b. 888  888Y88b. 888     X88Y88b. 888Y88b.        X88888       888  888888    888  888  888Y8b.    Y88b. Y8b.    888         X88
// "Y8888P"  "Y888"Y888888 "Y888888 88888P' "Y888888 "Y8888P 88888P'888       "Y888888888    888  888  888 "Y8888  "Y888 "Y8888 888     88888P'

type StatisticsParameters struct {
	StandardDesviation float64
	Average            float64
	Variance           float64
}

func MakeStatisticsMethods(values []float64) *StatisticsParameters {

	/*statistics := StatisticsParameters{
		StandardDesviation: 0,
		Average:            0,
		Variance:           0,
		//ProcessErorr:       nil,
	}*/

	/*statistics := new(StatisticsParameters)
	statistics.StandardDesviation = 0
	statistics.Average = 0
	statistics.Variance = 0*/

	var sumTot float64 = 0
	var average float64 = 0
	var variance float64 = 0
	var standardDeviation float64 = 0
	for _, value := range values {

		sumTot += value
	}

	average = float64(sumTot) / float64(len(values))

	for _, element := range values {

		variance += math.Pow((element - average), 2)

	}

	variance = (variance / float64(len(values)-1))
	standardDeviation = math.Sqrt(variance)

	return &StatisticsParameters{
		Average:            average,
		Variance:           variance,
		StandardDesviation: standardDeviation,
	}
}

func MakeCovariance(X []float64, Y []float64) (float64, float64, error) {

	var XAverage float64 = 0
	var YAverage float64 = 0
	var XSum float64 = 0
	var YSum float64 = 0
	var covariance float64 = 0
	var Sx float64 = 0
	var Sy float64 = 0
	var correlationCoefficient float64 = 0

	if len(X) != len(Y) {
		return 0, 0, errors.New("the size of ranges cannot be different")
	}

	var XVolatileList []float64
	var YVolatileList []float64

	for _, value := range X {
		XSum += value
	}
	for _, value := range Y {
		YSum += value
	}

	XAverage = XSum / float64(len(X))
	YAverage = YSum / float64(len(Y))

	for _, element := range X {
		XVolatileList = append(XVolatileList, (element - XAverage))
		Sx += math.Pow((element - XAverage), 2)
	}

	for _, element := range Y {
		YVolatileList = append(YVolatileList, (element - YAverage))
		Sy += math.Pow((element - YAverage), 2)
	}

	Sx = math.Sqrt(Sx / float64(len(X)-1))
	Sy = math.Sqrt(Sy / float64(len(Y)-1))

	for i := 0; i < len(X); i++ {
		covariance += XVolatileList[i] * YVolatileList[i]
	}

	covariance = covariance / float64(len(X)-1)

	correlationCoefficient = covariance / (Sx * Sy)

	return covariance, correlationCoefficient, nil

}
