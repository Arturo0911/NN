package math_process

import (
	"errors"
	"math"
)

func MakeStatisticsMethods(values []float64) (float64, float64, float64) {

	var sumTot float64 = 0
	var average float64 = 0
	var variance float64 = 0
	var standarDesviation float64 = 0
	for _, value := range values {

		sumTot += value
	}

	average = float64(sumTot) / float64(len(values))

	for _, element := range values {

		variance += math.Pow((element - average), 2)

	}

	variance = (variance / float64(len(values)-1))
	standarDesviation = math.Sqrt(variance)

	return variance, standarDesviation, average

}

func MakeCovariance(X []float64, Y []float64) (float64, float64, error) {

	var XAverage float64 = 0
	var YAverage float64 = 0
	var XSum float64 = 0
	var YSum float64 = 0
	var covariance float64 = 0
	var Sx float64 = 0
	var Sy float64 = 0
	var corrCoeff float64 = 0

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

	corrCoeff = covariance / (Sx * Sy)

	return covariance, corrCoeff, nil

}
