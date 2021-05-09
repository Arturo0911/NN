package math_process

import (
	"errors"
	"fmt"
	"log"
	"math"
	"strconv"
)

//
//	.d8888b. 888           888   d8b        888   d8b
//	d88P  Y88b888           888   Y8P        888   Y8P
//	Y88b.     888           888              888
//	 "Y888b.  888888 8888b. 888888888.d8888b 888888888 .d8888b.d8888b
//	    "Y88b.888       "88b888   88888K     888   888d88P"   88K
//	      "888888   .d888888888   888"Y8888b.888   888888     "Y8888b.
//	 Y88b  d88PY88b. 888  888Y88b. 888     X88Y88b. 888Y88b.        X8
//	 "Y8888P"  "Y888"Y888888 "Y888888 88888P' "Y888888 "Y8888P 88888P'

//	 8888888b.                                             888
//	 888   Y88b                                            888
//	 888    888                                            888
//	 888   d88P 8888b. 888d888 8888b. 88888b.d88b.  .d88b. 888888 .d88b. 888d888.d8888b
//	 8888888P"     "88b888P"      "88b888 "888 "88bd8P  Y8b888   d8P  Y8b888P"  88K
//	 888       .d888888888    .d888888888  888  88888888888888   88888888888    "Y8888b.
//	 888       888  888888    888  888888  888  888Y8b.    Y88b. Y8b.    888         X88
//	 888       "Y888888888    "Y888888888  888  888 "Y8888  "Y888 "Y8888 888     88888P'

type MathModel interface {
}

type StatisticsParameters struct {
	StandardDesviation     float64 // Standard desviation general
	CorrelationCoefficient float64 // Metric of how much it's correlationship betweem two variables
	Average                float64
	Variance               float64 // Variance General
	Covariance             float64 // covariance between values
	Sx                     float64 // standard desviations of X
	Sy                     float64 // Standard desviations of Y
	BetaOne                float64 // The first parameter of math model
	BetaZero               float64 // The second parameter
	Bias                   float64 // desviation

}

type NeuronalNetwork struct {
	Prediction float64
	XDataTrain []float64
	YDataTrain []float64
	XDataTest  []float64
	YDataTest  []float64
}

func MakeStatisticsMethods(values []float64) *StatisticsParameters {
	// this function is for init the struct with the parameters
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

	_average, err := strconv.ParseFloat(fmt.Sprintf("%.3f", average), 64)
	if err != nil {

		log.Fatal(err)
	}
	_variance, err := strconv.ParseFloat(fmt.Sprintf("%.3f", variance), 64)
	if err != nil {

		log.Fatal(err)
	}
	_standard, err := strconv.ParseFloat(fmt.Sprintf("%.3f", standardDeviation), 64)
	if err != nil {

		log.Fatal(err)
	}

	return &StatisticsParameters{
		Average:            _average,
		Variance:           _variance,
		StandardDesviation: _standard,
	}
}

func (s *StatisticsParameters) MakeCovariance(X []float64, Y []float64) error {

	// the third parameter to return is if the covariance is greater than
	// 0, that's mean exists straight correlations between variables
	// es decir a mayor valor en X mayor valor en Y

	var XAverage float64 = 0
	var YAverage float64 = 0
	var XSum float64 = 0
	var YSum float64 = 0
	var covariance float64 = 0
	var Sx float64 = 0
	var Sy float64 = 0

	var betaOne float64 = 0
	//var variance float64 = 0

	if len(X) != len(Y) {
		return errors.New("the size of ranges cannot be different")
	}

	for _, value := range X {
		XSum += value
	}
	for _, value := range Y {
		YSum += value
	}

	XAverage = XSum / float64(len(X))
	YAverage = YSum / float64(len(Y))

	for i := 0; i < len(X); i++ {

		Sx += math.Pow((X[i] - XAverage), 2)
		Sy += math.Pow((Y[i] - YAverage), 2)
		covariance += ((X[i] - XAverage) * (Y[i] - YAverage))
		betaOne += ((X[i] - XAverage) * (Y[i] - YAverage))
	}
	s.BetaOne = betaOne / Sx
	s.BetaZero = YAverage - (betaOne/Sx)*XAverage

	// in both cases, make sqrt root in the Variance under their sizes
	Sx = math.Sqrt((Sx / float64(len(X)-1)))

	Sy = math.Sqrt((Sy / float64(len(Y)-1)))
	s.Covariance = covariance / float64(len(X)-1)
	s.CorrelationCoefficient = s.Covariance / (Sx * Sy)

	s.Sx = Sx
	s.Sy = Sy

	return nil

}

//	'########:'########::'######::'########::
//	... ##..:: ##.....::'##... ##:... ##..:::
//	::: ##:::: ##::::::: ##:::..::::: ##:::::
//	::: ##:::: ######:::. ######::::: ##:::::
//	::: ##:::: ##...:::::..... ##:::: ##:::::
//	::: ##:::: ##:::::::'##::: ##:::: ##:::::
//	::: ##:::: ########:. ######::::: ##:::::
//	:::..:::::........:::......::::::..:::::.
//	'##::::'##::::'###::::'########:'##::::'##:
// : ###::'###:::'## ##:::... ##..:: ##:::: ##:
// : ####'####::'##:. ##::::: ##:::: ##:::: ##:
// : ## ### ##:'##:::. ##:::: ##:::: #########:
// : ##. #: ##: #########:::: ##:::: ##.... ##:
// : ##:.:: ##: ##.... ##:::: ##:::: ##:::: ##:
// : ##:::: ##: ##:::: ##:::: ##:::: ##:::: ##:
// : :::::..::..:::::..:::::..:::::..:::::..::.
// '##::::'##::'#######::'########::'########:'##:::::::
//  ###::'###:'##.... ##: ##.... ##: ##.....:: ##:::::::
//  ####'####: ##:::: ##: ##:::: ##: ##::::::: ##:::::::
//  ## ### ##: ##:::: ##: ##:::: ##: ######::: ##:::::::
//  ##. #: ##: ##:::: ##: ##:::: ##: ##...:::: ##:::::::
//  ##:.:: ##: ##:::: ##: ##:::: ##: ##::::::: ##:::::::
//  ##:::: ##:. #######:: ########:: ########: ########:
// .:::::..:::.......:::........:::........::........::

func InitTest(parameterToTest []float64, betaOne float64, betaZero float64) []float64 {

	//----------------------------------------------//
	//               Y = β0 + β1*x                  //
	//----------------------------------------------//

	preidctions := make([]float64, 0)

	for _, element := range parameterToTest {
		preidctions = append(preidctions, (betaZero + betaOne*element))
	}

	return preidctions
}

func CompareDataTrained(dataTrained []float64, dataTest []float64) (float64, error) {

	var mathErrorMetric float64

	if len(dataTest) != len(dataTrained) {
		return 0, errors.New("THe data should be the same size")
	}

	for i := 0; i < len(dataTest); i++ {
		mathErrorMetric += ((dataTrained[i] * 100) / (dataTest[i]))
	}

	return (mathErrorMetric / float64(len(dataTrained))), nil

}

func (s *StatisticsParameters) PresentingStatisticModel() {

	fmt.Println("\n	Average: ", s.Average)
	fmt.Println("	Beta one: ", s.BetaOne)
	fmt.Println("	Beta Zero: ", s.BetaZero)
	fmt.Println("	Bias: ", s.Bias)
	fmt.Println("	Correlation coefficient: ", s.CorrelationCoefficient)
	fmt.Println("	Covariance: ", s.Covariance)
	fmt.Println("	Standard desviation: ", s.StandardDesviation)
	fmt.Println("	X variance: ", s.Sx)
	fmt.Println("	Y variance: ", s.Sy)
	fmt.Println("	General Variance: ", s.Variance)

}
