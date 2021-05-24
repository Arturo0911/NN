package nn

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func InitProcess(x, wOutput, wHidden, bOutput, bHidden *mat.Dense) (output, hiddenActivation mat.Dense) {
	applySigmoid := func(_, _ int, v float64) float64 { // Getting the form func(int, int float64) float64{}
		return Sigmoid(v)
	}

	var hiddenInput mat.Dense // Init the Multiplying with the respective weights and values from the input
	hiddenInput.Mul(x, wHidden)

	hiddenInput.Apply(func(_, col int, v float64) float64 { // Adding the respective bias from the hiddenLayer
		return v + bHidden.At(0, col)
	}, &hiddenInput)

	hiddenActivation.Apply(applySigmoid, &hiddenInput) // Applying the sigmoid function

	var outputLayer mat.Dense                   // taking the activation layer generate the process multiplying by the weights but the outputLayer
	outputLayer.Mul(&hiddenActivation, wOutput) // adding the respective bias from the outputLayer
	outputLayer.Apply(func(_, col int, v float64) float64 {
		return v + bOutput.At(0, col)
	}, &outputLayer)

	output.Apply(applySigmoid, &outputLayer) // APplying the same activation Sigmoid function
	return output, hiddenActivation
}

//Back propagation side
func BackPropagationForm(y, wOutput *mat.Dense, output, hiddenLayerActivation mat.Dense) (mat.Dense, mat.Dense) {

	applySigmoidePrime := func(_, _ int, v float64) float64 { // Getting the respective SigmoidPrime for loss or errors
		return SigmoidePrime(v)
	}

	var (
		networkErr         mat.Dense // The error calculus
		slopeOutputLayer   mat.Dense // slopes
		slopeHiddenLayer   mat.Dense
		errorAtHiddenLayer mat.Dense // getting the respective error at the specific hiddenLayer
		dOutput            mat.Dense // derivatives
		dHiddenLayer       mat.Dense
	)

	networkErr.Sub(y, &output)
	slopeOutputLayer.Apply(applySigmoidePrime, &output)
	slopeHiddenLayer.Apply(applySigmoidePrime, &hiddenLayerActivation)

	dOutput.MulElem(&networkErr, &slopeOutputLayer)

	errorAtHiddenLayer.Mul(&dOutput, wOutput.T())

	dHiddenLayer.MulElem(&errorAtHiddenLayer, &slopeHiddenLayer)

	return dOutput, dHiddenLayer
}

// This function return the adjusted weights
func AdjustingWeightsAndBias(x *mat.Dense, hiddenActivation,
	dHiddenLayer, dOutput mat.Dense, lr float64) (mat.Dense, *mat.Dense, *mat.Dense, mat.Dense, error) {
	var (
		wAdjHidden mat.Dense
		wAdjOut    mat.Dense
	)
	wAdjOut.Mul(hiddenActivation.T(), &dOutput)
	wAdjOut.Scale(lr, &wAdjOut)
	//wAdjOut.Add(wOut, &wAdjOut)

	wAdjHidden.Mul(x.T(), &dHiddenLayer)
	wAdjHidden.Scale(lr, &wAdjHidden)
	//wAdjHidden.Add(wHidden, &wAdjHidden)

	bOutdAdj, err := SumAlongAxis(0, &dOutput)
	if err != nil {
		return *mat.NewDense(0, 0, nil), nil, nil, *mat.NewDense(0, 0, nil), err
	}

	bHiddenAdj, err := SumAlongAxis(0, &dHiddenLayer)
	if err != nil {
		return *mat.NewDense(0, 0, nil), nil, nil, *mat.NewDense(0, 0, nil), err
	}

	return wAdjHidden, bHiddenAdj, bOutdAdj, wAdjOut, nil
}

func CreateMatrixDense(h, i, o int) (wHidden, bHidden, wOutput, bOutput *mat.Dense) {

	// h -> number of hidden layers or nodes
	// i -> number of input layers or nodes
	// o -> number of output layers or nodes

	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHiddenRaw := make([]float64, h*i)
	bHiddenRaw := make([]float64, h)
	wOutputRaw := make([]float64, h*o)
	bOutputRaw := make([]float64, o)

	// make a loop to fill into these floats arrays
	for _, record := range [][]float64{wHiddenRaw, bHiddenRaw, wOutputRaw, bOutputRaw} {
		for i := range record {
			record[i] = randGen.Float64()
		}
	}
	wHidden = mat.NewDense(i, h, wHiddenRaw)
	bHidden = mat.NewDense(1, h, bHiddenRaw)
	wOutput = mat.NewDense(h, o, wOutputRaw)
	bOutput = mat.NewDense(1, o, bOutputRaw)

	return wHidden, bHidden, wOutput, bOutput
}
