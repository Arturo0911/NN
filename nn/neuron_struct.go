package nn

import (
	"errors"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Contain all the whole information about
// The neural network trained
// Here reference to the final value, that is, all the model trained
// included in a matrix of type Dense
type NeuralNet struct {
	Config  NeuralNetConfig
	BOutput *mat.Dense
	WOutput *mat.Dense
	BHidden *mat.Dense
	WHidden *mat.Dense
}

// NeuralNetconfig define the neural Network structure
// Basically the architecture explained
// Set the struct of the Neural net and
// the neural congfiguration
// Input neurons the numbers of attributes
// that we are going to feed in the network
// Outout neurons is the number of nodes at the end,
// previously setting up to make the classification in
// one of three classes
// Hidden Neuron, the hidden layers with nodes inside
// The number of Epochs is reference at the number of iteration
// that the train should to do
type NeuralNetConfig struct {
	InputNeurons  int
	OutputNeurons int
	HiddenNeurons int
	NumberEpochs  int
	LearningRate  float64
}

// Activation function
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivative sigmoid function
func SigmoidePrime(x float64) float64 {
	return x * (1.0 - x)
}

// This functions is used to fix the bias at the output
func SumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	nRows, nCol := m.Dims()
	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, nCol)
		for i := 0; i < nCol; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, nCol, data)
	case 1:
		data := make([]float64, nRows)
		for j := 0; j < nRows; j++ {
			row := mat.Row(nil, j, m)
			data[j] = floats.Sum(row)
		}
		output = mat.NewDense(nRows, 1, data)
	default:
		return nil, errors.New("invalid axis, should to be between 0 or 1")

	}

	return output, nil
}

func Prediction() {}

// Another way to fix the data
// even is a best way that use the backtracking tradicional
func SthocasticGradientDescendt() {}

//Back propagation side
func BackPropagationForm(y, wOutput *mat.Dense, output, hiddenLayerActivation mat.Dense) (mat.Dense,
	mat.Dense, mat.Dense, mat.Dense, mat.Dense, mat.Dense, error) {

	applySigmoidePrime := func(_, _ int, v float64) float64 {
		return SigmoidePrime(v)
	}

	var (
		networkErr         mat.Dense
		slopeOutputLayer   mat.Dense
		slopeHiddenLayer   mat.Dense
		errorAtHiddenLayer mat.Dense
		dOutput            mat.Dense
		dHiddenLayer       mat.Dense
	)

	networkErr.Sub(y, &output)
	slopeOutputLayer.Apply(applySigmoidePrime, &output)
	slopeHiddenLayer.Apply(applySigmoidePrime, &hiddenLayerActivation)

	dOutput.MulElem(&networkErr, &slopeOutputLayer)

	errorAtHiddenLayer.Mul(&dOutput, wOutput.T())

	dHiddenLayer.MulElem(&errorAtHiddenLayer, &slopeHiddenLayer)

	return networkErr, slopeOutputLayer, slopeHiddenLayer,
		errorAtHiddenLayer, dOutput, dHiddenLayer, nil
}

// This function return the adjusted weights
func AdjustingWeights(x, wHidden, hiddenLayer, wOut *mat.Dense,
	dHiddenLayer, dOutput mat.Dense, lr float64) (*mat.Dense, *mat.Dense, error) {
	var (
		wAdjHidden mat.Dense
		wAdjOut    mat.Dense
	)
	wAdjOut.Mul(hiddenLayer.T(), &dOutput)
	wAdjOut.Scale(lr, &wAdjOut)
	//wAdjOut.Add(wOut, &wAdjOut)

	wAdjHidden.Mul(x.T(), &dHiddenLayer)
	wAdjHidden.Scale(lr, &wAdjHidden)
	//wAdjHidden.Add(wHidden, &wAdjHidden)

	return &wAdjHidden, &wAdjOut, nil
}
func AdjustingBias(dOutput, dHiddenLayer mat.Dense) (*mat.Dense, *mat.Dense, error) {

	bOutdAdj, err := SumAlongAxis(0, &dOutput)
	if err != nil {
		return nil, nil, err
	}

	bHiddenAdj, err := SumAlongAxis(0, &dHiddenLayer)
	if err != nil {
		return nil, nil, err
	}

	return bOutdAdj, bHiddenAdj, nil

}

// Generate float random numbers
// to initialize the weights
func GenerateRandomSeed() float64 {

	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)
	return randGen.Float64()
}
