package nn

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Initialize the neuron with the parameters inside
func NewNeuron1(config NeuralNetConfig) *NeuralNet {
	return &NeuralNet{
		Config: config,
	}
}

//Activation function => Sigmoid
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidePrime(x float64) float64 {
	return x * (1.0 - x)
}

func SumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()
	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, should to be  1 or 0")
	}

	return output, nil
}

func (nn *NeuralNet) Train(x, y *mat.Dense) error {

	// Initialize a number of weights and biases
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHiddenRaw := make([]float64, nn.Config.InputNeurons*nn.Config.HiddenNeurons)
	wOutputRaw := make([]float64, nn.Config.HiddenNeurons*nn.Config.OutputNeurons)

	bHiddenRaw := make([]float64, nn.Config.HiddenNeurons)
	bOutputRaw := make([]float64, nn.Config.OutputNeurons)

	// Fill into every array random numbers
	for _, record := range [][]float64{wHiddenRaw, wOutputRaw, bHiddenRaw, bOutputRaw} {
		for index := range record {
			record[index] = randGen.Float64()
		}
	}

	// Now create a NewDense for every parameter and add the respective array previously fill it
	wHidden := mat.NewDense(nn.Config.InputNeurons, nn.Config.HiddenNeurons, wHiddenRaw)
	wOutput := mat.NewDense(nn.Config.OutputNeurons, nn.Config.HiddenNeurons, wOutputRaw)
	bHidden := mat.NewDense(1, nn.Config.HiddenNeurons, bHiddenRaw)
	bOutput := mat.NewDense(1, nn.Config.OutputNeurons, bOutputRaw)

	fmt.Println(wHidden)
	fmt.Println(bHidden)
	fmt.Println(wOutput)
	fmt.Println(bOutput)

	var output mat.Dense

	// Looping over the number of epochs
	for i := 0; i < nn.Config.NumberEpochs; i++ {

		var hiddenLayerInput mat.Dense

		fmt.Println(hiddenLayerInput)
		hiddenLayerInput.Mul(x, wHidden)

		addBHidden := func(_, col int, value float64) float64 {
			return (value + bHidden.At(0, col))
		}
		hiddenLayerInput.Apply(addBHidden, &hiddenLayerInput)

		var hiddenLayerActivation mat.Dense
		applySigmoid := func(_, _ int, value float64) float64 {
			return Sigmoid(value)
		}
		hiddenLayerActivation.Apply(applySigmoid, &hiddenLayerInput)

		var outputLayerInput mat.Dense
		outputLayerInput.Mul(&hiddenLayerActivation, wOutput)
		addBOut := func(_, col int, value float64) float64 {
			return value + bOutput.At(0, col)
		}
		outputLayerInput.Apply(addBOut, &outputLayerInput)
		output.Apply(applySigmoid, &outputLayerInput)

		// Backpropagation

		var networkErr mat.Dense
		networkErr.Sub(y, &output)

		var slopeOutputLayer mat.Dense
		applySigmoidePrime := func(_, _ int, value float64) float64 {
			return SigmoidePrime(value)
		}
		slopeOutputLayer.Apply(applySigmoidePrime, &output)

		var slopeHiddenLayer mat.Dense
		slopeHiddenLayer.Apply(applySigmoidePrime, &hiddenLayerActivation)

		var dOutput mat.Dense
		dOutput.MulElem(&networkErr, &slopeOutputLayer)

		var errorAtHiddenLayer mat.Dense
		errorAtHiddenLayer.Mul(&dOutput, wOutput.T())

		var dHiddenLayer mat.Dense
		dHiddenLayer.MulElem(&errorAtHiddenLayer, &slopeHiddenLayer)

		// ADjust parameters

		var wOutAdj mat.Dense
		wOutAdj.Mul(hiddenLayerActivation.T(), &dOutput)
		wOutAdj.Scale(nn.Config.LearningRate, &wOutAdj)
		wOutput.Add(wOutput, &wOutAdj)

		bOutAdj, err := SumAlongAxis(0, &dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.Config.LearningRate, bOutAdj)
		bOutAdj.Add(bOutput, bOutAdj)

		var wHiddenAdj mat.Dense
		wHiddenAdj.Mul(x.T(), &dHiddenLayer)
		wHiddenAdj.Scale(nn.Config.LearningRate, &wHiddenAdj)
		wHidden.Add(wHidden, &wHiddenAdj)

		bHiddenAdj, err := SumAlongAxis(0, &dHiddenLayer)
		if err != nil {
			return err
		}

		bHiddenAdj.Scale(nn.Config.LearningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)

	}

	nn.WHidden = wHidden
	nn.BHidden = bHidden
	nn.WOutput = wOutput
	nn.BOutput = bOutput

	return nil

}
