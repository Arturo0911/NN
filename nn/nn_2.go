package nn

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func NewNeuron2(config NeuralNetConfig) *NeuralNet {
	return &NeuralNet{
		Config: config,
	}
}

func (nn *NeuralNet) TrainNN2(x *mat.Dense, y *mat.Dense) error {

	// Initialize the weighs and bias
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	// Initialize the data raw
	wHiddenRaw := make([]float64, nn.Config.InputNeurons+nn.Config.HiddenNeurons)
	wOutputRaw := make([]float64, nn.Config.HiddenNeurons*nn.Config.OutputNeurons)

	bHiddenRaw := make([]float64, nn.Config.HiddenNeurons)
	bOutputRaw := make([]float64, nn.Config.OutputNeurons)

	for _, record := range [][]float64{wHiddenRaw, wOutputRaw, bHiddenRaw, bOutputRaw} {
		for i := range record {
			record[i] = randGen.Float64()
		}
	}

	wHidden := mat.NewDense(nn.Config.InputNeurons, nn.Config.HiddenNeurons, wHiddenRaw)
	wOutput := mat.NewDense(nn.Config.HiddenNeurons, nn.Config.OutputNeurons, wOutputRaw)
	bHidden := mat.NewDense(1, nn.Config.HiddenNeurons, bHiddenRaw)
	bOutput := mat.NewDense(1, nn.Config.OutputNeurons, bOutputRaw)

	// fill, process, backtrack and adjust the data
	var output mat.Dense
	for i := 0; i < nn.Config.NumberEpochs; i++ {

		// This variable is for multiplies the matrix
		// with the respective values of Matrix x
		var wHiddenLayerInput mat.Dense
		wHiddenLayerInput.Mul(x, wHidden)

		// Add the respective bias
		addBHidden := func(_, col int, v float64) float64 {
			return v + bHidden.At(0, col)
		}
		wHiddenLayerInput.Apply(addBHidden, &wHiddenLayerInput)

		// Generate the function to generate the activation function
		var hiddenLayerActivation mat.Dense
		applySigmoid := func(_, _ int, v float64) float64 {
			return Sigmoid(v)
		}
		hiddenLayerActivation.Apply(applySigmoid, &wHiddenLayerInput)

		var outputLayerInput mat.Dense
		outputLayerInput.Mul(&hiddenLayerActivation, wOutput)
		addBOutput := func(_, col int, v float64) float64 {
			return v + bOutput.At(0, col)
		}
		outputLayerInput.Apply(addBOutput, &outputLayerInput)
		output.Apply(applySigmoid, &outputLayerInput)
		// BACK-PROPAGATION

		var errorNetwork mat.Dense
		errorNetwork.Sub(y, &output)

		// Generate the slope at the output layer
		var slopeOutLayer mat.Dense
		applySigmoidPrime := func(_, _ int, v float64) float64 {
			return SigmoidePrime(v)
		}
		slopeOutLayer.Apply(applySigmoidPrime, &output)

		var slopeHiddLayer mat.Dense
		slopeHiddLayer.Apply(applySigmoidPrime, &hiddenLayerActivation)

		var dOutput mat.Dense
		dOutput.MulElem(&errorNetwork, &slopeOutLayer)

		var errorAtHidden mat.Dense
		errorAtHidden.Mul(&dOutput, wOutput.T())

		var dHidden mat.Dense
		dHidden.MulElem(&errorAtHidden, &slopeHiddLayer)

	}

	nn.BHidden = bHidden
	nn.WHidden = wHidden
	nn.BOutput = bOutput
	nn.WOutput = wOutput

	return nil
}
