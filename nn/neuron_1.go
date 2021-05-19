package nn

import (
	"fmt"
	"math"
	"math/rand"
	"time"

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

func (nn *NeuralNet) Train(x *mat.Dense, y *mat.Dense) error {

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

	//var output mat.Dense

	// Looping over the number of epochs
	for i := 0; i < nn.Config.NumberEpochs; i++ {

		var hiddenLayerInput mat.Dense
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

	}

	return nil

}
