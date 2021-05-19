package main

import (
	"github.com/Arturo0911/NN/nn"
	"gonum.org/v1/gonum/mat"
)

// @author: Arturo Negreiros (AKA Pxyl0xd)

func main() {

	// Initialize one neuron
	neuralConf := nn.NeuralNetConfig{
		InputNeurons:  3,
		HiddenNeurons: 3,
		OutputNeurons: 1,
		NumberEpochs:  5000,
		LearningRate:  0.3,
	}

	//var x *mat.Dense
	//var y *mat.Dense

	x := mat.NewDense(3, 4, []float64{
		1.0, 0.0, 1.0, 0.0,
		1.0, 0.0, 1.0, 1.0,
		0.0, 1.0, 0.0, 1.0,
	})

	y := mat.NewDense(3, 1, []float64{1.0, 1.0, 0.0})
	neuron := nn.NewNeuron1(neuralConf)
	if err := neuron.Train(x, y); err != nil {
		panic(err)
	}

}
