package main

import (
	"log"

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
	neuron := nn.NewNeuron1(neuralConf)

	var x *mat.Dense
	var y *mat.Dense

	value := neuron.Train(x, y)
	if value != nil {
		log.Fatal(value)
	}
}
