package main

import (
	"fmt"

	"github.com/Arturo0911/NN/nn"
)

func main() {

	// Initialize the neuron
	neuralConf := nn.NeuralNetConfig{
		InputNeurons:  3,
		HiddenNeurons: 3,
		OutputNeurons: 1,
		NumEpochs:     5000,
		LearningRate:  0.3,
	}
	neuron := nn.NewNeuron(neuralConf)
	fmt.Println(neuron)

}
