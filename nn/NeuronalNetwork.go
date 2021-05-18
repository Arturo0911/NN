package nn

import "gonum.org/v1/gonum/mat"

// Contain all the whole information about
// The neural network trained
type NeuralNet struct {
	Config  NeuralNetConfig
	Bhidden *mat.Dense
	WHidden *mat.Dense
	BOut    *mat.Dense
	WOut    *mat.Dense
}

// NeuralNetconfig define the neural Network structure
// Basically the architecture explained
type NeuralNetConfig struct {
	InputNeurons  int
	HiddenNeurons int
	OutputNeurons int
	NumEpochs     int
	LearningRate  float64
}
