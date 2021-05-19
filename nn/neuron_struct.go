package nn

import "gonum.org/v1/gonum/mat"

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
// Set the struct of the Neural net and the neural congfiguration
// Input neurons the numbers of attributes that we are going to feed in the network
// Outout neurons is the number of nodes at the end, previously setting up to make the classification in one of three classes
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
