package main

import (
	"github.com/Arturo0911/NN/model/plotting"
)

// @author: Arturo Negreiros (AKA Pxyl0xd)

func main() {

	// Initialize one neuron

	/*x := mat.NewDense(3, 4, []float64{
		1.0, 0.0, 1.0, 0.0,
		1.0, 0.0, 1.0, 1.0,
		0.0, 1.0, 0.0, 1.0,
	})

	y := mat.NewDense(3, 2, []float64{
		1.0, 1.0,
		0.0, 1.1,
		2.1, 1.1})

	neuralConf := nn.NeuralNetConfig{
		InputNeurons:  4,
		HiddenNeurons: 3,
		OutputNeurons: 2,
		NumberEpochs:  2000,
		LearningRate:  0.3,
	}
	neuron := nn.InitNeuron3(neuralConf)
	if err := neuron.InitTraining(x, y); err != nil {
		panic(err)
	}*/

	//if err := settings.MakingFiles("wineQualityReds.csv"); err != nil {
	//	log.Fatal(err)
	//}

	plotting.CreatePloting("wineQualityReds.csv")

}
