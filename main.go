package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// @author: Arturo Negreiros (AKA Pxyl0xd)

/*func main() {

	// Initialize one neuron

	x := mat.NewDense(3, 4, []float64{
		1.0, 0.0, 1.0, 0.0,
		1.0, 0.0, 1.0, 1.0,
		0.0, 1.0, 0.0, 1.0,
	})

	y := mat.NewDense(3, 1, []float64{1.0, 1.0, 0.0})

	neuralConf := nn.NeuralNetConfig{
		InputNeurons:  4,
		HiddenNeurons: 3,
		OutputNeurons: 1,
		NumberEpochs:  5000,
		LearningRate:  0.3,
	}
	neuron := nn.NewNeuron1(neuralConf)
	if err := neuron.Train(x, y); err != nil {
		panic(err)
	}

	f := mat.Formatted(neuron.WHidden, mat.Prefix(" "))
	fmt.Println(f)

}*/

func matPresentation(value *mat.Dense) {
	element := mat.Formatted(value, mat.Prefix(" "))
	fmt.Println(element)
}

func main() {
	var output mat.Dense

	x := mat.NewDense(2, 2, []float64{65, 43, 37, 66})
	y := mat.NewDense(2, 2, []float64{35, 55, 98, 78})

	fmt.Println("Showing matrix x and y respectevly")
	matPresentation(x)
	matPresentation(y)

	fmt.Printf("\n\n")
	output.MulElem(x, y)
	matPresentation(&output)
	fmt.Printf("\n\n")
	output.Add(x, y)
	matPresentation(&output)
	fmt.Printf("\n\n")
	output.Scale(1.0, x.T())
	matPresentation(&output)

}
