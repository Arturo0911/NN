package nn

import "fmt"

func InitNeuron(inNeurons int) {
	config := NeuralNetConfig{
		InputNeurons: inNeurons,
	}

	fmt.Println(config)
}
