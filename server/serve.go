package server

import (
	"fmt"
	"net/http"

	"github.com/Arturo0911/NN/nn"
	"gonum.org/v1/gonum/mat"
)

type FormPrediction struct {
	Accuracy float64 `json:"accuracy"`
	Loss     float64 `json:"loss"`
}

func initNN() (*FormPrediction, error) {

	neuron := nn.NeuralNetConfig{
		InputNeurons:  3,
		OutputNeurons: 3,
		HiddenNeurons: 1,
		NumberEpochs:  3000,
		LearningRate:  0.3,
	}

	x := mat.NewDense(3, 4, []float64{
		1.0, 1.1, 1.0, 0.1,
		0.2, 0.6, 1.0, 1.1, 1.0,
	})

	y := mat.NewDense(3, 4, []float64{
		1.0, 1.1, 1.0, 0.1,
		0.2, 0.6, 1.0, 1.1, 1.0,
	})

	network := nn.NewNeuron2(neuron)
	if err := network.Train(x, y); err != nil {
		return nil, err
	}
	return &FormPrediction{}, nil

}

func InitServer() {
	fmt.Println("Initializing server")
	http.HandleFunc("/neuron", func(w http.ResponseWriter, r *http.Request) {
		initNN()
	})
}
