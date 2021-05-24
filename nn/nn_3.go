package nn

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func InitNeuron3(config NeuralNetConfig) *NeuralNet {
	return &NeuralNet{
		Config: config,
	}
}

func printingMatDense(a *mat.Dense) {
	fmt.Println(mat.Formatted(a, mat.Prefix("")))
}

func (nn *NeuralNet) InitTraining(x, y *mat.Dense) error {

	wHidden, bHidden, wOutput, bOutput := CreateMatrixDense(nn.Config.HiddenNeurons, nn.Config.InputNeurons, nn.Config.OutputNeurons)
	printingMatDense(wHidden)
	printingMatDense(x)
	var response mat.Dense
	response.Mul(wHidden, x)
	printingMatDense(&response)
	/*for i := 0; i < nn.Config.NumberEpochs; i++ {

		output, hiddenLayerActivation := InitProcess(x, wOutput, wHidden, bOutput, bHidden)
		dOutput, dHidden := BackPropagationForm(y, wOutput, output, hiddenLayerActivation)
		wHiddenAdj, bHiddenAdj, bOutdAdj, wOutAdj, err := AdjustingWeightsAndBias(x, hiddenLayerActivation, dHidden, dOutput, nn.Config.LearningRate)
		if err != nil {
			return err
		}
		fmt.Println("")
		printingMatDense(&output)
		fmt.Println("")

		wOutput.Add(wOutput, &wOutAdj)
		wHidden.Add(wHidden, &wHiddenAdj)
		bOutput.Add(bOutput, bOutdAdj)
		bHidden.Add(bHidden, bHiddenAdj)

	}*/

	nn.WHidden = wHidden
	nn.BHidden = bHidden
	nn.WOutput = wOutput
	nn.BOutput = bOutput

	return nil
}

func RunNeuronThird() {

	config := NeuralNetConfig{
		InputNeurons:  3,
		HiddenNeurons: 3,
		OutputNeurons: 4,
		NumberEpochs:  2000,
		LearningRate:  0.2,
	}

	/*
		the weights dimensions are:
		[3x3] hidden
		[3x4] output
	*/
	fmt.Println(config)

}
