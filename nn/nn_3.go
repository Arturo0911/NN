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

	for i := 0; i < nn.Config.NumberEpochs; i++ {

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

	}

	nn.WHidden = wHidden
	nn.BHidden = bHidden
	nn.WOutput = wOutput
	nn.BOutput = bOutput

	return nil
}
