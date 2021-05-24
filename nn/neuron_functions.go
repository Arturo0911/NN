package nn

import (
	"errors"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func InitProcess(x, wOutput, wHidden, bOutput, bHidden *mat.Dense) (output, hiddenActivation mat.Dense) {
	applySigmoid := func(_, _ int, v float64) float64 { // Getting the form func(int, int float64) float64{}
		return Sigmoid(v)
	}

	var hiddenInput mat.Dense // Init the Multiplying with the respective weights and values from the input
	hiddenInput.Mul(x, wHidden)

	hiddenInput.Apply(func(_, col int, v float64) float64 { // Adding the respective bias from the hiddenLayer
		return v + bHidden.At(0, col)
	}, &hiddenInput)

	hiddenActivation.Apply(applySigmoid, &hiddenInput) // Applying the sigmoid function

	var outputLayer mat.Dense                   // taking the activation layer generate the process multiplying by the weights but the outputLayer
	outputLayer.Mul(&hiddenActivation, wOutput) // adding the respective bias from the outputLayer
	outputLayer.Apply(func(_, col int, v float64) float64 {
		return v + bOutput.At(0, col)
	}, &outputLayer)

	output.Apply(applySigmoid, &outputLayer) // APplying the same activation Sigmoid function
	return output, hiddenActivation
}

//
//	888888b.                  888     	8888888b.                                                888   d8b
//	888  "88b                 888     	888   Y88b                                               888   Y8P
//	888  .88P                 888     	888    888                                               888
//	8888888K.  8888b.  .d8888b888  888	888   d88P888d888 .d88b. 88888b.  8888b.  .d88b.  8888b. 888888888 .d88b. 88888b.
//	888  "Y88b    "88bd88P"   888 .88P	8888888P" 888P"  d88""88b888 "88b    "88bd88P"88b    "88b888   888d88""88b888 "88b
//	888    888.d888888888     888888K 	888       888    888  888888  888.d888888888  888.d888888888   888888  888888  888
//	888   d88P888  888Y88b.   888 "88b	888       888    Y88..88P888 d88P888  888Y88b 888888  888Y88b. 888Y88..88P888  888
//	8888888P" "Y888888 "Y8888P888  888	888       888     "Y88P" 88888P" "Y888888 "Y88888"Y888888 "Y888888 "Y88P" 888  888
//	                                  	                         888                  888
//	                                  	                         888             Y8b d88P
//	                                  	                         888              "Y88P"

func BackPropagationForm(y, wOutput *mat.Dense, output, hiddenLayerActivation mat.Dense) (mat.Dense, mat.Dense) {

	applySigmoidePrime := func(_, _ int, v float64) float64 { // Getting the respective SigmoidPrime for loss or errors
		return SigmoidePrime(v)
	}

	var (
		networkErr         mat.Dense // The error calculus
		slopeOutputLayer   mat.Dense // slopes
		slopeHiddenLayer   mat.Dense
		errorAtHiddenLayer mat.Dense // getting the respective error at the specific hiddenLayer
		dOutput            mat.Dense // derivatives
		dHiddenLayer       mat.Dense
	)

	networkErr.Sub(y, &output)                          // Calculating the difference between the data desired and the data obtained
	slopeOutputLayer.Apply(applySigmoidePrime, &output) // Applying the derivative sigmoid
	slopeHiddenLayer.Apply(applySigmoidePrime, &hiddenLayerActivation)

	dOutput.MulElem(&networkErr, &slopeOutputLayer)

	errorAtHiddenLayer.Mul(&dOutput, wOutput.T())

	dHiddenLayer.MulElem(&errorAtHiddenLayer, &slopeHiddenLayer)

	return dOutput, dHiddenLayer
}

//
//         d8888     888  d8b                888   		                                                    888
//        d88888     888  Y8P                888   		                                                    888
//       d88P888     888                     888   		                                                    888
//      d88P 888 .d88888 8888888  888.d8888b 888888		88888b.  8888b. 888d888 8888b. 88888b.d88b.  .d88b. 888888 .d88b. 888d888.d8888b
//     d88P  888d88" 888 "888888  88888K     888   		888 "88b    "88b888P"      "88b888 "888 "88bd8P  Y8b888   d8P  Y8b888P"  88K
//    d88P   888888  888  888888  888"Y8888b.888   		888  888.d888888888    .d888888888  888  88888888888888   88888888888    "Y8888b.
//   d8888888888Y88b 888  888Y88b 888     X88Y88b. 		888 d88P888  888888    888  888888  888  888Y8b.    Y88b. Y8b.    888         X88
//  d88P     888 "Y88888  888 "Y88888 88888P' "Y888		88888P" "Y888888888    "Y888888888  888  888 "Y8888  "Y888 "Y8888 888     88888P'
//		   				  888                      		888
//		   			     d88P                      		888
//		   			   888P"                       		888

func AdjustingWeightsAndBias(x *mat.Dense, hiddenActivation,
	dHiddenLayer, dOutput mat.Dense, lr float64) (mat.Dense, *mat.Dense, *mat.Dense, mat.Dense, error) {
	var (
		wAdjHidden mat.Dense
		wAdjOut    mat.Dense
	)
	wAdjOut.Mul(hiddenActivation.T(), &dOutput)
	wAdjOut.Scale(lr, &wAdjOut)
	//wAdjOut.Add(wOut, &wAdjOut)

	wAdjHidden.Mul(x.T(), &dHiddenLayer)
	wAdjHidden.Scale(lr, &wAdjHidden)
	//wAdjHidden.Add(wHidden, &wAdjHidden)

	bOutdAdj, err := SumAlongAxis(0, &dOutput)
	if err != nil {
		return *mat.NewDense(0, 0, nil), nil, nil, *mat.NewDense(0, 0, nil), err
	}

	bHiddenAdj, err := SumAlongAxis(0, &dHiddenLayer)
	if err != nil {
		return *mat.NewDense(0, 0, nil), nil, nil, *mat.NewDense(0, 0, nil), err
	}

	return wAdjHidden, bHiddenAdj, bOutdAdj, wAdjOut, nil
}

func CreateMatrixDense(h, i, o int) (wHidden, bHidden, wOutput, bOutput *mat.Dense) {

	// h -> number of hidden layers or nodes
	// i -> number of input layers or nodes
	// o -> number of output layers or nodes

	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHiddenRaw := make([]float64, h*i)
	bHiddenRaw := make([]float64, h)
	wOutputRaw := make([]float64, h*o)
	bOutputRaw := make([]float64, o)

	// make a loop to fill into these floats arrays
	for _, record := range [][]float64{wHiddenRaw, bHiddenRaw, wOutputRaw, bOutputRaw} {
		for i := range record {
			record[i] = randGen.Float64()
		}
	}
	wHidden = mat.NewDense(i, h, wHiddenRaw)
	bHidden = mat.NewDense(1, h, bHiddenRaw)
	wOutput = mat.NewDense(h, o, wOutputRaw)
	bOutput = mat.NewDense(1, o, bOutputRaw)

	return wHidden, bHidden, wOutput, bOutput
}

//	8888888        d8b888   	8888888b.
//	  888          Y8P888   	888   Y88b
//	  888             888   	888    888
//	  888  88888b. 888888888	888   d88P888d888 .d88b.  .d8888b .d88b. .d8888b .d8888b
//	  888  888 "88b888888   	8888888P" 888P"  d88""88bd88P"   d8P  Y8b88K     88K
//	  888  888  888888888   	888       888    888  888888     88888888"Y8888b."Y8888b.
//	  888  888  888888Y88b. 	888       888    Y88..88PY88b.   Y8b.         X88     X88
//	8888888888  888888 "Y888	888       888     "Y88P"  "Y8888P "Y8888  88888P' 88888P'

func StartLearning(trainFile, testFile string) (float64, error) {

	return 0, nil
}

//
//	8888888b.                     888d8b        888   d8b
//	888   Y88b                    888Y8P        888   Y8P
//	888    888                    888           888
//	888   d88P888d888 .d88b.  .d88888888 .d8888b888888888 .d88b. 88888b.
//	8888888P" 888P"  d8P  Y8bd88" 888888d88P"   888   888d88""88b888 "88b
//	888       888    88888888888  888888888     888   888888  888888  888
//	888       888    Y8b.    Y88b 888888Y88b.   Y88b. 888Y88..88P888  888
//	888       888     "Y8888  "Y88888888 "Y8888P "Y888888 "Y88P" 888  888

func (nn *NeuralNet) Prediction(x *mat.Dense) (*mat.Dense, error) {

	applySigmoid := func(_, _ int, v float64) float64 {
		return Sigmoid(v)
	}

	var output mat.Dense

	if nn.BHidden == nil || nn.WHidden == nil || nn.WOutput == nil || nn.BOutput == nil {
		return nil, errors.New("should be have data trained to make predictions")
	}

	var hiddenLayer mat.Dense
	hiddenLayer.Mul(x, nn.WHidden)
	hiddenLayer.Apply(func(_, col int, v float64) float64 {
		return v + nn.BHidden.At(0, col)
	}, &hiddenLayer)

	var hiddenLayerActivation mat.Dense
	hiddenLayerActivation.Apply(applySigmoid, &hiddenLayer)

	var outputLayer mat.Dense
	outputLayer.Mul(&hiddenLayerActivation, nn.WOutput)
	outputLayer.Apply(func(_, col int, v float64) float64 {
		return v + nn.BHidden.At(0, col)
	}, &outputLayer)

	output.Apply(applySigmoid, &outputLayer)

	return &output, nil

}
