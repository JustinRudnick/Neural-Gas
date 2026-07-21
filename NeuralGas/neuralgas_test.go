package neuralgas

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

func (ng *NeuralGas) TestStep(sample *mat.VecDense, rankedPrototypes []*RankedPrototype, iteration int, maxIterations int, maxCores int) {
	ng.step(sample, rankedPrototypes, iteration, maxIterations, maxCores)
	for _, prototype := range rankedPrototypes {
		fmt.Fprintln(os.Stdout, *prototype.prototype)
		fmt.Printf("dist: %f\n", prototype.distance)
	}
}
