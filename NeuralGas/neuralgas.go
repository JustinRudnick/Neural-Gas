package neuralgas

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type Params struct {
	LearningRate_start     float64
	LearningRate_end       float64
	InnerTemperature_start float64
	InnerTemperature_end   float64
}

type NeuralGas struct {
	samples                  []*mat.VecDense
	prototypes               []*mat.VecDense
	optimizingPrototypeCount uint // amount of nearest prototypes to a sample to optimize with step()
	randomizer               *rand.Rand

	constants Params
}

// returns a neural gas algorithm with normalized generated prototypes
// input dataset components must be normalized to interval [0, 1]
func NewNorm(dataset []*mat.VecDense,
	prototypeCount uint,
	randomizer *rand.Rand,
	params Params) NeuralGas {
	dimensions := (*dataset[0]).Len()
	prototypes := make([]*mat.VecDense, prototypeCount)

	for i := range prototypeCount {
		prototype := make([]float64, dimensions)
		for j := range dimensions {
			prototype[j] = randomizer.Float64()
		}
		prototypes[i] = mat.NewVecDense(dimensions, prototype)
	}

	return NeuralGas{samples: dataset,
		prototypes:               prototypes,
		optimizingPrototypeCount: prototypeCount,
		randomizer:               randomizer,
		constants:                params}
}

// changes the prototypes
func (ng *NeuralGas) step(sample *mat.VecDense, iteration int, maxIterations int) {

	sort.Slice(ng.prototypes, func(i, j int) bool {
		return (DistanceSq(sample, ng.prototypes[i]) < DistanceSq(sample, ng.prototypes[j]))
	})

	lambda := ng.GetInnerTemperature(iteration, maxIterations)
	epsilon := ng.GetStepWidth(iteration, maxIterations)

	for rank := range ng.optimizingPrototypeCount { //sort(sample, ng.prototypes, prototypeCount) {
		exp := math.Exp(-float64(rank) / lambda) // e^{-k/lambda}

		var offset *mat.VecDense = mat.NewVecDense(sample.Len(), nil)

		offset.SubVec(sample, ng.prototypes[rank]) // (v - w_iOld)
		offset.ScaleVec(epsilon*exp, offset)       // epsilon * e^{-k/lambda} * (v - w_iOld)

		ng.prototypes[rank].AddVec(ng.prototypes[rank], offset) // w_iOld + epsilon * e^{-k/lambda} * (v - w_iOld)
	}
}

func (ng *NeuralGas) Train(epochs uint) {
	iteration := 0
	totalIterations := int(epochs) * len(ng.samples)
	for epoch := range epochs {
		rand.Shuffle(len(ng.samples), ng.swap)
		for _, sample := range ng.samples {
			ng.step(sample, iteration, totalIterations)
			iteration++
		}
		if (epoch+1)%(epochs/uint(math.Min(float64(epochs), float64(10)))) == 0 {
			fmt.Printf("---------------------- EPOCH %d / %d -----------------------\n", epoch+1, epochs)
		}
		// ng.step(ng.prototypes[ng.randomizer.Intn(len(ng.samples))], ng.optimizingPrototypeCount, int(epoch), int(epochs))
	}
}

//###################### relevant functions ##############################################################

func (ng *NeuralGas) GetStepWidth(iteration int, maxIterations int) float64 {
	return calculation(ng.constants.LearningRate_start, ng.constants.LearningRate_end, iteration, maxIterations)
}

func (ng *NeuralGas) GetInnerTemperature(iteration int, maxIterations int) float64 {
	return calculation(ng.constants.InnerTemperature_start, ng.constants.InnerTemperature_end, iteration, maxIterations)
}

func (ng NeuralGas) GetPrototypes() []*mat.VecDense {
	return ng.prototypes
}

func (ng NeuralGas) GetSamples() []*mat.VecDense {
	return ng.samples
}

// func sort(sample *mat.VecDense, prototypes []*mat.VecDense, prototypeCount uint) []*mat.VecDense {
// 	array := make([]*mat.VecDense, prototypeCount)
// 	for i := range prototypeCount {
// 		array[i] = prototypes[i]
// 	}
// 	Bubblesort(sample, array) //bubblesort the initialized array
// 	for i := int(prototypeCount); i < len(prototypes); i++ {
// 		if DistanceSq(sample, prototypes[i]) < DistanceSq(sample, array[prototypeCount-1]) {
// 			array[prototypeCount-1] = prototypes[i]
// 			for j := prototypeCount - 1; DistanceSq(sample, array[j]) < DistanceSq(sample, array[j-1]); j-- {
// 				buffer := array[j-1]
// 				array[j-1] = array[j]
// 				array[j] = buffer
// 			}
// 		}
// 	}
// 	return array
// }

//###################### Helper functions ####################################################

// returns gI * (gF/gI)^(t/t_max)
func calculation(gI float64, gF float64, t int, tMax int) float64 {
	return gI * math.Pow(gF/gI, float64(t)/float64(tMax))
}

func fillVec(component float64, dimensions int) *mat.VecDense {
	array := make([]float64, dimensions)
	for i := range dimensions {
		array[i] = component
	}
	return mat.NewVecDense(dimensions, array)
}

func (ng NeuralGas) swap(i int, j int) {
	ng.samples[i], ng.samples[j] = ng.samples[j], ng.samples[i]
}

func DistanceSq(v1 mat.Vector, v2 mat.Vector) (dist float64) {
	r1, _ := v1.Dims()
	r2, _ := v2.Dims()
	if r1 != r2 {
		return -1
	}
	var sum float64 = 0
	for d := range r1 {
		dif := v1.AtVec(d) - v2.AtVec(d)
		sum += dif * dif
	}
	return sum
}

// sorts the passed array by comparing the distances from the sample point
// func Bubblesort(sample *mat.VecDense, arr []*mat.VecDense) {
// 	for i := range len(arr) {
// 		for j := len(arr) - 1; j > i; j-- {
// 			if DistanceSq(arr[j], sample) < DistanceSq(arr[j-1], sample) {
// 				buffer := arr[j-1]
// 				arr[j-1] = arr[j]
// 				arr[j] = buffer
// 			}
// 		}
// 	}
// }
