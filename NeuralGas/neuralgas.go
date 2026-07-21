package neuralgas

import (
	parallelize "NeuralGas/Parallelize"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Params struct {
	LearningRate_initial     float64
	LearningRate_final       float64
	InnerTemperature_initial float64
	InnerTemperature_final   float64
}

type NeuralGas struct {
	samples                  []*mat.VecDense
	prototypes               []*mat.VecDense
	optimizingPrototypeCount uint // amount of nearest prototypes to a sample to optimize with step()
	randomizer               *rand.Rand

	constants Params
	logger    *slog.Logger
	isLogged  bool
}

type RankedPrototype struct {
	prototype *mat.VecDense
	distance  float64 // a buffer for the distance at step function for the given sample
}

// returns a neural gas algorithm with normalized generated prototypes
// input dataset components must be normalized to interval [0, 1]
func NewNorm(dataset []*mat.VecDense,
	prototypeCount uint,
	randomizer *rand.Rand,
	params Params,
	logger *slog.Logger) *NeuralGas {

	dimensions := (*dataset[0]).Len()
	prototypes := make([]*mat.VecDense, prototypeCount)

	for i := range prototypeCount {
		prototype := make([]float64, dimensions)
		for j := range dimensions {
			prototype[j] = randomizer.Float64()
		}
		prototypes[i] = mat.NewVecDense(dimensions, prototype)
	}

	isLogged := logger != nil
	if isLogged {
		logger.Info("New normalized [NeuralGas] object has been created.")
	}

	return &NeuralGas{
		samples:                  dataset,
		prototypes:               prototypes,
		optimizingPrototypeCount: prototypeCount,
		randomizer:               randomizer,
		constants:                params,
		logger:                   logger,
		isLogged:                 isLogged}
}

func NewRankedPrototype(prototype *mat.VecDense, distance float64) *RankedPrototype {
	return &RankedPrototype{prototype: prototype, distance: distance}
}

/*
This function evaluates a learning step on the passed <rankedPrototypes> of neural gas algorithm.
The learning step function will only be applied for the
<ng.optimizingPrototypeCount> closest <rankedPrototypes> to the passed <sample> using euclidean distance.
This function destroys the identity of the <rankdedPrototypes> within the slice.
*/
func (ng *NeuralGas) step(sample *mat.VecDense, rankedPrototypes []*RankedPrototype, iteration int, maxIterations int, maxCores int) {

	parallelize.MultiThread(
		sample,
		rankedPrototypes,
		maxCores,
		func(sample *mat.VecDense, rankedPrototypes []*RankedPrototype, _ int, wg *sync.WaitGroup) {
			defer wg.Done()
			for i := range rankedPrototypes { // calculate distances from prototypes to the sample
				rankedPrototypes[i].distance = DistanceSq(sample, rankedPrototypes[i].prototype)
			}
		})

	// TODO EFFICIENT SORT
	sort.Slice(rankedPrototypes, func(i, j int) bool {
		return rankedPrototypes[i].distance < rankedPrototypes[j].distance
	})

	lambda := ng.InnerTemperature(iteration, maxIterations)
	epsilon := ng.StepWidth(iteration, maxIterations)

	parallelize.MultiThread(
		sample,
		rankedPrototypes[:ng.optimizingPrototypeCount],
		maxCores,
		func(sample *mat.VecDense, rankedPrototypes []*RankedPrototype, originalOffset int, wg *sync.WaitGroup) {
			defer wg.Done()
			for off := range rankedPrototypes {
				rank := originalOffset + off

				exp := math.Exp(-float64(rank) / lambda) // e^{-k/lambda}

				var offset *mat.VecDense = mat.NewVecDense(sample.Len(), nil)

				offset.SubVec(sample, rankedPrototypes[off].prototype) // (v - w_iOld)
				offset.ScaleVec(epsilon*exp, offset)                   // epsilon * e^{-k/lambda} * (v - w_iOld)

				rankedPrototypes[off].prototype.AddVec(rankedPrototypes[off].prototype, offset) // w_iOld + epsilon * e^{-k/lambda} * (v - w_iOld)
			}
		})
}

/*
Trains the prototypes of this [NeuralGas] for the amount of <epochs> using <maxCores> threads.
*/
func (ng *NeuralGas) Train(epochs uint, maxCores uint) {
	initialT := time.Now()
	if ng.isLogged {
		ng.logger.Info(fmt.Sprintf("Begin training for %d epoch(s) using %d threads.", epochs, maxCores))
	}

	iteration := 0
	totalIterations := int(epochs) * len(ng.samples)
	for epoch := range epochs {
		ng.ShuffleSamples()

		prototypeCount := len(ng.prototypes)
		rankedPrototypes := make([]*RankedPrototype, prototypeCount)
		for i := range prototypeCount {
			rankedPrototypes[i] = &RankedPrototype{ng.prototypes[i], 0}
		}

		for _, sample := range ng.samples {
			ng.step(sample, rankedPrototypes, iteration, totalIterations, int(maxCores))
			iteration++
		}

		if ng.isLogged && (epoch+1)%(epochs/uint(math.Min(float64(epochs), float64(10)))) == 0 {
			ng.logger.Info(fmt.Sprintf("---------------------- EPOCH %d / %d -----------------------", epoch+1, epochs))
		}
	}

	if ng.isLogged {
		ng.logger.Info(fmt.Sprintf("Training of %d epoch(s) in %f sec.", epochs, float64(time.Since(initialT))/float64(time.Second)))
	}

}

//###################### Getter functions ##############################################################

func (ng *NeuralGas) StepWidth(iteration int, maxIterations int) float64 {
	return calculation(ng.constants.LearningRate_initial, ng.constants.LearningRate_final, iteration, maxIterations)
}

func (ng *NeuralGas) InnerTemperature(iteration int, maxIterations int) float64 {
	return calculation(ng.constants.InnerTemperature_initial, ng.constants.InnerTemperature_final, iteration, maxIterations)
}

func (ng NeuralGas) Prototypes() []*mat.VecDense {
	return ng.prototypes
}

func (ng NeuralGas) Samples() []*mat.VecDense {
	return ng.samples
}

//###################### Helper functions ####################################################

// returns gI * (gF/gI)^(t/t_max)
func calculation(gI float64, gF float64, t int, tMax int) float64 {
	return gI * math.Pow(gF/gI, float64(t)/float64(tMax))
}

// fills a vector with
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

// returns the squared euclidian distance of the passed vectors
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

// Shuffle pseudo-randomizes the order of elements.
// n is the number of elements. Shuffle panics if n < 0.
// swap swaps the elements with indexes i and j.
//
// A modification of [rand.Shuffle] to set the pseudo-randomizer
func Shuffle(n int, randomizer *rand.Rand, swap func(i, j int)) {
	if n < 0 {
		panic("invalid argument to Shuffle")
	}

	// Fisher-Yates shuffle: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
	// Shuffle really ought not be called with n that doesn't fit in 32 bits.
	// Not only will it take a very long time, but with 2³¹! possible permutations,
	// there's no way that any PRNG can have a big enough internal state to
	// generate even a minuscule percentage of the possible permutations.
	// Nevertheless, the right API signature accepts an int n, so handle it as best we can.
	i := n - 1
	for ; i > 1<<31-1-1; i-- {
		j := int(randomizer.Int63n(int64(i + 1)))
		swap(i, j)
	}
	for ; i > 0; i-- {
		j := int(randomizer.Int31n(int32(i + 1)))
		swap(i, j)
	}
}

// ShuffleSamples pseudo-randomizes the order of ng.samples using the randomizer of ng.
//
// A modification of [rand.Shuffle] to set the pseudo-randomizer
func (ng *NeuralGas) ShuffleSamples() {
	Shuffle(len(ng.samples), ng.randomizer, ng.swap)
}
