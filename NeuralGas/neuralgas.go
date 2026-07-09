package neuralgas

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"

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

type RankedPrototype struct {
	prototype *mat.VecDense
	distance  float64 // a buffer for the distance at step function for the given sample
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

func NewRankedPrototype(prototype *mat.VecDense, distance float64) *RankedPrototype {
	return &RankedPrototype{prototype: prototype, distance: distance}
}

func (ng *NeuralGas) TestStep(sample *mat.VecDense, rankedPrototypes []*RankedPrototype, iteration int, maxIterations int, maxCores int) {
	ng.step(sample, rankedPrototypes, iteration, maxIterations, maxCores)
	for _, prototype := range rankedPrototypes {
		fmt.Fprintln(os.Stdout, *prototype.prototype)
		fmt.Printf("dist: %f\n", prototype.distance)
	}
}

// changes the prototypes
func (ng *NeuralGas) step(sample *mat.VecDense, rankedPrototypes []*RankedPrototype, iteration int, maxIterations int, maxCores int) {

	MultiThread(
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

	MultiThread(
		sample,
		rankedPrototypes,
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

func (ng *NeuralGas) Train(epochs uint, maxCores uint) {
	iteration := 0
	totalIterations := int(epochs) * len(ng.samples)
	for epoch := range epochs {
		// ng.ShuffleSamples()
		rand.Shuffle(len(ng.samples), ng.swap)

		prototypeCount := len(ng.prototypes)
		rankedPrototypes := make([]*RankedPrototype, prototypeCount)
		for i := range prototypeCount {
			rankedPrototypes[i] = &RankedPrototype{ng.prototypes[i], 0}
		}

		for _, sample := range ng.samples {
			ng.step(sample, rankedPrototypes, iteration, totalIterations, int(maxCores))
			iteration++
			// for i := range rankedPrototypes {
			// 	ng.prototypes[i] = rankedPrototypes[i].prototype
			// }
		}

		if (epoch+1)%(epochs/uint(math.Min(float64(epochs), float64(10)))) == 0 {
			fmt.Printf("---------------------- EPOCH %d / %d -----------------------\n", epoch+1, epochs)
		}
	}
}

//###################### relevant functions ##############################################################

func (ng *NeuralGas) StepWidth(iteration int, maxIterations int) float64 {
	return calculation(ng.constants.LearningRate_start, ng.constants.LearningRate_end, iteration, maxIterations)
}

func (ng *NeuralGas) InnerTemperature(iteration int, maxIterations int) float64 {
	return calculation(ng.constants.InnerTemperature_start, ng.constants.InnerTemperature_end, iteration, maxIterations)
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

// blocks until all goroutines are done
//
// Calls [runtime.SetDefaultGOMAXPROCS] in the end.
func MultiThread[K, T any](item K, items []T, maxCores int, function func(item K, subSlice []T, originalStartIndex int, wg *sync.WaitGroup)) {
	runtime.GOMAXPROCS(int(math.Min(float64(maxCores), float64(runtime.NumCPU()))))
	var wg sync.WaitGroup

	itemCount := len(items)
	routineCount := int(math.Min(float64(maxCores), float64(itemCount)))
	smallSubSliceSize := int(math.Floor(float64(itemCount) / float64(routineCount)))
	bigSubSliceSize := smallSubSliceSize + 1

	// println("items: ", itemCount)
	// println("routineCount: ", routineCount)
	wg.Add(routineCount)

	rest := itemCount % routineCount
	// fmt.Printf("big loops: %d\tsize: %d\n", rest, bigSubSliceSize)
	// fmt.Printf("small loops: %d\tsize: %d\n", routineCount-1-rest, smallSubSliceSize)

	for i := range rest {
		offset := i * bigSubSliceSize
		// println("big slice off ", offset)
		go function(item, items[offset:offset+bigSubSliceSize], offset, &wg)
	}

	for i := range routineCount - 1 - rest {
		offset := i*smallSubSliceSize + rest*bigSubSliceSize
		// println("small slice off ", offset)
		go function(item, items[offset:offset+smallSubSliceSize], offset, &wg)
	}

	offset := (routineCount-1)*smallSubSliceSize + rest
	// println("last slice off", offset)
	function(item, items[offset:], offset, &wg)

	wg.Wait()
	runtime.SetDefaultGOMAXPROCS()
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
