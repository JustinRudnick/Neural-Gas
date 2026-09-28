package neuralgas

import (
	parallelize "NeuralGas/Parallelize"
	sorting "NeuralGas/Parallelize/Sorting"
	plotting "NeuralGas/Plotting"
	util "NeuralGas/Util"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Params struct {
	LearningRate_initial     float64
	LearningRate_final       float64
	InnerTemperature_initial float64
	InnerTemperature_final   float64
	Threshold                float64 //threshold of learning step factor (if factor epsilon * exp(-k/lambda) gets below: learning step ends)

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

// returns a neural gas algorithm with normalized generated prototypes
// input dataset components must be normalized to interval [0, 1]
func NewNorm(
	dataset []*mat.VecDense,
	dimensions uint,
	prototypeCount uint,
	randomizer *rand.Rand,
	params Params,
	maxCores int,
	logger *slog.Logger,
) (ng *NeuralGas, err error) {

	prototypes := make([]*mat.VecDense, prototypeCount)

	for i := range prototypeCount {
		prototype := make([]float64, dimensions)
		for j := range dimensions {
			prototype[j] = randomizer.Float64()
		}
		prototypes[i] = mat.NewVecDense(int(dimensions), prototype)
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
		isLogged:                 isLogged}, nil
}

func NewRankedPrototype(prototype *mat.VecDense, distance float64) *util.RankedPrototype {
	return &util.RankedPrototype{Prototype: prototype, Distance: distance}
}

/*
This function evaluates a learning step on the passed <rankedPrototypes> of neural gas algorithm.
The learning step function will only be applied for the
<ng.optimizingPrototypeCount> closest <rankedPrototypes> to the passed <sample> using euclidean distance.
This function destroys the identity of the <rankdedPrototypes> within the slice.
*/
func (ng *NeuralGas) step(
	sample *mat.VecDense,
	rankedPrototypes []*util.RankedPrototype,
	iteration int,
	maxIterations int,
	maxCores int) (err error) {
	logger := ng.logger

	var errors chan error = make(chan error, 1)

	// compute distances of prototypes to sample point
	parallelize.MultiThread(
		sample,
		rankedPrototypes,
		maxCores,
		func(sample *mat.VecDense, rankedPrototypes []*util.RankedPrototype, startIdx int, wg *sync.WaitGroup) {
			var err error
			defer wg.Done()
			for i := range rankedPrototypes { // calculate distances from prototypes to the sample
				rankedPrototypes[i].Distance, err = DistanceSq(sample, rankedPrototypes[i].Prototype)
				if err != nil {
					select {
					case errors <- err: // non blocking
					default:
					}
					if ng.isLogged {
						totalIdx := startIdx + i
						logger.Error(fmt.Sprintf("Calculating distance failed for prototype idx: %d at iteration %d", totalIdx, iteration))
					}
				}
			}
		})

	select {
	case err = <-errors: // non blocking
		return fmt.Errorf("Calculating distances (threaded) in step function failed:\n\t%w", err)
	default:
	}

	// prototypes to be sorted
	var adjustedPrototypes int = 0

	factor := func(epsilon, lambda float64, rank int) float64 {
		return epsilon * math.Exp(-float64(rank)/lambda)
	}

	epsilon := ng.StepWidth(iteration, maxIterations)
	lambda := ng.InnerTemperature(iteration, maxIterations)

	for rank := 0; (factor(epsilon, lambda, rank) >= ng.constants.Threshold) && rank < ng.OptimizingPrototypeCount(); rank++ {
		adjustedPrototypes++
	}

	// rank prototypes
	// sections := 2 * maxCores
	// sorter, err := sorting.NewMaster(rankedPrototypes, sections, maxCores)
	// if err != nil {
	// 	return fmt.Errorf("Could not create sorting algorithm: %w", err)
	// }

	sortElem := func(slice []*util.RankedPrototype, i, j int) (err error) {
		if slice[i].Distance > slice[j].Distance {
			slice[i], slice[j] = slice[j], slice[i]
		}
		return nil
	}

	// sorter.BubbleSort(sortElem, adjustedPrototypes)
	sorting.Bubblesort[*util.RankedPrototype](rankedPrototypes, adjustedPrototypes, sortElem) //single core for unencrypted is faster (less overhead)

	//apply learning step
	parallelize.MultiThread(
		sample,
		rankedPrototypes[:adjustedPrototypes],
		maxCores,
		func(sample *mat.VecDense, rankedPrototypes []*util.RankedPrototype, originalOffset int, wg *sync.WaitGroup) {
			defer wg.Done()

			for off := range rankedPrototypes {
				rank := originalOffset + off

				exp := math.Exp(-float64(rank) / lambda) // e^{-k/lambda}
				koeff := epsilon * exp

				diff := mat.NewVecDense(sample.Len(), nil)
				diff.SubVec(sample, rankedPrototypes[off].Prototype)                                       // (v - w_iOld)
				rankedPrototypes[off].Prototype.AddScaledVec(rankedPrototypes[off].Prototype, koeff, diff) //w_iOld + epsilon * e^{-k/lambda} * (v - w_iOld)

			}
		})

	select {
	case err = <-errors: // non blocking
		return err
	default:
		return nil
	}
}

/*
Trains the prototypes of this [NeuralGas] for the amount of <epochs> using <maxCores> threads.
*/
func (ng *NeuralGas) Train(epochs uint, maxCores uint) (err error) {
	initialT := time.Now()
	if ng.isLogged {
		ng.logger.Info(fmt.Sprintf("Begin training for %d epoch(s) using %d threads.", epochs, maxCores))
	}

	iteration := 0
	totalIterations := int(epochs) * len(ng.samples)
	for epoch := range epochs {
		ng.ShuffleSamples()

		prototypeCount := len(ng.prototypes)
		rankedPrototypes := make([]*util.RankedPrototype, prototypeCount)
		for i := range prototypeCount {
			rankedPrototypes[i] = &util.RankedPrototype{Prototype: ng.prototypes[i], Distance: -1}
		}

		for _, sample := range ng.samples {
			err = ng.step(sample, rankedPrototypes, iteration, totalIterations, int(maxCores))
			if err != nil {
				return err
			}

			iteration++
		}

		if ng.isLogged && (epoch+1)%(epochs/uint(math.Min(float64(epochs), float64(10)))) == 0 {
			ng.logger.Info(fmt.Sprintf("---------------------- EPOCH %d / %d -----------------------", epoch+1, epochs))
		}
	}

	if ng.isLogged {
		ng.logger.Info(fmt.Sprintf("Training of %d epoch(s) in %f sec.", epochs, float64(time.Since(initialT))/float64(time.Second)))
	}

	return nil

}

func (ng *NeuralGas) TrainPlots(epochs, maxCores uint, filenames string, plotEpochs []int, logFileTrainedPrototypeCoundPerEpoch string) (err error) {
	// init log prototype count per epoch
	var logTPCPE *os.File = nil
	if logFileTrainedPrototypeCoundPerEpoch != "" {
		logTPCPE, err = os.OpenFile(logFileTrainedPrototypeCoundPerEpoch, os.O_CREATE|os.O_RDWR, 0644)
		if err != nil {
			return fmt.Errorf("Could not open file %s: %w", logFileTrainedPrototypeCoundPerEpoch, err)
		}
		defer logTPCPE.Close()

		err = logTPCPE.Truncate(0)
		if err != nil {
			return fmt.Errorf("Could not truncate file %s: %w", logFileTrainedPrototypeCoundPerEpoch, err)
		}
		_, err = logTPCPE.WriteString("epoch, adjusted prototypes\n")
		if err != nil {
			return fmt.Errorf("Could not write .csv header to file %s: %w", logFileTrainedPrototypeCoundPerEpoch, err)
		}
	}
	// end init log prototype count per epoch

	initialT := time.Now()
	if ng.isLogged {
		ng.logger.Info(fmt.Sprintf("Begin training for %d epoch(s) using %d threads.", epochs, maxCores))
	}

	logger := ng.logger

	iteration := 0
	totalIterations := int(epochs) * len(ng.samples)
	prototypeCount := len(ng.prototypes)

	if util.In(plotEpochs, 0) {
		plotting.Plot2D(ng.prototypes, fmt.Sprintf("%d epoch(s), %d prototypes", 0, prototypeCount), fmt.Sprintf("%s%d", filenames, 0))
	}

	for epoch := range epochs {
		ng.ShuffleSamples()

		rankedPrototypes := make([]*util.RankedPrototype, prototypeCount)
		for i := range prototypeCount {
			rankedPrototypes[i] = &util.RankedPrototype{Prototype: ng.prototypes[i], Distance: -1}
		}

		// log trained prototype count per epoch
		var adjustedPrototypes int
		epsilon := ng.StepWidth(iteration, totalIterations)
		lambda := ng.InnerTemperature(iteration, totalIterations)
		factor := func(rank int) float64 {
			return epsilon * math.Exp(-float64(rank)/lambda)
		}

		for adjustedPrototypes = 0; (factor(adjustedPrototypes) >= ng.constants.Threshold) && adjustedPrototypes < ng.OptimizingPrototypeCount(); adjustedPrototypes++ {
		}

		if logTPCPE != nil {
			_, err = logTPCPE.WriteString(fmt.Sprintf("%d,%d\n", epoch+1, adjustedPrototypes))
			if err != nil && logger != nil {
				logger.Warn(fmt.Sprintf("could not write to file %s: %s", logFileTrainedPrototypeCoundPerEpoch, err.Error()))
			}
		}
		// end log trained prototype count per epoch

		for _, sample := range ng.samples {

			err = ng.step(sample, rankedPrototypes, iteration, totalIterations, int(maxCores))
			if err != nil {
				return fmt.Errorf("Evaluating adaption step failed: %s", err.Error())
			}

			for i := range rankedPrototypes {
				ng.prototypes[i] = rankedPrototypes[i].Prototype
			}

			iteration++
		}

		//plotting
		if util.In(plotEpochs, int(epoch)+1) {
			plotting.Plot2D(ng.prototypes, fmt.Sprintf("%d epoch(s), %d prototypes", epoch+1, prototypeCount), fmt.Sprintf("%s%d", filenames, epoch+1))
		}

		if ng.isLogged && (epoch+1)%(epochs/uint(math.Min(float64(epochs), float64(10)))) == 0 {
			ng.logger.Info(fmt.Sprintf("---------------------- EPOCH %d / %d -----------------------", epoch+1, epochs))
		}
	}

	if ng.isLogged {
		ng.logger.Info(fmt.Sprintf("Training of %d epoch(s) in %f sec.", epochs, float64(time.Since(initialT))/float64(time.Second)))
	}

	return nil

}

//###################### Getter functions ##############################################################

// epsilon
func (ng *NeuralGas) StepWidth(iteration int, maxIterations int) float64 {
	var minEpsilon float64 = 0 //.5
	calEpsilon := calculation(ng.constants.LearningRate_initial, ng.constants.LearningRate_final, iteration, maxIterations)

	return math.Max(calEpsilon, minEpsilon)
}

// lambda - actual neighborhood range
func (ng *NeuralGas) InnerTemperature(iteration int, maxIterations int) float64 {
	var minLambda float64 = 0 //.1
	calLambda := calculation(ng.constants.InnerTemperature_initial, ng.constants.InnerTemperature_final, iteration, maxIterations)

	return math.Max(minLambda, calLambda)
}

func (ng NeuralGas) Prototypes() []*mat.VecDense {
	return ng.prototypes
}

func (ng NeuralGas) Samples() []*mat.VecDense {
	return ng.samples
}

func (ng NeuralGas) OptimizingPrototypeCount() int {
	return int(ng.optimizingPrototypeCount)
}

func (ng NeuralGas) SetOptimizingPrototypeCount(new uint) {
	ng.optimizingPrototypeCount = new
}

func (ng NeuralGas) PrototypeCount() int {
	return len(ng.prototypes)
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
//
//	The level of ciphertext distance is 1 level lower, than the levels of the ciphertexts v1 and v2.
func DistanceSq(v1 *mat.VecDense, v2 *mat.VecDense) (dist float64, err error) {
	if v1.Len() != v2.Len() {
		return -1, errors.New("v1 and v2 have different dimensions.")
	}

	var d []float64 = make([]float64, v1.Len())
	for i := range v1.RawVector().Data {
		d[i] = v1.AtVec(i) - v2.AtVec(i)
	}

	sum := util.Sum[float64](d, func(value float64, idx int) float64 { return value * value })

	return sum, nil
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
