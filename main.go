package main

import (
	input "NeuralGas/Input"
	neuralgas "NeuralGas/NeuralGas"
	plotting "NeuralGas/Plotting"
	"fmt"
	"image"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func main() {

	//-----------------
	//standard init
	//-----------------
	var err error

	var logger *slog.Logger = slog.New(slog.NewTextHandler(os.Stdout, nil))
	var seed int64
	randomizer := rand.New(rand.NewSource(rand.Int63()))
	trainCores := 1
	encCores := 1 //1 for deterministic purposes

	samplePlotPath := ".gitignore/imagePlots/"
	sampleName := "sample"

	var factor float64 = 1                                          //factor decides the likelyhood of creating a sample
	var samplePath string = ".gitignore/imageSamples/man_small.jpg" //".gitignore/imageSamples/DestroyerJhinIcon.jpeg"

	// resPath := "C:/GitHub/Neural-Gas-CKKS/files/"
	// resFile := "default.txt"

	isPlotted := false
	plotPath := ".gitignore/plots/"
	imageNumber := 0 //prefix for plots

	epochs := 10

	sampleCount := 40
	prototypeCount := 500

	var useRandomSet bool = false

	//-----------------
	//process input
	//-----------------

	for i, arg := range os.Args {
		switch arg[0] {
		case '-':
			switch strings.ToLower(arg[1:]) {
			case "plot":
				imageNumber, err = strconv.Atoi(os.Args[i+1])
				if err != nil {
					panic(err)
				}
				isPlotted = true
			case "cores", "c":
				trainCores, err = strconv.Atoi(os.Args[i+1])
				if err != nil {
					panic(err)
				}
			case "seed":
				seed, err = strconv.ParseInt(os.Args[i+1], 10, 64)
				if err != nil {
					panic(err)
				}
				randomizer = rand.New(rand.NewSource(seed))
			case "prototypes", "p":
				prototypeCount, err = strconv.Atoi(os.Args[i+1])
				if err != nil {
					panic(err)
				}
			case "samples", "s":
				sampleCount, err = strconv.Atoi(os.Args[i+1])
				if err != nil {
					panic(err)
				}
				useRandomSet = true
			case "epochs", "e":
				epochs, err = strconv.Atoi(os.Args[i+1])
				if err != nil {
					panic(err)
				}
			case "help", "h", "?":
				printHelpInfo()
				return
			// case "file", "f":
			// 	resFile = os.Args[i+1]
			// case "path":
			// 	resPath = os.Args[i+1]
			default:
			}
		case '?':
			printHelpInfo()
			return
		default:
		}
	}

	var sampleSet []*mat.VecDense
	if useRandomSet {
		sampleSet = make([]*mat.VecDense, sampleCount)
		fillDataset(sampleSet, randomizer)
	} else {
		sampleSetRed := input.ImageToSampleSetReverse(samplePath, func(x, y int, img *image.Image) bool {
			r, _, _, a := (*img).At(x, y).RGBA()
			return randomizer.Float64() > factor*float64(r)/float64(0xffff)*float64(a)/float64(0xffff)
		})
		sampleSetGreen := input.ImageToSampleSetReverse(samplePath, func(x, y int, img *image.Image) bool {
			_, g, _, a := (*img).At(x, y).RGBA()
			return randomizer.Float64() > factor*float64(g)/float64(0xffff)*float64(a)/float64(0xffff)
		})
		sampleSetBlue := input.ImageToSampleSetReverse(samplePath, func(x, y int, img *image.Image) bool {
			_, _, b, a := (*img).At(x, y).RGBA()
			return randomizer.Float64() > factor*float64(b)/float64(0xffff)*float64(a)/float64(0xffff)
		})
		sampleSetAvg := input.ImageToSampleSetReverse(samplePath, func(x, y int, img *image.Image) bool {
			r, _, _, a := (*img).At(x, y).RGBA()
			// r = (r + g + b) / 3
			value := factor * float64(r) * float64(a) / float64(0xffff)
			return (x*y)%2 == 1 && value < 0x6000
		})

		plotting.Plot2D(sampleSetRed, fmt.Sprintf("%s, %d samples", "red filter", len(sampleSetRed)), fmt.Sprintf("%s/%s", samplePlotPath, fmt.Sprintf("%d%s_red", imageNumber, sampleName)))
		println("sample generated: ", len(sampleSetRed), " data points")
		plotting.Plot2D(sampleSetGreen, fmt.Sprintf("%s, %d samples", "green filter", len(sampleSetGreen)), fmt.Sprintf("%s/%s", samplePlotPath, fmt.Sprintf("%d%s_green", imageNumber, sampleName)))
		println("sample generated")
		plotting.Plot2D(sampleSetBlue, fmt.Sprintf("%s, %d samples", "blue filter", len(sampleSetBlue)), fmt.Sprintf("%s/%s", samplePlotPath, fmt.Sprintf("%d%s_blue", imageNumber, sampleName)))
		println("sample generated")
		plotting.Plot2D(sampleSetAvg, fmt.Sprintf("%s, %d samples", "average filter", len(sampleSetAvg)), fmt.Sprintf("%s/%s", samplePlotPath, fmt.Sprintf("%d%s_avg", imageNumber, sampleName)))
		println("sample generated: ", len(sampleSetAvg), " data points")

		sampleSet = sampleSetAvg
	}

	params := neuralgas.Params{
		LearningRate_initial:     0.5,
		LearningRate_final:       0.005,
		InnerTemperature_initial: float64(prototypeCount) / 2.0,
		InnerTemperature_final:   0.01}

	ng, err := neuralgas.NewNorm(sampleSet,
		uint(prototypeCount),
		randomizer,
		params,
		encCores,
		logger)
	if err != nil {
		panic(err)
	}

	plotEpochs := make([]int, 10)
	for i := range 10 {
		plotEpochs[i] = epochs / (i + 1)
	}
	if isPlotted {
		err = ng.TrainPlots(uint(epochs), uint(trainCores), fmt.Sprintf("%s%dplot", plotPath, imageNumber), append(plotEpochs, 0, 1, 2, 3, 4, 5, 6, 7, 10))
	} else {
		err = ng.Train(uint(epochs), uint(trainCores))
	}
	if err != nil {
		panic(err)
	}
}

func randArr(dimensions int, randomizer rand.Rand) []float64 {
	arr := make([]float64, dimensions)
	for i := range dimensions {
		arr[i] = randomizer.Float64()
	}
	return arr
}

func printVecs(sample *mat.VecDense, arr []*mat.VecDense) {
	for i := range len(arr) {
		fmt.Println(mat.Formatted(arr[i]))
		dist, err := neuralgas.DistanceSq(sample, arr[i])
		if err != nil {
			panic(err)
		}
		println("Distance: ", dist)
	}
}

func printHelpInfo() {
	println("commands:")
	println("-plot <int>\t...plots the results with given prefix. Default: no plotting")
	println("-cores -c <int>\t...number of threads created. Default: 1")
	println("-seed <int64>\t...seed for randomizer. Default: random")
	println("-prototypes -p <int>\t...amount of prototypes created. Default: 500")
	println("-samples -s <int>\t...generates random sample set of passed amount of samples. Default: use image")
	println("-epochs -e <int>\t...amount of epochs used for training.")
	println("-help -h -? ?\t...prints this.")
}

func fillDataset(dataset []*mat.VecDense, RNG *rand.Rand) {
	for i := range len(dataset) {
		rng := RNG.Float64()
		dataset[i] = mat.NewVecDense(2, []float64{0.5*math.Sin(rng*2*math.Pi) + 0.5, 0.5*math.Cos(rng*2*math.Pi) + 0.5}) //circle
		// dataset[2*i] = mat.NewVecDense(2, []float64{rng, math.Cos(rng)})	// sin cos (1/2)
		// dataset[2*i+1] = mat.NewVecDense(2, []float64{rng, math.Sin(rng)}) // sin cos (2/2)
		// dataset[i] = mat.NewVecDense(2, []float64{0.5*rng + 0.2, 0.2*rand.Float64() + 0.4}) // rectangle area
	}

}
