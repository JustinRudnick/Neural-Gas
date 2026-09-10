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

type inputFunctionalities struct {
	Logger     *slog.Logger
	Seed       int64
	TrainCores int
	InitCores  int

	SamplePlotPath string
	SampleName     string

	SampleImgPath string
	SampleImgFile string

	ResPath string
	ResFile string

	PlotPrefix string

	EpochCount     int
	SampleCount    int
	PrototypeCount int
}

func main() {

	//-----------------
	//standard init
	//-----------------
	var err error

	// standard initialization
	in := &inputFunctionalities{
		Logger:     slog.New(slog.NewTextHandler(os.Stdout, nil)),
		TrainCores: 1, //for deterministic purposes
		InitCores:  1,

		SamplePlotPath: ".gitignore/imagePlots/",
		SampleName:     "sample",

		SampleImgPath: ".gitignore/imageSamples",
		SampleImgFile: "man_small.jpg",

		ResPath: "./",

		PlotPrefix: "0",

		EpochCount:     10,
		SampleCount:    500,
		PrototypeCount: 50,
	}

	var seed int64
	randomizer := rand.New(rand.NewSource(rand.Int63()))

	var factor float64 = 1 //factor decides the likelyhood of creating a sample

	isPlotted := false
	plotPath := ".gitignore/plots/"

	var useRandomSet bool = false
	var isFiled bool = false

	//-----------------
	//process input
	//-----------------

	for i, arg := range os.Args {
		switch arg[0] {
		case '-':
			switch strings.ToLower(arg[1:]) {
			case "plot":
				in.PlotPrefix = os.Args[i+1]
				if err != nil {
					panic(err)
				}
				isPlotted = true
			case "cores", "c":
				in.TrainCores, err = strconv.Atoi(os.Args[i+1])
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
				in.PrototypeCount, err = strconv.Atoi(os.Args[i+1])
				if err != nil {
					panic(err)
				}
			case "samples", "s":
				in.SampleCount, err = strconv.Atoi(os.Args[i+1])
				if err != nil {
					panic(err)
				}
				useRandomSet = true
			case "sampleimg", "si":
				in.SampleImgFile = os.Args[i+1]
			case "samplepath", "sp":
				in.SampleImgPath = os.Args[i+1]
			case "epochs", "e":
				in.EpochCount, err = strconv.Atoi(os.Args[i+1])
				if err != nil {
					panic(err)
				}
			case "help", "h", "?":
				printHelpInfo(in)
				return
			case "file", "f":
				in.ResFile = os.Args[i+1]
				isFiled = true
			case "path":
				in.ResPath = os.Args[i+1]
			default:
			}
		case '?':
			printHelpInfo(in)
			return
		default:
		}
	}

	var sampleSet []*mat.VecDense
	if useRandomSet {
		sampleSet = make([]*mat.VecDense, in.SampleCount)
		fillDataset(sampleSet, randomizer)
		plotting.Plot2D(sampleSet, fmt.Sprintf("%d samples", len(sampleSet)), fmt.Sprintf("%s%s", in.SamplePlotPath, fmt.Sprintf("%s%s", in.PlotPrefix, in.SampleName)))
	} else {
		sampleSet, err = input.ImageToSampleSetReverse(fmt.Sprintf("%s%s", in.SampleImgPath, in.SampleImgFile), func(x, y int, img *image.Image) bool {
			r, _, _, a := (*img).At(x, y).RGBA()
			value := factor * float64(r) * float64(a) / float64(0xffff)
			return (x*y)%2 == 1 && value < 0x6000
		})

		plotting.Plot2D(sampleSet, fmt.Sprintf("%s, %d samples", "average filter", len(sampleSet)), fmt.Sprintf("%s%s", in.SamplePlotPath, fmt.Sprintf("%s%s", in.PlotPrefix, in.SampleName)))
		println("sample generated: ", len(sampleSet), " data points")
	}

	params := neuralgas.Params{
		LearningRate_initial:     0.5,
		LearningRate_final:       0.005,
		InnerTemperature_initial: float64(in.PrototypeCount) / 2.0,
		InnerTemperature_final:   0.01}

	ng, err := neuralgas.NewNorm(sampleSet,
		uint(in.PrototypeCount),
		randomizer,
		params,
		in.InitCores,
		in.Logger)
	if err != nil {
		panic(err)
	}

	if isPlotted {
		plotEpochs := make([]int, 20)
		for i := range 10 {
			plotEpochs[2*i] = in.EpochCount / (i + 1)
			plotEpochs[2*i+1] = int(math.Round(float64(i+1) / float64(10) * float64(in.EpochCount)))
		}
		err = ng.TrainPlots(uint(in.EpochCount), uint(in.TrainCores), fmt.Sprintf("%s%splot", plotPath, in.PlotPrefix), append(plotEpochs, 0))
	} else {
		err = ng.Train(uint(in.EpochCount), uint(in.TrainCores))
	}
	if err != nil {
		panic(err)
	}

	if !isFiled {
		return
	}

	// open / create file and write down the contents
	file, err := os.OpenFile(fmt.Sprintf("%s%s", in.ResPath, in.ResFile), os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	err = file.Truncate(0)
	if err != nil {
		panic(err)
	}

	var s string
	for i := range ng.Prototypes() {
		s = ""
		sampleDims := 0

		if len(sampleSet) > 0 {
			sampleDims = len(sampleSet[0].RawVector().Data)
		}

		for dim := range sampleDims {
			s += fmt.Sprintf("%.17f", ng.Prototypes()[i].RawVector().Data[dim])
			if dim < sampleDims-1 {
				s += ", "
			} else {
				s += "\n"
			}
		}
		file.WriteString(s)
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

func printHelpInfo(in *inputFunctionalities) {
	println("commands:")
	println("--- learning ---")
	println("-cores -c <int>\t\t...number of threads created. Default: ", in.TrainCores)
	println("-seed <int64>\t\t...seed for randomizer. Default: random")
	println("-samples -s <int>\t...generates random sample set of passed amount of samples. Default: use image")
	println("-sampleimg -si <string>\t...image to use as sample. Default: ", in.SampleImgFile)
	println("-samplepath -sp <string>\t...path to sample image. Default: ", in.SampleImgPath)
	println("-prototypes -p <int>\t...amount of prototypes created. Default: ", in.PrototypeCount)
	println("-epochs -e <int>\t...amount of epochs used for training.")
	println("\n--- logging ---")
	println("-plot <int>\t\t...plots the results with given prefix. Default: no plotting")
	println("-file -f <string>\t...file to store decrypted prototype results. Default: no logging of results")
	println("-path <string>\t\t...path to store the file created with -file in. Default: ", in.ResPath)
	println("-help -h -? ?\t\t...prints this.")
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
