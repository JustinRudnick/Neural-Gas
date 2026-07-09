package main

import (
	input "NeuralGas/Input"
	neuralgas "NeuralGas/NeuralGas"
	plotting "NeuralGas/Plotting"
	"fmt"
	"image"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func main() {

	var seed int64 = 22
	randomizer := rand.New(rand.NewSource(seed))

	var factor float64 = 4
	var imagePath string = "imageSamples/man_and_woman.jpg"

	sampleSetRed := input.ImageToSampleSetReverse(imagePath, func(x, y int, img *image.Image) bool {
		r, _, _, a := (*img).At(x, y).RGBA()
		return rand.Float64() > factor*float64(r)/float64(0xffff)*float64(a)/float64(0xffff)
	})
	// sampleSetGreen := input.ImageToSampleSetReverse(imagePath, func(x, y int, img *image.Image) bool {
	// 	_, g, _, a := (*img).At(x, y).RGBA()
	// 	return rand.Float64() > factor*float64(g)/float64(0xffff)*float64(a)/float64(0xffff)
	// })
	// sampleSetBlue := input.ImageToSampleSetReverse(imagePath, func(x, y int, img *image.Image) bool {
	// 	_, _, b, a := (*img).At(x, y).RGBA()
	// 	return rand.Float64() > factor*float64(b)/float64(0xffff)*float64(a)/float64(0xffff)
	// })
	// sampleSetAvg := input.ImageToSampleSetReverse(imagePath, func(x, y int, img *image.Image) bool {
	// 	r, g, b, a := (*img).At(x, y).RGBA()
	// 	r = uint32(float64(r+g+b) / 3.0)
	// 	return rand.Float64() > factor*float64(b)/float64(0xffff)*float64(a)/float64(0xffff)
	// })

	// if sampleSet == nil {
	// 	println("Error at sample Set")
	// }

	// plotting.Plot2D(sampleSetRed, "red filter", "imagePlots/sample41")
	// println("sample generated")
	// plotting.Plot2D(sampleSetGreen, "green filter", "imagePlots/sample42")
	// println("sample generated")
	// plotting.Plot2D(sampleSetBlue, "blue filter", "imagePlots/sample43")
	// println("sample generated")
	// plotting.Plot2D(sampleSetAvg, "average filter", "imagePlots/sample44")
	// println("sample generated")

	prototypeCount := 200

	params := neuralgas.Params{
		LearningRate_start:     0.5,
		LearningRate_end:       0.005,
		InnerTemperature_start: float64(prototypeCount) / 2.0,
		InnerTemperature_end:   0.01}

	for i := range 5 {
		ng := neuralgas.NewNorm(sampleSetRed,
			uint(prototypeCount),
			randomizer,
			params)
		ng.Train(uint(i), 8)
		plotting.Plot2D(ng.Prototypes(), fmt.Sprintf("%d epoch(s)", i), fmt.Sprintf("imagePlots/%dimg_0000%d", 4, i))
	}

}

// func main() {
// 	var seed int64 = 22

// 	randomizer := rand.New(rand.NewSource(seed))
// 	var dataset []*mat.VecDense = make([]*mat.VecDense, 100)
// 	// dataset[0] = mat.NewVecDense(2, []float64{0.1, 0.2})
// 	// dataset[1] = mat.NewVecDense(2, []float64{0.2, 0.1})
// 	// dataset[2] = mat.NewVecDense(2, []float64{0.15, 0.18})
// 	// dataset[3] = mat.NewVecDense(2, []float64{0.8, 0.9})
// 	// dataset[4] = mat.NewVecDense(2, []float64{0.9, 0.8})
// 	// dataset[5] = mat.NewVecDense(2, []float64{0.85, 0.88})

// 	for i := range len(dataset) {
// 		rng := rand.Float64()
// 		// dataset[i] = mat.NewVecDense(2, []float64{0.5*math.Sin(rng*2*math.Pi) + 0.5, 0.5*math.Cos(rng*2*math.Pi) + 0.5}) //circle
// 		// dataset[2*i] = mat.NewVecDense(2, []float64{rng, math.Cos(rng)})	// sin cos (1/2)
// 		// dataset[2*i+1] = mat.NewVecDense(2, []float64{rng, math.Sin(rng)}) // sin cos (2/2)
// 		dataset[i] = mat.NewVecDense(2, []float64{0.5*rng + 0.2, 0.2*rand.Float64() + 0.4}) // rectangle area
// 	}

// 	prototypeCount := 300

// 	params := neuralgas.Params{
// 		LearningRate_start:     0.5,
// 		LearningRate_end:       0.005,
// 		InnerTemperature_start: float64(prototypeCount) / 2.0,
// 		InnerTemperature_end:   0.01}

// 	ng500 := neuralgas.NewNorm(dataset, uint(prototypeCount), randomizer, params)
// 	ng2000 := neuralgas.NewNorm(dataset, uint(prototypeCount), randomizer, params)
// 	// ng10000 := neuralgas.NewNorm(dataset, uint(prototypeCount), randomizer, params)

// 	plotting.Plot2D(dataset, "Samples", "plots/samples")

// 	testreihe := 0
// 	for i := range 10 {
// 		ng := neuralgas.NewNorm(dataset,
// 			uint(prototypeCount),
// 			randomizer,
// 			params)
// 		ng.Train(uint(i))
// 		plotting.Plot2D(ng.GetPrototypes(), fmt.Sprintf("%d epoch(s)", i), fmt.Sprintf("plots/%dimg_0000%d", testreihe, i))
// 	}

// 	ng500.Train(500)
// 	plotting.Plot2D(ng500.GetPrototypes(), "500 epochs", fmt.Sprintf("plots/%dimg_00500", testreihe))
// 	ng2000.Train(2000)
// 	plotting.Plot2D(ng2000.GetPrototypes(), "2000 epochs", fmt.Sprintf("plots/%dimg_02000", testreihe))
// 	// ng10000.Train(10000)
// 	// plotting.Plot2D(ng10000.GetPrototypes(), "10000 epochs", fmt.Sprintf("plots/%dimg_10000", testreihe))

// }

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
		println("Distance: ", neuralgas.DistanceSq(sample, arr[i]))
	}
}
