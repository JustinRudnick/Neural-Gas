package util

import (
	"gonum.org/v1/gonum/mat"
)

type RankedPrototype struct {
	Prototype *mat.VecDense
	Distance  float64 // a buffer for the distance at step function for the given sample
}

// accuracy cannot be grater than 255 - logScalingFactor
func CkksLevels(logScalingFactor, logAccuracy, levels int) []float64 {
	arr := make([]float64, levels+1)
	for i := range arr {
		arr[i] = float64(logScalingFactor)
	}
	arr[0] += float64(logAccuracy)

	return arr
}

func FillSlice(value int, dims int) []int {
	arr := make([]int, dims)
	for i := range dims {
		arr[i] = value
	}
	return arr
}
