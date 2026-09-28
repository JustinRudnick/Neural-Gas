package util

import (
	"gonum.org/v1/gonum/mat"
)

type RankedPrototype struct {
	Prototype *mat.VecDense
	Distance  float64 // a buffer for the distance at step function for the given sample
}

func FillSlice[T any](value T, dims int) []T {
	arr := make([]T, dims)
	for i := range dims {
		arr[i] = value
	}
	return arr
}
