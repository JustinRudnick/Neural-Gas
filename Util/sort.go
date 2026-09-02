package util

import (
	"log/slog"
)

func BubbleSort(
	array []*RankedPrototype,
	sortingElements int,
	logger *slog.Logger,
) (err error) {
	if logger != nil {
		logger.Info("Starting bubble sort.")
	}

	for i := 0; i < sortingElements; i++ {
		for j := len(array) - 2; j >= i; j-- {
			if array[j+1].Distance < array[j].Distance {
				array[j], array[j+1] = array[j+1], array[j]
			}
		}
	}

	if logger != nil {
		logger.Info("Ending bubble sort.")
	}
	return err
}

/*
returns the first index of the passed element within the passed slice or -1 if the element is not within the slice.
*/
func In[In comparable](slice []In, elem In) int {
	for i, e := range slice {
		if e == elem {
			return i
		}
	}
	return -1
}
