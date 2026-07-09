// // Parallelisierung mittels Goroutines (eine abstrakte Schicht über den Betriebssystem-Threads, zur besseren Auslastung dieser)
package main

// import (
// 	"fmt"
// 	"runtime"

// 	//"time"
// 	"math"
// 	"sync"
// )

// func main() {
// 	runtime.GOMAXPROCS(2) //setzt die maximal erlaubte # an Betriebssystem-Threads
// 	// runtime.GOMAXPROCS(0) //getter für GOMAXPROCS
// 	sublen := 10
// 	arr := make([]int, 3333)
// 	for i := range len(arr) {
// 		arr[i] = i + 1
// 	}
// 	var wg sync.WaitGroup
// 	amount := int(math.Ceil(float64(len(arr)) / float64(sublen)))
// 	println("amount: ", amount)
// 	wg.Add(amount)

// 	for i := range len(arr) / sublen {
// 		go add(arr[i*sublen:i*sublen+sublen], 2, &wg)
// 	}
// 	if rest := len(arr) % sublen; rest != 0 {
// 		println("special case")
// 		go add(arr[len(arr)-rest:], 2, &wg)
// 	}

// 	wg.Wait()

// 	for i := range len(arr) {
// 		fmt.Printf("idx: %d\tWert: %d\n", i, arr[i])
// 	}
// }

// func add(arr []int, a int, wg *sync.WaitGroup) {
// 	for i := range len(arr) {
// 		fmt.Printf("%d -> %d\n", arr[i], arr[i]+a)
// 		arr[i] += a
// 	}
// 	defer wg.Done()
// }

// func say(s string, wg *sync.WaitGroup) {
// 	//time.Sleep(100 * time.Millisecond)
// 	fmt.Println(s)
// 	if wg != nil {
// 		defer wg.Done()
// 	}
// }
