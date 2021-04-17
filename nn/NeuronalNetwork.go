package nn

import (
	"fmt"
	"io/ioutil"
)

/*type Slice struct {
	Id        int
	attribute string
}*/

func LoadFile(pathName string) {

	fileReader, err := ioutil.ReadFile(pathName)

	if err != nil {
		panic(err)
	}

	fmt.Println(string(fileReader))
	//fmt.Println(reflect.ValueOf(fileReader).Kind())

}

func GradientDescent() int {
	return 0
}

// Metrics

func EuclidianDistance() int {
	return 0
}

func CosinDistance() int {
	return 0
}

func ManhattanDistance() int {
	return 0
}

func Greeting() {

	fmt.Println("Hi")
}
