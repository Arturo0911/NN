package nn

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
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

func LoopFile(fileName string) ([]string, error) {

	var lines []string
	file, err := os.Open(fileName)

	if err != nil {

		return nil, err
	}

	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		lines = append(lines, line)
	}

	err = file.Close()

	if scanner.Err() != nil {
		return nil, scanner.Err()
	}

	return lines, err

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
