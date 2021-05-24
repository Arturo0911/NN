package plotting

import (
	"fmt"
	"os"

	"github.com/go-gota/gota/dataframe"
)

// HISTOGRAMS
func CreatePloting(pathFile string) error {
	file, err := os.Open(pathFile)
	if err != nil {
		return err
	}
	defer file.Close()

	readerDF := dataframe.ReadCSV(file)
	fmt.Println("READING DATAFRAME")
	fmt.Println(readerDF.Describe())

	return nil
}
