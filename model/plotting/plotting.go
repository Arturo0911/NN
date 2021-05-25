package plotting

import (
	"log"
	"os"

	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// HISTOGRAMS
func CreatePloting(pathFile string) error {
	file, err := os.Open(pathFile)
	if err != nil {
		return err
	}
	defer file.Close()

	readerDF := dataframe.ReadCSV(file)
	//fmt.Println("READING DATAFRAME")
	//fmt.Println(readerDF)

	/*
		[X0 fixed.acidity volatile.acidity
		citric.acid residual.sugar
		chlorides free.sulfur.dioxide
		total.sulfur.dioxide density
		pH sulphates alcohol quality]
	*/
	//fmt.Println(readerDF.Names())
	yVals := readerDF.Col("density").Float()

	for _, colNames := range readerDF.Names() {
		pts := make(plotter.XYs, readerDF.Nrow())

		for i, floatVal := range readerDF.Col(colNames).Float() {
			pts[i].X = floatVal
			pts[i].Y = yVals[i]
		}

		p := plot.New()

		p.X.Label.Text = colNames
		p.Y.Label.Text = "y"

		p.Add(plotter.NewGrid())

		s, err := plotter.NewScatter(pts)
		if err != nil {
			log.Fatal(err)
		}
		s.GlyphStyle.Radius = vg.Points(3)
		p.Add(s)

		if err := p.Save(8*vg.Inch, 8*vg.Inch, colNames+"_scatter.png"); err != nil {
			log.Fatal(err)
		}
	}

	return nil
}
