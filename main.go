package main

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
)

// @author: Arturo Negreiros (AKA Pxyl0xd)

func main() {
	g := gorgonia.NewGraph()

	var x, y, z *gorgonia.Node
	var err error

	// Define the expression
	x = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
	if z, err = gorgonia.Add(x, y); err != nil {
		log.Fatal(err)
	}

	// Create a VM to run the program on
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	// Set initial alues then run
	gorgonia.Let(x, 2.0)
	gorgonia.Let(y, 2.5)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}
    fmt.Println("hi Arturo ")
	fmt.Printf("%v\n\n", z.Value())

}
