package server

import (
	"fmt"
	"net/http"
)

func InitServer() {
	fmt.Println("Initializing server")
	http.HandleFunc("/neuron", func(w http.ResponseWriter, r *http.Request) {

	})
}
