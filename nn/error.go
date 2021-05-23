package nn

import (
	"fmt"
	"strings"
)

type Errors []error

// Errors is a list error
// useful in a loop if you don't wanto to return the error right
// away and you want to display after the loop.
// all the errors that happened during the loop.
func (errorList Errors) Error() string {
	if len(errorList) < 1 {
		return ""
	}

	out := make([]string, len(errorList))
	for i := range errorList {
		out[i] = errorList[i].Error()
	}
	return strings.Join(out, ", ")
}

// StatusError reports an unsuccessful exit.
type StatusError struct {
	Status      string
	StatuscCode int
}

func (e StatusError) Error() string {
	return fmt.Sprintf("Status: %s, Code: %d", e.Status, e.StatuscCode)
}
