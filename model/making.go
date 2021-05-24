package model

import "github.com/Arturo0911/NN/model/settings"

func MainProcess(pathFile string) error {
	if err := settings.MakingFiles(pathFile); err != nil {
		return err
	}
	return nil

}
