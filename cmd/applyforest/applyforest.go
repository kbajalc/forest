package main

import (
	"flag"

	"ecg.mk/rfx"
)

func main() {
	af := rfx.NewApplyForest()
	af.Mount()
	flag.Parse()
	af.Run()
}
