package main

import (
	"flag"

	"ecg.mk/rfx"
)

func main() {
	g := rfx.NewGrowForest()
	g.Mount("")
	flag.Parse()
	g.Fit()
}
