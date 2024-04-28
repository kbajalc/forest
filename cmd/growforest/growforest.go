package main

import (
	"flag"

	"ecg.mk/learn"
)

func main() {
	g := learn.GrowForestFlags("")
	flag.Parse()
	g.Fit()
}
