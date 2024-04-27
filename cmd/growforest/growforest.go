package main

import (
	"ecg.mk/learn"
)

func main() {
	gf := learn.GrowForestParse()
	gf.Fit()
}
