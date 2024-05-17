package rfx

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
)

type ApplyForest struct {
	MatrixFile string
	ForestFile string
	PredFile   string
	VotesFile  string

	UseNum   bool
	UseSum   bool
	UseExpit bool
	UseCat   bool
}

func NewApplyForest() *ApplyForest {
	return &ApplyForest{
		MatrixFile: "featurematrix.afm",
		ForestFile: "rface.sf",
		PredFile:   "",
		VotesFile:  "",

		UseNum:   false,
		UseSum:   false,
		UseExpit: false,
		UseCat:   false,
	}
}

func (af *ApplyForest) Mount() {
	flag.StringVar(&af.MatrixFile, "fm", af.MatrixFile, "AFM formated feature matrix containing data.")
	flag.StringVar(&af.ForestFile, "rfpred", af.ForestFile, "A predictor forest.")
	flag.StringVar(&af.PredFile, "preds", af.PredFile, "The name of a file to write the predictions into.")
	flag.StringVar(&af.VotesFile, "votes", af.VotesFile, "The name of a file to write categorical vote totals to.")

	flag.BoolVar(&af.UseNum, "mean", af.UseNum, "Force numeric (mean) voting.")
	flag.BoolVar(&af.UseSum, "sum", af.UseSum, "Force numeric sum voting (for gradient boosting etc).")
	flag.BoolVar(&af.UseExpit, "expit", af.UseExpit, "Expit (inverst logit) transform data (for gradient boosting classification).")
	flag.BoolVar(&af.UseCat, "mode", af.UseCat, "Force categorical (mode) voting.")
}

func (af *ApplyForest) Run() {
	//Parse Data
	data, err := LoadAFM(af.MatrixFile)
	if err != nil {
		log.Fatal(err)
	}

	forestfile, err := os.Open(af.ForestFile) // For read access.
	if err != nil {
		log.Fatal(err)
	}
	defer forestfile.Close()
	forestreader := NewForestReader(forestfile)
	forest, err := forestreader.ReadForest()
	if err != nil {
		log.Fatal(err)
	}

	var predfile *os.File
	if af.PredFile != "" {
		predfile, err = os.Create(af.PredFile)
		if err != nil {
			log.Fatal(err)
		}
		defer predfile.Close()
	}

	var tally VoteTallyer
	switch {
	case af.UseSum:
		tally = NewSumBallotBox(data.Data[0].Length())

	case !af.UseCat && (af.UseNum || strings.HasPrefix(forest.Target, "N")):
		tally = NewNumBallotBox(data.Data[0].Length())

	default:
		tally = NewCatBallotBox(data.Data[0].Length())
	}

	// Trees execution
	for _, tree := range forest.Trees {
		tree.Vote(data, tally)
	}

	targeti, hasTarget := data.Map[forest.Target]
	if hasTarget {
		fmt.Printf("Target is %v in feature %v\n", forest.Target, targeti)
		er := tally.TallyError(data.Data[targeti])
		fmt.Printf("Error: %v\n", er)
	}

	if af.PredFile != "" {
		fmt.Printf("Outputting label predicted actual tsv to %v\n", af.PredFile)
		for i, l := range data.Cases {
			actual := "NA"
			if hasTarget {
				actual = data.Data[targeti].GetStr(i)
			}

			result := ""
			if af.UseSum || forest.Intercept != 0.0 {
				numresult := 0.0
				if af.UseSum {
					numresult = tally.(*SumBallotBox).TallyNum(i) + forest.Intercept
				} else {
					numresult = tally.(*NumBallotBox).TallyNum(i) + forest.Intercept
				}
				if af.UseExpit {
					numresult = Expit(numresult)
				}
				result = fmt.Sprintf("%v", numresult)
			} else {
				result = tally.Tally(i)
			}
			fmt.Fprintf(predfile, "%v\t%v\t%v\n", l, result, actual)
		}
	}

	// Not thread safe code!
	if af.VotesFile != "" {
		fmt.Printf("Outputting vote totals to %v\n", af.VotesFile)
		cbb := tally.(*CatBallotBox)
		votefile, err := os.Create(af.VotesFile)
		if err != nil {
			log.Fatal(err)
		}
		defer votefile.Close()
		fmt.Fprintf(votefile, ".")

		for _, lable := range cbb.CatMap.Back {
			fmt.Fprintf(votefile, "\t%v", lable)
		}
		fmt.Fprintf(votefile, "\n")

		for i, box := range cbb.Box {
			fmt.Fprintf(votefile, "%v", data.Cases[i])
			for j, _ := range cbb.CatMap.Back {
				total := 0.0
				total = box.Map[j]
				fmt.Fprintf(votefile, "\t%v", total)
			}
			fmt.Fprintf(votefile, "\n")
		}
	}
}
