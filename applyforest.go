package rfx

import (
	"flag"
	"fmt"
	"log"
	"os"
	"slices"
	"strings"
	"sync"
)

type ApplyForest struct {
	DataFile   string
	ForestFile string
	PredFile   string
	VotesFile  string

	Progress bool

	UseNum   bool
	UseSum   bool
	UseExpit bool
	UseCat   bool
}

func NewApplyForest() *ApplyForest {
	return &ApplyForest{
		DataFile:   "featurematrix.afm",
		ForestFile: "rface.sf",
		PredFile:   "",
		VotesFile:  "",

		Progress: false,

		UseNum:   false,
		UseSum:   false,
		UseExpit: false,
		UseCat:   false,
	}
}

func (af *ApplyForest) Clone() *ApplyForest {
	return &ApplyForest{
		DataFile:   af.DataFile,
		ForestFile: af.ForestFile,
		PredFile:   af.PredFile,
		VotesFile:  af.VotesFile,

		Progress: af.Progress,

		UseNum:   af.UseNum,
		UseSum:   af.UseSum,
		UseExpit: af.UseExpit,
		UseCat:   af.UseCat,
	}
}

func (af *ApplyForest) Mount(pref string) {
	flag.StringVar(&af.DataFile, pref+"fm", af.DataFile, "AFM formated feature matrix containing data.")
	flag.StringVar(&af.ForestFile, pref+"rfpred", af.ForestFile, "A predictor forest.")
	flag.StringVar(&af.PredFile, pref+"preds", af.PredFile, "The name of a file to write the predictions into.")
	flag.StringVar(&af.VotesFile, pref+"votes", af.VotesFile, "The name of a file to write categorical vote totals to.")

	// progress bool
	flag.BoolVar(&af.Progress, pref+"progress", af.Progress, "Report progress.")

	flag.BoolVar(&af.UseNum, pref+"mean", af.UseNum, "Force numeric (mean) voting.")
	flag.BoolVar(&af.UseSum, pref+"sum", af.UseSum, "Force numeric sum voting (for gradient boosting etc).")
	flag.BoolVar(&af.UseExpit, pref+"expit", af.UseExpit, "Expit (inverst logit) transform data (for gradient boosting classification).")
	flag.BoolVar(&af.UseCat, pref+"mode", af.UseCat, "Force categorical (mode) voting.")
}

func (af *ApplyForest) Run() {
	fmt.Println("--------------------------------------------------------------------")

	//Parse Data
	fmt.Println("> Data: ", af.DataFile)
	data, err := LoadAFM(af.DataFile)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("> Forest: ", af.ForestFile)
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

	targeti, hasTarget := data.Map[forest.Target]
	if hasTarget {
		fmt.Printf("< Target is %v in feature %v\n", forest.Target, targeti)
	}

	fmt.Println("--------------------------------------------------------------------")

	var tally VoteTallyer
	switch {
	case af.UseSum:
		tally = NewSumBallotBox(data.Data[0].Length())
		fmt.Printf("@ Performing numeric sum vote with %v cases.\n", data.Data[0].Length())
	case !af.UseCat && (af.UseNum || strings.HasPrefix(forest.Target, "N")):
		tally = NewNumBallotBox(data.Data[0].Length())
		fmt.Printf("@ Performing numeric mean vote with %v cases.\n", data.Data[0].Length())
	default:
		//classification
		tally = NewCatBallotBox(data.Data[0].Length())
		fmt.Printf("@ Performing categorical (mode) vote %v cases.\n", data.Data[0].Length())
	}

	// Trees execution
	var wg sync.WaitGroup
	for i, tree := range forest.Trees {
		wg.Add(1)
		go func(i int, tree *Tree) {
			tree.Vote(data, tally)
			wg.Done()
			if af.Progress {
				fmt.Printf("%% Processed tree: %v\n", i+1)
			}
		}(i, tree)
	}
	wg.Wait()

	if af.Progress {
		fmt.Println("--------------------------------------------------------------------")
	}

	if af.PredFile != "" {
		fmt.Printf("> Outputting label predicted actual to: %v\n", af.PredFile)
		fmt.Fprintf(predfile, "%v\t%v\t%v\n", ".", "Result", "Actual")
		for i, index := range data.Cases {
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
			fmt.Fprintf(predfile, "%v\t%v\t%v\n", index, result, actual)
		}
	}

	// Not thread safe code!
	if af.VotesFile != "" {
		fmt.Printf("> Outputting vote totals to: %v\n", af.VotesFile)
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
			for j := range cbb.CatMap.Back {
				total := 0.0
				total = box.Map[j]
				fmt.Fprintf(votefile, "\t%v", total)
			}
			fmt.Fprintf(votefile, "\n")
		}
	}

	if hasTarget {
		fmt.Println()
		er := tally.TallyError(data.Data[targeti])
		fmt.Printf("= Error: %v\n", er)
	}

	var tester = data.Data[targeti]
	if hasTarget && tester.NCats() != 0 {
		reftotals := map[string]int{}
		predtotals := map[string]int{}

		true_pos := map[string]int{}
		false_pos := map[string]int{}
		false_neg := map[string]int{}

		conf := map[string]map[string]int{}

		correct := 0
		errors := 0
		nas := 0
		length := tester.Length()
		for i := 0; i < length; i++ {
			// refi := tester.(*DenseCatFeature).Geti(i)
			pred := tally.Tally(i)
			orig := tester.GetStr(i)

			reftotals[orig]++

			if _, ok := conf[orig]; !ok {
				conf[orig] = map[string]int{}
			}
			if _, ok := conf[pred]; !ok {
				conf[pred] = map[string]int{}
			}

			if _, ok := conf[pred][orig]; !ok {
				conf[pred][orig] = 0
			}

			conf[pred][orig]++

			if pred == "NA" {
				nas++
				continue
			}

			// predi := tester.(*DenseCatFeature).CatToNum(pred)
			predtotals[pred]++
			if pred == orig {
				correct++
				true_pos[orig]++
			} else {
				errors++
				false_pos[pred]++
				false_neg[orig]++
			}
		}

		fmt.Printf("= Classified: %v / %v = %.3f\n", correct, length, float64(correct)*100/float64(length))
		if nas != 0 {
			fmt.Printf("= Couldn't predict %v cases due to missing values.\n", nas)
		}

		// fmt.Println()
		// for i, v := range testtarget.(*DenseCatFeature).Back {
		// 	fmt.Printf("- Label [%v] Percision (Actuall/Predicted): %v / %v = %.3f\n", v, false_pos[i], predtotals[i], float64(false_pos[i])*100/float64(predtotals[i]))
		// 	falses := reftotals[i] - true_pos[i]
		// 	fmt.Printf("- Label [%v] Missed/Actuall Rate: %v / %v = %.3f\n", v, falses, reftotals[i], float64(falses)*100/float64(reftotals[i]))
		// }

		fmt.Println()

		cts := []string{}
		for _, k := range tester.(*DenseCatFeature).Back {
			cts = append(cts, k)
		}
		slices.Sort(cts)

		err_pos := 0
		err_neg := 0

		fmt.Println("--------------------------------------------------------------------")
		fmt.Println("CAT           TP      FP      FN      ER        SE       PP       F1")
		fmt.Println("--------------------------------------------------------------------")
		for _, i := range cts {
			se := float64(true_pos[i]) / float64(true_pos[i]+false_pos[i])
			pp := float64(true_pos[i]) / float64(true_pos[i]+false_neg[i])
			f1 := 2 * se * pp / (se + pp)
			err_pos += false_pos[i]
			err_neg += false_neg[i]

			fmt.Printf("%-8v  %6d  %6d  %6d  %6d   %7.3f  %7.3f  %7.3f\n",
				i, true_pos[i], false_pos[i], false_neg[i], false_pos[i]+false_neg[i], se*100, pp*100, f1*100)
		}
		fmt.Println("--------------------------------------------------------------------")
		se := float64(correct) / float64(correct+err_pos)
		pp := float64(correct) / float64(correct+err_neg)
		f1 := 2 * se * pp / (se + pp)
		fmt.Printf("%-8v  %6d  %6d  %6d  %6d   %7.3f  %7.3f  %7.3f\n",
			"ALL", correct, err_pos, err_neg, errors, se*100, pp*100, f1*100)
		fmt.Println("--------------------------------------------------------------------")

		fmt.Println()

		fmt.Printf("%6s", "")
		for _, o := range cts {
			fmt.Printf("%8s", o)
		}
		fmt.Printf("%10s", "count")
		fmt.Printf("%10s\n", "c.err.")
		for _, p := range cts {
			fmt.Printf("%6s", p)
			s := 0
			e := 0
			for _, o := range cts {
				v := conf[p][o]
				s += v
				if p != o {
					e += v
				}
				fmt.Printf("%8d", v)
			}
			fmt.Printf("%10d%10.6f\n", s, float64(e)/float64(s))
		}

		fmt.Println()
	}
}
