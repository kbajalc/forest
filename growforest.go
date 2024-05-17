package rfx

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"regexp"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"time"
)

type GrowForest struct {
	// ========================================================================================================
	// GENERAL OPTIONS

	// nCores: The number of cores to use.
	Cores int

	// oob: Calculate and report OOB error.
	OOB bool

	// ace: Number Ace permutations to do. Output Ace style importance and p values.
	Ace int

	// cutoff: P-value Cutoff to apply to features for last forest after ACE.
	Cutoff float64

	// progress: Report tree number and running oob error.
	Progress bool

	// multiboost: Allow multi-threaded boosting which may have unexpected results. (highly experimental)
	Multiboost bool

	// noseed: Don't seed the random number generator from time.
	NoSeed bool

	// ========================================================================================================
	// FEATURE MATRIX

	// train: AFM formated feature matrix containing training data.
	TrainFile string

	// target: The row header of the target in the feature matrix.
	TargetName string

	// nContrasts: The number of randomized artificial contrast features to include in the feature matrix.
	Contrasts int

	// Contrastall: Include a shuffled artificial contrast copy of every feature.
	ContrastAll bool

	// blacklist: A list of feature id's to exclude from the set of predictors.
	Blacklist string

	// blockRE: A regular expression to identify features that should be filtered out.
	BlockRE string

	// includeRE: Filter features that DON'T match this RE.
	IncludeRE string

	// impute: Impute missing values to feature mean/mode before growth.
	Impute bool

	// permute: Permute the target feature (to establish random predictive power).
	Permute bool

	// shuffleRE: A regular expression to identify features that should be shuffled.
	ShuffleRE string

	// balance: Balance bagging of samples by target class for unbalanced classification.
	Balance bool

	// balanceby: Roughly balanced bag the target within each class of this feature.
	BalanceBy string

	// ========================================================================================================
	// FOREST OPTIONS

	// nTrees: 100, Number of trees to grow in the predictor.
	Trees int

	// jungle: Grow unserializable and experimental decision jungle with node recombination.
	Jungle bool

	// nobag: Don't bag samples for each tree.
	NoBag bool

	// mTry: Number of candidate features for each split as a count (ex: 10) or portion of total (ex: .5). Ceil(sqrt(nFeatures)) if <=0.
	MTry string

	// leafSize: The minimum number of cases on a leaf node. If <=0 will be inferred to 1 for classification 4 for regression.
	LeafSize string

	// nSamples: The number of cases to sample (with replacement) for each tree as a count (ex: 10) or portion of total (ex: .5). If <=0 set to total number of cases.
	NSamples string

	// maxDepth: Maximum tree depth. Ignored if 0.
	MaxDepth int

	// splitmissing: Split missing values onto a third branch at each node (experimental).
	SplitMissing bool

	// force: Force at least one non constant feature to be tested for each split.
	Force bool

	// vet: Penalize potential splitter impurity decrease by subtracting the best split of a permuted target.
	Vet bool

	// evaloob: Evaluate potential splitting features on OOB cases after finding split value in bag.
	EvalOOB bool

	// extra: Grow Extra Random Trees (supports learning from numerical variables only).
	Extra bool

	// ========================================================================================================
	// DENSITY

	// density: Build density estimating trees instead of classification/regression trees.
	Density bool

	// ========================================================================================================
	// REGRESSION

	// l1: Use l1 norm regression (target must be numeric).
	UseL1 bool

	// ordinal: Use ordinal regression (target must be numeric).
	UseOrdinal bool

	// ========================================================================================================
	// CLASIFICATION

	// NP: Use approximate Neyman-Pearson classification.
	UseNP bool

	// NP_pos: 1, Class label to constrain percision in NP classification.
	NP_pos string

	// NP_a: 0.1, Constraint on percision in NP classification [0,1]
	NP_a float64

	// NP_k: 100, Weight of constraint in NP classification [0,Inf+)
	NP_k float64

	// cost: For categorical targets, a json string to float map of the cost of falsely identifying each category.
	UseCosts string

	// entropy: Use entropy minimizing classification (target must be categorical).
	UseEntropy bool

	// dentropy: Class disutilities for disutility entropy.
	UseDentropy string

	// rfweights: For categorical targets, a json string to float map of the weights to use for each category in Weighted RF.
	UseRfWeights string

	// adacost: Json costs for cost sentive AdaBoost.
	UseAdaCosts string

	// adaboost: Use Adaptive boosting for regression/classification.
	UseAdaBoost bool

	// hellinger: Build trees using UseHellinger distance.
	UseHellinger bool

	// positive: true, Positive class to output probabilities for.
	Positive string

	// gbt: Use gradient boosting with the specified learning rate.
	UseGradBoost float64

	// trans-unlabeled: Class to treat as TransUnlabeled for transduction forests.
	TransUnlabeled string

	// trans-alpha: 10.0, Weight of unsupervised term in transduction impurity.
	TransAlpha float64

	// trans-beta: Multiple to penalize unlabeled class by.
	TransBeta float64

	// ========================================================================================================
	// TESTING

	// selftest: Test the forest on the data and report accuracy.
	SelfTest bool

	// test: Data to test the model on.
	TestFile string

	// oobpreds: Calculate and report oob predictions in the file specified.
	CaseOOB string

	// ========================================================================================================
	// OUTOUT

	// rfpred: File name to output predictor forest in sf format.
	ForestFile string

	// importance: File name to output importance.
	Importance string

	// cpuprofile: write cpu profile to file
	CpuProfile string

	// scikitforest: Write out a (partially complete) scikit style forest in json.
	ScikitForest string
}

func NewGrowForest() *GrowForest {
	g := GrowForest{
		Cores:      1,
		Trees:      100,
		TransAlpha: 10.0,
		Positive:   "True",
		NP_pos:     "1",
		NP_a:       0.1,
		NP_k:       100,
	}
	return &g
}

func (g *GrowForest) Apply(train string, target string, test string) *GrowForest {
	g.TrainFile = train
	g.TargetName = target
	g.TestFile = test

	if g.TestFile != "" {
		g.SelfTest = true
	}

	pref := strings.TrimSuffix(train, ".tsv")
	pref = strings.TrimSuffix(pref, ".fm")

	g.Cores = runtime.NumCPU()
	g.OOB = true
	g.ForestFile = pref + ".model.sf"
	g.Importance = pref + ".imp.tsv"
	g.CaseOOB = pref + ".oob.tsv"

	return g
}

func (gf *GrowForest) Clone() *GrowForest {
	copy := *gf
	return &copy
}

func (gf *GrowForest) Mount(px string) {
	// ========================================================================================================
	// GENERAL OPTIONS

	// nCores int
	flag.IntVar(&gf.Cores, px+"cores", gf.Cores, "The number of cores to use.")

	// oob bool
	flag.BoolVar(&gf.OOB, px+"oob", gf.OOB, "Calculate and report oob error.")

	// ace int
	flag.IntVar(&gf.Ace, px+"ace", gf.Ace, "Number ace permutations to do. Output ace style importance and p values.")

	// cutoff float64
	flag.Float64Var(&gf.Cutoff, px+"cutoff", gf.Cutoff, "P-value cutoff to apply to features for last forest after ACE.")

	// progress bool
	flag.BoolVar(&gf.Progress, px+"progress", gf.Progress, "Report tree number and running oob error.")

	// multiboost bool
	flag.BoolVar(&gf.Multiboost, px+"multiboost", gf.Multiboost, "Allow multi-threaded boosting which may have unexpected results. (highly experimental)")

	// noseed bool
	flag.BoolVar(&gf.NoSeed, px+"noseed", gf.NoSeed, "Don't seed the random number generator from time.")

	// ========================================================================================================
	// FEATURE MATRIX

	// fm string
	flag.StringVar(&gf.TrainFile, px+"train", gf.TrainFile, "AFM formated feature matrix containing training data.")

	// targetname string
	flag.StringVar(&gf.TargetName, px+"target", gf.TargetName, "The row header of the target in the feature matrix.")

	// nContrasts int
	flag.IntVar(&gf.Contrasts, px+"contrasts", gf.Contrasts, "The number of randomized artificial contrast features to include in the feature matrix.")

	// contrastAll bool
	flag.BoolVar(&gf.ContrastAll, px+"contrastall", gf.ContrastAll, "Include a shuffled artificial contrast copy of every feature.")

	// blacklist string
	flag.StringVar(&gf.Blacklist, px+"blacklist", gf.Blacklist, "A list of feature id's to exclude from the set of predictors.")

	// blockRE string
	flag.StringVar(&gf.BlockRE, px+"blockRE", gf.BlockRE, "A regular expression to identify features that should be filtered out.")

	// includeRE string
	flag.StringVar(&gf.IncludeRE, px+"includeRE", gf.IncludeRE, "Filter features that DON'T match this RE.")

	// impute bool
	flag.BoolVar(&gf.Impute, px+"impute", gf.Impute, "Impute missing values to feature mean/mode before growth.")

	// permutate bool
	flag.BoolVar(&gf.Permute, px+"permute", gf.Permute, "Permute the target feature (to establish random predictive power).")

	// shuffleRE string
	flag.StringVar(&gf.ShuffleRE, px+"shuffleRE", gf.ShuffleRE, "A regular expression to identify features that should be shuffled.")

	// balance bool
	flag.BoolVar(&gf.Balance, px+"balance", gf.Balance, "Balance bagging of samples by target class for unbalanced classification.")

	// balanceby string
	flag.StringVar(&gf.BalanceBy, px+"balanceby", gf.BalanceBy, "Roughly balanced bag the target within each class of this feature.")

	// ========================================================================================================
	// FOREST OPTIONS

	// nTrees int
	flag.IntVar(&gf.Trees, px+"trees", gf.Trees, "Number of trees to grow in the predictor.")

	// jungle bool
	flag.BoolVar(&gf.Jungle, px+"jungle", gf.Jungle, "Grow unserializable and experimental decision jungle with node recombination.")

	// nobag bool
	flag.BoolVar(&gf.NoBag, px+"nobag", gf.NoBag, "Don't bag samples for each tree.")

	// StringmTry string
	flag.StringVar(&gf.MTry, px+"mtry", gf.MTry, "Number of candidate features for each split as a count (ex: 10) or portion of total (ex: .5). Ceil(sqrt(nFeatures)) if <=0.")

	// StringleafSize string
	flag.StringVar(&gf.LeafSize, px+"leaf-size", gf.LeafSize, "The minimum number of cases on a leaf node. If <=0 will be inferred to 1 for classification 4 for regression.")

	// StringnSamples string
	flag.StringVar(&gf.NSamples, px+"nsamples", gf.NSamples, "The number of cases to sample (with replacement) for each tree as a count (ex: 10) or portion of total (ex: .5). If <=0 set to total number of cases.")

	// maxDepth int
	flag.IntVar(&gf.MaxDepth, px+"max-depth", gf.MaxDepth, "Maximum tree depth. Ignored if 0.")

	// splitmissing bool
	flag.BoolVar(&gf.SplitMissing, px+"split-missing", gf.SplitMissing, "Split missing values onto a third branch at each node (experimental).")

	// force bool
	flag.BoolVar(&gf.Force, px+"force", gf.Force, "Force at least one non constant feature to be tested for each split.")

	// vet bool
	flag.BoolVar(&gf.Vet, px+"vet", gf.Vet, "Penalize potential splitter impurity decrease by subtracting the best split of a permuted target.")

	// evaloob bool
	flag.BoolVar(&gf.EvalOOB, px+"evaloob", gf.EvalOOB, "Evaluate potential splitting features on OOB cases after finding split value in bag.")

	// extra bool
	flag.BoolVar(&gf.Extra, px+"extra", gf.Extra, "Grow Extra Random Trees (supports learning from numerical variables only).")

	// ========================================================================================================
	// DENSITY

	flag.BoolVar(&gf.Density, px+"density", gf.Density, "Build density estimating trees instead of classification/regression trees.")

	// ========================================================================================================
	// REGRESSION

	// - Using l1/absolute deviance error.
	flag.BoolVar(&gf.UseL1, px+"l1", gf.UseL1, "Use l1 norm regression (target must be numeric).")

	// ordinal bool
	flag.BoolVar(&gf.UseOrdinal, px+"ordinal", gf.UseOrdinal, "Use ordinal regression (target must be numeric).")

	// ========================================================================================================
	// CLASIFICATION

	// NP bool
	flag.BoolVar(&gf.UseNP, px+"NP", gf.UseNP, "Use approximate Neyman-Pearson classification.")

	// NP_pos string
	flag.StringVar(&gf.NP_pos, px+"NP-pos", gf.NP_pos, "Class label to constrain percision in NP classification.")

	// NP_a float64
	flag.Float64Var(&gf.NP_a, px+"NP-a", gf.NP_a, "Constraint on percision in NP classification [0,1]")

	// NP_k float64
	flag.Float64Var(&gf.NP_k, px+"NP-k", gf.NP_k, "Weight of constraint in NP classification [0,Inf+)")

	// entropy bool
	flag.BoolVar(&gf.UseEntropy, px+"entropy", gf.UseEntropy, "Use entropy minimizing classification (target must be categorical).")

	// costs string
	flag.StringVar(&gf.UseCosts, px+"cost", gf.UseCosts, "For categorical targets, a json string to float map of the cost of falsely identifying each category.")

	// dentropy string
	flag.StringVar(&gf.UseDentropy, px+"dentropy", gf.UseDentropy, "Class disutilities for disutility entropy.")

	// rfweights string
	flag.StringVar(&gf.UseRfWeights, px+"rfweights", gf.UseRfWeights, "For categorical targets, a json string to float map of the weights to use for each category in Weighted RF.")

	// adacosts string
	flag.StringVar(&gf.UseAdaCosts, px+"adacost", gf.UseAdaCosts, "Json costs for cost sentive AdaBoost.")

	// adaboost bool
	flag.BoolVar(&gf.UseAdaBoost, px+"adaboost", gf.UseAdaBoost, "Use Adaptive boosting for regression/classification.")

	// hellinger bool
	flag.BoolVar(&gf.UseHellinger, px+"hellinger", gf.UseHellinger, "Build trees using hellinger distance.")

	// positive string
	flag.StringVar(&gf.Positive, px+"positive", gf.Positive, "Positive class to output probabilities for.")

	// gradboost float64
	flag.Float64Var(&gf.UseGradBoost, px+"gbt", gf.UseGradBoost, "Use gradient boosting with the specified learning rate.")

	// unlabeled string
	flag.StringVar(&gf.TransUnlabeled, px+"trans-unlabeled", gf.TransUnlabeled, "Class to treat as unlabeled for transduction forests.")

	// trans_alpha float64
	flag.Float64Var(&gf.TransAlpha, px+"trans-alpha", gf.TransAlpha, "Weight of unsupervised term in transduction impurity.")

	// trans_beta float64
	flag.Float64Var(&gf.TransBeta, px+"trans-beta", gf.TransBeta, "Multiple to penalize unlabeled class by.")

	// ========================================================================================================
	// TESTING

	// testfm string
	flag.StringVar(&gf.TestFile, px+"test", gf.TestFile, "Data to test the model on.")

	// dotest bool
	flag.BoolVar(&gf.SelfTest, px+"selftest", gf.SelfTest, "Test the forest on the data and report accuracy.")

	// caseoob string
	flag.StringVar(&gf.CaseOOB, px+"oobpreds", gf.CaseOOB, "Calculate and report oob predictions in the file specified.")

	// ========================================================================================================
	// OUTOUT

	// rf string
	flag.StringVar(&gf.ForestFile, px+"save", gf.ForestFile, "File name to output predictor forest in sf format.")
	flag.StringVar(&gf.ForestFile, px+"rfpred", gf.ForestFile, "File name to output predictor forest in sf format.")

	// imp string
	flag.StringVar(&gf.Importance, px+"importance", gf.Importance, "File name to output importance.")

	// cpuprofile string
	flag.StringVar(&gf.CpuProfile, px+"cpuprofile", gf.CpuProfile, "write cpu profile to file")

	// scikitforest string
	flag.StringVar(&gf.ScikitForest, px+"scikitforest", gf.ScikitForest, "Write out a (partially complete) scikit style forest in json.")
}

func (g *GrowForest) Fit() {
	nForest := 1

	fmt.Println("--------------------------------------------------------------------")

	if !g.NoSeed {
		rand.Seed(time.Now().UTC().UnixNano())
	}

	if g.TestFile != "" {
		g.SelfTest = true
	}

	if g.Multiboost {
		fmt.Println("! MULTIBOOST!!!!1!!!!1!!11 (things may break).")
	}
	var boostMutex sync.Mutex
	boost := (g.UseAdaBoost || g.UseGradBoost != 0.0)
	if boost && !g.Multiboost {
		g.Cores = 1
	}

	// if nCores > 1 {
	// 	runtime.GOMAXPROCS(nCores)
	// }

	fmt.Printf("! Cores: %v\n", g.Cores)
	fmt.Printf("! Trees: %v\n", g.Trees)

	if g.CpuProfile != "" {
		fmt.Printf("> CPU Profile: %v\n", g.CpuProfile)
		f, err := os.Create(g.CpuProfile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	//Parse Data
	fmt.Printf("< Data: %v\n", g.TrainFile)
	data, err := LoadAFM(g.TrainFile)
	if err != nil {
		log.Fatal(err)
	}

	if g.Contrasts > 0 {
		fmt.Printf("- Adding %v Random Contrasts\n", g.Contrasts)
		data.AddContrasts(g.Contrasts)
	}
	if g.ContrastAll {
		fmt.Printf("- Adding Random Contrasts for All Features.\n")
		data.ContrastAll()
	}

	blacklisted := 0
	blacklistis := make([]bool, len(data.Data))
	if g.Blacklist != "" {
		fmt.Printf("- Blacklist: %v\n", g.Blacklist)
		blackfile, err := os.Open(g.Blacklist)
		if err != nil {
			log.Fatal(err)
		}
		tsv := csv.NewReader(blackfile)
		tsv.Comma = '\t'
		for {
			id, err := tsv.Read()
			if err == io.EOF {
				break
			} else if err != nil {
				log.Fatal(err)
			}
			i, ok := data.Map[id[0]]
			if !ok {
				fmt.Printf("- Ignoring blacklist feature not found in data: %v\n", id[0])
				continue
			}
			if !blacklistis[i] {
				blacklisted += 1
				blacklistis[i] = true
			}
		}
		blackfile.Close()
	}

	//find the target feature
	fmt.Printf("< Target: %v\n", g.TargetName)
	targeti, ok := data.Map[g.TargetName]
	if !ok {
		log.Fatal("Target not found in data.")
	}

	// Blacklist all T that are not target
	for i, feature := range data.Data {
		if targeti != i && strings.HasPrefix(feature.GetName(), "T:") {
			if !blacklistis[i] {
				blacklisted += 1
				blacklistis[i] = true
			}
		}
	}

	if g.BlockRE != "" {
		re := regexp.MustCompile(g.BlockRE)
		for i, feature := range data.Data {
			if targeti != i && re.MatchString(feature.GetName()) {
				if !blacklistis[i] {
					blacklisted += 1
					blacklistis[i] = true
				}
			}
		}
	}

	if g.IncludeRE != "" {
		re := regexp.MustCompile(g.IncludeRE)
		for i, feature := range data.Data {
			if targeti != i && !re.MatchString(feature.GetName()) {
				if !blacklistis[i] {
					blacklisted += 1
					blacklistis[i] = true
				}
			}
		}
	}

	nFeatures := len(data.Data) - blacklisted - 1
	fmt.Printf("< Features: %v\n", nFeatures)

	if g.Impute {
		fmt.Println("- Imputing missing values to feature mean/mode.")
		data.ImputeMissing()
	}

	if g.Permute {
		fmt.Println("- Permuting target feature.")
		data.Data[targeti].Shuffle()
	}

	if g.ShuffleRE != "" {
		re := regexp.MustCompile(g.ShuffleRE)
		shuffled := 0
		for i, feature := range data.Data {
			if targeti != i && re.MatchString(feature.GetName()) {
				data.Data[i].Shuffle()
				shuffled += 1

			}
		}
		fmt.Printf("- Shuffled %v features matching %v\n", shuffled, g.ShuffleRE)
	}

	targetf := data.Data[targeti]
	unboostedTarget := targetf.Copy()

	var bSampler Bagger
	if g.Balance {
		bSampler = NewBalancedSampler(targetf.(*DenseCatFeature))
	}

	if g.BalanceBy != "" {
		bSampler = NewSecondaryBalancedSampler(targetf.(*DenseCatFeature), data.Data[data.Map[g.BalanceBy]].(*DenseCatFeature))
		g.Balance = true
	}

	nNonMissing := 0

	for i := 0; i < targetf.Length(); i++ {
		if !targetf.IsMissing(i) {
			nNonMissing += 1
		}
	}
	fmt.Printf("< Non-Missing cases: %v\n", nNonMissing)

	mTry := ParseAsIntOrFractionOfTotal(g.MTry, nFeatures)
	if mTry <= 0 {
		mTry = int(math.Ceil(math.Sqrt(float64(nFeatures))))
	}
	fmt.Printf("< mTry: %v\n", mTry)

	leafSize := ParseAsIntOrFractionOfTotal(g.LeafSize, nNonMissing)
	if leafSize <= 0 {
		if boost {
			leafSize = nNonMissing / 3
		} else if targetf.NCats() == 0 {
			//regression
			leafSize = 4
		} else {
			//classification
			leafSize = 1
		}
	}
	fmt.Printf("< Leaf size: %v\n", leafSize)

	//infer nSamples and mTry from data if they are 0
	nSamples := ParseAsIntOrFractionOfTotal(g.NSamples, nNonMissing)
	if nSamples <= 0 {
		nSamples = nNonMissing
	}
	fmt.Printf("< nSamples: %v\n", nSamples)

	if g.Progress {
		g.OOB = true
	}
	if g.CaseOOB != "" {
		g.OOB = true
	}

	fmt.Println("--------------------------------------------------------------------")

	//****** Set up Target for Alternative Impurity  if needed *******//
	var target Target
	if g.Density {
		fmt.Println("@ Estimating Density.")
		target = &DensityTarget{&data.Data, nNonMissing}
	} else {
		switch tf := targetf.(type) {
		case NumFeature:
			fmt.Println("@ Performing regression.")
			if g.UseL1 {
				fmt.Println("- Using l1/absolute deviance error.")
				targetf = &L1Target{tf}
			}
			if g.UseOrdinal {
				fmt.Println("- Using Ordinal (mode) prediction.")
				targetf = NewOrdinalTarget(tf)
			}
			switch {
			case g.UseGradBoost != 0.0:
				fmt.Println("- Using Gradient Boosting.")
				targetf = NewGradBoostTarget(tf, g.UseGradBoost)

			case g.UseAdaBoost:
				fmt.Println("- Using Numeric Adaptive Boosting.")
				targetf = NewNumAdaBoostTarget(tf)
			}
			target = targetf

		case CatFeature:
			fmt.Printf("@ Performing classification with %v categories.\n", targetf.NCats())
			switch {
			case g.UseNP:
				fmt.Printf("- Performing Approximate Neyman-Pearson Classification with constrained false \"%v\".\n", g.NP_pos)
				fmt.Printf("- False %v constraint: %v, constraint weight: %v.\n", g.NP_pos, g.NP_a, g.NP_k)
				targetf = NewNPTarget(tf, g.NP_pos, g.NP_a, g.NP_k)

			case g.UseCosts != "":
				fmt.Println("- Using misclassification costs: ", g.UseCosts)
				costmap := make(map[string]float64)
				err := json.Unmarshal([]byte(g.UseCosts), &costmap)
				if err != nil {
					log.Fatal(err)
				}

				regTarg := NewRegretTarget(tf)
				regTarg.SetCosts(costmap)
				targetf = regTarg

			case g.UseEntropy:
				fmt.Println("- Using entropy minimization.")
				targetf = &EntropyTarget{tf}

			case g.UseDentropy != "":
				fmt.Println("- Using entropy with disutilities: ", g.UseDentropy)
				costmap := make(map[string]float64)
				err := json.Unmarshal([]byte(g.UseDentropy), &costmap)
				if err != nil {
					log.Fatal(err)
				}

				deTarg := NewDEntropyTarget(tf)
				deTarg.SetCosts(costmap)
				targetf = deTarg

			case g.UseRfWeights != "":
				fmt.Println("- Using rf weights: ", g.UseRfWeights)
				weightmap := make(map[string]float64)
				err := json.Unmarshal([]byte(g.UseRfWeights), &weightmap)
				if err != nil {
					log.Fatal(err)
				}

				wrfTarget := NewWRFTarget(tf, weightmap)
				targetf = wrfTarget

			case g.UseAdaCosts != "":
				fmt.Println("- Using cost sensative AdaBoost costs: ", g.UseAdaCosts)
				costmap := make(map[string]float64)
				err := json.Unmarshal([]byte(g.UseAdaCosts), &costmap)
				if err != nil {
					log.Fatal(err)
				}

				actarget := NewAdaCostTarget(tf)
				actarget.SetCosts(costmap)
				targetf = actarget

			case g.UseAdaBoost:
				fmt.Println("- Using Adaptive Boosting.")
				targetf = NewAdaBoostTarget(tf)

			case g.UseHellinger:
				fmt.Println("- Using Hellinger Distance with postive class:", g.Positive)
				targetf = NewHDistanceTarget(tf, g.Positive)

			case g.UseGradBoost != 0.0:
				fmt.Println("- Using Gradient Boosting Classification with postive class:", g.Positive)
				targetf = NewGradBoostClassTarget(tf, g.UseGradBoost, g.Positive)

			}

			if g.TransUnlabeled != "" {
				fmt.Println("+ Using traduction forests with unlabeled class: ", g.TransUnlabeled)
				targetf = NewTransTarget(tf, &data.Data, g.TransUnlabeled, g.TransAlpha, g.TransBeta, nNonMissing)

			}
			target = targetf
		}
	}

	//****************** Setup For ACE ********************************//
	var aceImps [][]float64
	firstace := len(data.Data)

	if g.Ace > 0 {
		fmt.Printf("- Performing ACE analysis with %v forests/permutations.\n", g.Ace)

		data.ContrastAll()

		for i := 0; i < firstace; i++ {
			blacklistis = append(blacklistis, blacklistis[i])
		}
		blacklistis[targeti+firstace] = true

		aceImps = make([][]float64, len(data.Data))
		for i := 0; i < len(data.Data); i++ {
			aceImps[i] = make([]float64, g.Ace)
		}
		nForest = g.Ace
		if g.Cutoff > 0 {
			nForest++
		}
	}

	var forestwriter *ForestWriter
	if g.ForestFile != "" {
		fmt.Printf("> Model: %s\n", g.ForestFile)
		forestfile, err := os.Create(g.ForestFile)
		if err != nil {
			log.Fatal(err)
		}
		defer forestfile.Close()
		forestwriter = NewForestWriter(forestfile)
		switch t := target.(type) {
		case TargetWithIntercept:
			forestwriter.WriteForestHeader(0, g.TargetName, t.Intercept())
		}
	}

	//****************** Needed Collections and vars ******************//
	var trees []*Tree
	trees = make([]*Tree, 0, g.Trees)

	var imppnt *[]*RunningMean
	var mmdpnt *[]*RunningMean
	if g.Importance != "" {
		fmt.Println("- Recording Importance scores")

		imppnt = NewRunningMeans(len(data.Data))
		mmdpnt = NewRunningMeans(len(data.Data))
	} else if g.Ace > 0 {
		fmt.Println("- Recording Ace importance and p values")
		imppnt = NewRunningMeans(len(data.Data))
	}

	var scikikittrees []ScikitTree
	if g.ScikitForest != "" {
		scikikittrees = make([]ScikitTree, 0, g.Trees)
	}

	//****************** Good Stuff Stars Here ;) ******************//

	var oobVotes VoteTallyer
	if g.OOB {
		fmt.Println("- Recording OOB error")
		if targetf.NCats() == 0 {
			//regression
			oobVotes = NewNumBallotBox(data.Data[0].Length())
		} else {
			//classification
			oobVotes = NewCatBallotBox(data.Data[0].Length())
		}
	}

	trainingStart := time.Now()

	for foresti := 0; foresti < nForest; foresti++ {
		var recordingTree sync.Mutex
		var treesStarted int
		var treesFinished int
		var prevErr = 1.0
		treesStarted = g.Cores
		var waitGroup sync.WaitGroup

		waitGroup.Add(g.Cores)
		treechan := make(chan *Tree)
		//fmt.Println("forest ", foresti)
		//Grow a single forest on nCores
		for core := 0; core < g.Cores; core++ {
			grow := func() {
				weight := -1.0
				canidates := make([]int, 0, len(data.Data))
				for i := 0; i < len(data.Data); i++ {
					if i != targeti && !blacklistis[i] {
						canidates = append(canidates, i)
					}
				}

				tree := NewTree()
				tree.Target = g.TargetName
				cases := make([]int, 0, nNonMissing)
				oobcases := make([]int, 0, nNonMissing)

				if g.NoBag {
					for i := 0; i < nNonMissing; i++ {
						if !targetf.IsMissing(i) {
							cases = append(cases, i)
						}
					}
				}

				var depthUsed *[]int
				if mmdpnt != nil {
					du := make([]int, len(data.Data))
					depthUsed = &du
				}

				allocs := NewBestSplitAllocs(nSamples, targetf)
				for {
					nCases := data.Data[0].Length()
					//sample nCases case with replacement
					if !g.NoBag {
						cases = cases[0:0]

						if g.Balance {
							bSampler.Sample(&cases, nSamples)

						} else {
							for j := 0; len(cases) < nSamples; j++ {
								r := rand.Intn(nCases)
								if !targetf.IsMissing(r) {
									cases = append(cases, r)
								}
							}
						}

					}

					if g.NoBag && nSamples != nCases {
						cases = cases[0:0]
						for i := 0; i < nCases; i++ {
							if !targetf.IsMissing(i) {
								cases = append(cases, i)
							}
						}
						SampleFirstN(&cases, &cases, nSamples, 0)

					}

					if g.OOB || g.EvalOOB {
						ibcases := make([]bool, nCases)
						for _, v := range cases {
							ibcases[v] = true
						}
						oobcases = oobcases[0:0]
						for i, v := range ibcases {
							if !v {
								oobcases = append(oobcases, i)
							}
						}
					}

					if g.Jungle {
						tree.GrowJungle(data, target, cases, canidates, oobcases, mTry, leafSize, g.MaxDepth, g.SplitMissing, g.Force, g.Vet, g.EvalOOB, g.Extra, imppnt, depthUsed, allocs)

					} else {
						tree.Grow(data, target, cases, canidates, oobcases, mTry, leafSize, g.MaxDepth, g.SplitMissing, g.Force, g.Vet, g.EvalOOB, g.Extra, imppnt, depthUsed, allocs)
					}
					if mmdpnt != nil {
						for i, v := range *depthUsed {
							if v != 0 {
								(*mmdpnt)[i].Add(float64(v))
								(*depthUsed)[i] = 0
							}

						}
					}

					if boost {
						boostMutex.Lock()
						ls, ps := tree.Partition(data)
						weight = targetf.(BoostingTarget).Boost(ls, ps)
						boostMutex.Unlock()
						if weight == math.Inf(1) {
							fmt.Printf("- Boosting Reached Weight of %v\n", weight)
							close(treechan)
							break
						}

						tree.Weight = weight
					}

					if g.OOB && foresti == nForest-1 {
						tree.VoteCases(data, oobVotes, oobcases)
					}

					////////////// Lock mutext to ouput tree ////////
					if g.Cores > 1 {
						recordingTree.Lock()
					}

					if forestwriter != nil && foresti == nForest-1 {
						forestwriter.WriteTree(tree, treesFinished)
					}

					if g.ScikitForest != "" {
						skt := NewScikitTree(nFeatures)
						BuildScikitTree(0, tree.Root, skt)
						scikikittrees = append(scikikittrees, *skt)
					}

					if g.SelfTest && foresti == nForest-1 {
						trees = append(trees, tree)

						if treesStarted < g.Trees-1 {
							//newtree := new(Tree)
							tree = NewTree()
							tree.Target = g.TargetName
						}
					}
					if g.Progress {
						treesFinished++
						berr := oobVotes.TallyError(unboostedTarget)
						diff := prevErr - berr
						prevErr = berr
						fmt.Printf("%% OOB error after tree %4v :  %.4f  %+10.6f\n", treesFinished, berr, diff)
					}
					if treesStarted < g.Trees {
						treesStarted++
					} else {
						if g.Cores > 1 {
							recordingTree.Unlock()
							waitGroup.Done()
						}
						break

					}
					if g.Cores > 1 {
						recordingTree.Unlock()
					}
					//////// Unlock //////////////////////////
					// treechan <- tree
					// tree = <-treechan
				}
			}

			if g.Cores > 1 {
				go grow()
			} else {
				grow()
			}
		}
		if g.Cores > 1 {
			waitGroup.Wait()
		}
		// for i := 0; i < nTrees; i++ {
		// 	tree := <-treechan
		// 	if tree == nil {
		// 		break
		// 	}
		// 	if forestwriter != nil && foresti == nForest-1 {
		// 		forestwriter.WriteTree(tree, i)
		// 	}

		// 	if dotest && foresti == nForest-1 {
		// 		trees = append(trees, tree)

		// 		if i < nTrees-1 {
		// 			//newtree := new(Tree)
		// 			treechan <- NewTree()
		// 		}
		// 	} else {
		// 		if i < nTrees-1 {
		// 			treechan <- tree
		// 		}
		// 	}
		// 	if progress {
		// 		fmt.Printf("Model oob error after tree %v : %v\n", i, oobVotes.TallyError(unboostedTarget))
		// 	}

		// }
		//Single forest growth is over.

		//Record importance scores from this forest for ace
		if g.Ace > 0 && (g.Cutoff == 0.0 || foresti < nForest-1) {
			if foresti < nForest-1 {
				fmt.Printf("- Finished ACE forest %v.\n", foresti)
			}
			//Record Importance scores
			for i := 0; i < len(data.Data); i++ {
				mean, count := (*imppnt)[i].Read()
				aceImps[i][foresti] = mean * float64(count) / float64(g.Trees)
			}

			//Reset importance scores
			imppnt = NewRunningMeans(len(data.Data))

			//Reshuffle contrast features
			for i := firstace; i < len(data.Data); i++ {
				if !blacklistis[i] {
					data.Data[i].Shuffle()
				}
			}

			if g.Cutoff > 0 && foresti == nForest-2 {
				sigcount := 0
				for i := range blacklistis {
					if i < firstace && !blacklistis[i] {
						p, _, _, m := Ttest(&aceImps[i], &aceImps[i+firstace])
						if p < g.Cutoff && m > 0.0 && i != targeti {
							blacklistis[i] = false
							sigcount++
						} else {
							blacklistis[i] = true
						}
					}
					if i >= firstace {
						blacklistis[i] = true
					}

				}
				mTry = ParseAsIntOrFractionOfTotal(g.MTry, sigcount)
				if mTry <= 0 {
					mTry = int(math.Ceil(math.Sqrt(float64(sigcount))))
				}
				fmt.Printf("- Growing non-ACE forest with %v features with p-value < %v.\nmTry: %v\n", sigcount, g.Cutoff, mTry)
			}
		}
	}

	trainingEnd := time.Now()
	if g.Progress {
		fmt.Println("--------------------------------------------------------------------")
	}
	fmt.Printf("= Total training time (seconds): %v\n", trainingEnd.Sub(trainingStart).Seconds())

	if g.OOB {
		fmt.Printf("= Out of Bag Error : %v\n", oobVotes.TallyError(unboostedTarget))
	}

	fmt.Println("--------------------------------------------------------------------")

	if g.ScikitForest != "" {
		fmt.Printf("> Scikit Forest: %v\n", g.ScikitForest)
		skfile, err := os.Create(g.ScikitForest)
		if err != nil {
			log.Fatal(err)
		}
		defer skfile.Close()
		skencoder := json.NewEncoder(skfile)
		err = skencoder.Encode(scikikittrees)
		if err != nil {
			log.Fatal(err)
		}
	}

	if g.CaseOOB != "" {
		fmt.Printf("> OOB predictions: %v\n", g.CaseOOB)
		caseoobfile, err := os.Create(g.CaseOOB)
		if err != nil {
			log.Fatal(err)
		}
		defer caseoobfile.Close()
		for i := 0; i < unboostedTarget.Length(); i++ {
			fmt.Fprintf(caseoobfile, "%v\t%v\t%v\n", data.Cases[i], oobVotes.Tally(i), unboostedTarget.GetStr(i))
		}
	}

	if g.Importance != "" {
		fmt.Printf("> Importance scores: %v\n", g.Importance)
		impfile, err := os.Create(g.Importance)
		if err != nil {
			log.Fatal(err)
		}
		defer impfile.Close()
		if g.Ace > 0 {
			// Header: target | predictor | p-value         | mean importance
			fmt.Fprintf(impfile, "Target\tPredictor\tP-Value\tMean Importance\n")
			for i := 0; i < firstace; i++ {
				p, _, _, m := Ttest(&aceImps[i], &aceImps[i+firstace])
				fmt.Fprintf(impfile, "%v\t%v\t%v\t%v\n", g.TargetName, data.Data[i].GetName(), p, m)
			}
		} else {
			// Write standard importance file
			// Header: Feature | Decrease Per Use | Use Count | Decrease Per Tree | Decrease Per Tree Used | Tree Used Count | Mean Minimal Depth
			fmt.Println("\n--------------------------------------------------------------------")
			fmt.Printf("Feature         Dc/Use    Used   Dc/Tree   Dc/TreeU   TreeU   MMinDp\n")
			fmt.Println("--------------------------------------------------------------------")

			fmt.Fprintf(impfile, "Feature\tDecrease Per Use\tUse Count\tDecrease Per Tree\tDecrease Per Tree Used\tTree Used Count\tMean Minimal Depth\n")
			for i, v := range *imppnt {
				mean, count := v.Read()
				meanMinDepth, treeCount := (*mmdpnt)[i].Read()
				fmt.Fprintf(impfile, "%v\t%v\t%v\t%v\t%v\t%v\t%v\n",
					data.Data[i].GetName(), mean, count, mean*float64(count)/float64(g.Trees), mean*float64(count)/float64(treeCount), treeCount, meanMinDepth)

				fmt.Printf("%-12v   %7.3f  %6v   %7.3f    %7.3f   %5v  %7.3f\n",
					data.Data[i].GetName(), mean, count, mean*float64(count)/float64(g.Trees), mean*float64(count)/float64(treeCount), treeCount, meanMinDepth)
			}

			fmt.Println("--------------------------------------------------------------------")
		}
	}

	if g.SelfTest {
		var bb VoteTallyer
		fmt.Println()

		testdata := data
		testtarget := unboostedTarget
		if g.TestFile != "" {
			var err error
			testdata, err = LoadAFM(g.TestFile)
			if err != nil {
				log.Fatal(err)
			}
			targeti, ok = testdata.Map[g.TargetName]
			if !ok {
				log.Fatal("Target not found in test data.")
			}
			testtarget = testdata.Data[targeti]

			for _, tree := range trees {
				tree.StripCodes()

			}
		}

		if unboostedTarget.NCats() == 0 {
			//regression
			bb = NewNumBallotBox(testdata.Data[0].Length())
		} else {
			//classification
			bb = NewCatBallotBox(testdata.Data[0].Length())
		}

		for _, tree := range trees {
			tree.Vote(testdata, bb)
		}

		fmt.Printf("= Error: %v\n", bb.TallyError(testtarget))

		if testtarget.NCats() != 0 {
			reftotals := make([]int, testtarget.NCats())
			predtotals := make([]int, testtarget.NCats())

			true_pos := make([]int, testtarget.NCats())
			false_pos := make([]int, testtarget.NCats())
			false_neg := make([]int, testtarget.NCats())

			correct := 0
			errors := 0
			nas := 0
			length := testtarget.Length()
			for i := 0; i < length; i++ {
				refi := testtarget.(*DenseCatFeature).Geti(i)
				reftotals[refi]++
				pred := bb.Tally(i)
				orig := testtarget.GetStr(i)
				if pred == "NA" {
					nas++
				} else {
					predi := testtarget.(*DenseCatFeature).CatToNum(pred)
					predtotals[predi]++
					if pred == orig {
						correct++
						true_pos[refi]++
					} else {
						errors++
						false_pos[predi]++
						false_neg[refi]++
					}
				}
			}

			fmt.Printf("= Classified: %v / %v = %.3f\n", correct, length, float64(correct)*100/float64(length))

			// fmt.Println()
			// for i, v := range testtarget.(*DenseCatFeature).Back {
			// 	fmt.Printf("- Label [%v] Percision (Actuall/Predicted): %v / %v = %.3f\n", v, false_pos[i], predtotals[i], float64(false_pos[i])*100/float64(predtotals[i]))
			// 	falses := reftotals[i] - true_pos[i]
			// 	fmt.Printf("- Label [%v] Missed/Actuall Rate: %v / %v = %.3f\n", v, falses, reftotals[i], float64(falses)*100/float64(reftotals[i]))
			// }
			if nas != 0 {
				fmt.Printf("= Couldn't predict %v cases due to missing values.\n", nas)
			}

			fmt.Println()

			err_pos := 0
			err_neg := 0

			fmt.Println("--------------------------------------------------------------------")
			fmt.Println("CAT           TP      FP      FN      ER        SE       PP       F1")
			fmt.Println("--------------------------------------------------------------------")
			for i, v := range testtarget.(*DenseCatFeature).Back {
				se := float64(true_pos[i]) / float64(true_pos[i]+false_pos[i])
				pp := float64(true_pos[i]) / float64(true_pos[i]+false_neg[i])
				f1 := 2 * se * pp / (se + pp)
				err_pos += false_pos[i]
				err_neg += false_neg[i]

				fmt.Printf("%-8v  %6d  %6d  %6d  %6d   %7.3f  %7.3f  %7.3f\n",
					v, true_pos[i], false_pos[i], false_neg[i], false_pos[i]+false_neg[i], se*100, pp*100, f1*100)
			}
			fmt.Println("--------------------------------------------------------------------")
			se := float64(correct) / float64(correct+err_pos)
			pp := float64(correct) / float64(correct+err_neg)
			f1 := 2 * se * pp / (se + pp)
			fmt.Printf("%-8v  %6d  %6d  %6d  %6d   %7.3f  %7.3f  %7.3f\n",
				"ALL", correct, err_pos, err_neg, errors, se*100, pp*100, f1*100)
			fmt.Println("--------------------------------------------------------------------")
		}
	}
}
