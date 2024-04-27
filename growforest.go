package learn

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
	"runtime/pprof"
	"sync"
	"time"
)

type GrowForest struct {
	// nTrees: 100, Number of trees to grow in the predictor.
	Trees int

	// train: AFM formated feature matrix containing training data.
	TrainFile string

	// rfpred: File name to output predictor forest in sf format.
	ForestFile string

	// target: The row header of the target in the feature matrix.
	TargetName string

	// importance: File name to output importance.
	Importance string

	// cost: For categorical targets, a json string to float map of the cost of falsely identifying each category.
	Costs string

	// Dentropy: Class disutilities for disutility entropy.
	Dentropy string

	// adacost: Json costs for cost sentive AdaBoost.
	AdaCosts string

	// rfweights: For categorical targets, a json string to float map of the weights to use for each category in Weighted RF.
	RfWeights string

	// blacklist: A list of feature id's to exclude from the set of predictors.
	Blacklist string

	// nCores: The number of cores to use.
	Cores int

	// nSamples: The number of cases to sample (with replacement) for each tree as a count (ex: 10) or portion of total (ex: .5). If <=0 set to total number of cases.
	NSamples string

	// mTry: Number of candidate features for each split as a count (ex: 10) or portion of total (ex: .5). Ceil(sqrt(nFeatures)) if <=0.
	MTry string

	// leafSize: The minimum number of cases on a leaf node. If <=0 will be inferred to 1 for classification 4 for regression.
	LeafSize string

	// maxDepth: Maximum tree depth. Ignored if 0.
	MaxDepth int

	// shuffleRE: A regular expression to identify features that should be shuffled.
	ShuffleRE string

	// blockRE: A regular expression to identify features that should be filtered out.
	BlockRE string

	// includeRE: Filter features that DON'T match this RE.
	IncludeRE string

	// trans_unlabeled: Class to treat as TransUnlabeled for transduction forests.
	TransUnlabeled string

	// trans_alpha: 10.0, Weight of unsupervised term in transduction impurity.
	TransAlpha float64

	// trans_beta: Multiple to penalize unlabeled class by.
	TransBeta float64

	// ace: Number Ace permutations to do. Output Ace style importance and p values.
	Ace int

	// cutoff: P-value Cutoff to apply to features for last forest after ACE.
	Cutoff float64

	// nContrasts: The number of randomized artificial contrast features to include in the feature matrix.
	Contrasts int

	// cpuprofile: write cpu profile to file
	CpuProfile string

	// Contrastall: Include a shuffled artificial contrast copy of every feature.
	ContrastAll bool

	// impute: Impute missing values to feature mean/mode before growth.
	Impute bool

	// extra: Grow Extra Random Trees (supports learning from numerical variables only).
	Extra bool

	// splitmissing: Split missing values onto a third branch at each node (experimental).
	SplitMissing bool

	// l1: Use l1 norm regression (target must be numeric).
	L1 bool

	// density: Build density estimating trees instead of classification/regression trees.
	Density bool

	// vet: Penalize potential splitter impurity decrease by subtracting the best split of a permuted target.
	Vet bool

	// positive: true, Positive class to output probabilities for.
	positive string

	// NP: Do approximate Neyman-Pearson classification.
	NP bool

	// NP_pos: 1, Class label to constrain percision in NP classification.
	NP_pos string

	// NP_a: 0.1, Constraint on percision in NP classification [0,1]
	NP_a float64

	// NP_k: 100, Weight of constraint in NP classification [0,Inf+)
	NP_k float64

	// evaloob: Evaluate potential splitting features on OOB cases after finding split value in bag.
	EvalOOB bool

	// force: Force at least one non constant feature to be tested for each split.
	Force bool

	// entropy: Use entropy minimizing classification (target must be categorical).
	Entropy bool

	// oob: Calculate and report OOB error.
	OOB bool

	// jungle: Grow unserializable and experimental decision jungle with node recombination.
	Jungle bool

	// oobpreds: Calculate and report oob predictions in the file specified.
	CaseOOB string

	// progress: Report tree number and running oob error.
	Progress bool

	// adaboost: Use Adaptive boosting for regression/classification.
	AdaBoost bool

	// hellinger: Build trees using Hellinger distance.
	Hellinger bool

	// gbt: Use gradient boosting with the specified learning rate.
	GradBoost float64

	// multiboost: Allow multi-threaded boosting which may have unexpected results. (highly experimental)
	Multiboost bool

	// nobag: Don't bag samples for each tree.
	NoBag bool

	// balance: Balance bagging of samples by target class for unbalanced classification.
	Balance bool

	// balanceby: Roughly balanced bag the target within each class of this feature.
	BalanceBy string

	// ordinal: Use ordinal regression (target must be numeric).
	Ordinal bool

	// permute: Permute the target feature (to establish random predictive power).
	Permute bool

	// selftest: Test the forest on the data and report accuracy.
	SelfTest bool

	// test: Data to test the model on.
	TestFile string

	// scikitforest: Write out a (partially complete) scikit style forest in json.
	ScikitForest string

	// noseed: Don't seed the random number generator from time.
	NoSeed bool
}

func NewGrowForest() *GrowForest {
	g := GrowForest{
		Trees:      100,
		TransAlpha: 10.0,
		positive:   "true",
		NP_pos:     "1",
		NP_a:       0.1,
		NP_k:       100,
	}
	return &g
}

func GrowForestParse() *GrowForest {
	g := NewGrowForest()

	// fm string
	flag.StringVar(&g.TrainFile, "train", "featurematrix.afm", "AFM formated feature matrix containing training data.")

	// testfm string
	flag.StringVar(&g.TestFile, "test", "", "Data to test the model on.")

	// rf string
	flag.StringVar(&g.ForestFile, "save", "", "File name to output predictor forest in sf format.")
	flag.StringVar(&g.ForestFile, "rfpred", "", "File name to output predictor forest in sf format.")

	// targetname string
	flag.StringVar(&g.TargetName, "target", "", "The row header of the target in the feature matrix.")

	// imp string
	flag.StringVar(&g.Importance, "importance", "", "File name to output importance.")

	// costs string
	flag.StringVar(&g.Costs, "cost", "", "For categorical targets, a json string to float map of the cost of falsely identifying each category.")

	// dentropy string
	flag.StringVar(&g.Dentropy, "dentropy", "", "Class disutilities for disutility entropy.")

	// adacosts string
	flag.StringVar(&g.AdaCosts, "adacost", "", "Json costs for cost sentive AdaBoost.")

	// rfweights string
	flag.StringVar(&g.RfWeights, "rfweights", "", "For categorical targets, a json string to float map of the weights to use for each category in Weighted RF.")

	// blacklist string
	flag.StringVar(&g.Blacklist, "blacklist", "", "A list of feature id's to exclude from the set of predictors.")

	// nCores int
	flag.IntVar(&g.Cores, "cores", 1, "The number of cores to use.")

	// StringnSamples string
	flag.StringVar(&g.NSamples, "nSamples", "0", "The number of cases to sample (with replacement) for each tree as a count (ex: 10) or portion of total (ex: .5). If <=0 set to total number of cases.")

	// StringmTry string
	flag.StringVar(&g.MTry, "mTry", "0", "Number of candidate features for each split as a count (ex: 10) or portion of total (ex: .5). Ceil(sqrt(nFeatures)) if <=0.")

	// StringleafSize string
	flag.StringVar(&g.LeafSize, "leafSize", "0", "The minimum number of cases on a leaf node. If <=0 will be inferred to 1 for classification 4 for regression.")

	// maxDepth int
	flag.IntVar(&g.MaxDepth, "maxDepth", 0, "Maximum tree depth. Ignored if 0.")

	// shuffleRE string
	flag.StringVar(&g.ShuffleRE, "shuffleRE", "", "A regular expression to identify features that should be shuffled.")

	// blockRE string
	flag.StringVar(&g.BlockRE, "blockRE", "", "A regular expression to identify features that should be filtered out.")

	// includeRE string
	flag.StringVar(&g.IncludeRE, "includeRE", "", "Filter features that DON'T match this RE.")

	// unlabeled string
	flag.StringVar(&g.TransUnlabeled, "trans_unlabeled", "", "Class to treat as unlabeled for transduction forests.")

	// trans_alpha float64
	flag.Float64Var(&g.TransAlpha, "trans_alpha", 10.0, "Weight of unsupervised term in transduction impurity.")

	// trans_beta float64
	flag.Float64Var(&g.TransBeta, "trans_beta", 0.0, "Multiple to penalize unlabeled class by.")

	// nTrees int
	flag.IntVar(&g.Trees, "trees", 100, "Number of trees to grow in the predictor.")

	// ace int
	flag.IntVar(&g.Ace, "ace", 0, "Number ace permutations to do. Output ace style importance and p values.")

	// cutoff float64
	flag.Float64Var(&g.Cutoff, "cutoff", 0.0, "P-value cutoff to apply to features for last forest after ACE.")

	// nContrasts int
	flag.IntVar(&g.Contrasts, "contrasts", 0, "The number of randomized artificial contrast features to include in the feature matrix.")

	// cpuprofile string
	flag.StringVar(&g.CpuProfile, "cpuprofile", "", "write cpu profile to file")

	// contrastAll bool
	flag.BoolVar(&g.ContrastAll, "contrastall", false, "Include a shuffled artificial contrast copy of every feature.")

	// impute bool
	flag.BoolVar(&g.Impute, "impute", false, "Impute missing values to feature mean/mode before growth.")

	// extra bool
	flag.BoolVar(&g.Extra, "extra", false, "Grow Extra Random Trees (supports learning from numerical variables only).")

	// splitmissing bool
	flag.BoolVar(&g.SplitMissing, "splitmissing", false, "Split missing values onto a third branch at each node (experimental).")

	// l1 bool
	flag.BoolVar(&g.L1, "l1", false, "Use l1 norm regression (target must be numeric).")

	// density bool
	flag.BoolVar(&g.Density, "density", false, "Build density estimating trees instead of classification/regression trees.")

	// vet bool
	flag.BoolVar(&g.Vet, "vet", false, "Penalize potential splitter impurity decrease by subtracting the best split of a permuted target.")

	// positive string
	flag.StringVar(&g.positive, "positive", "True", "Positive class to output probabilities for.")

	// NP bool
	flag.BoolVar(&g.NP, "NP", false, "Do approximate Neyman-Pearson classification.")

	// NP_pos string
	flag.StringVar(&g.NP_pos, "NP_pos", "1", "Class label to constrain percision in NP classification.")

	// NP_a float64
	flag.Float64Var(&g.NP_a, "NP_a", 0.1, "Constraint on percision in NP classification [0,1]")

	// NP_k float64
	flag.Float64Var(&g.NP_k, "NP_k", 100, "Weight of constraint in NP classification [0,Inf+)")

	// evaloob bool
	flag.BoolVar(&g.EvalOOB, "evaloob", false, "Evaluate potential splitting features on OOB cases after finding split value in bag.")

	// force bool
	flag.BoolVar(&g.Force, "force", false, "Force at least one non constant feature to be tested for each split.")

	// entropy bool
	flag.BoolVar(&g.Entropy, "entropy", false, "Use entropy minimizing classification (target must be categorical).")

	// oob bool
	flag.BoolVar(&g.OOB, "oob", false, "Calculate and report oob error.")

	// jungle bool
	flag.BoolVar(&g.Jungle, "jungle", false, "Grow unserializable and experimental decision jungle with node recombination.")

	// caseoob string
	flag.StringVar(&g.CaseOOB, "oobpreds", "", "Calculate and report oob predictions in the file specified.")

	// progress bool
	flag.BoolVar(&g.Progress, "progress", false, "Report tree number and running oob error.")

	// adaboost bool
	flag.BoolVar(&g.AdaBoost, "adaboost", false, "Use Adaptive boosting for regression/classification.")

	// hellinger bool
	flag.BoolVar(&g.Hellinger, "hellinger", false, "Build trees using hellinger distance.")

	// gradboost float64
	flag.Float64Var(&g.GradBoost, "gbt", 0.0, "Use gradient boosting with the specified learning rate.")

	// multiboost bool
	flag.BoolVar(&g.Multiboost, "multiboost", false, "Allow multi-threaded boosting which may have unexpected results. (highly experimental)")

	// nobag bool
	flag.BoolVar(&g.NoBag, "nobag", false, "Don't bag samples for each tree.")

	// balance bool
	flag.BoolVar(&g.Balance, "balance", false, "Balance bagging of samples by target class for unbalanced classification.")

	// balanceby string
	flag.StringVar(&g.BalanceBy, "balanceby", "", "Roughly balanced bag the target within each class of this feature.")

	// ordinal bool
	flag.BoolVar(&g.Ordinal, "ordinal", false, "Use ordinal regression (target must be numeric).")

	// permutate bool
	flag.BoolVar(&g.Permute, "permute", false, "Permute the target feature (to establish random predictive power).")

	// dotest bool
	flag.BoolVar(&g.SelfTest, "selftest", false, "Test the forest on the data and report accuracy.")

	// scikitforest string
	flag.StringVar(&g.ScikitForest, "scikitforest", "", "Write out a (partially complete) scikit style forest in json.")

	// noseed bool
	flag.BoolVar(&g.NoSeed, "noseed", false, "Don't seed the random number generator from time.")

	flag.Parse()

	return g
}

func (g *GrowForest) Fit() {
	nForest := 1

	if !g.NoSeed {
		rand.Seed(time.Now().UTC().UnixNano())
	}

	if g.TestFile != "" {
		g.SelfTest = true
	}

	if g.Multiboost {
		fmt.Println("MULTIBOOST!!!!1!!!!1!!11 (things may break).")
	}
	var boostMutex sync.Mutex
	boost := (g.AdaBoost || g.GradBoost != 0.0)
	if boost && !g.Multiboost {
		g.Cores = 1
	}

	// if nCores > 1 {
	// 	runtime.GOMAXPROCS(nCores)
	// }

	fmt.Printf("Threads : %v\n", g.Cores)
	fmt.Printf("nTrees : %v\n", g.Trees)

	//Parse Data
	fmt.Printf("Loading data from: %v\n", g.TrainFile)
	data, err := LoadAFM(g.TrainFile)
	if err != nil {
		log.Fatal(err)
	}

	if g.CpuProfile != "" {
		f, err := os.Create(g.CpuProfile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	if g.Contrasts > 0 {
		fmt.Printf("Adding %v Random Contrasts\n", g.Contrasts)
		data.AddContrasts(g.Contrasts)
	}
	if g.ContrastAll {
		fmt.Printf("Adding Random Contrasts for All Features.\n")
		data.ContrastAll()
	}

	blacklisted := 0
	blacklistis := make([]bool, len(data.Data))
	if g.Blacklist != "" {
		fmt.Printf("Loading blacklist from: %v\n", g.Blacklist)
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
				fmt.Printf("Ignoring blacklist feature not found in data: %v\n", id[0])
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
	fmt.Printf("Target : %v\n", g.TargetName)
	targeti, ok := data.Map[g.TargetName]
	if !ok {
		log.Fatal("Target not found in data.")
	}

	if g.BlockRE != "" {
		re := regexp.MustCompile(g.BlockRE)
		for i, feature := range data.Data {
			if targeti != i && re.MatchString(feature.GetName()) {
				if blacklistis[i] == false {
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
				if blacklistis[i] == false {
					blacklisted += 1
					blacklistis[i] = true
				}
			}
		}
	}

	nFeatures := len(data.Data) - blacklisted - 1
	fmt.Printf("Non Target Features : %v\n", nFeatures)

	mTry := ParseAsIntOrFractionOfTotal(g.MTry, nFeatures)
	if mTry <= 0 {
		mTry = int(math.Ceil(math.Sqrt(float64(nFeatures))))
	}
	fmt.Printf("mTry : %v\n", mTry)

	if g.Impute {
		fmt.Println("Imputing missing values to feature mean/mode.")
		data.ImputeMissing()
	}

	if g.Permute {
		fmt.Println("Permuting target feature.")
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
		fmt.Printf("Shuffled %v features matching %v\n", shuffled, g.ShuffleRE)
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
	fmt.Printf("non-missing cases: %v\n", nNonMissing)

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
	fmt.Printf("leafSize : %v\n", leafSize)

	//infer nSamples and mTry from data if they are 0
	nSamples := ParseAsIntOrFractionOfTotal(g.NSamples, nNonMissing)
	if nSamples <= 0 {
		nSamples = nNonMissing
	}
	fmt.Printf("nSamples : %v\n", nSamples)

	if g.Progress {
		g.OOB = true
	}
	if g.CaseOOB != "" {
		g.OOB = true
	}
	var oobVotes VoteTallyer
	if g.OOB {
		fmt.Println("Recording oob error.")
		if targetf.NCats() == 0 {
			//regression
			oobVotes = NewNumBallotBox(data.Data[0].Length())
		} else {
			//classification
			oobVotes = NewCatBallotBox(data.Data[0].Length())
		}
	}

	//****** Set up Target for Alternative Impurity  if needed *******//
	var target Target
	if g.Density {
		fmt.Println("Estimating Density.")
		target = &DensityTarget{&data.Data, nNonMissing}
	} else {

		switch targetf.(type) {

		case NumFeature:
			fmt.Println("Performing regression.")
			if g.L1 {
				fmt.Println("Using l1/absolute deviance error.")
				targetf = &L1Target{targetf.(NumFeature)}
			}
			if g.Ordinal {
				fmt.Println("Using Ordinal (mode) prediction.")
				targetf = NewOrdinalTarget(targetf.(NumFeature))
			}
			switch {
			case g.GradBoost != 0.0:
				fmt.Println("Using Gradient Boosting.")
				targetf = NewGradBoostTarget(targetf.(NumFeature), g.GradBoost)

			case g.AdaBoost:
				fmt.Println("Using Numeric Adaptive Boosting.")
				targetf = NewNumAdaBoostTarget(targetf.(NumFeature))
			}
			target = targetf

		case CatFeature:
			fmt.Printf("Performing classification with %v categories.\n", targetf.NCats())
			switch {
			case g.NP:
				fmt.Printf("Performing Approximate Neyman-Pearson Classification with constrained false \"%v\".\n", g.NP_pos)
				fmt.Printf("False %v constraint: %v, constraint weight: %v.\n", g.NP_pos, g.NP_a, g.NP_k)
				targetf = NewNPTarget(targetf.(CatFeature), g.NP_pos, g.NP_a, g.NP_k)
			case g.Costs != "":
				fmt.Println("Using misclassification costs: ", g.Costs)
				costmap := make(map[string]float64)
				err := json.Unmarshal([]byte(g.Costs), &costmap)
				if err != nil {
					log.Fatal(err)
				}

				regTarg := NewRegretTarget(targetf.(CatFeature))
				regTarg.SetCosts(costmap)
				targetf = regTarg
			case g.Dentropy != "":
				fmt.Println("Using entropy with disutilities: ", g.Dentropy)
				costmap := make(map[string]float64)
				err := json.Unmarshal([]byte(g.Dentropy), &costmap)
				if err != nil {
					log.Fatal(err)
				}

				deTarg := NewDEntropyTarget(targetf.(CatFeature))
				deTarg.SetCosts(costmap)
				targetf = deTarg
			case g.AdaCosts != "":
				fmt.Println("Using cost sensative AdaBoost costs: ", g.AdaCosts)
				costmap := make(map[string]float64)
				err := json.Unmarshal([]byte(g.AdaCosts), &costmap)
				if err != nil {
					log.Fatal(err)
				}

				actarget := NewAdaCostTarget(targetf.(CatFeature))
				actarget.SetCosts(costmap)
				targetf = actarget

			case g.RfWeights != "":
				fmt.Println("Using rf weights: ", g.RfWeights)
				weightmap := make(map[string]float64)
				err := json.Unmarshal([]byte(g.RfWeights), &weightmap)
				if err != nil {
					log.Fatal(err)
				}

				wrfTarget := NewWRFTarget(targetf.(CatFeature), weightmap)
				targetf = wrfTarget

			case g.Entropy:
				fmt.Println("Using entropy minimization.")
				targetf = &EntropyTarget{targetf.(CatFeature)}

			case g.AdaBoost:
				fmt.Println("Using Adaptive Boosting.")
				targetf = NewAdaBoostTarget(targetf.(CatFeature))

			case g.Hellinger:
				fmt.Println("Using Hellinger Distance with postive class:", g.positive)
				targetf = NewHDistanceTarget(targetf.(CatFeature), g.positive)

			case g.GradBoost != 0.0:
				fmt.Println("Using Gradient Boosting Classification with postive class:", g.positive)
				targetf = NewGradBoostClassTarget(targetf.(CatFeature), g.GradBoost, g.positive)

			}

			if g.TransUnlabeled != "" {
				fmt.Println("Using traduction forests with unlabeled class: ", g.TransUnlabeled)
				targetf = NewTransTarget(targetf.(CatFeature), &data.Data, g.TransUnlabeled, g.TransAlpha, g.TransBeta, nNonMissing)

			}
			target = targetf
		}
	}

	var forestwriter *ForestWriter
	if g.ForestFile != "" {
		forestfile, err := os.Create(g.ForestFile)
		if err != nil {
			log.Fatal(err)
		}
		defer forestfile.Close()
		forestwriter = NewForestWriter(forestfile)
		switch target.(type) {
		case TargetWithIntercept:
			forestwriter.WriteForestHeader(0, g.TargetName, target.(TargetWithIntercept).Intercept())
		}
	}
	//****************** Setup For ACE ********************************//
	var aceImps [][]float64
	firstace := len(data.Data)

	if g.Ace > 0 {
		fmt.Printf("Performing ACE analysis with %v forests/permutations.\n", g.Ace)

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

	//****************** Needed Collections and vars ******************//
	var trees []*Tree
	trees = make([]*Tree, 0, g.Trees)

	var imppnt *[]*RunningMean
	var mmdpnt *[]*RunningMean
	if g.Importance != "" {
		fmt.Println("Recording Importance Scores.")

		imppnt = NewRunningMeans(len(data.Data))
		mmdpnt = NewRunningMeans(len(data.Data))
	} else if g.Ace > 0 {
		imppnt = NewRunningMeans(len(data.Data))
	}

	var scikikittrees []ScikitTree
	if g.ScikitForest != "" {
		scikikittrees = make([]ScikitTree, 0, g.Trees)
	}

	//****************** Good Stuff Stars Here ;) ******************//

	trainingStart := time.Now()

	for foresti := 0; foresti < nForest; foresti++ {
		var treesStarted, treesFinished int
		treesStarted = g.Cores
		var recordingTree sync.Mutex
		var waitGroup sync.WaitGroup

		waitGroup.Add(g.Cores)
		treechan := make(chan *Tree, 0)
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
							fmt.Printf("Boosting Reached Weight of %v\n", weight)
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
						fmt.Printf("Model oob error after tree %v : %v\n", treesFinished, oobVotes.TallyError(unboostedTarget))
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
				fmt.Printf("Finished ACE forest %v.\n", foresti)
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
				fmt.Printf("Growing non-ACE forest with %v features with p-value < %v.\nmTry: %v\n", sigcount, g.Cutoff, mTry)
			}
		}
	}

	trainingEnd := time.Now()
	fmt.Printf("Total training time (seconds): %v\n", trainingEnd.Sub(trainingStart).Seconds())

	if g.ScikitForest != "" {
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

	if g.OOB {
		fmt.Printf("Out of Bag Error : %v\n", oobVotes.TallyError(unboostedTarget))
	}
	if g.CaseOOB != "" {
		caseoobfile, err := os.Create(g.CaseOOB)
		if err != nil {
			log.Fatal(err)
		}
		defer caseoobfile.Close()
		for i := 0; i < unboostedTarget.Length(); i++ {
			fmt.Fprintf(caseoobfile, "%v\t%v\t%v\n", data.CaseLabels[i], oobVotes.Tally(i), unboostedTarget.GetStr(i))
		}
	}

	if g.Importance != "" {

		impfile, err := os.Create(g.Importance)
		if err != nil {
			log.Fatal(err)
		}
		defer impfile.Close()
		if g.Ace > 0 {

			for i := 0; i < firstace; i++ {

				p, _, _, m := Ttest(&aceImps[i], &aceImps[i+firstace])

				fmt.Fprintf(impfile, "%v\t%v\t%v\t%v\n", g.TargetName, data.Data[i].GetName(), p, m)

			}
		} else {
			//Write standard importance file
			for i, v := range *imppnt {
				mean, count := v.Read()
				meanMinDepth, treeCount := (*mmdpnt)[i].Read()
				fmt.Fprintf(impfile, "%v\t%v\t%v\t%v\t%v\t%v\t%v\n", data.Data[i].GetName(), mean, count, mean*float64(count)/float64(g.Trees), mean*float64(count)/float64(treeCount), treeCount, meanMinDepth)

			}
		}
	}

	if g.SelfTest {
		var bb VoteTallyer

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

		fmt.Printf("Error: %v\n", bb.TallyError(testtarget))

		if testtarget.NCats() != 0 {
			falsesbypred := make([]int, testtarget.NCats())
			predtotals := make([]int, testtarget.NCats())

			truebytrue := make([]int, testtarget.NCats())
			truetotals := make([]int, testtarget.NCats())

			correct := 0
			nas := 0
			length := testtarget.Length()
			for i := 0; i < length; i++ {
				truei := testtarget.(*DenseCatFeature).Geti(i)
				truetotals[truei]++
				pred := bb.Tally(i)
				if pred == "NA" {
					nas++
				} else {
					predi := testtarget.(*DenseCatFeature).CatToNum(pred)
					predtotals[predi]++
					if pred == testtarget.GetStr(i) {
						correct++
						truebytrue[truei]++
					} else {

						falsesbypred[predi]++
					}
				}

			}
			fmt.Printf("Classified: %v / %v = %v\n", correct, length, float64(correct)/float64(length))
			for i, v := range testtarget.(*DenseCatFeature).Back {
				fmt.Printf("Label %v Percision (Actuall/Predicted): %v / %v = %v\n", v, falsesbypred[i], predtotals[i], float64(falsesbypred[i])/float64(predtotals[i]))
				falses := truetotals[i] - truebytrue[i]
				fmt.Printf("Label %v Missed/Actuall Rate: %v / %v = %v\n", v, falses, truetotals[i], float64(falses)/float64(truetotals[i]))

			}
			if nas != 0 {
				fmt.Printf("Couldn't predict %v cases due to missing values.\n", nas)
			}
		}
	}
}
