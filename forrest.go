package rfx

// Forest represents a collection of decision trees grown to predict Target.
type Forest struct {
	//Forest string
	Target    string
	Trees     []*Tree
	Intercept float64
}

/*
GrowRandomForest grows a forest using Brieman and Cutler's method. For many cases it
it will yield better performance to re-implment this method to write trees directly to disk or grow
trees in parallel. See the grow forest command line utility for an example of this.

target is the feature to predict.

nSamples is the number of cases to sample (with replacement) for each tree.

mTry is the number of candidate features to evaluate at each node.

nTrees is the number of trees to grow.

leafSize is the minimum number of cases that should end up on a leaf.

itter indicates weather to use iterative splitting for all categorical features or only those
with more then 6 categories.
*/

func GrowRandomForest(fm *FeatureMatrix,
	target Target,
	candidates []int,
	nSamples int,
	mTry int,
	nTrees int,
	leafSize int,
	maxDepth int,
	splitmissing bool,
	force bool,
	vet bool,
	evaloob bool,
	importance *[]*RunningMean) (f *Forest) {

	f = &Forest{target.GetName(), make([]*Tree, 0, nTrees), 0.0}

	switch t := target.(type) {
	case TargetWithIntercept:
		f.Intercept = t.Intercept()
	}

	//Slices for reuse during search for best splitter.
	allocs := NewBestSplitAllocs(nSamples, target)

	for i := 0; i < nTrees; i++ {
		nCases := fm.Data[0].Length()
		cases := SampleWithReplacment(nSamples, nCases)

		f.Trees = append(f.Trees, NewTree())
		f.Trees[i].Grow(fm, target, cases, candidates, nil, mTry, leafSize, maxDepth, splitmissing, force, vet, evaloob, false, importance, nil, allocs)
		switch t := target.(type) {
		case BoostingTarget:
			ls, ps := f.Trees[i].Partition(fm)
			f.Trees[i].Weight = t.Boost(ls, ps)
		}
	}
	return
}
