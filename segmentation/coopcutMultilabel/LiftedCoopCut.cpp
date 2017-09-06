#include <iostream>
#include <vector>
#include <fstream>

#include "LiftedCoopCut.h"

#include "CoopCutMultilabel.h"
#include "../gcoLibrary/gco-v3.0/GCoptimization.h"

#include <vector>
using std::vector;

#include <stdio.h>
#include <iostream>
#include <sys/stat.h>
#include <algorithm>
#include <time.h>
#include <assert.h>

using namespace std;

/*
	This file is organized into sections as:
	auxillary-functions 
	normal-lifting
	repeated-lifting
	alpha-expansion
	greedy
*/

bool LiftedCoopCut::initialize_factor_graph(FactorGraph& fg) {
	cout << ("Starting initialize_factor_graph") << endl;
	clock_t start = clock();
	fg.initialized_unary_colors = false;
	fg.vecVariables.resize(nodeNumber_);
	fg.adjListOfVariables.resize(nodeNumber_);
	fg.vecFactors.resize(nodeNumber_ + edgeNumber_);
	fg.adjListOfFactors.resize(nodeNumber_ + edgeNumber_);

	fg.variableSignatures.resize(nodeNumber_);
	fg.factorSignatures.resize(nodeNumber_ + edgeNumber_);
	fg.nowVariableSignatures.resize(nodeNumber_);
	fg.nowFactorSignatures.resize(nodeNumber_ + edgeNumber_);

	for(int iNode=0; iNode<nodeNumber_; ++iNode) {
		fg.vecVariables[iNode] = iNode; // sure. why not.
		fg.vecFactors[iNode] = iNode;
		fg.adjListOfFactors[iNode] = std::vector<int>(1, iNode);
		fg.adjListOfVariables[iNode] = std::vector<int>(1, iNode);
	}
	for(int iEdge=0; iEdge<edgeNumber_; ++iEdge) {
		int node0 = edges_[ getEdgeEndIndex(iEdge, 0) ];
		assert(node0 >= 0 && node0 < nodeNumber_);

		int node1 = edges_[ getEdgeEndIndex(iEdge, 1) ];
		assert(node1 >= 0 && node1 < nodeNumber_);

		fg.adjListOfFactors[nodeNumber_ + iEdge] = std::vector<int>(2);
		fg.adjListOfFactors[nodeNumber_ + iEdge][0] = min(node0, node1);
		fg.adjListOfFactors[nodeNumber_ + iEdge][1] = max(node0, node1);
		
		fg.adjListOfVariables[node0].push_back(nodeNumber_ + iEdge);
		fg.adjListOfVariables[node1].push_back(nodeNumber_ + iEdge);
	}
	
	for(int iNode=0; iNode<nodeNumber_; ++iNode) {
		fg.variableSignatures[iNode] = std::vector<int>(1 + fg.adjListOfVariables[iNode].size() , -1);
		fg.factorSignatures[iNode] = std::vector<int>(1 + fg.adjListOfFactors[iNode].size(),-1);
	}
	for(int iEdge=0; iEdge<edgeNumber_; ++iEdge) {
		fg.factorSignatures[nodeNumber_ + iEdge] = std::vector<int>(1 + fg.adjListOfFactors[nodeNumber_ + iEdge].size(), -1);
	}
	cout << "Took "<< ((double)(clock()-start))/CLOCKS_PER_SEC<< " seconds for initialize_factor_graph" << endl;
	return true;
}

void LiftedCoopCut::unlift(std::vector<int>& src, FactorGraph& fg, std::vector<int>& dst) {
	for(int i=0; i<dst.size(); ++i) {
		dst[i] = src[ fg.invsyms[i] ];
	}
}


void LiftedCoopCut::saveLifting(FactorGraph& fg, string& name) {
	// For each pixel, write the group number. It becomes a simple vector.
	ofstream file;
	
	string directory_temp = "syms/" + name + "/";
	wstring directory(directory_temp.begin(), directory_temp.end());
	CreateDirectory(directory.c_str(), NULL);

	file.open("syms/" + name + "/syms."
		+ to_string(nRanksConsidered) + "." + to_string(nCBPIterations) + "." + to_string(nSqrtSpatialGridCount) 
		+ ".txt", fstream::out);
	
	for(int iNode=0; iNode<nodeNumber_; ++iNode) {
		file << fg.invsyms[iNode] << ",";
	}
	
	file << endl;
	file.close();
}

void LiftedCoopCut::setUpLiftedNeighborhood(GCoptimizationGeneralGraph *gc) {
	cout << "Starting setUpLiftedNeighborhood" << endl;
	clock_t start = clock();
	pair2liftedEdge_t::const_iterator iter = pair2liftedEdge_.begin();
	int supernode0, supernode1;
	while(iter != pair2liftedEdge_.end()) {
		supernode0 = iter->first.first;
		supernode1 = iter->first.second;
		gc -> setNeighbors(supernode0, supernode1);
		iter++;
	}
	cout << "Finished setUpLiftedNeighborhood in " << ((double)(clock() - start)/CLOCKS_PER_SEC) << endl;
}

void LiftedCoopCut::initialize_variable_colors(FactorGraph& fg) {
	for(int i=0; i<nodeNumber_; ++i) {
		std::fill(fg.variableSignatures[i].begin(), fg.variableSignatures[i].end(), -1);
		fg.variableSignatures[i][0] = 0;
	}
}

struct sort_pred {
	bool operator()(const std::pair<double,int> &left, const std::pair<double,int> &right) {
		return left.first < right.first;
	}
};
int LiftedCoopCut::initialize_unary_colors(FactorGraph& fg) {
	typedef boost::unordered_map< std::vector<int> , int > unordered_map;
	cout << "Starting initialize_unary_colors" << endl;
	clock_t start = clock();
	unordered_map ranks2color;
	unordered_map::const_iterator find_ranks;
	std::vector< std::pair<double, int> > potential_and_index(labelNumber_);
	std::vector<int> ranks(nRanksConsidered,-1);
	int color;
	int ncolors=0;
	for(int v=0; v<nodeNumber_; ++v) {
		// read potential_and_index
		for(int l=0; l<labelNumber_; ++l) {
			potential_and_index[l] = std::make_pair(unaryTerms_[getUnaryTermIndex(v,l)], l);
		}
		std::sort(potential_and_index.begin(), potential_and_index.end(),sort_pred());
		// prepare ranks
		for(int r=0; r<nRanksConsidered; ++r) {
			ranks[r] = potential_and_index[r].second;
		}
		// if ranks has been seen, then assign that color.
		find_ranks = ranks2color.find(ranks);
		if(find_ranks==ranks2color.end()) {
			ranks2color[ranks] = ncolors;
			color = ncolors;
			ncolors++;
		} else {
			color = find_ranks->second;
		}
		std::fill(fg.factorSignatures[v].begin(), fg.factorSignatures[v].end(), -1);
		fg.factorSignatures[v][0] = color;
	}
	cout << "initialize_unary_colors took " << ((double)(clock()-start)/CLOCKS_PER_SEC) << endl;
	return color;
}

#define THRESHOLD_SEARCH_WINDOW 10000
int LiftedCoopCut::initialize_unary_colors_threshold(FactorGraph& fg) {
	cout << "Started initialize_unary_colors_threshold()" << endl;
	clock_t start = clock();
	std::vector<int> unary_id2color(nodeNumber_,-1);
	double dist = 0.0;
	int color = 0;
	for(int v1=0; v1<nodeNumber_; ++v1) {
		for (int v2=v1-THRESHOLD_SEARCH_WINDOW; v2<v1; ++v2) {
			if (v2<0) continue; // Dont be stoopid.

			for (int l=0; l<labelNumber_; ++l) {
				dist += fabs(unaryTerms_[getUnaryTermIndex(v1,l)] - unaryTerms_[getUnaryTermIndex(v2,l)]);
			}
			dist /= labelNumber_;

			if (dist < thresholdForSymmetry) {
				unary_id2color[v1] = unary_id2color[v2];
				continue;
			}
		}
		if (unary_id2color[v1]==-1) {
			unary_id2color[v1] = color;
			color++;
		}
		std::fill(fg.factorSignatures[v1].begin(), fg.factorSignatures[v1].end(), -1);
		fg.factorSignatures[v1][0] = unary_id2color[v1];
	}
	cout << "initialize_unary_colors_threshold() took " << ((double)(clock()-start)/CLOCKS_PER_SEC) << endl;
	return color;
}


void LiftedCoopCut::prelift(FactorGraph& fg) {
	cout << "Starting prelift(FactorGraph& fg)" << endl;
	clock_t start = clock();
	// ASSERT: !fg.initialized_unary_colors
	nUnaryColorsUsed = 0;
	nUnaryColorsUsed = initialize_unary_colors(fg);
	initialize_variable_colors(fg); // copies colors from unary
	fg.nowFactorSignatures = fg.factorSignatures;
	fg.nowVariableSignatures = fg.variableSignatures;
	fg.initialized_unary_colors = true;
	fg.syms.clear();
	fg.invsyms.resize(nodeNumber_, -1);
	// ASSERT: fg.initialized_unary_colors
}


void LiftedCoopCut::prelift_threshold(FactorGraph& fg) {
	cout << "Starting prelift_threshold(FactorGraph& fg)" << endl;
	clock_t start = clock();
	// ASSERT: !fg.initialized_unary_colors
	nUnaryColorsUsed = 0;
	nUnaryColorsUsed = initialize_unary_colors_threshold(fg);
	initialize_variable_colors(fg); // copies colors from unary
	fg.nowFactorSignatures = fg.factorSignatures;
	fg.nowVariableSignatures = fg.variableSignatures;
	fg.initialized_unary_colors = true;
	fg.syms.clear();
	fg.invsyms.resize(nodeNumber_, -1);
	// ASSERT: fg.initialized_unary_colors	
}

void LiftedCoopCut::set_groups(FactorGraph& fg) {
	// Simply go over variables and put them into syms, invsyms
	// ASSERT: variableSignatures[v][0] is the color of variable v.
	cout << "Starting set_groups" << endl;
	clock_t start = clock();
	fg.syms.clear();
	fg.invsyms.resize(fg.vecVariables.size());
	std::vector<int> color2group(fg.vecVariables.size(),-1); // at worst, one color for each variable.
	int color, group;
	int total_groups=0;
	for(int v=0; v<fg.vecVariables.size(); ++v) {
		color = fg.nowVariableSignatures[v][0];
		assert(color>=0 && color< fg.vecVariables.size());
		group = color2group[color];
		if(group==-1) {
			color2group[color] = total_groups;
			group = total_groups;
			total_groups++;
			fg.syms.push_back(std::vector<int>());
		}
		fg.syms[group].push_back(v);
		fg.invsyms[v] = group;
	}
	cout << "Done with set_groups in " << ((double)(clock() - start)/CLOCKS_PER_SEC) << endl;
	return;
}

void LiftedCoopCut::recolorVariables(FactorGraph& fg) {
	typedef boost::unordered_map< std::vector<int> , int > unordered_map;
	clock_t tick = clock();
	unordered_map sig2color;
	unordered_map::const_iterator find_sig;
	std::vector<int> signature;
	int ncolors = 0;
	int color;
	for(int v=0; v<nodeNumber_; ++v) {
		signature = fg.nowVariableSignatures[v];
		std::sort(signature.begin()+1, signature.end());
		find_sig = sig2color.find(signature);
		if(find_sig == sig2color.end()) {
			sig2color[signature] = ncolors;
			color = ncolors;
			ncolors++;
		} else {
			color = find_sig->second;
		}
		fg.nowVariableSignatures[v][0] = color;
	}
	cout << "recolorVariables took " << ((double)(clock()-tick)/CLOCKS_PER_SEC) << endl;
	return;
}

void LiftedCoopCut::recolorFactors(struct FactorGraph& fg) {
	typedef boost::unordered_map< std::vector<int> , int > unordered_map;
	clock_t tick = clock();
	unordered_map sig2color;
	unordered_map::const_iterator find_sig;
	std::vector<int> signature;
	int ncolors = 0;
	int color;
	for(int f=0; f<(nodeNumber_ + edgeNumber_); ++f) {
		// signature = fg.nowFactorSignatures[f]; // ew, copy constructor.
		std::sort(fg.nowFactorSignatures[f].begin()+1, fg.nowFactorSignatures[f].end());
		find_sig = sig2color.find(fg.nowFactorSignatures[f]);
		if(find_sig==sig2color.end()) {
			sig2color[fg.nowFactorSignatures[f]] = ncolors;
			color = ncolors;
			ncolors++;
		} else {
			color = find_sig->second;
		}
		fg.nowFactorSignatures[f][0] = color;
	}
	cout << "recolorFactors took " << ((double)(clock()-tick)/CLOCKS_PER_SEC) << endl;
	return;
}

void LiftedCoopCut::doCBPIteration(FactorGraph& fg, bool last) {
	// pass messages from factors to variables
	cout << "Starting CBP iteration" << endl;
	clock_t tick = clock();
	for(int v=0; v<nodeNumber_; ++v) {
		for(int fidx=0; fidx< fg.adjListOfVariables[v].size(); ++fidx) {
			int f = fg.adjListOfVariables[v][fidx];
			fg.nowVariableSignatures[v][1+fidx] = fg.nowFactorSignatures[f][0];
		}
	}
	cout << "fac->var took " << ((double)(clock()-tick))/CLOCKS_PER_SEC << endl;
	// recolor variables
	recolorVariables(fg);
	// pass messages from variables to factors
	if(!last) {
		tick = clock();
		for(int f=0; f<(nodeNumber_ + edgeNumber_); ++f) { // need to speed this up anyway.
			for(int vidx=0; vidx< fg.adjListOfFactors[f].size(); ++vidx) {
				int v = fg.adjListOfFactors[f][vidx];
				fg.nowFactorSignatures[f][1+vidx] = fg.nowVariableSignatures[v][0];
			}
		}
		cout << "var->fac took " << ((double)(clock()-tick))/CLOCKS_PER_SEC << endl;
		// recolor factors
		recolorFactors(fg);
	} else {
		cout << "Last iteration, hence no factor recoloring" << endl;
	}
	return;
}

// Section2: normal-lifting ---------------------------------------------------

void LiftedCoopCut::set_lifted_graph(FactorGraph& fg) {
	// ASSERT: syms, invsyms have been set.
	// aux variables
	cout << "Starting set_lifted_graph " << endl;
	clock_t start = clock();
	liftedNodeNumber_ = fg.syms.size();
	liftedUnaryTerms_ = (double*) malloc(liftedNodeNumber_*labelNumber_*sizeof(double));	

	// unary terms
	for(int g=0; g<liftedNodeNumber_; ++g) {
		for(int l=0; l<labelNumber_; ++l) {
			liftedUnaryTerms_[ g*labelNumber_ + l ] = 0.0;
		}
		for(int i=0; i<fg.syms[g].size(); ++i) {
			int v = fg.syms[g][i];
			for(int l=0; l<labelNumber_; ++l) {
				liftedUnaryTerms_[ g*labelNumber_ + l ] += unaryTerms_[v*labelNumber_ + l];
			}
		}
	}
	cout << "\tUnary terms are ready" << endl;
	// pairwise terms
	// need to populate pair2liftedEdge_
	pair2liftedEdge_.clear();
	int node0, node1, supernode0, supernode1, class0, class1/*, class_idx*/;
	std::vector<int>::iterator class_finder;
	pair2liftedEdge_t::iterator p2le_iter;
	double w;
	for(int iEdge=0; iEdge<edgeNumber_; ++iEdge) {
		node0 = edges_[ 2*iEdge + 0 ];
		supernode0 = fg.invsyms[node0];
		class0 = edgeClasses_[2*iEdge + 0];
		node1 = edges_[ 2*iEdge + 1 ];
		supernode1 = fg.invsyms[node1];
		class1 = edgeClasses_[2*iEdge + 1];
		if(supernode0==supernode1) continue;

		if( supernode0 > supernode1 ) {
			int t = supernode1;
			supernode1 = supernode0;
			supernode0 = t;
			t = class1;
			class1 = class0;
			class0 = t;
		}
		std::pair<int,int> supernode_pair = std::make_pair(supernode0,supernode1);
		p2le_iter = pair2liftedEdge_.find(supernode_pair);
		if(p2le_iter == pair2liftedEdge_.end()) {
			p2le_iter = pair2liftedEdge_.insert(std::make_pair(supernode_pair, LiftedEdge(supernode0, supernode1, 0.0, 1 + edgeClassNumber_, labelNumber_))).first;
		}
		w = edgeWeights_[iEdge];
		p2le_iter->second.total_weight += w;

		if (class0>= edgeClassNumber_) class0 = edgeClassNumber_;
		if (class1>= edgeClassNumber_) class1 = edgeClassNumber_; // IMPORTANT. Hack to make computation easier.

		// Forward class.
		bool found = false;
		for(int i=0; i<p2le_iter->second.forwards.size(); ++i) {
			if (class0==p2le_iter->second.forwards[i].edge_class) {
				p2le_iter->second.forwards[i].edge_weight += w;
				found = true;
				break;
			}
		}
		if(!found) {
			class_data_t new_class;
			new_class.edge_class = class0;
			new_class.edge_weight = w;
			p2le_iter->second.forwards.push_back(new_class);
		}
		
		// Backward class.	
		found = false;
		for(int i=0; i<p2le_iter->second.backwards.size(); ++i) {
			if (class1==p2le_iter->second.backwards[i].edge_class) {
				p2le_iter->second.backwards[i].edge_weight += w;
				found = true;
				break;
			}
		}
		if(!found) {
			class_data_t new_class;
			new_class.edge_class = class1;
			new_class.edge_weight = w;
			p2le_iter->second.backwards.push_back(new_class);
		}
	}

	liftedEdgeNumber_ = pair2liftedEdge_.size();
	cout << "Ending set_lifted_graph in " << ((double)(clock()-start)/CLOCKS_PER_SEC) << endl;
}

void LiftedCoopCut::initialize_pairwise_colors(FactorGraph& fg) {
	// Divide purely based on geography... ignore edge classes?
	// need to set fg.nowFactorSignatures[nodeNumber_ + iEdge][0] according to some heuristic.
	// For AE, lets just do geography.
	// we get to use the parameter nSqrtSpatialGridCount
	// Figure out the location of the edge's center, see which grid it falls into.
	cout << "Starting initialize_pairwise_colors " << endl;
	clock_t start = clock();
	int node0, node1, class0, class1, /*class_coord,*/ temp;
	// int gx,gy,grid_coord; // the xy themselves dont matter.
	typedef boost::unordered_map< std::pair<int,int> ,int> unordered_map; // goes from <class,class> to color.
	std::pair<int,int> key;
	unordered_map edge2color;
	unordered_map::const_iterator iter;

	int nextcolor = nUnaryColorsUsed;

	for(int iEdge=0; iEdge<edgeNumber_; ++iEdge) {
		node0 = edges_[iEdge*2 + 0];
		node1 = edges_[iEdge*2 + 1];
		class0 = edgeClasses_[iEdge*2 + 0];
		class1 = edgeClasses_[iEdge*2 + 1];
		// swap
		if(node0>node1) {
			temp = node1;
			node1 = node0;
			node0 = temp;
			temp = class1;
			class1 = class0;
			class0 = temp;
		}
		// gx = (node1%nx_ + node0%nx_)/(2*nSqrtSpatialGridCount);
		// gy = (node1/nx_ + node0/nx_)/(2*nSqrtSpatialGridCount);
		// class_coord = class0*edgeClassNumber_ + class1;
		// grid_coord = gx*nSqrtSpatialGridCount + gy;
		// grid_coord = 1; // TODO, make it pointless to color on geography.
		key = std::make_pair(class0, class1);
		if ((iter = edge2color.find(key)) == edge2color.end()) {
			edge2color[key] = nextcolor;
			std::fill(fg.nowFactorSignatures[ nodeNumber_ + iEdge ].begin(), fg.nowFactorSignatures[ nodeNumber_ + iEdge ].end(), -1);
			fg.nowFactorSignatures[ nodeNumber_ + iEdge ][0] = nextcolor;
			nextcolor++;
		} else {
			std::fill(fg.nowFactorSignatures[ nodeNumber_ + iEdge ].begin(), fg.nowFactorSignatures[ nodeNumber_ + iEdge ].end(), -1);
			fg.nowFactorSignatures[ nodeNumber_ + iEdge ][0] = iter->second;
		}
	}
	cout << "Finished initialize_pairwise_colors in " << ((double)(clock()-start)/CLOCKS_PER_SEC) << endl;
}

void LiftedCoopCut::lift(FactorGraph& fg) { // Doesn't take hValue because it doesn't care about hValue.
	// ASSERT: prelift has been done.
	cout << "Starting lift" << endl;
	clock_t start = clock();
	initialize_pairwise_colors(fg);
	bool last;
	for(int i=0; i<nCBPIterations; ++i) {
		clock_t starti = clock();
		last = (i==(nCBPIterations-1));
		doCBPIteration(fg, last);
		cout << "Iteration " << i << " of cbp took " << ((double)(clock() - starti)/CLOCKS_PER_SEC) << endl;
	}
	set_groups(fg);
	set_lifted_graph(fg);
	cout << "Done lift() in " << ((double)(clock() - start)/CLOCKS_PER_SEC) << endl;
	cout << "   Old: " << nodeNumber_ << " nodes and " << edgeNumber_ << " edges" << endl;
	cout << "   New: " << liftedNodeNumber_ << " nodes and " << liftedEdgeNumber_ << " edges" << endl;
}

// Section 3: repeated lifting ---------------------------------------------------

void LiftedCoopCut::initializeAllEdgeLookups(vector<vector<int>>& hValue) {
	// Go over each lifted edge, and initialize its values.
	return;
	clock_t start = clock();
	oldH_value = hValue;
	initialized_hvalues = true;
	// double edgeWeight;
	for(pair2liftedEdge_t::iterator iter=pair2liftedEdge_.begin(); iter!=pair2liftedEdge_.end(); iter++) {
		// Modify iter->second.forwardWeightsGivenH and iter->second.backwardWeightsGivenH
		for(int l=0; l<labelNumber_; ++l) {
			for(int c=0; c<edgeClassNumber_; ++c) {
				// edgeWeight = iter->second.forwardWeights[c];
				// iter->second.forwardWeightsGivenH[l] += (hValue[ l ][ c ] == 1) ? edgeWeight : edgeWeight * alpha_;
				// edgeWeight = iter->second.backwardWeights[c];
				// iter->second.backwardWeightsGivenH[l] += (hValue[ l ][ c ] == 1) ? edgeWeight : edgeWeight * alpha_;
			}
			// edgeWeight = iter->second.forwardWeights[edgeClassNumber_];
			// iter->second.forwardWeightsGivenH[l] += edgeWeight;
			// edgeWeight = iter->second.backwardWeights[edgeClassNumber_];
			// iter->second.backwardWeightsGivenH[l] += edgeWeight;			
		}
	}
	cout << "initializeAllEdgeLookups took " << ((double)(clock()-start)/CLOCKS_PER_SEC) << endl;
}

void LiftedCoopCut::updateLiftedWeights_byToggling(int l, int c) {
	// Modify each edge
	return;
/*	clock_t start = clock();
	int newval = 1 - oldH_value[l][c];
	double edgeWeight;
	for(pair2liftedEdge_t::iterator iter=pair2liftedEdge_.begin(); iter!=pair2liftedEdge_.end(); iter++) {
		// Modify iter->second.forwardWeightsGivenH and iter->second.backwardWeightsGivenH
		// edgeWeight = iter->second.forwardWeights[c];
		// iter->second.forwardWeightsGivenH[l] += edgeWeight * (1 - alpha_) * ( (newval==1)? +1 : -1 );
		
		// edgeWeight = iter->second.backwardWeights[c];
		// iter->second.backwardWeightsGivenH[l] += edgeWeight * (1 - alpha_) * ( (newval==1)? +1 : -1 );
	}
	// update oldH_value
	oldH_value[l][c] = newval;

	cout << "updateLiftedWeights_byToggling took " << ((double)(clock()-start)/CLOCKS_PER_SEC) << endl;*/
}
// --- end of section 3.


// Section 4: alpha-expansion ---------------------------------------------------

GCoptimization::EnergyTermType LiftedCoopCut::liftedComputePairwiseCost(
			GCoptimization::SiteID s1, GCoptimization::SiteID s2,
			GCoptimization::LabelID l1, GCoptimization::LabelID l2
			) {
	if (l1==l2) return 0;
	if(s1>s2) {
		int t = s1;
		s1 = s2;
		s2 = t;
	}
	std::pair<int, int> node_pair = std::make_pair(s1,s2);
	return pair2liftedEdge_[node_pair].total_weight * lambda_;
}
GCoptimization::EnergyTermType liftedGlobalPairwiseCost(
			GCoptimization::SiteID s1, 	GCoptimization::SiteID s2, 
			GCoptimization::LabelID l1,	GCoptimization::LabelID l2,
			void * objectPtr
		) {
	LiftedCoopCut* objectPtrCorType = (LiftedCoopCut*) objectPtr;
	return objectPtrCorType -> liftedComputePairwiseCost(s1, s2, l1, l2);	
}

bool LiftedCoopCut::minimizeLiftedEnergy_alphaExpansion() {
	double tAlgoStart = clock();
	double tStart = clock();
	cout << "In LiftedAlphaExpansion- Preparing energy" << endl;

	if ( !pairwiseLoaded_ || !unaryLoaded_ ) {
		return false;
	}


	// unaryTerms_  - array of unary potentials

	// prepare grid data structure
	// const int maxDegree_ = 8; // this number is a hack!!! DONE: compute it from the input
	labeling_.resize( nodeNumber_ );   // stores result of optimization
	constructIndexingDataStructure( maxDegree_ );

	// And lift!
	initialize_factor_graph(fg_);
	prelift(fg_); // initializes signatures and colors unary factors
	lift(fg_); // this does pairwise color and counting BP...also creates fg.syms, fg.invsyms and liftedNodeNumber etc.
	liftedLabeling_.resize(liftedNodeNumber_);
	saveLifting(fg_, name);
	
	try{	
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(liftedNodeNumber_, labelNumber_);
		gc->setDataCost(liftedUnaryTerms_);
		
		gc->setSmoothCost( liftedGlobalPairwiseCost, this );

		//gc->setSmoothCost(smooth);
		
		cout << "Time: " << (clock() - tStart) / CLOCKS_PER_SEC << "s" << endl;

		// now set up a grid neighborhood system
		cout << "Setting the neighborhood" << endl;
		tStart = clock();
		setUpLiftedNeighborhood(gc); // to implement

		// gc -> setAllNeighbors( neighborNumber_, neighboringNode_, neighboringEdgeWeight_);

		cout << "Time: "<< (clock() - tStart) / CLOCKS_PER_SEC << endl;
		
		cout << "Before optimization energy is " << gc->compute_energy() << endl;
		
		cout << "Running lifted-alpha-expansion" << endl;
		tStart = clock();
		gc->expansion(10);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		cout << "Only expansion Time: " << (clock() - tStart) / CLOCKS_PER_SEC << endl;

		cout << "Computing the energy and labeling" << endl;
		tStart = clock();

		energy_ = gc->compute_energy();
		cout << "After optimization energy is "<< energy_ << endl;

		for ( int  iNode = 0; iNode < liftedNodeNumber_; ++iNode ) {
			liftedLabeling_[ iNode ] = gc->whatLabel( iNode );
		}
		cout << "Time: "<< (clock() - tStart) / CLOCKS_PER_SEC << endl;

		cout << "Deleting GCO object" << endl;
		tStart = clock();

		delete gc;

		cout << "Time: "<< (clock() - tStart) / CLOCKS_PER_SEC << endl;

	}
	catch (GCException e){
		e.Report();
	}
	unlift(liftedLabeling_, fg_, labeling_);

	elapsedTime_ = (clock() - tAlgoStart) / CLOCKS_PER_SEC;
	resultsComputed_ = true;

	double energy = computeCoopEnergy( labeling_ );
	if( true || fabs( energy - energy_ ) > 1e-3 ){
		cout << "Correct energy: "<< energy << endl;
		energy_ = energy;
	}
	return true;
}


// Section 4: greedy - 
GCoptimization::EnergyTermType liftedGlobalPairwiseCostOneBreakPoint
	(
	GCoptimization::SiteID s1, 
	GCoptimization::SiteID s2, 
	GCoptimization::LabelID l1, 
	GCoptimization::LabelID l2,
	void * objectPtr
	)
{
	LiftedCoopCut* objectPtrCorType = (LiftedCoopCut*) objectPtr;
	return objectPtrCorType -> liftedComputePairwiseCostOneBreakPoint(s1, s2, l1, l2);
}


GCoptimization::EnergyTermType LiftedCoopCut::liftedComputePairwiseCostOneBreakPoint
	(
	GCoptimization::SiteID s1, 
	GCoptimization::SiteID s2, 
	GCoptimization::LabelID l1, 
	GCoptimization::LabelID l2
	)
{
	if (l1 == l2) return 0;	
	if (s1>s2) {
		int t = s1;
		s1 = s2;
		s2 = t;
		t = l1;
		l1 = l2;
		l2 = t;
	}

	std::pair<int,int> key=std::make_pair(s1,s2);
	pair2liftedEdge_t::iterator iter = pair2liftedEdge_.find(key);
	double weightForward = 0;
	double weightBackward = 0;
	// int edgeClassForward, edgeClassBackward;
	double edge_weight;
	int edge_class;
	for(int i=0; i<iter->second.forwards.size(); ++i) {
		edge_weight = iter->second.forwards[i].edge_weight;
		if ( (edge_class = iter->second.forwards[i].edge_class) < edgeClassNumber_ ) {
			edge_weight = edge_weight*(( hValue_[l1][edge_class]==1 )?(1.0):(alpha_));
		}
		weightForward += edge_weight;
	}

	for(int i=0; i<iter->second.backwards.size(); ++i) {
		edge_weight = iter->second.backwards[i].edge_weight;
		if ( (edge_class = iter->second.backwards[i].edge_class) < edgeClassNumber_ ) {
			edge_weight = edge_weight*(( hValue_[l2][edge_class]==1 )?(1.0):(alpha_));
		}
		weightBackward += edge_weight;
	}

	return lambda_ * (weightForward + weightBackward) / 2;
}

double LiftedCoopCut::oneRunLiftedAlphaExpansion(std::vector<int> &labeling) 
{
	double energy = 0.0;
	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(liftedNodeNumber_, labelNumber_);
		gc->setDataCost(liftedUnaryTerms_);
		gc->setSmoothCost( liftedGlobalPairwiseCostOneBreakPoint, this );
		setUpLiftedNeighborhood(gc);
		
		// gc -> setAllNeighbors( neighborNumber_, neighboringNode_, neighboringEdgeWeight_);

		for(int iNode = 0; iNode < liftedNodeNumber_; ++iNode)
			gc -> setLabel(iNode, 0);
		
		// cout << "Still have enough memory!" << endl;		
		gc->expansion(10);
		
		energy = gc->compute_energy();
	
		// labeling.resize(liftedNodeNumber_);
		for ( int  iNode = 0; iNode < liftedNodeNumber_; ++iNode )
			labeling[ iNode ] = gc->whatLabel( iNode );
		
		delete gc;
		
	}
	catch (GCException e){
		e.Report();
		return 1e+20;
	}
	return energy;
}

bool LiftedCoopCut::minimizeLiftedEnergy_greedy() {
	double tAlgoStart = clock();
	double tStart = clock();

	hyperparameters_t ground_params;
		ground_params.k = ground_params.ncbp = 0;
	hyperparameters_t start_params;
		start_params.k = start_params.ncbp = 1;
	hyperparameters_t intermediate_params;
		intermediate_params.k = 1 + int(labelNumber_/2);
		intermediate_params.ncbp = 1;
	hyperparameters_t intermediate_params2;
		intermediate_params2.k = 3;
		intermediate_params2.ncbp = 1;
	
	std::vector<hyperparameters_t> hyperparameters = {/*start_params, */intermediate_params, /*intermediate_params2, ground_params*/};
	volatile int active_idx = 0;
	int max_active_idx = hyperparameters.size()-1;
	// setLiftingParameters(hyperparameters[active_idx].k, hyperparameters[active_idx].ncbp);
	name = name + "." + to_string(nRanksConsidered) + "." + to_string(nCBPIterations) + ".bin";
	cout << "Name: " << name << endl;
	cout << "Starting hybrid greedy ... Preparing energy" << endl;
	if ( !pairwiseLoaded_ || !unaryLoaded_ ) {
		return false;
	}
	labeling_.resize( nodeNumber_ );   // stores result of optimization
	constructIndexingDataStructure( maxDegree_ ); 	
	initialize_factor_graph(fg_);

	hValue_.resize( labelNumber_, vector<int>( edgeClassNumber_, 1) );
	double hSum = 0.0;
	// initialize hValue_
	prelift(fg_); // initializes signatures and colors unary factors
	lift(fg_);
	

	liftedLabeling_.resize(liftedNodeNumber_);
	vector<int> curLabeling(liftedNodeNumber_);
	std::vector<int> swapLabeling_(nodeNumber_); // used as buffer while copying liftedLabeling_

	saveLifting(fg_, name);
	initializeAllEdgeLookups(hValue_);

	int energies_idx = 0;
	int NITERATIONS_ENERGIES = 1;

	// std::vector< vector< vector<double> > > energies(NITERATIONS_ENERGIES, vector< vector<double> >( labelNumber_, vector<double>(edgeClassNumber_, 0.0) ) );
	std::vector<double> energies(labelNumber_*edgeClassNumber_*NITERATIONS_ENERGIES, 0.0);

	energy_ = oneRunLiftedAlphaExpansion( liftedLabeling_ );
	energies[energies_idx] = energy_;
	energies_idx++;

	for(int iIter = 0; iIter < greedyMaxIter_; ++iIter) {
		bool changed = false;
		double greedy_iteration_time = 0.0;
		// Do iteration
		for(int iLabel = 0; iLabel < labelNumber_; ++iLabel) {
			for(int iEdgeClass = 0; iEdgeClass < edgeClassNumber_; ++ iEdgeClass) {
				cout << "Params: (" << nRanksConsidered << "," << nCBPIterations << ") " << endl;
				// toggle hValue_
				hValue_[iLabel][iEdgeClass] = 1 - hValue_[iLabel][iEdgeClass];
				if (hValue_[iLabel][iEdgeClass] == 1) {
					hSum -= (1 - alpha_) * classThreshold_[iEdgeClass];
				} else {
					hSum += (1 - alpha_) * classThreshold_[iEdgeClass];
				}

				// AE
				double energy;
				energy = oneRunLiftedAlphaExpansion(curLabeling);

				// accept or reject hValue
				if (energy + lambda_ * hSum < energy_) {
					// accept the change 
					energy_ = energy + lambda_ * hSum;
					changed = true;
					liftedLabeling_ = curLabeling;
					

					printf("Changing h(%d, %d, %d) to %d, energy is %f", iLabel, iEdgeClass, iIter, hValue_[iLabel][iEdgeClass], energy_);
					cout << endl;
				} else {
					// return h back
					printf("Changing h(%d, %d, %d) to %d REJECTED, energy is %f", iLabel, iEdgeClass, iIter, hValue_[iLabel][iEdgeClass], energy + lambda_  * hSum);
					cout << endl;
					hValue_[iLabel][iEdgeClass] = 1 - hValue_[iLabel][iEdgeClass];
					if (hValue_[iLabel][iEdgeClass] == 1) {
						hSum -= (1 - alpha_) * classThreshold_[iEdgeClass];
					} else {
						hSum += (1 - alpha_) * classThreshold_[iEdgeClass];
					}
				}

				// update energies
				energies[energies_idx] = energy_;
				energies_idx = (energies_idx+1)%energies.size();

				// save labeling_
				unlift(liftedLabeling_, fg_, labeling_);
				saveLabels(iIter, iLabel, iEdgeClass);
				
			}
		}
		if (!changed)
			break;
	}

	// ASSERT: labeling_ is ready.
	elapsedTime_ = (clock() - tAlgoStart) / CLOCKS_PER_SEC;
	printf("Energy: %f, time: %f", energy_, elapsedTime_);
	cout << endl;

	double energy = computeCoopEnergy( labeling_ );
	if( true || fabs( energy - energy_ ) > 1e-3 ){
		printf("Correct energy: %f", energy);
		cout << endl;
		energy_ = energy;
	}

	resultsComputed_ = true;

	return true;
}

double energy_change_threshold_fraction = 0.0;
int energies_window_size = 0;

typedef struct hyperparameters_t hyperparameters_t;
bool LiftedCoopCut::minimizeHybridEnergy_greedy() {
	energies_window_size = labelNumber_;
	double tAlgoStart = clock();
	double tStart = clock();
	cout << "Name: " << name << endl;
	cout << "Starting hybrid greedy ... Preparing energy" << endl;
	if ( !pairwiseLoaded_ || !unaryLoaded_ ) {
		return false;
	}
	labeling_.resize( nodeNumber_ );   // stores result of optimization
	constructIndexingDataStructure( maxDegree_ ); 	
	initialize_factor_graph(fg_);

	hValue_.resize( labelNumber_, vector<int>( edgeClassNumber_, 1) );
	double hSum = 0.0;

	std::vector<hyperparameters_t> hyperparameters;

	hyperparameters_t ground_params;
		ground_params.k = ground_params.ncbp = 0;
	hyperparameters_t start_params;
		start_params.k = int((labelNumber_+1)/2); // 2 if labelNumber_==3 or 4. 3 if labelNumber_==5.
		start_params.ncbp = 1;
	/*
		What we want:
			1_11_s: 2,1 -> 2,2 -> ground. labelNumber_ = 3
			2_21_s: 2,1 -> 2,2 -> ground. labelNumber_ = 3
			3_24_s: 2/3,1 -> ground? labelNumber_ = 4
			4_26_s: 2/3,1 -> ground? labelNumber_ = 4
			17_12_s: 3,1 -> 3,2 -> ground. labelNumber_ = 5
	*/
	hyperparameters.push_back(start_params);
	if(labelNumber_==3) {
		hyperparameters_t mid;
			mid.k = int((labelNumber_+1)/2);
			mid.ncbp=2;
		hyperparameters.push_back(mid);
	} else if(labelNumber_==4) {
		// Do nothing.
	} else {
		hyperparameters_t mid;
			mid.k = int((labelNumber_+1)/2);
			mid.ncbp=2;
		hyperparameters.push_back(mid);
	}
	hyperparameters.push_back(ground_params);

	volatile int active_idx = 0;

	int max_active_idx = hyperparameters.size()-1;
	setLiftingParameters(hyperparameters[active_idx].k, hyperparameters[active_idx].ncbp);
	// initialize hValue_
	prelift(fg_); // initializes signatures and colors unary factors
	lift(fg_);
	

	liftedLabeling_.resize(liftedNodeNumber_);
	vector<int> curLabeling(liftedNodeNumber_);
	std::vector<int> swapLabeling_(nodeNumber_); // used as buffer while copying liftedLabeling_

	saveLifting(fg_, name);
	initializeAllEdgeLookups(hValue_);

	int energies_idx = 0;
	int NITERATIONS_ENERGIES = 1;

	// std::vector< vector< vector<double> > > energies(NITERATIONS_ENERGIES, vector< vector<double> >( labelNumber_, vector<double>(edgeClassNumber_, 0.0) ) );
	std::vector<double> energies(labelNumber_*edgeClassNumber_*NITERATIONS_ENERGIES, 0.0);

	energy_ = oneRunLiftedAlphaExpansion( liftedLabeling_ );
	energies[energies_idx] = energy_;
	energies_idx++;
	int lastLabelThatCausedChange = 0;
	int lastEdgeClassThatCausedChange = 0;
	for(int iIter = 0; iIter < greedyMaxIter_; ++iIter) {
		bool changed = false;
		double greedy_iteration_time = 0.0;
		// Do iteration
		for(int iLabel = 0; iLabel < labelNumber_; ++iLabel) {
			for(int iEdgeClass = 0; iEdgeClass < edgeClassNumber_; ++iEdgeClass) {
				cout << "Params: (" << hyperparameters[active_idx].k << "," << hyperparameters[active_idx].ncbp << ") " << endl;
				// toggle hValue_
				hValue_[iLabel][iEdgeClass] = 1 - hValue_[iLabel][iEdgeClass];
				if (hValue_[iLabel][iEdgeClass] == 1) {
					hSum -= (1 - alpha_) * classThreshold_[iEdgeClass];
				} else {
					hSum += (1 - alpha_) * classThreshold_[iEdgeClass];
				}

				// AE
				double energy;
				if (active_idx < max_active_idx) {
					energy = oneRunLiftedAlphaExpansion(curLabeling);
				} else {
					energy = oneRunAlphaExpansion(curLabeling);
				}

				// accept or reject hValue
				if (energy + lambda_ * hSum < energy_) {
					// accept the change 
					energy_ = energy + lambda_ * hSum;
					changed = true;
					if(active_idx < max_active_idx) {
						liftedLabeling_ = curLabeling;
					} else {
						labeling_ = curLabeling;
					}
					

					printf("Changing h(%d, %d, %d) to %d, energy is %f", iLabel, iEdgeClass, iIter, hValue_[iLabel][iEdgeClass], energy_);
					lastLabelThatCausedChange = iLabel;
					lastEdgeClassThatCausedChange = iEdgeClass;
					cout << endl;
				} else {
					// return h back
					printf("Changing h(%d, %d, %d) to %d REJECTED, energy is %f", iLabel, iEdgeClass, iIter, hValue_[iLabel][iEdgeClass], energy + lambda_  * hSum);
					cout << endl;
					hValue_[iLabel][iEdgeClass] = 1 - hValue_[iLabel][iEdgeClass];
					if (hValue_[iLabel][iEdgeClass] == 1) {
						hSum -= (1 - alpha_) * classThreshold_[iEdgeClass];
					} else {
						hSum += (1 - alpha_) * classThreshold_[iEdgeClass];
					}
				}

				// update energies
				energies[energies_idx] = energy_;
				energies_idx = (energies_idx+1)%energies.size();

				// save labeling_
				if(active_idx < max_active_idx) {
					unlift(liftedLabeling_, fg_, labeling_);
				}
				saveLabels(iIter, iLabel, iEdgeClass);

				if(isToggleTime(energies_idx, active_idx, max_active_idx, energies)) {
					active_idx++;
					setLiftingParameters(hyperparameters[active_idx].k, hyperparameters[active_idx].ncbp);
					cout << "Beginning toggle" << endl;
					if (active_idx < max_active_idx) {
						cout << "active_idx" << active_idx << " < max_active_idx" << max_active_idx << endl;
						unlift(liftedLabeling_, fg_, swapLabeling_);
						clock_t start = clock();
						prelift(fg_); // initializes signatures and colors unary factors
						lift(fg_);
						clock_t end = clock();
						saveLifting(fg_, name);
						liftedLabeling_.resize(liftedNodeNumber_);
						curLabeling.resize(liftedNodeNumber_);
						lift_labeling(swapLabeling_, fg_, liftedLabeling_);
						cout << "Toggling from (" << hyperparameters[active_idx-1].k << "," << hyperparameters[active_idx-1].ncbp 
							<< ") to (" << hyperparameters[active_idx].k << "," << hyperparameters[active_idx].ncbp << ") in " << ((double)(end-start)/CLOCKS_PER_SEC) << endl; 
						changed = true;
					} else {
						unlift(liftedLabeling_, fg_, labeling_);
						curLabeling.resize(nodeNumber_);
						cout << "Toggling from (" << hyperparameters[active_idx-1].k << "," << hyperparameters[active_idx-1].ncbp 
							<< ") to ground" << endl; 
						iLabel = lastLabelThatCausedChange;
						iEdgeClass = lastEdgeClassThatCausedChange;
						cout << "Setting iLabel to " << iLabel << " and iEdgeClass to " << iEdgeClass << endl;
						changed = true;
					}
				}
			}
		}
		if (!changed)
			break;
	}

	// ASSERT: labeling_ is ready.
	elapsedTime_ = (clock() - tAlgoStart) / CLOCKS_PER_SEC;
	printf("Energy: %f, time: %f", energy_, elapsedTime_);
	cout << endl;

	double energy = computeCoopEnergy( labeling_ );
	if( true || fabs( energy - energy_ ) > 1e-3 ){
		printf("Correct energy: %f", energy);
		cout << endl;
		energy_ = energy;
	}

	resultsComputed_ = true;

	return true;
}

void LiftedCoopCut::saveLabels(int& iterationNumber, int& iLabel, int& iEdgeClass) {
	if( doNotSaveLabels ) return;
	// Saves labeling_
	using namespace cimg_library;
	CImg<int> img(nx_, ny_, 1, 1);
	for(int x=0; x<nx_; ++x) {
		for(int y=0; y<ny_; ++y) {
			img.data(x,y)[0] = labeling_[x*ny_ + y];
		}
	}
	img.normalize(0,255);
	string filename = "logs/hybrid_iteration/" + name + "/labels"
		+ to_string(iterationNumber) + "." + to_string(iLabel) + "." + to_string(iEdgeClass) + ".params." 
		+ to_string(nRanksConsidered) + "." + to_string(nCBPIterations) + "." + to_string(nSqrtSpatialGridCount) 
		+ ".bmp";


	// create the directory!
	string directory_temp = "logs/hybrid_iteration/" + name + "/";
	wstring directory(directory_temp.begin(), directory_temp.end());
	CreateDirectory(directory.c_str(), NULL);
	img.save(filename.c_str());
}


void LiftedCoopCut::lift_labeling(std::vector<int>& ground_labeling, FactorGraph& fg, std::vector<int>& lifted_labeling) {
	lifted_labeling.resize(liftedNodeNumber_);
	for(int i=0; i<liftedNodeNumber_; ++i) {
		lifted_labeling[i] = ground_labeling[ fg.syms[i][0] ]; // it needs to be consistent.
	}
}

static int time_since_last_toggle = 0;


bool LiftedCoopCut::isToggleTime(int energies_idx, int active_idx, int max_active_idx, vector<double>& energies) {

	time_since_last_toggle++;

	if( active_idx == max_active_idx ) return false;
	if( active_idx==0 && energies_idx < (energies_window_size) ) return false;
	if( active_idx== (max_active_idx-1) && time_since_last_toggle < energies_window_size ) return false;
	// cout << "Got here!" << endl;
	// Check if its time to switch
	// we look at energies from energies_idx - (edgeClassNumber_*labelNumber_/2) to energies_idx
	energies_idx -= min(time_since_last_toggle, energies_window_size);
	energies_idx += energies.size();
	// for(int i=1; i<(energies_window_size); ++i) {
	// 	cout << energies[(energies_idx + i)%energies.size()] << " ";
	// }
	// cout << endl;
	for(int i=1; i<(energies_window_size); ++i) {
		double decrease = energies[(energies_idx + i)%energies.size()] - energies[(energies_idx + i-1)%energies.size()];
		if (fabs(decrease) > energy_change_threshold_fraction * fabs(energies[(energies_idx + i)%energies.size()])) {
			return false;
		}
	}
	time_since_last_toggle = 0;
	return true;
}

#include <iostream>
#include <sstream>
#include <fstream>
void LiftedCoopCut::read_slic(FactorGraph& fg, string slicSegments) {
	std::cerr << " Entered read_slic(" << slicSegments << ") " << std::endl;
	std::ifstream infile(slicSegments);
	fg.invsyms.resize(nodeNumber_);
	std::string line;
	std::string grpStr;
	int grp;
	int maxGroup = 0;

	int vidx = 0;
	while(!infile.eof()) {
		getline(infile, line);
		if(line.length()==0) {
			std::cerr << "line.length=0" << std::endl;
			break;
		}
		grp = atoi(line.c_str());
		// Add to invsyms
		fg.invsyms[vidx] = grp;
		maxGroup = (grp>maxGroup)?(grp):maxGroup;
		vidx++;
	}
	fg.syms.resize(maxGroup+1);
	for(int v=0; v<fg.invsyms.size(); ++v) {
		fg.syms[ fg.invsyms[v] ].push_back(v);
	}
	infile.close();
	std::cerr << "Leaving read_slic with fg.syms.size=" << fg.syms.size() << std::endl;
}


bool LiftedCoopCut::minimizeSLICEnergy_greedy() {
	double tAlgoStart = clock();
	double tStart = clock();	

	std::string temp_name = name + ".slic" /*+ to_string(nRanksConsidered) + "." + to_string(nCBPIterations) */ + ".bin";
	cout << "Name: " << temp_name << endl;
	cout << "Starting slic-greedy ... Preparing energy" << endl;
	if ( !pairwiseLoaded_ || !unaryLoaded_ ) {
		return false;
	}

	hValue_.resize( labelNumber_, vector<int>( edgeClassNumber_, 1) );
	double hSum = 0.0;

	labeling_.resize( nodeNumber_ );   // stores result of optimization
	constructIndexingDataStructure( maxDegree_ ); 	
	initialize_factor_graph(fg_);
	// Read fg_ from symmetry file.
	read_slic(fg_, name + ".slic.csv");
	// fg_ is now ready. Simply lift is on the basis of fg_
	int temp = nCBPIterations;
	nCBPIterations = 0;
	set_lifted_graph(fg_);
	nCBPIterations = temp;


	liftedLabeling_.resize(liftedNodeNumber_);
	vector<int> curLabeling(liftedNodeNumber_);
	std::vector<int> swapLabeling_(nodeNumber_); // used as buffer while copying liftedLabeling_

	saveLifting(fg_, name);
	initializeAllEdgeLookups(hValue_);

	int energies_idx = 0;
	int NITERATIONS_ENERGIES = 1;

	// std::vector< vector< vector<double> > > energies(NITERATIONS_ENERGIES, vector< vector<double> >( labelNumber_, vector<double>(edgeClassNumber_, 0.0) ) );
	std::vector<double> energies(labelNumber_*edgeClassNumber_*NITERATIONS_ENERGIES, 0.0);

	energy_ = oneRunLiftedAlphaExpansion( liftedLabeling_ );
	energies[energies_idx] = energy_;
	energies_idx++;
	int lastLabelThatCausedChange = 0;
	int lastEdgeClassThatCausedChange = 0;
	for(int iIter = 0; iIter < greedyMaxIter_; ++iIter) {
		bool changed = false;
		double greedy_iteration_time = 0.0;
		// Do iteration
		for(int iLabel = 0; iLabel < labelNumber_; ++iLabel) {
			for(int iEdgeClass = 0; iEdgeClass < edgeClassNumber_; ++iEdgeClass) {
				
				// toggle hValue_
				hValue_[iLabel][iEdgeClass] = 1 - hValue_[iLabel][iEdgeClass];
				if (hValue_[iLabel][iEdgeClass] == 1) {
					hSum -= (1 - alpha_) * classThreshold_[iEdgeClass];
				} else {
					hSum += (1 - alpha_) * classThreshold_[iEdgeClass];
				}

				// AE
				double energy;
				energy = oneRunLiftedAlphaExpansion(curLabeling);
				
				// accept or reject hValue
				if (energy + lambda_ * hSum < energy_) {
					// accept the change 
					energy_ = energy + lambda_ * hSum;
					changed = true;
					liftedLabeling_ = curLabeling;
					

					printf("Changing h(%d, %d, %d) to %d, energy is %f", iLabel, iEdgeClass, iIter, hValue_[iLabel][iEdgeClass], energy_);
					lastLabelThatCausedChange = iLabel;
					lastEdgeClassThatCausedChange = iEdgeClass;
					cout << endl;
				} else {
					// return h back
					printf("Changing h(%d, %d, %d) to %d REJECTED, energy is %f", iLabel, iEdgeClass, iIter, hValue_[iLabel][iEdgeClass], energy + lambda_  * hSum);
					cout << endl;
					hValue_[iLabel][iEdgeClass] = 1 - hValue_[iLabel][iEdgeClass];
					if (hValue_[iLabel][iEdgeClass] == 1) {
						hSum -= (1 - alpha_) * classThreshold_[iEdgeClass];
					} else {
						hSum += (1 - alpha_) * classThreshold_[iEdgeClass];
					}
				}

				// update energies
				energies[energies_idx] = energy_;
				energies_idx = (energies_idx+1)%energies.size();

				// save labeling_
				unlift(liftedLabeling_, fg_, labeling_);
				saveLabels(iIter, iLabel, iEdgeClass);

			}
		}
	}

	// ASSERT: labeling_ is ready.
	elapsedTime_ = (clock() - tAlgoStart) / CLOCKS_PER_SEC;
	printf("Energy: %f, time: %f", energy_, elapsedTime_);
	cout << endl;

	double energy = computeCoopEnergy( labeling_ );
	if( true || fabs( energy - energy_ ) > 1e-3 ){
		printf("Correct energy: %f", energy);
		cout << endl;
		energy_ = energy;
	}

	resultsComputed_ = true;
	return true;
}

bool LiftedCoopCut::minimizeThresholdEnergy_greedy() {
	// generic 
	double tAlgoStart = clock();
	double tStart = clock();	

	std::string temp_name = name + ".thr" + to_string(thresholdForSymmetry) /*+ to_string(nRanksConsidered) + "." + to_string(nCBPIterations) */ + ".bin";
	cout << "Name: " << temp_name << endl;
	cout << "Starting threshold-greedy ... Preparing energy" << endl;
	cout << "thresholdForSymmetry=" << thresholdForSymmetry << endl;
	if ( !pairwiseLoaded_ || !unaryLoaded_ ) {
		return false;
	}

	hValue_.resize( labelNumber_, vector<int>( edgeClassNumber_, 1) );
	double hSum = 0.0;

	labeling_.resize( nodeNumber_ );   // stores result of optimization
	constructIndexingDataStructure( maxDegree_ ); 	
	initialize_factor_graph(fg_);

	// Based on parameters, obtain the symmetries
	initialize_factor_graph(fg_);
	prelift_threshold(fg_); // initializes signatures and colors unary factors
	lift(fg_); // this does pairwise color and counting BP...also creates fg.syms, fg.invsyms and liftedNodeNumber etc.
	liftedLabeling_.resize(liftedNodeNumber_);
	saveLifting(fg_, temp_name);


	// lift the graph
	// set_lifted_graph(fg_);	

	vector<int> curLabeling(liftedNodeNumber_);
	std::vector<int> swapLabeling_(nodeNumber_); // used as buffer while copying liftedLabeling_

	initializeAllEdgeLookups(hValue_);

	int energies_idx = 0;
	int NITERATIONS_ENERGIES = 1;

	// std::vector< vector< vector<double> > > energies(NITERATIONS_ENERGIES, vector< vector<double> >( labelNumber_, vector<double>(edgeClassNumber_, 0.0) ) );
	std::vector<double> energies(labelNumber_*edgeClassNumber_*NITERATIONS_ENERGIES, 0.0);

	energy_ = oneRunLiftedAlphaExpansion( liftedLabeling_ );
	energies[energies_idx] = energy_;
	energies_idx++;
	int lastLabelThatCausedChange = 0;
	int lastEdgeClassThatCausedChange = 0;
	cout << "greedyMaxIter_=" << greedyMaxIter_ << endl;
	// perform inference
	for(int iIter = 0; iIter < greedyMaxIter_; ++iIter) {
		cout << "Iteration " << iIter << endl;
		bool changed = false;
		double greedy_iteration_time = 0.0;
		// Do iteration
		for(int iLabel = 0; iLabel < labelNumber_; ++iLabel) {
			for(int iEdgeClass = 0; iEdgeClass < edgeClassNumber_; ++iEdgeClass) {
				
				// toggle hValue_
				hValue_[iLabel][iEdgeClass] = 1 - hValue_[iLabel][iEdgeClass];
				if (hValue_[iLabel][iEdgeClass] == 1) {
					hSum -= (1 - alpha_) * classThreshold_[iEdgeClass];
				} else {
					hSum += (1 - alpha_) * classThreshold_[iEdgeClass];
				}

				// AE
				double energy;
				energy = oneRunLiftedAlphaExpansion(curLabeling);
				
				// accept or reject hValue
				if (energy + lambda_ * hSum < energy_) {
					// accept the change 
					energy_ = energy + lambda_ * hSum;
					changed = true;
					liftedLabeling_ = curLabeling;
					

					printf("Changing h(%d, %d, %d) to %d, energy is %f", iLabel, iEdgeClass, iIter, hValue_[iLabel][iEdgeClass], energy_);
					lastLabelThatCausedChange = iLabel;
					lastEdgeClassThatCausedChange = iEdgeClass;
					cout << endl;
				} else {
					// return h back
					printf("Changing h(%d, %d, %d) to %d REJECTED, energy is %f", iLabel, iEdgeClass, iIter, hValue_[iLabel][iEdgeClass], energy + lambda_  * hSum);
					cout << endl;
					hValue_[iLabel][iEdgeClass] = 1 - hValue_[iLabel][iEdgeClass];
					if (hValue_[iLabel][iEdgeClass] == 1) {
						hSum -= (1 - alpha_) * classThreshold_[iEdgeClass];
					} else {
						hSum += (1 - alpha_) * classThreshold_[iEdgeClass];
					}
				}

				// update energies
				energies[energies_idx] = energy_;
				energies_idx = (energies_idx+1)%energies.size();

				// save labeling_
				unlift(liftedLabeling_, fg_, labeling_);
				saveLabels(iIter, iLabel, iEdgeClass);

			}
		}
	}

	// ASSERT: labeling_ is ready.
	elapsedTime_ = (clock() - tAlgoStart) / CLOCKS_PER_SEC;
	printf("Energy: %f, time: %f", energy_, elapsedTime_);
	cout << endl;

	double energy = computeCoopEnergy( labeling_ );
	if( true || fabs( energy - energy_ ) > 1e-3 ){
		printf("Correct energy: %f", energy);
		cout << endl;
		energy_ = energy;
	}

	resultsComputed_ = true;
	return true;	
}



void LiftedCoopCut::read_superpixel_hybrid(std::vector<FactorGraph>& fgs, string filename) {
	std::cerr << "read_superpixel_hybrid() started" << std::endl;
	std::ifstream infile(filename);
	std::string line,str_grp;
	int granularity_idx = 0;
	while(!infile.eof()) {
		getline(infile, line);
		if(line.length()==0) {
			std::cerr << "line.length=0" << std::endl;
			break;
		}
		// Create a new factor graph!
		FactorGraph fg;
		fgs.push_back(fg);
		initialize_factor_graph(fgs[granularity_idx]);
		fgs[granularity_idx].invsyms.resize(nodeNumber_);

		std::stringstream line_ss(line);
		int grp;
		int maxGroup=-1;
		int v=0;

		while(!line_ss.eof() && (v<nodeNumber_)) {
			getline(line_ss, str_grp, ',');
			// cout << str_grp << endl;
			grp = stoi(str_grp);
			fgs[granularity_idx].invsyms[v] = grp;
			
			if (grp>maxGroup) maxGroup = grp;

			v++;
		}

		fgs[granularity_idx].syms.resize(maxGroup+1);
		for(int v=0; v<fgs[granularity_idx].invsyms.size(); ++v) {
			fgs[granularity_idx].syms[ fgs[granularity_idx].invsyms[v] ].push_back(v);
		}

		granularity_idx++;
	}
	infile.close();
	return;
}

bool LiftedCoopCut::minimizeSLICHybrid_greedy() {
	energies_window_size = labelNumber_;
	string slic_c2f_filename = name + ".slic.csv";
	double tAlgoStart = clock();
	double tStart = clock();
	cout << "Name: " << name << endl;
	cout << "Starting hybrid greedy ... Preparing energy" << endl;
	if ( !pairwiseLoaded_ || !unaryLoaded_ ) {
		return false;
	}
	labeling_.resize( nodeNumber_ );   // stores result of optimization
	constructIndexingDataStructure( maxDegree_ ); 	

	hValue_.resize( labelNumber_, vector<int>( edgeClassNumber_, 1) );
	double hSum = 0.0;

	// Load SLIC liftings
	volatile int active_idx = 0;
	std::vector<FactorGraph> fgs;
	read_superpixel_hybrid(fgs, slic_c2f_filename);
	set_lifted_graph(fgs[0]);
	int max_active_idx = fgs.size();
	cout << "max_active_idx= " << max_active_idx << endl;
	// Need to lift!

	liftedLabeling_.resize(liftedNodeNumber_);
	vector<int> curLabeling(liftedNodeNumber_);
	std::vector<int> swapLabeling_(nodeNumber_); // used as buffer while copying liftedLabeling_

	saveLifting(fgs[0], name);
	initializeAllEdgeLookups(hValue_);

	int energies_idx = 0;
	int NITERATIONS_ENERGIES = 1;

	// std::vector< vector< vector<double> > > energies(NITERATIONS_ENERGIES, vector< vector<double> >( labelNumber_, vector<double>(edgeClassNumber_, 0.0) ) );
	std::vector<double> energies(labelNumber_*edgeClassNumber_*NITERATIONS_ENERGIES, 0.0);

	energy_ = oneRunLiftedAlphaExpansion( liftedLabeling_ );
	energies[energies_idx] = energy_;
	energies_idx++;
	int lastLabelThatCausedChange = 0;
	int lastEdgeClassThatCausedChange = 0;
	for(int iIter = 0; iIter < greedyMaxIter_; ++iIter) {
		bool changed = false;
		double greedy_iteration_time = 0.0;
		// Do iteration
		for(int iLabel = 0; iLabel < labelNumber_; ++iLabel) {
			for(int iEdgeClass = 0; iEdgeClass < edgeClassNumber_; ++iEdgeClass) {
				// cout << "Params: (" << hyperparameters[active_idx].k << "," << hyperparameters[active_idx].ncbp << ") " << endl;
				// toggle hValue_
				hValue_[iLabel][iEdgeClass] = 1 - hValue_[iLabel][iEdgeClass];
				if (hValue_[iLabel][iEdgeClass] == 1) {
					hSum -= (1 - alpha_) * classThreshold_[iEdgeClass];
				} else {
					hSum += (1 - alpha_) * classThreshold_[iEdgeClass];
				}

				// AE
				double energy;
				if (active_idx < max_active_idx) {
					energy = oneRunLiftedAlphaExpansion(curLabeling);
				} else {
					energy = oneRunAlphaExpansion(curLabeling);
				}

				// accept or reject hValue
				if (energy + lambda_ * hSum < energy_) {
					// accept the change 
					energy_ = energy + lambda_ * hSum;
					changed = true;
					if(active_idx < max_active_idx) {
						liftedLabeling_ = curLabeling;
					} else {
						labeling_ = curLabeling;
					}
					

					printf("Changing h(%d, %d, %d) to %d, energy is %f", iLabel, iEdgeClass, iIter, hValue_[iLabel][iEdgeClass], energy_);
					lastLabelThatCausedChange = iLabel;
					lastEdgeClassThatCausedChange = iEdgeClass;
					cout << endl;
				} else {
					// return h back
					printf("Changing h(%d, %d, %d) to %d REJECTED, energy is %f", iLabel, iEdgeClass, iIter, hValue_[iLabel][iEdgeClass], energy + lambda_  * hSum);
					cout << endl;
					hValue_[iLabel][iEdgeClass] = 1 - hValue_[iLabel][iEdgeClass];
					if (hValue_[iLabel][iEdgeClass] == 1) {
						hSum -= (1 - alpha_) * classThreshold_[iEdgeClass];
					} else {
						hSum += (1 - alpha_) * classThreshold_[iEdgeClass];
					}
				}

				// update energies
				energies[energies_idx] = energy_;
				energies_idx = (energies_idx+1)%energies.size();

				// save labeling_
				if(active_idx < max_active_idx) {
					unlift(liftedLabeling_, fgs[active_idx], labeling_);
				}
				saveLabels(iIter, iLabel, iEdgeClass);

				if(isToggleTime(energies_idx, active_idx, max_active_idx, energies)) {
					active_idx++;
					
					cout << "Beginning toggle" << endl;
					if (active_idx < max_active_idx) {
						cout << "active_idx" << active_idx << " < max_active_idx" << max_active_idx << endl;
						unlift(liftedLabeling_, fgs[active_idx-1], swapLabeling_);
						clock_t start = clock();
						set_lifted_graph(fgs[active_idx]);
						clock_t end = clock();
						saveLifting(fgs[active_idx], name);
						liftedLabeling_.resize(liftedNodeNumber_);
						curLabeling.resize(liftedNodeNumber_);
						lift_labeling(swapLabeling_, fgs[active_idx], liftedLabeling_);
						changed = true;
					} else {
						unlift(liftedLabeling_, fgs[active_idx-1], labeling_);
						curLabeling.resize(nodeNumber_);
						// cout << "Toggling from (" << hyperparameters[active_idx-1].k << "," << hyperparameters[active_idx-1].ncbp 
						// 	<< ") to ground" << endl; 
						cout << "Switching to ground" << endl;
						iLabel = lastLabelThatCausedChange;
						iEdgeClass = lastEdgeClassThatCausedChange;
						cout << "Setting iLabel to " << iLabel << " and iEdgeClass to " << iEdgeClass << endl;
						changed = true;
					}
				}
			}
		}
		if (!changed)
			break;
	}

	// ASSERT: labeling_ is ready.
	elapsedTime_ = (clock() - tAlgoStart) / CLOCKS_PER_SEC;
	printf("Energy: %f, time: %f", energy_, elapsedTime_);
	cout << endl;

	double energy = computeCoopEnergy( labeling_ );
	if( true || fabs( energy - energy_ ) > 1e-3 ){
		printf("Correct energy: %f", energy);
		cout << endl;
		energy_ = energy;
	}

	resultsComputed_ = true;

	return true;
}

