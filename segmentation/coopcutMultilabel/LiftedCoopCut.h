#pragma once
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

#include <boost/unordered_map.hpp>
// little struct to help haroun make things happen.


struct hyperparameters_t {
	int k;
	int ncbp;
};
struct FactorGraph {
	/*
		variables: 0-nodeNumber_ variables
		factors: 0-nodeNumber_ unaries and nodeNumber_ - nodeNumber_ + edgeNumber pairwise.
	*/
	std::vector<int> vecVariables; // my idx -> idx in original
	std::vector<int> vecFactors; // my idx -> idx in original 
	std::vector< std::vector<int> > adjListOfVariables; // my idx -> vector<my idx of factors>
	std::vector< std::vector<int> > adjListOfFactors;  // my idx -> vector<my idx of variables>
	/*
		signatures: 0 is for self color. rest are colors for neighbours.
	*/
	std::vector< std::vector<int> > variableSignatures;
	std::vector< std::vector<int> > factorSignatures;
	bool initialized_unary_colors;

	// the ones to use.
	std::vector< std::vector<int> > nowVariableSignatures;
	std::vector< std::vector<int> > nowFactorSignatures;
	std::vector< std::vector<int> > syms;
	std::vector<int> invsyms;
}; 
typedef struct FactorGraph FactorGraph; //making life easier.

struct class_data_t {
	int edge_class;
	double edge_weight;
};
typedef struct class_data_t class_data_t;
class LiftedEdge {
public:
	int supernode0;
	int supernode1;
	double total_weight; // For alpha expansion... ignores edge-classes afterall.
	
	std::vector<class_data_t> forwards;
	std::vector<class_data_t> backwards;
	LiftedEdge() {};
	LiftedEdge(int sn0, int sn1, double w, int c, int l) : supernode0(sn0), supernode1(sn1), total_weight(w), 
		forwards(0), backwards(0) { };

};
typedef boost::unordered_map< std::pair<int,int> , LiftedEdge > pair2liftedEdge_t;


class LiftedCoopCut : public CoopCutMultilabel {
public:

	LiftedCoopCut(string outputLabelFile) : CoopCutMultilabel(outputLabelFile) { initialized_hvalues=false; doNotSaveLabels = true;};
	
	// code options
	bool doNotSaveLabels;

	// a more comfortable interface with graphs.
	FactorGraph fg_;
	int liftedNodeNumber_;
	int liftedEdgeNumber_;
	double *liftedUnaryTerms_;

	// lifted edge representation.
	// int getLiftedEdgeIndex(int supernode0, int supernode1);
	pair2liftedEdge_t pair2liftedEdge_;

	// lifting variables
	int nRanksConsidered;
	int nCBPIterations;
	int nSqrtSpatialGridCount; // the number of grids along each axis is equal for now. That's stupid, but it works :)
	double thresholdForSymmetry;

	void setLiftingParameters(int k, int ncbp, int sqrt_spatial_grid_size=1, double threshold_for_symmetry=0.1) {
		nRanksConsidered = k; nCBPIterations=ncbp; nSqrtSpatialGridCount=sqrt_spatial_grid_size;
		thresholdForSymmetry = threshold_for_symmetry;
		cout << "setLiftingParameters(" << k <<"," << nCBPIterations << "," << nSqrtSpatialGridCount << "," << thresholdForSymmetry << ")" << endl;
	};

	// Section 1
	bool initialize_factor_graph(FactorGraph& fg);
	void setUpLiftedNeighborhood(GCoptimizationGeneralGraph *gc);	
	
	void initialize_variable_colors(FactorGraph& fg);
	
	int nUnaryColorsUsed;
	int initialize_unary_colors(FactorGraph& fg);
	int initialize_unary_colors_threshold(FactorGraph& fg);
	

	void read_slic(FactorGraph& fg, string slic_name);
	void read_superpixel_hybrid(std::vector<FactorGraph>& fgs, string filename); // Figure out how and what to do later.

	void prelift(FactorGraph& fg);
	void prelift_threshold(FactorGraph& fg);

	void set_groups(FactorGraph& fg);
	void recolorVariables(FactorGraph& fg);
	void recolorFactors(struct FactorGraph& fg);
	void doCBPIteration(FactorGraph& fg, bool flag=false);
	void set_lifted_graph(FactorGraph& fg);
	
	void initialize_pairwise_colors(FactorGraph& fg);

	void lift(FactorGraph& fg);

	// stuff to make greedy lifted efficient.
	vector<vector<int>> oldH_value;
	bool initialized_hvalues;
	void initializeAllEdgeLookups(vector<vector<int>>& oldH_value);
	void updateLiftedWeights_byToggling(int label, int nclass);
	
	// void lift(FactorGraph& fg, vector<vector<int>>& hValues); // deprecated.

	void unlift(std::vector<int>& src, FactorGraph& fg, std::vector<int>& dst);

	void saveLifting(FactorGraph& fg, string& name);
	// stuff to store labeling.
	std::vector<int> liftedLabeling_;

	// some functions to actually do some minimization.

	bool minimizeLiftedEnergy_alphaExpansion(); // lifts at the beginning.
	// callback for alpha-exp that ignore cooperation
	GCoptimization::EnergyTermType liftedComputePairwiseCost(
				GCoptimization::SiteID s1, GCoptimization::SiteID s2,
				GCoptimization::LabelID l1, GCoptimization::LabelID l2
				);
	friend	GCoptimization::EnergyTermType liftedGlobalPairwiseCost(
				GCoptimization::SiteID s1, 	GCoptimization::SiteID s2, 
				GCoptimization::LabelID l1,	GCoptimization::LabelID l2,
				void * objectPtr
			);
	// ---- done with alpha-expansion



	bool minimizeSLICEnergy_greedy(); // basically the same as minimizeLiftedEnergy_greedy
	bool minimizeLiftedEnergy_greedy(); // lifts on each step.
	// callbacks that take coopreration into account
	GCoptimization::EnergyTermType liftedComputePairwiseCostOneBreakPoint(
				GCoptimization::SiteID s1, GCoptimization::SiteID s2,
				GCoptimization::LabelID l1, GCoptimization::LabelID l2
				);
	friend	GCoptimization::EnergyTermType liftedGlobalPairwiseCostOneBreakPoint(
				GCoptimization::SiteID s1, 	GCoptimization::SiteID s2, 
				GCoptimization::LabelID l1,	GCoptimization::LabelID l2,
				void * objectPtr
			);
	double oneRunLiftedAlphaExpansion(std::vector<int>& labeling);
	// ---- done with greedy expansion.

	// ----- threshold
	bool minimizeThresholdEnergy_greedy();
	// done with threshold


	// ---- hybrid
	bool minimizeHybridEnergy_greedy();
	bool minimizeSLICHybrid_greedy();
	// need a function to save image per iteration
	// We can post process the accuracies
	void saveLabels(int& iterationNumber,int& iLabel,int& iEdgeClass); // Saves the image using name + iterationNumber.
	bool isToggleTime(int energies_idx, int active_idx, int max_active_idx, vector< double >& energies);
	void lift_labeling(std::vector<int>& ground_labeling, FactorGraph& fg, std::vector<int>& lifted_labeling);


	// ---- done with hybrid
};