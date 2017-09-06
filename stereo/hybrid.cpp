#ifndef _HYBRID_CPP_
#define _HYBRID_CPP_

double delta_threshold = 0.0;
int lastswitch=0;

// This file contains functions that support our coarse-to-fine algorithm
bool detectPlateau1(int iter, std::vector< std::vector<double> >& energies) {
	if(iter==0) lastswitch=0;
	if ((iter<70) || ((iter-lastswitch)<energies[0].size())) return false; // Give the algorithm a chance man.
	int n = energies[0].size();
	// assert(n>2);
	// Calculate difference.
	std::vector<double> deltas(2,0.0);
	double temp;
	for(int lr=0; lr<2; ++lr) {
		deltas[lr] = abs(energies[lr][(iter+1)%n] - energies[lr][(iter)%n]);
		// If deltas[lr] is too huge, ditch
		if (deltas[lr]>delta_threshold) return false;
		// Else if its small then check the next one.
		for(int i=1; i<(n-1); ++i) {
			deltas[lr] = abs(energies[lr][(iter+i+1)%n] - energies[lr][(iter+i)%n]);
			if ( (deltas[lr]>delta_threshold) ) {
				return false;
			}
		}
	}
	lastswitch = iter;
	std::cout << "i should be switching..." << std::endl;
	// for(int lr=0; lr<2; ++lr) {
	// 	std::cout << "lr:" << lr <<  " " << std::endl;
	// 	for(int i=0; i<n; ++i) {
	// 		std::cout << energies[lr][i] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	
	return true;
}

bool toGroundDetect(int iter, std::vector< std::vector<double> >& energies) {
	return (iter > 200); // We to switch to ground on iteration 
}

void updateLabelsWithChangedSymmetries(
		std::vector<int>& srcLabels, 
		std::vector<std::vector<int> >& srcSyms,
		std::vector<int>& dstLabels,
		std::vector<int>& dstInvSyms
	) {
	for(int i=0; i<srcSyms.size(); ++i) { // for each group
		for(int vidx=0; vidx<srcSyms[i].size(); ++vidx) {
			// read this by writing it out. sorry.
			dstLabels[ dstInvSyms[ srcSyms[i][vidx] ] ] = srcLabels[i];
		}
	}
}

#endif