#include <iostream>
#include <fstream>


void EDP::read_slic_hybrid(png::image< png::rgb_pixel >& img, double* m_cost, std::string slic_file, AEFInferType::Parameter& params,// IN
		std::map<int, Model>& vecgms,
		std::map<int, sym_t>& vecsyms,
		std::map<int, invsym_t>& vecinvsyms,
		std::map<int, labels_t>& veclabels,
		std::map<int, AEFInferType*>& vecaesptr // OUT
	) {
	std::cerr << "Entered read_slic_hybrid(...)" << std::endl;
	std::ifstream infile(slic_file);
	std::string line, str_grp;
	int grp;
	int maxGroup=-1;
	
	int nPixels = this->m_nPixels;

	int granularity_idx = 0;
	
	while(!infile.eof()) {
		getline(infile, line);
		
		if(line.length()==0) { // empty line
			std::cerr << "line.length()==0" << std::endl;
			continue;
		}
		// read invsyms first
		vecinvsyms[granularity_idx] = (std::vector<int>(nPixels,-1));
		// read invsyms
		std::stringstream line_ss(line);
		
		int v=0;

		while(/*!line_ss.eof() && */(v<nPixels)) {
			getline(line_ss, str_grp, ',');
			grp = stoi(str_grp);
			vecinvsyms[granularity_idx][v] = grp;
			if(grp>maxGroup) maxGroup = grp;
			v++;
		}

		// prepare vecsyms
		vecsyms[granularity_idx] = ( std::vector< std::vector<int> >(maxGroup+1, std::vector<int>()) );
		for (int v=0; v<nPixels; ++v) {
			vecsyms[granularity_idx][ vecinvsyms[granularity_idx][v] ].push_back(v);
		}

		// prepare vecgms, vecaesptr
		vecgms[granularity_idx] = (constructGraphicalModel(img, this->m_nLabels, m_cost, vecsyms[granularity_idx], vecinvsyms[granularity_idx]));
		veclabels[granularity_idx] = ( std::vector<int>(vecgms[granularity_idx].numberOfVariables(), 0));
		vecaesptr[granularity_idx] = ( new AEFInferType(vecgms[granularity_idx], params) );
		
		std::cerr << vecgms[granularity_idx].numberOfVariables() << "," << vecgms[granularity_idx].numberOfFactors() << "," << vecgms[granularity_idx].factorOrder() << std::endl;

		granularity_idx++;
	}
	infile.close();
	std::cerr << "Closed infile, dealing with ground now" << std::endl;
	// Add ground to the mix.
	vecinvsyms[granularity_idx] = (std::vector<int>( nPixels ));
	vecsyms[granularity_idx] = (std::vector< std::vector<int> >( nPixels , std::vector<int>(1) ));
	for(int v=0; v<nPixels; ++v) {
		vecinvsyms[granularity_idx][v] = v;
		vecsyms[granularity_idx][v][0] = v;
	}
	vecgms[granularity_idx] = (constructGraphicalModel(img, this->m_nLabels, m_cost)); // ground
	veclabels[granularity_idx] = ( std::vector<int>(vecgms[granularity_idx].numberOfVariables(), 0) );
	vecaesptr[granularity_idx] = ( new AEFInferType(vecgms[granularity_idx], params) );
	std::cerr << "Done with read_slic_hybrid(...)" << std::endl;
	return;
}

void EDP::slic_hybrid_MRF__z(int itr, int * dL, int * dR, png::image< png::ga_pixel >& truthimg) {
	// Step0 : Initializations
	using namespace std;
	clock_t start,end; // timers.
	std::cout << "Entered slic_hybrid_MRF__z" << std::endl;
	AEFInferType::Parameter params(1); // we're gonna step through this.
	std::vector<AEFInferType> aes; // Empty for now.
	int nx=m_width;
	int ny=m_height;
	int nLabels=this->m_nLabels;
	std::cerr << "nLabels=" << nLabels << std::endl;
	Smthnss = new DELTA_D [5];
	float c_lbl = 10, occ =0;
	Size_fw = 18;
	float sigma_c_mul = 0.75/255;
	float sigmax = Size_fw;
	float sigmay = Size_fw ;
	int cl_Q = (N_LB <30) ? 2000 : 4000;/*100*(1+iii);*/
	int n_bufs =m_nLabels;
	Lmbd_trn = 0.5;
	REAL * Mrgnl = new REAL [m_nPixels*this->m_nLabels];
	double *intr_cst[2];
	double * m_cost = new double [m_nPixels*this->m_nLabels];
	if(N_LB>30) {
		Make_Gr_inf_pp(0.011,15,  7, 15);
	} else {
		Make_Gr_inf(0,  7, 15);
	}
	float Lm_tr[3] = {2.5*0.125, 1.25*0.125,1.*0.125};
	float ml[3] = {1./6, 1./6, 1./6};
	for(int q =0; q<3; q++){
		Smthnss[q].max = Lm_tr[q];
		Smthnss[q].semi = Lm_tr[q]*ml[q];
	}

	// Model 

	// std::vector< hyperparameter_t > hyperparameters = {std::make_pair(1,0), std::make_pair(1,1), std::make_pair(2,0), std::make_pair(3,0), std::make_pair(0,0)};
	// int ground_parameter_idx = hyperparameters.size()-1;
	std::vector< std::map<int, Model> > vecgms(2, std::map<int, Model>());
	std::vector< std::map<int, sym_t> > vecsyms(2, std::map<int, sym_t>());
	std::vector< std::map<int, invsym_t> > vecinvsyms(2, std::map<int, invsym_t>());
	std::vector< std::map<int, labels_t> > veclabels(2, std::map<int, labels_t>());
	std::vector< std::map<int, AEFInferType*> > vecaesptr(2, std::map<int, AEFInferType*>());
	
	// Step1 : upper mrf - kmeans?
	png::image< png::rgb_pixel > img(nx, ny);
	for(int lr = 0; lr<2;lr++) {
		int * rez =  (lr)? dR : dL;
		BYTE * I_b = I_ims[lr];
		copy_m_D(m_cost, lr);
		int thr = 5;
		float alp =0.;
		
		Make_Gr_fl_buf(  thr, lr,m_cost, N_LB, alp );
		/*if(Sc_out == 1)*/
		K_mean_Flt_Add_new(I_b, sigma_c_mul, sigmax,sigmay, cl_Q, m_cost, n_bufs);
		
		// Need RVO here to speed it up. FIXME: use Return-by-value optimization.
		ucharArray2pngImage(img, I_b, nx, ny);
		
		// TODO - create different graphical models with symmetries
		std::string slic_file;
		if (lr==0) {
			slic_file = lview_slic;
		} else {
			slic_file = rview_slic;
		}
		
		read_slic_hybrid(img, m_cost, slic_file, params, // IN
			vecgms[lr], vecsyms[lr], vecinvsyms[lr], veclabels[lr], vecaesptr[lr]); // OUT
		
			/*{
				std::cerr << "Entered read_slic_hybrid(...)" << std::endl;
				std::ifstream infile(slic_file);
				std::string line, str_grp;
				int grp;
				int maxGroup=-1;
				
				int nPixels = this->m_nPixels;

				int granularity_idx = 0;
				
				while(!infile.eof()) {
					getline(infile, line);
					
					if(line.length()==0) { // empty line
						std::cerr << "line.length()==0" << std::endl;
						continue;
					}
					// read invsyms first
					vecinvsyms[lr].push_back(std::vector<int>(nPixels,-1));
					// read invsyms
					std::stringstream line_ss(line);
					
					int v=0;

					while(!line_ss.eof() && (v<nPixels)) {
						getline(line_ss, str_grp, ',');
						grp = stoi(str_grp);
						vecinvsyms[lr][granularity_idx][v] = grp;
						if(grp>maxGroup) maxGroup = grp;
						v++;
					}

					// prepare vecsyms
					vecsyms[lr].push_back( std::vector< std::vector<int> >(maxGroup+1, std::vector<int>()) );
					for (int v=0; v<nPixels; ++v) {
						vecsyms[lr][granularity_idx][ vecinvsyms[lr][granularity_idx][v] ].push_back(v);
					}

					// prepare vecgms, vecaesptr
					vecgms[lr].push_back(constructGraphicalModel(img, this->m_nLabels, m_cost, vecsyms[lr][granularity_idx], vecinvsyms[lr][granularity_idx]));
					veclabels[lr].push_back( std::vector<int>(vecgms[lr][granularity_idx].numberOfVariables(), 0));
					vecaesptr[lr].push_back( new AEFInferType(vecgms[lr][granularity_idx], params) );
					
					std::cerr << vecgms[lr][granularity_idx].numberOfVariables() << "," << vecgms[lr][granularity_idx].numberOfFactors() << "," << vecgms[lr][granularity_idx].factorOrder() << std::endl;

					granularity_idx++;
				}
				infile.close();
				std::cerr << "Closed infile, dealing with ground now" << std::endl;
				// Add ground to the mix.
				vecinvsyms[lr].push_back(std::vector<int>( nPixels ));
				vecsyms[lr].push_back(std::vector< std::vector<int> >( nPixels , std::vector<int>(1) ));
				for(int v=0; v<nPixels; ++v) {
					vecinvsyms[lr][granularity_idx][v] = v;
					vecsyms[lr][granularity_idx][v][0] = v;
				}
				vecgms[lr].push_back(constructGraphicalModel(img, this->m_nLabels, m_cost)); // ground
				veclabels[lr].push_back( std::vector<int>(vecgms[lr][granularity_idx].numberOfVariables(), 0) );
				vecaesptr[lr].push_back( new AEFInferType(vecgms[lr][granularity_idx], params) );
				std::cerr << "Done with read_slic_hybrid(...)" << std::endl;
				// return;
			}*/
	} // end for(lr=0...)
	
	std::cerr << "Models created. Now starting inference" << std::endl;
	std::cout << "Models created. Now starting inference" << std::endl;
	// Step3 : inference
	AEFInferType::TimingVisitorType visitorl;
	AEFInferType::TimingVisitorType visitorr;
	// hyperparameter_t active_hyperparameter = std::make_pair(1,0); // start coarse
	int active_hyperparameter_idx = 0;
	std::vector< std::vector<double> > energies(2, std::vector<double>(4,0.0));

	int maxiter=400;
	int nLevels = vecgms[0].size();
	int ground_parameter_idx = nLevels-1;
	for(int iter=0; iter<maxiter; ++iter) {
		// We don't really understand the magic that happens in the original MRF__z
		// hence, we'll just use the entire TSGO algorithm here...
		
		// IMPORTANT - create a function that does this.
		if((active_hyperparameter_idx<(vecgms[0].size()-1) && (detectPlateau1(iter, energies)))
			/*|| ((active_hyperparameter_idx == (hyperparameters.size()-2)) && toGroundDetect(iter, energies))*/
			) {
			// TODO - check if you need to switch.
			active_hyperparameter_idx++;
			// std::cout << "Switching hyperparameters from ";
			// std::cout << active_hyperparameter.first << " " << active_hyperparameter.second << " to ";
			// active_hyperparameter = hyperparameters[active_hyperparameter_idx];
			// std::cout << active_hyperparameter.first << " " << active_hyperparameter.second << std::endl;
			// copy labels over correctly from veclabels[lr][active_hyperparameter] to veclabels[lr][new_hyperparameter]
			for(int lr=0; lr<2; ++lr){
				updateLabelsWithChangedSymmetries(veclabels[lr][active_hyperparameter_idx-1], vecsyms[lr][active_hyperparameter_idx-1], veclabels[lr][active_hyperparameter_idx], vecinvsyms[lr][active_hyperparameter_idx]);
			}
			if(active_hyperparameter_idx == ground_parameter_idx) {
				std::cerr << "Switching to ground on " << iter << std::endl;
			}
		}
		// std::cout << "Starting iteration=" << iter << " using parameters=(" << active_hyperparameter.first << "," << active_hyperparameter.second << ")" << std::endl;
		// std::cerr << "Starting iteration=" << iter << " using parameters=(" << vecsyms[0][active_hyperparameter_idx].size() << ")" << std::endl;
		std::cout << "Starting iteration=" << iter << " using parameters=(" << vecsyms[0][active_hyperparameter_idx].size() << ")" << std::endl;

		for(int lr = 0; lr<2;lr++) {
			int * rez =  (lr)? dR : dL;
			// Do a single step of inference on gms[lr]
			// std::cerr << "#brk1" << std::endl;
			vecaesptr[lr][active_hyperparameter_idx]->setStartingPoint( veclabels[lr][active_hyperparameter_idx].begin() );
			// std::cerr << "#brk2" << std::endl;
			std::cout << "lr: " << lr << "  active_hyperparameter: " << vecsyms[0][active_hyperparameter_idx].size() << std::endl;
			if(lr==0) vecaesptr[lr][active_hyperparameter_idx]->infer(visitorl);
			else vecaesptr[lr][active_hyperparameter_idx]->infer(visitorr);
			// std::cerr << "#brk3" << std::endl;
			vecaesptr[lr][active_hyperparameter_idx]->arg(veclabels[lr][active_hyperparameter_idx]);
			// std::cerr << "#brk4" << std::endl;
			// Copy the result into veclabels[lr], rez
			copyVecLabels2Rez(veclabels[lr][active_hyperparameter_idx], vecinvsyms[lr][active_hyperparameter_idx], rez);
			// save energy
			// std::cerr << "#brk5" << std::endl;
			energies[lr][iter%(energies[lr].size())] = vecgms[lr][active_hyperparameter_idx].evaluate(veclabels[lr][active_hyperparameter_idx]);
			// std::cerr << "#brk6" << std::endl;
		} // end for(lr=0...)
		
		// Postprocessing
		{
			sigma_img_get(dL);
		
			if(occ > c_lbl) {
				MkDspCorr_RL(  dL, dR);
			}

			{
				if(occ > c_lbl)
					for(int i=0; i<1; i++) {
						int r_w = 4;
						float cl_prc = sigma_img;//0.23;
						Gss_wnd_( r_w,  cl_prc, dL, 0, m_nLabels );
						Gss_wnd_( r_w,  cl_prc, dR, 1, m_nLabels );
					}
			}

			{
				if(occ > c_lbl) {
					MkDspCorr_RL(  dL, dR);
				}
				int r_w1 = 4;
				float cl_prc1 = sigma_img;//0.23;
				Gss_wnd_( r_w1,  cl_prc1, dL, 0, m_nLabels );
			}

			Outlr_reg_filter(0.001, 3, dL, 0, 10, 0.01);

			if(0)  {
				Intermed(dL, dR);
			}
			int r = 4;//(N_LB >16) ? 4:2;
			if(occ > c_lbl) {
				CorrectSub_new(dL, dR, /*intr_cst[0],*/ r, 1);
			} else {
				for(int i =0; i < m_nPixels; i++ ){
					dL[i] *= 1;
				}
			}
		}
		double badpixelscore;
		// if(iter==99)
		badpixelscore = getBadPixelScore(dL, truthimg, nx, ny, nLabels, badpixel_threshold);
		std::cout << "On step " << iter << " error was " << badpixelscore << std::endl;
	}
	std::cerr << "Inference complete" << std::endl;
	// Step4: clean up
	delete [] m_cost;
	delete [] Mrgnl;
}
