CFLAGS=-O3 -std=c++11 -lpng
CFLAGSAE=  -std=c++11 -lpng
CC=g++
original:
	$(CC) main.cpp -DORIGINAL -o original.out $(CFLAGS) -I ../libs/opengm-master/src/external/QPBO-v1.3.src-patched/ -I ../libs/opengm-master/src/external/TRWS-v1.3.src-patched/ -lexternal-library-qpbo-shared -lexternal-library-trws-shared  -Wfatal-errors
alp_exp:
	$(CC) main.cpp -DAE -o ae.out $(CFLAGSAE) -I ../libs/opengm-master/src/external/QPBO-v1.3.src-patched/ -I ../libs/opengm-master/src/external/TRWS-v1.3.src-patched/ -lexternal-library-qpbo-shared -lexternal-library-trws-shared  -Wfatal-errors
sym_ae:
	$(CC) main.cpp -DSYM_AE -o sym.out $(CFLAGS) -I ../libs/opengm-master/src/external/QPBO-v1.3.src-patched/ -I ../libs/opengm-master/src/external/TRWS-v1.3.src-patched/ -lexternal-library-qpbo-shared -lexternal-library-trws-shared  -Wfatal-errors
hybrid:
	$(CC) main.cpp -DHYBRID -o hyb.out $(CFLAGS) -I ../libs/opengm-master/src/external/QPBO-v1.3.src-patched/ -I ../libs/opengm-master/src/external/TRWS-v1.3.src-patched/ -lexternal-library-qpbo-shared -lexternal-library-trws-shared  -Wfatal-errors
threshold:
	$(CC) main.cpp -DTHRESH_AE -o thr.out $(CFLAGS) -I ../libs/opengm-master/src/external/QPBO-v1.3.src-patched/ -I ../libs/opengm-master/src/external/TRWS-v1.3.src-patched/ -lexternal-library-qpbo-shared -lexternal-library-trws-shared  -Wfatal-errors
slic_lift:
	$(CC) main.cpp -DSLIC -o sli.out $(CFLAGS) -I ../libs/opengm-master/src/external/QPBO-v1.3.src-patched/ -I ../libs/opengm-master/src/external/TRWS-v1.3.src-patched/ -lexternal-library-qpbo-shared -lexternal-library-trws-shared  -Wfatal-errors	
slic_hyb:
	$(CC) main.cpp -DSH -o slihyb.out $(CFLAGS) -I ../libs/opengm-master/src/external/QPBO-v1.3.src-patched/ -I ../libs/opengm-master/src/external/TRWS-v1.3.src-patched/ -lexternal-library-qpbo-shared -lexternal-library-trws-shared  -Wfatal-errors		
all: alp_exp sym_ae hybrid threshold slic_lift slic_hyb
clean:
	rm *.out