GCO_FILES=../lib/gco-3.0/GCoptimization.cpp ../lib/gco-3.0/graph.cpp ../lib/gco-3.0/maxflow.cpp ../lib/gco-3.0/LinkedBlockList.cpp

#
# #CYTHON_CMD=g++ -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing  `python-config --libs` -L`python-config --prefix`/lib -static `python-config --ldflags` `python-config --includes` -I `python -c "import numpy; print numpy.get_include()"`
# CYTHON_CMD=g++ -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing  `python-config --libs` -L`python-config --prefix`/lib  `python-config --ldflags` `python-config --includes` -I `python -c "import numpy; print numpy.get_include()"`
#
CYTHON_CMD=g++ -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing  `python-config --libs` -L`python-config --prefix`/lib `python-config --ldflags` `python-config --includes` -I `python -c "import numpy; print numpy.get_include()"`


#-I `python-config --prefix`/lib/python2.7/site-packages/numpy/core/include

mrf : mrf.pyx mrf.cc mrf.h
	g++ -c -O3 -fPIC -I ../lib/gco-3.0 mrf.cc $(GCO_FILES) 
	cython mrf.pyx -o mrf_.c
	$(CYTHON_CMD) mrf_.c mrf.o GCoptimization.o LinkedBlockList.o graph.o maxflow.o -o mrf.so

