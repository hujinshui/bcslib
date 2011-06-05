# The make file for BCSLib

# compiler configuration

ifndef CC
CC=gcc
endif

ifndef CXX
CXX=g++
endif

ifndef BOOST_HOME
$(error "The environment variable BOOST_HOME is not defined")
endif


WARNING_FLAGS = -Wall -Wextra -Wconversion -Wformat -Wno-unused-parameter
BOOST_WARNING_FLAGS = -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter

ifeq ($(CXX), clang++)
	CFLAGS = -std=c++0x -stdlib=libc++ -pedantic -U__STRICT_ANSI__ $(WARNING_FLAGS) -I. 
else
	CFLAGS = -std=c++0x -pedantic $(WARNING_FLAGS) -I. 
endif

BOOST_CFLAGS = -ansi $(BOOST_WARNING_FLAGS) -I$(BOOST_HOME) -I. 

BASE_HEADERS = bcslib/base/config.h bcslib/base/basic_defs.h bcslib/base/basic_funcs.h bcslib/base/basic_mem.h bcslib/base/enumerate.h 

TEST_HEADERS = bcslib/test/test_assertion.h bcslib/test/test_units.h bcslib/test/execution_mon.h

VEC_COMP_HEADERS = bcslib/veccomp/veccalc.h bcslib/veccomp/vecnorm.h bcslib/veccomp/vecstat.h

ARRAY_BASIC_HEADERS = bcslib/array/array_base.h bcslib/array/index_selection.h bcslib/array/array_index.h bcslib/array/array1d.h bcslib/array/array2d.h

ARRAY_COMP_HEADERS = bcslib/array/array_calc.h bcslib/array/array_eval.h

ARRAY_SPARSE_HEADERS = bcslib/array/sparse_vector.h bcslib/array/dynamic_sparse_vector.h

GRAPH_BASIC_HEADERS = bcslib/graph/graph_base.h bcslib/graph/gr_edgelist.h bcslib/graph/gr_adjlist.h bcslib/graph/bgl_port.h

GEOMETRY_BASIC_HEADERS = bcslib/geometry/geometry_base.h bcslib/geometry/poly_scan.h bcslib/geometry/triangle_mesh.h

IMAGE_BASIC_HEADERS = bcslib/image/image_base.h bcslib/image/image.h

PROB_BASIC_HEADERS = bcslib/prob/pdistribution.h bcslib/prob/sampling.h bcslib/prob/discrete_distr.h 

all: test_c0x_support 
others: test_array test_geometry test_image test_graph test_prob

test_c0x_support: bin/test_c0x_support.out
bin/test_c0x_support.out: test/test_c0x_support.cpp
	$(CXX) $(CFLAGS) -c test/test_c0x_support.cpp -o bin/test_c0x_support.out

test_array : bin/test_array_basics bin/test_array_comp bin/test_array_sparse bin/test_access_performance 
test_geometry : bin/test_geometry_basics
test_image: bin/test_image_basics
test_graph : bin/test_graph_basics
test_prob: bin/test_prob_basics bin/test_psampling

ARRAY_BASIC_TESTS = test/test_array_basics.cpp test/test_index_selection.cpp test/test_array1d.cpp test/test_array2d.cpp
bin/test_array_basics: $(BASE_HEADERS) $(TEST_HEADERS) $(ARRAY_BASIC_HEADERS) $(ARRAY_BASIC_TESTS) 
	$(CXX) $(CFLAGS) $(ARRAY_BASIC_TESTS) -o bin/test_array_basics
	
	
ARRAY_COMP_TESTS = test/test_array_comp.cpp test/test_array_calc.cpp test/test_array_eval.cpp
bin/test_array_comp: $(BASE_HEADERS) $(TEST_HEADERS) $(ARRAY_BASIC_HEADERS) $(VEC_COMP_HEADERS) $(ARRAY_COMP_HEADERS) $(ARRAY_COMP_TESTS) 
	$(CXX) $(CFLAGS) $(ARRAY_COMP_TESTS) -o bin/test_array_comp
	
ARRAY_SPARSE_TESTS = test/test_array_sparse.cpp test/test_spvec.cpp test/test_dynamic_spvec.cpp
bin/test_array_sparse: $(BASE_HEADERS) $(TEST_HEADERS) $(ARRAY_BASIC_HEADERS) $(ARRAY_SPARSE_HEADERS) $(ARRAY_SPARSE_TESTS)
	$(CXX) $(CFLAGS) $(ARRAY_SPARSE_TESTS) -o bin/test_array_sparse	


bin/test_access_performance: $(BASE_HEADERS) $(ARRAY_BASIC_HEADERS) test/test_access_performance.cpp
	$(CXX) $(CFLAGS) -O3 test/test_access_performance.cpp -o bin/test_access_performance


GEOMETRY_BASIC_TESTS = test/test_geometry_basics.cpp test/test_geometry_primitives.cpp test/test_poly_scan.cpp
bin/test_geometry_basics: $(BASE_HEADERS) $(TEST_HEADERS) $(GEOMETRY_BASIC_HEADERS) $(GEOMETRY_BASIC_TESTS)
	$(CXX) $(CFLAGS) $(GEOMETRY_BASIC_TESTS) -o bin/test_geometry_basics

	
IMAGE_BASIC_TESTS = test/test_image_basics.cpp test/test_image_views.cpp
bin/test_image_basics: $(BASE_HEADERS) $(TEST_HEADERS) $(IMAGE_BASIC_HEADERS) $(IMAGE_BASIC_TESTS)
	$(CXX) $(CFLAGS) $(IMAGE_BASIC_TESTS) -o bin/test_image_basics
	 

GRAPH_BASIC_TESTS = test/test_graph_basics.cpp test/test_gr_edgelist.cpp test/test_gr_adjlist.cpp test/test_graph_basic_alg.cpp
bin/test_graph_basics: $(BASE_HEADERS) $(TEST_HEADERS) $(GRAPH_BASIC_HEADERS) $(GRAPH_BASIC_TESTS)
	$(CXX) $(BOOST_CFLAGS) $(GRAPH_BASIC_TESTS) -o bin/test_graph_basics	
	
		
PROB_BASIC_TESTS = test/test_prob_basics.cpp test/test_discrete_distr.cpp
bin/test_prob_basics: $(BASE_HEADERS) $(TEST_HEADERS) $(PROB_BASIC_HEADERS) $(PROB_BASIC_TESTS)
	$(CXX) $(BOOST_CFLAGS) $(PROB_BASIC_TESTS) -o bin/test_prob_basics
	
	
bin/test_psampling: $(BASE_HEADERS) $(TEST_HEADERS) $(PROB_BASIC_HEADERS) test/test_psampling.cpp
	$(CXX) $(BOOST_CFLAGS) -O3 test/test_psampling.cpp -o bin/test_psampling
	

clean:
	rm bin/*

