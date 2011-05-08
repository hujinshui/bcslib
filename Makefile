# The make file for BCSLib

# compiler configuration

CC = g++
WARNING_FLAGS = -Wall -Wextra -Wconversion -Wformat -Wno-unused-parameter
CFLAGS = -std=c++0x -pedantic $(WARNING_FLAGS) -I.

BASE_HEADERS = bcslib/base/config.h bcslib/base/basic_defs.h bcslib/base/basic_funcs.h bcslib/base/basic_mem.h bcslib/base/enumerate.h 

TEST_HEADERS = bcslib/test/test_assertion.h bcslib/test/test_units.h bcslib/test/execution_mon.h

VEC_COMP_HEADERS = bcslib/veccomp/veccalc.h bcslib/veccomp/vecnorm.h bcslib/veccomp/vecstat.h

ARRAY_BASIC_HEADERS = bcslib/array/array_base.h bcslib/array/index_selection.h bcslib/array/array_index.h bcslib/array/array1d.h bcslib/array/array2d.h

ARRAY_COMP_HEADERS = bcslib/array/array_calc.h bcslib/array/array_eval.h

GRAPH_BASIC_HEADERS = bcslib/graph/graph_base.h bcslib/graph/gr_edgelist.h bcslib/graph/gr_adjlist.h bcslib/graph/bgl_port.h

GEOMETRY_BASIC_HEADERS = bcslib/geometry/geometry_base.h bcslib/geometry/poly_scan.h bcslib/geometry/triangle_mesh.h

IMAGE_BASIC_HEADERS = bcslib/image/image_base.h bcslib/image/image.h

PROB_BASIC_HEADERS = bcslib/prob/pdistribution.h bcslib/prob/sampling.h bcslib/prob/discrete_distr.h 


all: test_array test_graph test_geometry test_image test_prob

test_array : bin/test_array_basics bin/test_array_comp bin/test_access_performance
test_graph : bin/test_graph_basics
test_geometry : bin/test_geometry_basics
test_image: bin/test_image_basics
test_prob: bin/test_prob_basics bin/test_psampling

ARRAY_BASIC_TESTS = test/test_array_basics.cpp test/test_index_selection.cpp test/test_array1d.cpp test/test_array2d.cpp
bin/test_array_basics: $(BASE_HEADERS) $(TEST_HEADERS) $(ARRAY_BASIC_HEADERS) $(ARRAY_BASIC_TESTS) 
	$(CC) $(CFLAGS) $(ARRAY_BASIC_TESTS) -o bin/test_array_basics
	
	
ARRAY_COMP_TESTS = test/test_array_comp.cpp test/test_array_calc.cpp test/test_array_eval.cpp
bin/test_array_comp: $(BASE_HEADERS) $(TEST_HEADERS) $(ARRAY_BASIC_HEADERS) $(VEC_COMP_HEADERS) $(ARRAY_COMP_HEADERS) $(ARRAY_COMP_TESTS) 
	$(CC) $(CFLAGS) $(ARRAY_COMP_TESTS) -o bin/test_array_comp
	

bin/test_access_performance: $(BASE_HEADERS) $(ARRAY_BASIC_HEADERS) test/test_access_performance.cpp
	$(CC) $(CFLAGS) -O2 test/test_access_performance.cpp -o bin/test_access_performance

GRAPH_BASIC_TESTS = test/test_graph_basics.cpp test/test_gr_edgelist.cpp test/test_gr_adjlist.cpp test/test_graph_basic_alg.cpp
bin/test_graph_basics: $(BASE_HEADERS) $(TEST_HEADERS) $(GRAPH_BASIC_HEADERS) $(GRAPH_BASIC_TESTS)
	$(CC) $(CFLAGS) $(GRAPH_BASIC_TESTS) -o bin/test_graph_basics	
	
GEOMETRY_BASIC_TESTS = test/test_geometry_basics.cpp test/test_geometry_primitives.cpp test/test_poly_scan.cpp
bin/test_geometry_basics: $(BASE_HEADERS) $(TEST_HEADERS) $(GEOMETRY_BASIC_HEADERS) $(GEOMETRY_BASIC_TESTS)
	$(CC) $(CFLAGS) $(GEOMETRY_BASIC_TESTS) -o bin/test_geometry_basics

	
IMAGE_BASIC_TESTS = test/test_image_basics.cpp test/test_image_views.cpp
bin/test_image_basics: $(BASE_HEADERS) $(TEST_HEADERS) $(IMAGE_BASIC_HEADERS) $(IMAGE_BASIC_TESTS)
	$(CC) $(CFLAGS) $(IMAGE_BASIC_TESTS) -o bin/test_image_basics
	 
bin/test_psampling: $(BASE_HEADERS) $(TEST_HEADERS) $(PROB_BASIC_HEADERS) test/test_psampling.cpp
	$(CC) $(CFLAGS) -O2 test/test_psampling.cpp -o bin/test_psampling
	
PROB_BASIC_TESTS = test/test_prob_basics.cpp test/test_discrete_distr.cpp
bin/test_prob_basics: $(BASE_HEADERS) $(TEST_HEADERS) $(PROB_BASIC_HEADERS) $(PROB_BASIC_TESTS)
	$(CC) $(CFLAGS) $(PROB_BASIC_TESTS) -o bin/test_prob_basics


clean:
	rm bin/*

