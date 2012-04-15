# The make file for BCSLib

# Detect platform
#--------------------

UNAME := $(shell uname -s)
MACH_TYPE := $(shell uname -m)

ifneq ($(MACH_TYPE), x86_64)
    $(error Only 64-bit platform is supported currently)
endif


# Compiler configuration
#-------------------------

WARNING_FLAGS = -Wall -Wextra -Wconversion -Wformat -Wno-unused-parameter
CPPFLAGS = -I. -isystem $(ARMA_HOME)/include

ifeq ($(UNAME), Linux)
	CXX=g++
	CXXFLAGS = -std=c++0x -pedantic $(WARNING_FLAGS) $(CPPFLAGS)
endif
ifeq ($(UNAME), Darwin)
	CXX=clang++
	CXXFLAGS = -std=c++0x -stdlib=libc++ -pedantic $(WARNING_FLAGS) $(CPPFLAGS)
endif

OFLAGS=-O3 -ffast-math


# Intel MKL configuration

USE_MKL=yes

MKL_INC_PATH = -I$(MKLROOT)/include

ifeq ($(UNAME), Linux)
    MKL_LNK_PATH = -L$(MKLROOT)/../../lib/intel64 -L$(MKLROOT)/lib/intel64
	MKL_LNK = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
endif
ifeq ($(UNAME), Darwin)
    MKL_LNK_PATH = -L$(MKLROOT)/../../lib -L$(MKLROOT)/lib
	MKL_LNK = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
endif
 
ifeq ($(USE_MKL), yes)
   BLAS_PATHS = $(MKL_INC_PATH) $(MKL_LNK_PATH)
   BLAS_LNKS = $(MKL_LNK)
endif


# Testing setup

MAIN_TEST_PRE=-isystem $(GTEST_HOME)/include test/bcs_test_main.cpp
MAIN_TEST_POST=$(GTEST_HOME)/lib/libgtest.a -lpthread


# directory configuration

INC=bcslib

#------ Output directory ----------

BIN=bin

#------ Header groups ----------

BASE_HEADERS = \
	$(INC)/base/config.h \
	$(INC)/base/basic_defs.h \
	$(INC)/base/arg_check.h \
	$(INC)/base/basic_math.h \
	$(INC)/base/mem_op.h \
	$(INC)/base/key_map.h \
	$(INC)/base/smart_ptr.h \
	$(INC)/base/tr1_containers.h \
	$(INC)/base/type_traits.h \
	$(INC)/base/basic_algorithms.h \
	$(INC)/base/iterator_wrappers.h \
	$(INC)/base/block.h \
	$(INC)/base/monitored_allocator.h \
	$(INC)/base/timer.h
	
MATRIX_HEADERS = \
	$(INC)/matrix/matrix_base.h \
	$(INC)/matrix/dense_matrix.h	


ARRAY_BASIC_HEADERS = $(BASE_HEADERS) \
	$(INC)/array/aview_base.h \
	$(INC)/array/aindex.h \
	$(INC)/array/aview1d_base.h \
	$(INC)/array/aview1d.h \
	$(INC)/array/aview1d_ops.h \
	$(INC)/array/array1d.h \
	$(INC)/array/amap.h \
	$(INC)/array/aview2d_base.h \
	$(INC)/array/aview2d_slices.h \
	$(INC)/array/aview2d.h \
	$(INC)/array/aview2d_ops.h \
	$(INC)/array/transpose2d.h \
	$(INC)/array/array2d.h
	
LINALG_HEADERS = $(ARRAY_BASIC_HEADERS) \
	$(INC)/extern/blas_select.h \
	$(INC)/array/aview_blas.h
	
DATASTRUCT_BASIC_HEADERS = $(BASE_HEADERS) \
	$(INC)/data_structs/hash_accumulator.h \
	$(INC)/data_structs/binary_heap.h \
	$(INC)/data_structs/disjoint_sets.h
		
GRAPH_BASIC_HEADERS = $(BASE_HEADERS) \
	$(INC)/graph/gview_base.h \
	$(INC)/graph/gedgelist_view.h \
	$(INC)/graph/ginclist_view.h \
	$(INC)/graph/ginclist.h
	
GRAPH_ALGORITHM_HEADERS = $(GRAPH_BASIC_HEADERS) \
	$(INC)/graph/graph_traversal.h \
	$(INC)/graph/graph_shortest_paths.h \
	$(INC)/graph/graph_minimum_span_trees.h


#---------- Target groups -------------------

.PHONY: all
all: test
# all: test bench

.PHONY: test
test: test_basics test_matrix
# test: test_basics test_array test_data_structs test_graph

.PHONY: benchnelems, 
bench: bench_array

.PHONY: clean

clean:
	-rm $(BIN)/*


#------ Basic tests ----------

.PHONY: test_basics
test_basics: $(BIN)/test_basics

#------ Matrix tests ----------

.PHONY: test_matrix
test_matrix: $(BIN)/test_matrix_basics


#------ Array tests (declaration) -----------

.PHONY: test_array
.PHONY: bench_array	

test_array: \
	$(BIN)/test_array_basics
	
test_array_pending: \
	$(BIN)/test_array_comp \
	$(BIN)/test_array_comp_intel \
	$(BIN)/test_linalg_intel

bench_array: \
	$(BIN)/bench_array_access
	
	
#------ Data struct tests (declaration) -----------	
	
.PHONY: test_data_structs

test_data_structs: \
	$(BIN)/test_data_struct_basics	
	
	
#------ Graph tests (declaration) -----------

.PHONY: test_graph

test_graph: \
	$(BIN)/test_graph_basics \
	$(BIN)/test_graph_algorithms



#_________________________________________________________________________
#
#  BELOW ARE DETAILS!
#

#----------------------------------------------------------
#
#   Basics test (details)
#
#----------------------------------------------------------


TEST_BASICS_SOURCES = \
	test/test_basic_concepts.cpp \
	test/test_basic_algorithms.cpp \
	test/test_basic_memory.cpp \
	
$(BIN)/test_basics: $(BASE_HEADERS) $(TEST_BASICS_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_BASICS_SOURCES) $(MAIN_TEST_POST) -o $@


#----------------------------------------------------------
#
#   Matrix (details)
#
#----------------------------------------------------------

TEST_MATRIX_BASICS_SOURCES = \
	test/test_dense_matrix.cpp


$(BIN)/test_matrix_basics: $(MATRIX_HEADERS) $(TEST_MATRIX_BASICS_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_MATRIX_BASICS_SOURCES) $(MAIN_TEST_POST) -o $@



#----------------------------------------------------------
#
#   Array test & bench (details)
#
#----------------------------------------------------------

# array tests

TEST_ARRAY_BASICS_SOURCES = \
	test/test_aindex.cpp \
	test/test_array1d.cpp \
	test/test_array2d.cpp \
	test/test_array_transpose.cpp
			
$(BIN)/test_array_basics: $(ARRAY_BASIC_HEADERS) $(TEST_ARRAY_BASICS_SOURCES) 
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_ARRAY_BASICS_SOURCES) $(MAIN_TEST_POST) -o $@
	
TEST_LINALG_SOURCES = \
	test/test_array_blas.cpp

$(BIN)/test_linalg_intel: $(LINALG_HEADERS) $(ARRAY_TEST_HEADERS) $(TEST_LINALG_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(INTEL_FLAGS) $(TEST_LINALG_SOURCES) $(MAIN_TEST_POST) -o $@

# array bench

$(BIN)/bench_array_access: $(ARRAY_BASIC_HEADERS) bench/bench_array_access.cpp
	$(CXX) $(CXXFLAGS) $(OFLAGS) bench/bench_array_access.cpp -o $@


#----------------------------------------------------------
#
#   Data structure test & bench
#
#----------------------------------------------------------

# Data structure tests

TEST_DATASTRUCT_BASICS_SOURCES = \
	test/test_binary_heap.cpp \
	test/test_disjoint_sets.cpp

$(BIN)/test_data_struct_basics: $(DATASTRUCT_BASIC_HEADERS) $(TEST_DATASTRUCT_BASICS_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_DATASTRUCT_BASICS_SOURCES) $(MAIN_TEST_POST) -o $@



#----------------------------------------------------------
#
#   Graph test & bench
#
#----------------------------------------------------------

# Graph tests

TEST_GRAPH_BASICS_SOURCES = \
	test/test_gview_base.cpp \
	test/test_gedgelist.cpp \
	test/test_ginclist.cpp
	
TEST_GRAPH_ALGORITHMS_SOURCES = \
	test/test_graph_traversal.cpp \
	test/test_graph_shortest_paths.cpp \
	test/test_graph_minimum_span_trees.cpp
	
$(BIN)/test_graph_basics: $(GRAPH_BASIC_HEADERS) $(TEST_GRAPH_BASICS_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_GRAPH_BASICS_SOURCES) $(MAIN_TEST_POST) -o $@

$(BIN)/test_graph_algorithms: $(GRAPH_ALGORITHM_HEADERS) $(TEST_GRAPH_ALGORITHMS_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_GRAPH_ALGORITHMS_SOURCES) $(MAIN_TEST_POST) -o $@

