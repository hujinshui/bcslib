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
CPPFLAGS = -I. 

ifeq ($(UNAME), Linux)
	CXX=g++
	CXXFLAGS = -std=c++0x -pedantic $(WARNING_FLAGS) $(CPPFLAGS) 
endif
ifeq ($(UNAME), Darwin)
	CXX=clang++
	CXXFLAGS = -std=c++0x -stdlib=libc++ -pedantic $(WARNING_FLAGS) $(CPPFLAGS)
endif

CXX_FAST=icpc
CXXFLAGS_FAST= -march=native -pedantic -O3 -DBCSLIB_NO_DEBUG $(WARNING_FLAGS) $(CPPFLAGS)
VECT_REPORT=-vec-report2

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

CORE_H = \
	$(INC)/config/user_config.h \
	$(INC)/config/platform_config.h \
	$(INC)/config/config.h \
	$(INC)/core/basic_types.h \
	$(INC)/core/syntax.h \
	$(INC)/core/functional.h \
	$(INC)/core/range.h \
	$(INC)/core/basic_defs.h \
	$(INC)/core/type_traits.h \
	$(INC)/core/iterator.h \
	$(INC)/core/mem_op.h \
	$(INC)/core/bits/mem_op_impl.h \
	$(INC)/core/bits/mem_op_impl_static.h \
	$(INC)/core/block.h \
	$(INC)/core.h
	
MATH_H = \
	$(INC)/math/scalar_math.h \
	$(INC)/math/arithmetic_functors.h \
	$(INC)/math/elementary_functors.h \
	$(INC)/math/basic_reductors.h
	
	
MATRIX_BASE_H = $(CORE_H) \
	$(INC)/matrix/matrix_fwd.h \
	$(INC)/matrix/matrix_ctf.h \
	$(INC)/matrix/matrix_concepts.h \
	$(INC)/matrix/matrix_properties.h \
	$(INC)/matrix/matrix_memop.h \
	$(INC)/matrix/matrix_xpr.h \
	$(INC)/matrix/matrix_assign.h \
	$(INC)/matrix/matrix_io.h \
	$(INC)/matrix/ref_matrix.h \
	$(INC)/matrix/dense_matrix.h \
	$(INC)/matrix/vector_accessors.h \
	$(INC)/matrix/vector_operations.h \
	$(INC)/matrix/ref_grid2d.h \
	$(INC)/matrix/matrix_capture.h \
	$(INC)/matrix/bits/matrix_memop_internal.h \
	$(INC)/matrix/bits/offset_helper.h \
	$(INC)/matrix/bits/ref_matrix_internal.h \
	$(INC)/matrix/bits/ref_grid2d_internal.h \
	$(INC)/matrix/bits/dense_matrix_internal.h \
	$(INC)/matrix/matrix_base.h
	
MATRIX_EXT_H = $(MATRIX_BASE_H) \
	$(INC)/matrix/matrix_subviews.h \
	$(INC)/matrix/matrix_transpose.h \
	$(INC)/matrix/bits/matrix_transpose_internal.h
	
	
MATRIX_EVAL_H = $(MATRIX_EXT_H) $(MATH_H) \
	$(INC)/matrix/ewise_matrix_expr.h \
	$(INC)/matrix/ewise_matrix_eval.h \
	$(INC)/matrix/matrix_arithmetic.h \
	$(INC)/matrix/matrix_elfuns.h \
	$(INC)/matrix/repeat_vectors.h \
	$(INC)/matrix/slicewise_proxy.h \
	$(INC)/matrix/matrix_broadcast.h \
	$(INC)/matrix/matrix_reduction.h \
	$(INC)/matrix/matrix_par_reduc.h \
	$(INC)/matrix/bits/ewise_matrix_eval_internal.h \
	$(INC)/matrix/bits/repeat_vectors_internal.h \
	$(INC)/matrix/bits/matrix_reduction_internal.h \
	$(INC)/matrix/bits/matrix_par_reduc_internal.h
	
LINALG_H = $(MATRIX_EXT_H) \
	$(INC)/engine/blas_extern.h \
	$(INC)/engine/blas.h \
	$(INC)/linalg/linalg_base.h \
	$(INC)/linalg/matrix_blas.h	\
	$(INC)/linalg/matrix_prod.h \
	$(INC)/linalg/bits/matrix_prod_internal.h \
	$(INC)/linalg.h
	

#---------- Target groups -------------------

.PHONY: all
all: test

.PHONY: test
test: test_core test_matrix test_linalg

.PHONY: bench
bench: bench_matrix

.PHONY: clean

clean:
	-rm $(BIN)/*


#------ Core tests ----------

.PHONY: test_core
test_core: $(BIN)/test_memory


#------ Matrix tests --------

.PHONY: test_matrix
test_matrix: \
	$(BIN)/test_matrix_basics \
	$(BIN)/test_matrix_ext \
	$(BIN)/test_matrix_eval \
	$(BIN)/test_matrix_reduc


bench_matrix: \
	$(BIN)/bench_matrix_access \
	$(BIN)/bench_ewise_calc \
	$(BIN)/bench_matrix_reduction 
	
	
#------ Linear Algebra tests --------

.PHONY: test_linalg
test_linalg: \
	$(BIN)/test_matrix_blas


#_________________________________________________________________________
#
#  BELOW ARE DETAILS!
#


#----------------------------------------------------------
#
#   Core test (details)
#
#----------------------------------------------------------


TEST_MEMORY_SOURCES = \
	test/core/test_mem_op.cpp \
	test/core/test_blocks.cpp
	
$(BIN)/test_memory: $(CORE_H) $(TEST_MEMORY_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_MEMORY_SOURCES) $(MAIN_TEST_POST) -o $@


#----------------------------------------------------------
#
#   Matrix test (details)
#
#----------------------------------------------------------

TEST_MATRIX_BASICS_SOURCES = \
	test/matrix/test_dense_matrix.cpp \
	test/matrix/test_ref_matrix.cpp \
	test/matrix/test_ref_grid2d.cpp \
	test/matrix/test_vector_access.cpp \
	test/matrix/test_matrix_memop.cpp \
	test/matrix/test_matrix_assign.cpp
	
$(BIN)/test_matrix_basics: $(MATRIX_BASE_H) $(TEST_MATRIX_BASICS_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_MATRIX_BASICS_SOURCES) $(MAIN_TEST_POST) -o $@
	
TEST_MATRIX_EXT_SOURCES = \
	test/matrix/test_matrix_slices.cpp \
	test/matrix/test_matrix_blocks.cpp \
	test/matrix/test_matrix_transpose.cpp

$(BIN)/test_matrix_ext: $(MATRIX_EXT_H) $(TEST_MATRIX_EXT_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_MATRIX_EXT_SOURCES) $(MAIN_TEST_POST) -o $@

TEST_MATRIX_EVAL_SOURCES = \
	test/matrix/test_matrix_ewise.cpp \
	test/matrix/test_matrix_arithmetic.cpp \
	test/matrix/test_matrix_elfuns.cpp \
	test/matrix/test_repeat_vecs.cpp \
	test/matrix/test_matrix_broadcast.cpp
	
$(BIN)/test_matrix_eval: $(MATRIX_EVAL_H) $(TEST_MATRIX_EVAL_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_MATRIX_EVAL_SOURCES) $(MAIN_TEST_POST) -o $@
	
TEST_MATRIX_REDUC_SOURCES = \
	test/matrix/test_matrix_unary_reduction.cpp \
	test/matrix/test_matrix_binary_reduction.cpp \
	test/matrix/test_matrix_unary_par_reduc.cpp \
	test/matrix/test_matrix_binary_par_reduc.cpp 
	
$(BIN)/test_matrix_reduc: $(MATRIX_EVAL_H) $(TEST_MATRIX_REDUC_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_MATRIX_REDUC_SOURCES) $(MAIN_TEST_POST) -o $@
	
	
$(BIN)/bench_matrix_access: $(MATRIX_BASE_H) bench/bench_matrix_access.cpp
	$(CXX_FAST) $(CXXFLAGS_FAST) bench/bench_matrix_access.cpp -o $@
	
$(BIN)/bench_ewise_calc: $(MATRIX_BASE_H) bench/bench_ewise_calc.cpp
	$(CXX_FAST) $(CXXFLAGS_FAST) bench/bench_ewise_calc.cpp -o $@
	
$(BIN)/bench_matrix_reduction: $(MATRIX_BASE_H) bench/bench_matrix_reduction.cpp
	$(CXX_FAST) $(CXXFLAGS_FAST) bench/bench_matrix_reduction.cpp -o $@

	
#----------------------------------------------------------
#
#   Linear algebra test (details)
#
#----------------------------------------------------------	

TEST_MATRIX_BLAS_SOURCES = \
	test/linalg/test_blas1.cpp \
	test/linalg/test_blas2.cpp \
	test/linalg/test_blas3.cpp \
	test/linalg/test_gen_matrix_prod.cpp
	
$(BIN)/test_matrix_blas: $(LINALG_H) $(TEST_MATRIX_BLAS_SOURCES)
	$(CXX) $(CXXFLAGS) $(BLAS_PATHS) $(MAIN_TEST_PRE) $(TEST_MATRIX_BLAS_SOURCES) $(BLAS_LNKS) $(MAIN_TEST_POST) -o $@
	



