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

CORE_H = \
	$(INC)/config/user_config.h \
	$(INC)/config/platform_config.h \
	$(INC)/config/config.h \
	$(INC)/core/basic_types.h \
	$(INC)/core/syntax.h \
	$(INC)/core/scalar_math.h \
	$(INC)/core/basic_defs.h \
	$(INC)/core/type_traits.h \
	$(INC)/core/iterator.h \
	$(INC)/core/mem_op.h \
	$(INC)/engine/mem_op_impl.h \
	$(INC)/engine/mem_op_impl_static.h \
	$(INC)/core/block.h \
	$(INC)/core.h

MATRIX_H = $(CORE_H) \
	$(INC)/matrix/matrix_base.h \
	$(INC)/matrix/matrix_fwd.h \
	$(INC)/matrix/matrix_xpr.h \
	$(INC)/matrix/dense_matrix.h \
	$(INC)/matrix/ref_matrix.h \
	$(INC)/matrix/bits/matrix_helpers.h \
	$(INC)/matrix/bits/dense_matrix_internal.h \
	$(INC)/matrix/bits/ref_matrix_internal.h \
	$(INC)/matrix.h
	
	

#---------- Target groups -------------------

.PHONY: all
all: test
# all: test bench

.PHONY: test
test: test_core test_matrix

.PHONY: clean

clean:
	-rm $(BIN)/*


#------ Core tests ----------

.PHONY: test_core
test_core: $(BIN)/test_memory


#------ Matrix tests --------

.PHONY: test_matrix
test_matrix: $(BIN)/test_matrix_basics


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
	test/matrix/test_ref_matrix.cpp
	
$(BIN)/test_matrix_basics: $(MATRIX_H) $(TEST_MATRIX_BASICS_SOURCES)
	$(CXX) $(CXXFLAGS) $(MAIN_TEST_PRE) $(TEST_MATRIX_BASICS_SOURCES) $(MAIN_TEST_POST) -o $@




