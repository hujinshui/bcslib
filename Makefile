# The make file for BCSLib

# compiler configuration

CC = g++
CFLAGS = -I. -Wall

BASE_HEADERS = bcslib/base/config.h bcslib/base/basic_defs.h bcslib/base/basic_funcs.h bcslib/base/basic_mem.h bcslib/base/enumerate.h 

TEST_HEADERS = bcslib/test/test_assertion.h bcslib/test/test_units.h bcslib/test/execution_mon.h

VEC_COMP_HEADERS = bcslib/veccomp/veccalc.h bcslib/veccomp/vecnorm.h bcslib/veccomp/vecstat.h

ARRAY_BASIC_HEADERS = bcslib/array/array_base.h bcslib/array/index_selection.h bcslib/array/array_index.h bcslib/array/array1d.h bcslib/array/array2d.h

ARRAY_COMP_HEADERS = bcslib/array/array_calc.h

all: test_array

test_array : bin/test_array_basics bin/test_array_comp

ARRAY_BASIC_TESTS = test/test_array_basics.cpp test/test_index_selection.cpp test/test_array1d.cpp test/test_array2d.cpp
bin/test_array_basics: $(BASE_HEADERS) $(TEST_HEADERS) $(ARRAY_BASIC_HEADERS) $(ARRAY_BASIC_TESTS) 
	$(CC) $(CFLAGS) $(ARRAY_BASIC_TESTS) -o bin/test_array_basics
	
	
ARRAY_COMP_TESTS = test/test_array_comp.cpp test/test_array_calc.cpp 
bin/test_array_comp: $(BASE_HEADERS) $(TEST_HEADERS) $(ARRAY_BASIC_HEADERS) $(VEC_COMP_HEADERS) $(ARRAY_COMP_HEADERS) $(ARRAY_COMP_TESTS) 
	$(CC) $(CFLAGS) $(ARRAY_COMP_TESTS) -o bin/test_array_comp