/**
 * @file test_linalg.cpp
 *
 * Testing of Linear Algebra computation
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/execution_mon.h>

using namespace bcs::test;

extern test_suite* test_array_blas_suite();

test_suite* master_suite()
{
	test_suite* msuite = new test_suite( "linear_algebra" );

	msuite->add( test_array_blas_suite() );

	return msuite;
}


BCS_TEST_MAIN_FUNCTION

