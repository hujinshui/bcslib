/**
 * @file test_array_sparse.cpp
 *
 * Test the sparse vector & sparse matrix classes
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/execution_mon.h>

using namespace bcs::test;

extern test_suite* test_spvec_suite();

test_suite* master_suite()
{
	test_suite* msuite = new test_suite( "array_sparse" );

	msuite->add( test_spvec_suite() );

	return msuite;
}


BCS_TEST_MAIN_FUNCTION



