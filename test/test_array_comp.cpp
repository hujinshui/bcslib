/**
 * @file test_array_comp.cpp
 *
 * The master file for testing array computation
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/execution_mon.h>

using namespace bcs::test;

extern test_suite* test_array_calc_suite();
extern test_suite* test_array_stat_suite();
extern test_suite* test_logical_array_ops_suite();

test_suite* master_suite()
{
	test_suite* msuite = new test_suite( "array_computation" );

	msuite->add( test_array_calc_suite() );
	msuite->add( test_array_stat_suite() );
	msuite->add( test_logical_array_ops_suite() );

	return msuite;
}


BCS_TEST_MAIN_FUNCTION

