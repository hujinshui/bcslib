/**
 * @file test_basics.cpp
 *
 * The master test file for testing basic facilities of the library
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/execution_mon.h>

using namespace bcs::test;

extern test_suite* test_basic_algorithms_suite();

test_suite* master_suite()
{
	test_suite* msuite = new test_suite( "array_basics" );

	msuite->add( test_basic_algorithms_suite() );

	return msuite;
}


BCS_TEST_MAIN_FUNCTION


