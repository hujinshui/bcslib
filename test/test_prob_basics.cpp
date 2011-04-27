/**
 * @file test_prob_basics.cpp
 *
 * The Unit testing of basics of probability stuff
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/execution_mon.h>

using namespace bcs::test;

extern test_suite* test_discrete_distr_suite();

test_suite* master_suite()
{
	test_suite* msuite = new test_suite( "prob_basics" );

	msuite->add( test_discrete_distr_suite() );

	return msuite;
}


BCS_TEST_MAIN_FUNCTION


