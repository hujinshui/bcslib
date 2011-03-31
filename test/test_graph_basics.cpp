/**
 * @file test_graph_basics.cpp
 *
 * Unit testing for basic aspects of graph classes
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/execution_mon.h>

using namespace bcs::test;

// extern test suites

extern test_suite *test_gr_edgelist_suite();
extern test_suite *test_gr_adjlist_suite();

test_suite* master_suite()
{
	test_suite* msuite = new test_suite( "graph_basics" );

	msuite->add( test_gr_edgelist_suite() );
	msuite->add( test_gr_adjlist_suite() );

	return msuite;
}


BCS_TEST_MAIN_FUNCTION
