/**
 * @file test_arbb_basics.cpp
 *
 * The testing of ArBB-based functionalities
 *
 * @author Dahua Lin
 */


#include <bcslib/test/execution_mon.h>

using namespace bcs::test;

extern std::shared_ptr<test_suite> test_arbb_port_suite();
extern std::shared_ptr<test_suite> test_arbb_linalg_suite();

test_suite* master_suite()
{
	test_suite* msuite = new test_suite( "arbb_basics" );

	msuite->add( test_arbb_port_suite() );
	msuite->add( test_arbb_linalg_suite() );

	return msuite;
}


BCS_TEST_MAIN_FUNCTION



