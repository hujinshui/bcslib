/**
 * @file test_array_basics.cpp
 *
 * The master test file for testing basic aspects of array classes
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/execution_mon.h>

using namespace bcs::test;

extern std::shared_ptr<test_suite> test_index_selection_suite();
extern std::shared_ptr<test_suite> test_array1d_suite();
extern std::shared_ptr<test_suite> test_array2d_suite();
extern std::shared_ptr<test_suite> test_array_transposition_suite();
extern std::shared_ptr<test_suite> test_array_eval_suite();

test_suite* master_suite()
{
	test_suite* msuite = new test_suite( "array_basics" );

	msuite->add( test_index_selection_suite() );
	msuite->add( test_array1d_suite() );
	msuite->add( test_array2d_suite() );
	msuite->add( test_array_transposition_suite() );
	msuite->add( test_array_eval_suite() );

	return msuite;
}


BCS_TEST_MAIN_FUNCTION



