/**
 * @file test_image_basics.cpp
 *
 * Test the basics of image classes
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/execution_mon.h>

using namespace bcs::test;

extern test_suite* test_image_views_suite();

test_suite* master_suite()
{
	test_suite* msuite = new test_suite( "image_basics" );

	msuite->add( test_image_views_suite() );

	return msuite;
}


BCS_TEST_MAIN_FUNCTION

