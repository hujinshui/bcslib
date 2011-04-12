/**
 * @file test_geometry_basics.cpp
 *
 * Some tests on geometry
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/execution_mon.h>

using namespace bcs::test;

extern test_suite* test_geometry_prim_suite();
extern test_suite* test_poly_scan_suite();

test_suite* master_suite()
{
	test_suite* msuite = new test_suite( "image_basics" );

	msuite->add( test_geometry_prim_suite() );
	// msuite->add( test_poly_scan_suite() );

	return msuite;
}


BCS_TEST_MAIN_FUNCTION
