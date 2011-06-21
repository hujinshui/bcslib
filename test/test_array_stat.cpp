/**
 * @file test_array_stat.cpp
 *
 * Unit testing of array stat evaluation
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/test/test_array_aux.h>

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>
#include <bcslib/array/array_stat.h>

using namespace bcs;
using namespace bcs::test;


BCS_TEST_CASE( test_sum_and_mean )
{
	const size_t Nmax = 24;
	double src[Nmax];
	for (size_t i = 0; i < Nmax; ++i) src[i] = i + 1;

	BCS_CHECK_EQUAL( sum(get_aview1d(src, 6)),    21.0 );
	BCS_CHECK_EQUAL( sum(get_aview2d_rm(src, 2, 3)), 21.0 );
	BCS_CHECK_EQUAL( sum(get_aview2d_cm(src, 2, 3)), 21.0 );
}


test_suite* test_array_stat_suite()
{
	test_suite* suite = new test_suite("test_array_stat");

	suite->add( new test_sum_and_mean() );

	return suite;
}

