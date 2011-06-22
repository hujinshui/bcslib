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


// auxiliary functions for testing

struct dsum
{
	typedef double result_type;

	double s;

	dsum() : s(0) { }
	void put(double x) { s += x; }
	double get() const { return s; }
};



BCS_TEST_CASE( test_sum_and_mean )
{
	const size_t Nmax = 24;
	double *src = new double[Nmax];
	for (size_t i = 0; i < Nmax; ++i) src[i] = (double)(i+1);

	// prepare views

	aview1d<double> v1d = get_aview1d(src, 6);  // 1 2 3 4 5 6
	aview1d<double, step_ind> v1d_s = get_aview1d_ex(src, step_ind(6, 2));  // 1 3 5 7 9 11

	aview2d<double, row_major_t>    v2d_rm = get_aview2d_rm(src, 2, 3);  // 1 2 3; 4 5 6
	aview2d<double, column_major_t> v2d_cm = get_aview2d_cm(src, 2, 3);  // 1 3 5; 2 4 6

	aview2d<double, row_major_t,    step_ind, step_ind> v2d_rm_s = get_aview2d_rm_ex(src, 4, 6, step_ind(2, 2), step_ind(3, 2)); // 1 3 5; 13 15 17
	aview2d<double, column_major_t, step_ind, step_ind> v2d_cm_s = get_aview2d_cm_ex(src, 4, 6, step_ind(2, 2), step_ind(3, 2)); // 1 9 17; 3 11 19

	// sum

	BCS_CHECK_EQUAL( sum(v1d), accum_all(v1d, dsum()) );
	BCS_CHECK_EQUAL( sum(v2d_rm), accum_all(v2d_rm, dsum()) );
	BCS_CHECK_EQUAL( sum(v2d_cm), accum_all(v2d_cm, dsum()) );

	BCS_CHECK_EQUAL( sum(v2d_rm, per_row()), accum_prow(v2d_rm, dsum()) );
	BCS_CHECK_EQUAL( sum(v2d_cm, per_row()), accum_prow(v2d_cm, dsum()) );
	BCS_CHECK_EQUAL( sum(v2d_rm, per_col()), accum_pcol(v2d_rm, dsum()) );
	BCS_CHECK_EQUAL( sum(v2d_cm, per_col()), accum_pcol(v2d_cm, dsum()) );

	BCS_CHECK_EQUAL( sum(v1d_s), accum_all(v1d_s, dsum()) );
	BCS_CHECK_EQUAL( sum(v2d_rm_s), accum_all(v2d_rm_s, dsum()) );
	BCS_CHECK_EQUAL( sum(v2d_cm_s), accum_all(v2d_cm_s, dsum()) );

	BCS_CHECK_EQUAL( sum(v2d_rm_s, per_row()), accum_prow(v2d_rm_s, dsum()) );
	BCS_CHECK_EQUAL( sum(v2d_cm_s, per_row()), accum_prow(v2d_cm_s, dsum()) );
	BCS_CHECK_EQUAL( sum(v2d_rm_s, per_col()), accum_pcol(v2d_rm_s, dsum()) );
	BCS_CHECK_EQUAL( sum(v2d_cm_s, per_col()), accum_pcol(v2d_cm_s, dsum()) );

}


test_suite* test_array_stat_suite()
{
	test_suite* suite = new test_suite("test_array_stat");

	suite->add( new test_sum_and_mean() );

	return suite;
}

