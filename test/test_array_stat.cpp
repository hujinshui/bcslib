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

#include <limits>

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


struct dvdot
{
	typedef double result_type;

	double s;

	dvdot() : s(0) { }
	void put(double x, double y) { s += x * y; }
	double get() const { return s; }
};


struct dsum_log
{
	typedef double result_type;

	double s;

	dsum_log() : s(0) { }
	void put(double x) { s += std::log(x); }
	double get() const { return s; }
};


struct dsum_xlogy
{
	typedef double result_type;

	double s;

	dsum_xlogy() : s(0) { }
	void put(double x, double y) { s += x * std::log(y); }
	double get() const { return s; }
};


struct dmean
{
	typedef double result_type;

	double s;
	int n;

	dmean() : s(0), n(0) { }
	void put(double x) { s += x; ++n; }
	double get() const { return s / (double)n; }
};


struct dmin
{
	typedef double result_type;

	double s;

	dmin() : s(std::numeric_limits<double>::infinity()) { }
	void put(double x) { if (x < s) s = x; }
	double get() const { return s; }
};


struct dmax
{
	typedef double result_type;

	double s;

	dmax() : s(-std::numeric_limits<double>::infinity()) { }
	void put(double x) { if (x > s) s = x; }
	double get() const { return s; }
};


struct dminmax
{
	typedef std::pair<double, double> result_type;

	bool started;
	std::pair<double, double> s;

	dminmax()
	{
		s.first = std::numeric_limits<double>::infinity();
		s.second = -std::numeric_limits<double>::infinity();
	}

	void put(double x)
	{
		if (x < s.first) s.first = x;
		if (x > s.second) s.second = x;
	}

	result_type get() const { return s; }
};




BCS_TEST_CASE( test_sum_dot_and_mean )
{
	const size_t Nmax = 36;
	double src[Nmax];
	double src2[Nmax];
	for (size_t i = 0; i < Nmax; ++i) src[i] = (double)(i+1);
	for (size_t i = 0; i < Nmax; ++i) src2[i] = (double)(i * 2 + 1);

	// prepare views

	aview1d<double> v1d = get_aview1d(src, 6);
	aview1d<double, step_ind> v1d_s = get_aview1d_ex(src, step_ind(6, 2));

	aview2d<double, row_major_t>    v2d_rm = get_aview2d_rm(src, 2, 3);
	aview2d<double, column_major_t> v2d_cm = get_aview2d_cm(src, 2, 3);

	aview2d<double, row_major_t,    step_ind, step_ind> v2d_rm_s = get_aview2d_rm_ex(src, 4, 6, step_ind(2, 2), step_ind(3, 2));
	aview2d<double, column_major_t, step_ind, step_ind> v2d_cm_s = get_aview2d_cm_ex(src, 4, 6, step_ind(2, 2), step_ind(3, 2));

	aview1d<double> w1d = get_aview1d(src2, 6);
	aview1d<double, step_ind> w1d_s = get_aview1d_ex(src2, step_ind(6, 2));

	aview2d<double, row_major_t>    w2d_rm = get_aview2d_rm(src2, 2, 3);
	aview2d<double, column_major_t> w2d_cm = get_aview2d_cm(src2, 2, 3);

	aview2d<double, row_major_t,    step_ind, step_ind> w2d_rm_s = get_aview2d_rm_ex(src2, 4, 6, step_ind(2, 2), step_ind(3, 2));
	aview2d<double, column_major_t, step_ind, step_ind> w2d_cm_s = get_aview2d_cm_ex(src2, 4, 6, step_ind(2, 2), step_ind(3, 2));

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

	// vdot

	BCS_CHECK_EQUAL( vdot(v1d, w1d), accum_all(v1d, w1d, dvdot()) );
	BCS_CHECK_EQUAL( vdot(v2d_rm, w2d_rm), accum_all(v2d_rm, w2d_rm, dvdot()) );
	BCS_CHECK_EQUAL( vdot(v2d_cm, w2d_cm), accum_all(v2d_cm, w2d_cm, dvdot()) );

	BCS_CHECK_EQUAL( vdot(v2d_rm, w2d_rm, per_row()), accum_prow(v2d_rm, w2d_rm, dvdot()) );
	BCS_CHECK_EQUAL( vdot(v2d_cm, w2d_cm, per_row()), accum_prow(v2d_cm, w2d_cm, dvdot()) );
	BCS_CHECK_EQUAL( vdot(v2d_rm, w2d_rm, per_col()), accum_pcol(v2d_rm, w2d_rm, dvdot()) );
	BCS_CHECK_EQUAL( vdot(v2d_cm, w2d_cm, per_col()), accum_pcol(v2d_cm, w2d_cm, dvdot()) );

	BCS_CHECK_EQUAL( vdot(v1d_s, w1d_s), accum_all(v1d_s, w1d_s, dvdot()) );
	BCS_CHECK_EQUAL( vdot(v2d_rm_s, w2d_rm_s), accum_all(v2d_rm_s, w2d_rm_s, dvdot()) );
	BCS_CHECK_EQUAL( vdot(v2d_cm_s, w2d_cm_s), accum_all(v2d_cm_s, w2d_cm_s, dvdot()) );

	BCS_CHECK_EQUAL( vdot(v2d_rm_s, w2d_rm_s, per_row()), accum_prow(v2d_rm_s, w2d_rm_s, dvdot()) );
	BCS_CHECK_EQUAL( vdot(v2d_cm_s, w2d_cm_s, per_row()), accum_prow(v2d_cm_s, w2d_cm_s, dvdot()) );
	BCS_CHECK_EQUAL( vdot(v2d_rm_s, w2d_rm_s, per_col()), accum_pcol(v2d_rm_s, w2d_rm_s, dvdot()) );
	BCS_CHECK_EQUAL( vdot(v2d_cm_s, w2d_cm_s, per_col()), accum_pcol(v2d_cm_s, w2d_cm_s, dvdot()) );

	// sum_log

	BCS_CHECK_EQUAL( sum_log(v1d), accum_all(v1d, dsum_log()) );
	BCS_CHECK_EQUAL( sum_log(v2d_rm), accum_all(v2d_rm, dsum_log()) );
	BCS_CHECK_EQUAL( sum_log(v2d_cm), accum_all(v2d_cm, dsum_log()) );

	BCS_CHECK_EQUAL( sum_log(v2d_rm, per_row()), accum_prow(v2d_rm, dsum_log()) );
	BCS_CHECK_EQUAL( sum_log(v2d_cm, per_row()), accum_prow(v2d_cm, dsum_log()) );
	BCS_CHECK_EQUAL( sum_log(v2d_rm, per_col()), accum_pcol(v2d_rm, dsum_log()) );
	BCS_CHECK_EQUAL( sum_log(v2d_cm, per_col()), accum_pcol(v2d_cm, dsum_log()) );

	BCS_CHECK_EQUAL( sum_log(v1d_s), accum_all(v1d_s, dsum_log()) );
	BCS_CHECK_EQUAL( sum_log(v2d_rm_s), accum_all(v2d_rm_s, dsum_log()) );
	BCS_CHECK_EQUAL( sum_log(v2d_cm_s), accum_all(v2d_cm_s, dsum_log()) );

	BCS_CHECK_EQUAL( sum_log(v2d_rm_s, per_row()), accum_prow(v2d_rm_s, dsum_log()) );
	BCS_CHECK_EQUAL( sum_log(v2d_cm_s, per_row()), accum_prow(v2d_cm_s, dsum_log()) );
	BCS_CHECK_EQUAL( sum_log(v2d_rm_s, per_col()), accum_pcol(v2d_rm_s, dsum_log()) );
	BCS_CHECK_EQUAL( sum_log(v2d_cm_s, per_col()), accum_pcol(v2d_cm_s, dsum_log()) );

	// sum_xlogy

	BCS_CHECK_EQUAL( sum_xlogy(v1d, w1d), accum_all(v1d, w1d, dsum_xlogy()) );
	BCS_CHECK_EQUAL( sum_xlogy(v2d_rm, w2d_rm), accum_all(v2d_rm, w2d_rm, dsum_xlogy()) );
	BCS_CHECK_EQUAL( sum_xlogy(v2d_cm, w2d_cm), accum_all(v2d_cm, w2d_cm, dsum_xlogy()) );

	BCS_CHECK_EQUAL( sum_xlogy(v2d_rm, w2d_rm, per_row()), accum_prow(v2d_rm, w2d_rm, dsum_xlogy()) );
	BCS_CHECK_EQUAL( sum_xlogy(v2d_cm, w2d_cm, per_row()), accum_prow(v2d_cm, w2d_cm, dsum_xlogy()) );
	BCS_CHECK_EQUAL( sum_xlogy(v2d_rm, w2d_rm, per_col()), accum_pcol(v2d_rm, w2d_rm, dsum_xlogy()) );
	BCS_CHECK_EQUAL( sum_xlogy(v2d_cm, w2d_cm, per_col()), accum_pcol(v2d_cm, w2d_cm, dsum_xlogy()) );

	BCS_CHECK_EQUAL( sum_xlogy(v1d_s, w1d_s), accum_all(v1d_s, w1d_s, dsum_xlogy()) );
	BCS_CHECK_EQUAL( sum_xlogy(v2d_rm_s, w2d_rm_s), accum_all(v2d_rm_s, w2d_rm_s, dsum_xlogy()) );
	BCS_CHECK_EQUAL( sum_xlogy(v2d_cm_s, w2d_cm_s), accum_all(v2d_cm_s, w2d_cm_s, dsum_xlogy()) );

	BCS_CHECK_EQUAL( sum_xlogy(v2d_rm_s, w2d_rm_s, per_row()), accum_prow(v2d_rm_s, w2d_rm_s, dsum_xlogy()) );
	BCS_CHECK_EQUAL( sum_xlogy(v2d_cm_s, w2d_cm_s, per_row()), accum_prow(v2d_cm_s, w2d_cm_s, dsum_xlogy()) );
	BCS_CHECK_EQUAL( sum_xlogy(v2d_rm_s, w2d_rm_s, per_col()), accum_pcol(v2d_rm_s, w2d_rm_s, dsum_xlogy()) );
	BCS_CHECK_EQUAL( sum_xlogy(v2d_cm_s, w2d_cm_s, per_col()), accum_pcol(v2d_cm_s, w2d_cm_s, dsum_xlogy()) );

	// mean

	BCS_CHECK_EQUAL( mean(v1d), accum_all(v1d, dmean()) );
	BCS_CHECK_EQUAL( mean(v2d_rm), accum_all(v2d_rm, dmean()) );
	BCS_CHECK_EQUAL( mean(v2d_cm), accum_all(v2d_cm, dmean()) );

	BCS_CHECK_EQUAL( mean(v2d_rm, per_row()), accum_prow(v2d_rm, dmean()) );
	BCS_CHECK_EQUAL( mean(v2d_cm, per_row()), accum_prow(v2d_cm, dmean()) );
	BCS_CHECK_EQUAL( mean(v2d_rm, per_col()), accum_pcol(v2d_rm, dmean()) );
	BCS_CHECK_EQUAL( mean(v2d_cm, per_col()), accum_pcol(v2d_cm, dmean()) );

	BCS_CHECK_EQUAL( mean(v1d_s), accum_all(v1d_s, dmean()) );
	BCS_CHECK_EQUAL( mean(v2d_rm_s), accum_all(v2d_rm_s, dmean()) );
	BCS_CHECK_EQUAL( mean(v2d_cm_s), accum_all(v2d_cm_s, dmean()) );

	BCS_CHECK_EQUAL( mean(v2d_rm_s, per_row()), accum_prow(v2d_rm_s, dmean()) );
	BCS_CHECK_EQUAL( mean(v2d_cm_s, per_row()), accum_prow(v2d_cm_s, dmean()) );
	BCS_CHECK_EQUAL( mean(v2d_rm_s, per_col()), accum_pcol(v2d_rm_s, dmean()) );
	BCS_CHECK_EQUAL( mean(v2d_cm_s, per_col()), accum_pcol(v2d_cm_s, dmean()) );

}


BCS_TEST_CASE( test_min_and_max )
{
	const size_t Nmax = 36;
	double src[Nmax] = {
			3, 1, 8, 4, 6, 2,
			2, 5, 7, 3, 1, 9,
			4, 3, 6, 9, 2, 8,
			9, 4, 2, 7, 3, 1,
			1, 7, 9, 2, 8, 4,
			5, 6, 3, 8, 2, 7
	};

	// prepare views

	aview1d<double> v1d = get_aview1d(src, 6);
	aview1d<double, step_ind> v1d_s = get_aview1d_ex(src, step_ind(6, 2));

	aview2d<double, row_major_t>    v2d_rm = get_aview2d_rm(src, 2, 3);
	aview2d<double, column_major_t> v2d_cm = get_aview2d_cm(src, 2, 3);

	aview2d<double, row_major_t,    step_ind, step_ind> v2d_rm_s = get_aview2d_rm_ex(src, 4, 6, step_ind(2, 2), step_ind(3, 2));
	aview2d<double, column_major_t, step_ind, step_ind> v2d_cm_s = get_aview2d_cm_ex(src, 4, 6, step_ind(2, 2), step_ind(3, 2));


	// min

	BCS_CHECK_EQUAL( min(v1d), accum_all(v1d, dmin()) );
	BCS_CHECK_EQUAL( min(v2d_rm), accum_all(v2d_rm, dmin()) );
	BCS_CHECK_EQUAL( min(v2d_cm), accum_all(v2d_cm, dmin()) );

	BCS_CHECK_EQUAL( min(v2d_rm, per_row()), accum_prow(v2d_rm, dmin()) );
	BCS_CHECK_EQUAL( min(v2d_cm, per_row()), accum_prow(v2d_cm, dmin()) );
	BCS_CHECK_EQUAL( min(v2d_rm, per_col()), accum_pcol(v2d_rm, dmin()) );
	BCS_CHECK_EQUAL( min(v2d_cm, per_col()), accum_pcol(v2d_cm, dmin()) );

	BCS_CHECK_EQUAL( min(v1d_s), accum_all(v1d_s, dmin()) );
	BCS_CHECK_EQUAL( min(v2d_rm_s), accum_all(v2d_rm_s, dmin()) );
	BCS_CHECK_EQUAL( min(v2d_cm_s), accum_all(v2d_cm_s, dmin()) );

	BCS_CHECK_EQUAL( min(v2d_rm_s, per_row()), accum_prow(v2d_rm_s, dmin()) );
	BCS_CHECK_EQUAL( min(v2d_cm_s, per_row()), accum_prow(v2d_cm_s, dmin()) );
	BCS_CHECK_EQUAL( min(v2d_rm_s, per_col()), accum_pcol(v2d_rm_s, dmin()) );
	BCS_CHECK_EQUAL( min(v2d_cm_s, per_col()), accum_pcol(v2d_cm_s, dmin()) );

	// max

	BCS_CHECK_EQUAL( max(v1d), accum_all(v1d, dmax()) );
	BCS_CHECK_EQUAL( max(v2d_rm), accum_all(v2d_rm, dmax()) );
	BCS_CHECK_EQUAL( max(v2d_cm), accum_all(v2d_cm, dmax()) );

	BCS_CHECK_EQUAL( max(v2d_rm, per_row()), accum_prow(v2d_rm, dmax()) );
	BCS_CHECK_EQUAL( max(v2d_cm, per_row()), accum_prow(v2d_cm, dmax()) );
	BCS_CHECK_EQUAL( max(v2d_rm, per_col()), accum_pcol(v2d_rm, dmax()) );
	BCS_CHECK_EQUAL( max(v2d_cm, per_col()), accum_pcol(v2d_cm, dmax()) );

	BCS_CHECK_EQUAL( max(v1d_s), accum_all(v1d_s, dmax()) );
	BCS_CHECK_EQUAL( max(v2d_rm_s), accum_all(v2d_rm_s, dmax()) );
	BCS_CHECK_EQUAL( max(v2d_cm_s), accum_all(v2d_cm_s, dmax()) );

	BCS_CHECK_EQUAL( max(v2d_rm_s, per_row()), accum_prow(v2d_rm_s, dmax()) );
	BCS_CHECK_EQUAL( max(v2d_cm_s, per_row()), accum_prow(v2d_cm_s, dmax()) );
	BCS_CHECK_EQUAL( max(v2d_rm_s, per_col()), accum_pcol(v2d_rm_s, dmax()) );
	BCS_CHECK_EQUAL( max(v2d_cm_s, per_col()), accum_pcol(v2d_cm_s, dmax()) );

	// minmax

	BCS_CHECK_EQUAL( minmax(v1d), accum_all(v1d, dminmax()) );
	BCS_CHECK_EQUAL( minmax(v2d_rm), accum_all(v2d_rm, dminmax()) );
	BCS_CHECK_EQUAL( minmax(v2d_cm), accum_all(v2d_cm, dminmax()) );

	BCS_CHECK_EQUAL( minmax(v2d_rm, per_row()), accum_prow(v2d_rm, dminmax()) );
	BCS_CHECK_EQUAL( minmax(v2d_cm, per_row()), accum_prow(v2d_cm, dminmax()) );
	BCS_CHECK_EQUAL( minmax(v2d_rm, per_col()), accum_pcol(v2d_rm, dminmax()) );
	BCS_CHECK_EQUAL( minmax(v2d_cm, per_col()), accum_pcol(v2d_cm, dminmax()) );

	BCS_CHECK_EQUAL( minmax(v1d_s), accum_all(v1d_s, dminmax()) );
	BCS_CHECK_EQUAL( minmax(v2d_rm_s), accum_all(v2d_rm_s, dminmax()) );
	BCS_CHECK_EQUAL( minmax(v2d_cm_s), accum_all(v2d_cm_s, dminmax()) );

	BCS_CHECK_EQUAL( minmax(v2d_rm_s, per_row()), accum_prow(v2d_rm_s, dminmax()) );
	BCS_CHECK_EQUAL( minmax(v2d_cm_s, per_row()), accum_prow(v2d_cm_s, dminmax()) );
	BCS_CHECK_EQUAL( minmax(v2d_rm_s, per_col()), accum_pcol(v2d_rm_s, dminmax()) );
	BCS_CHECK_EQUAL( minmax(v2d_cm_s, per_col()), accum_pcol(v2d_cm_s, dminmax()) );

}


test_suite* test_array_stat_suite()
{
	test_suite* suite = new test_suite("test_array_stat");

	suite->add( new test_sum_dot_and_mean() );
	suite->add( new test_min_and_max() );

	return suite;
}

