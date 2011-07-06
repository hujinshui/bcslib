/**
 * @file test_array_eval.cpp
 *
 * 
 *
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/test/test_array_aux.h>

#include <bcslib/array/array_eval.h>

#include <functional>

using namespace bcs;
using namespace bcs::test;

template<typename T>
struct mysum
{
	typedef T result_type;

	T operator() (size_t n, const T *x)
	{
		T s = 0;
		for (size_t i = 0; i < n; ++i)
		{
			s += x[i];
		}
		return s;
	}
};

template<typename T>
struct mydot
{
	typedef T result_type;

	T operator() (size_t n, const T *x1, const T *x2)
	{
		T s = 0;
		for (size_t i = 0; i < n; ++i)
		{
			s += x1[i] * x2[i];
		}
		return s;
	}
};



BCS_TEST_CASE( test_array_map )
{
	const size_t N = 6;
	const size_t m = 2;
	const size_t n = 3;

	double src1[N] = {1, 4, 6, 7, 8, 2};
	double src2[N] = {4, 3, 5, 8, 1, 9};

	caview1d<double> x1(src1, N);
	caview1d<double> x2(src2, N);
	caview2d<double, row_major_t> X1_rm(src1, m, n);
	caview2d<double, row_major_t> X2_rm(src2, m, n);
	caview2d<double, column_major_t> X1_cm(src1, m, n);
	caview2d<double, column_major_t> X2_cm(src2, m, n);

	double rneg[N] = {-1, -4, -6, -7, -8, -2};
	double radd[N] = {5, 7, 11, 15, 9, 11};

	// 1D

	array1d<double> y1 = map(std::negate<double>(), x1);
	BCS_CHECK( array_view_equal(y1, rneg, N) );

	array1d<double> y2 = map(std::plus<double>(), x1, x2);
	BCS_CHECK( array_view_equal(y2, radd, N) );

	// 2D row major

	array2d<double, row_major_t> Y1_rm = map(std::negate<double>(), X1_rm);
	BCS_CHECK( array_view_equal(Y1_rm, rneg, m, n) );

	array2d<double, row_major_t> Y2_rm = map(std::plus<double>(), X1_rm, X2_rm);
	BCS_CHECK( array_view_equal(Y2_rm, radd, m, n) );

	// 2D column major

	array2d<double, column_major_t> Y1_cm = map(std::negate<double>(), X1_cm);
	BCS_CHECK( array_view_equal(Y1_cm, rneg, m, n) );

	array2d<double, column_major_t> Y2_cm = map(std::plus<double>(), X1_cm, X2_cm);
	BCS_CHECK( array_view_equal(Y2_cm, radd, m, n) );
}


BCS_TEST_CASE( test_array_vreduce )
{
	const size_t N = 6;
	const size_t m = 2;
	const size_t n = 3;

	double src1[N] = {1, 4, 6, 7, 8, 2};
	double src2[N] = {4, 3, 5, 8, 1, 9};

	caview1d<double> x1(src1, N);
	caview1d<double> x2(src2, N);
	caview2d<double, row_major_t> X1_rm(src1, m, n);
	caview2d<double, row_major_t> X2_rm(src2, m, n);
	caview2d<double, column_major_t> X1_cm(src1, m, n);
	caview2d<double, column_major_t> X2_cm(src2, m, n);

	// 1D

	double sv = 28;
	double dv = 128;

	BCS_CHECK_EQUAL( vreduce(mysum<double>(), x1),  sv );
	BCS_CHECK_EQUAL( vreduce(mydot<double>(), x1, x2),  dv );

	// 2D

	BCS_CHECK_EQUAL( vreduce(mysum<double>(), X1_rm),  sv );
	BCS_CHECK_EQUAL( vreduce(mydot<double>(), X1_rm, X2_rm),  dv );

	BCS_CHECK_EQUAL( vreduce(mysum<double>(), X1_cm),  sv );
	BCS_CHECK_EQUAL( vreduce(mydot<double>(), X1_cm, X2_cm),  dv );

	// 2D per_row

	double s_pr_rm[m] = {11, 17};
	double s_pr_cm[m] = {15, 13};
	double d_pr_rm[m] = {46, 82};
	double d_pr_cm[m] = {42, 86};

	BCS_CHECK( array_view_equal( vreduce(mysum<double>(), per_row(), X1_rm), s_pr_rm,  m) );
	BCS_CHECK( array_view_equal( vreduce(mysum<double>(), per_row(), X1_cm), s_pr_cm,  m) );
	BCS_CHECK( array_view_equal( vreduce(mydot<double>(), per_row(), X1_rm, X2_rm), d_pr_rm,  m) );
	BCS_CHECK( array_view_equal( vreduce(mydot<double>(), per_row(), X1_cm, X2_cm), d_pr_cm,  m) );

	// 2D per_col

	double s_pc_rm[n] = {8, 12, 8};
	double s_pc_cm[n] = {5, 13, 10};
	double d_pc_rm[n] = {60, 20, 48};
	double d_pc_cm[n] = {16, 86, 26};

	BCS_CHECK( array_view_equal( vreduce(mysum<double>(), per_col(), X1_rm), s_pc_rm,  n) );
	BCS_CHECK( array_view_equal( vreduce(mysum<double>(), per_col(), X1_cm), s_pc_cm,  n) );
	BCS_CHECK( array_view_equal( vreduce(mydot<double>(), per_col(), X1_rm, X2_rm), d_pc_rm,  n) );
	BCS_CHECK( array_view_equal( vreduce(mydot<double>(), per_col(), X1_cm, X2_cm), d_pc_cm,  n) );
}

std::shared_ptr<test_suite> test_array_eval_suite()
{
	BCS_NEW_TEST_SUITE( suite, "test_array_eval" );

	BCS_ADD_TEST_CASE( suite, test_array_map() );
	BCS_ADD_TEST_CASE( suite, test_array_vreduce() );

	return suite;
}




