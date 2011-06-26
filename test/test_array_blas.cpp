/**
 * @file test_array_blas.cpp
 *
 * Unit Testing of BLAS routines on arrays
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/test/test_array_aux.h>

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>
#include <bcslib/array/generic_blas.h>

using namespace bcs;
using namespace bcs::test;

BCS_TEST_CASE( test_blas_level1 )
{
	// prepare data

	const size_t N = 6;
	double x1d_src[N] = {3, -4, 5, -6, 8, 2};
	double x2d_src[N] = {1, 2, 3, 4, 5, 6};
	float  x1f_src[N] = {3, -4, 5, -6, 8, 2};
	float  x2f_src[N] = {1, 2, 3, 4, 5, 6};
	const size_t m = 2;
	const size_t n = 3;

	caview1d<double> x1d = get_aview1d(x1d_src, N);
	caview1d<double> x2d = get_aview1d(x2d_src, N);
	caview1d<float>  x1f = get_aview1d(x1f_src, N);
	caview1d<float>  x2f = get_aview1d(x2f_src, N);

	caview2d<double, row_major_t> X1d_rm = get_aview2d_rm(x1d_src, m, n);
	caview2d<double, row_major_t> X2d_rm = get_aview2d_rm(x2d_src, m, n);
	caview2d<float,  row_major_t> X1f_rm = get_aview2d_rm(x1f_src, m, n);
	caview2d<float,  row_major_t> X2f_rm = get_aview2d_rm(x2f_src, m, n);

	caview2d<double, column_major_t> X1d_cm = get_aview2d_cm(x1d_src, m, n);
	caview2d<double, column_major_t> X2d_cm = get_aview2d_cm(x2d_src, m, n);
	caview2d<float,  column_major_t> X1f_cm = get_aview2d_cm(x1f_src, m, n);
	caview2d<float,  column_major_t> X2f_cm = get_aview2d_cm(x2f_src, m, n);

	array1d<double> y1d(N);
	array1d<float>  y1f(N);
	array2d<double, row_major_t> Y1d_rm(m, n);
	array2d<float,  row_major_t> Y1f_rm(m, n);
	array2d<double, column_major_t> Y1d_cm(m, n);
	array2d<float,  column_major_t> Y1f_cm(m, n);

	array1d<double> y2d(N);
	array1d<float>  y2f(N);
	array2d<double, row_major_t> Y2d_rm(m, n);
	array2d<float,  row_major_t> Y2f_rm(m, n);
	array2d<double, column_major_t> Y2d_cm(m, n);
	array2d<float,  column_major_t> Y2f_cm(m, n);

	double td = 1e-14;
	float  tf = 1e-6f;

	// asum

	double asum_rd = 28.0;
	float  asum_rf = 28.0f;

	BCS_CHECK_APPROX_( blas::asum(x1d),    asum_rd, td );
	BCS_CHECK_APPROX_( blas::asum(x1f),    asum_rf, tf );
	BCS_CHECK_APPROX_( blas::asum(X1d_rm), asum_rd, td );
	BCS_CHECK_APPROX_( blas::asum(X1f_rm), asum_rf, tf );
	BCS_CHECK_APPROX_( blas::asum(X1d_cm), asum_rd, td );
	BCS_CHECK_APPROX_( blas::asum(X1f_cm), asum_rf, tf );

	// axpy

	double axpy_rd[N] = {7, -6, 13, -8, 21, 10};
	float  axpy_rf[N] = {7, -6, 13, -8, 21, 10};

	y1d << x2d; blas::axpy(2.0, x1d, y1d);
	BCS_CHECK( array_view_equal(y1d, axpy_rd, N) );

	y1f << x2f; blas::axpy(2.0f, x1f, y1f);
	BCS_CHECK( array_view_equal(y1f, axpy_rf, N) );

	Y1d_rm << X2d_rm; blas::axpy(2.0, X1d_rm, Y1d_rm);
	BCS_CHECK( array_view_equal(Y1d_rm, axpy_rd, m, n) );

	Y1f_rm << X2f_rm; blas::axpy(2.0, X1f_rm, Y1f_rm);
	BCS_CHECK( array_view_equal(Y1f_rm, axpy_rf, m, n) );

	Y1d_cm << X2d_cm; blas::axpy(2.0, X1d_cm, Y1d_cm);
	BCS_CHECK( array_view_equal(Y1d_cm, axpy_rd, m, n) );

	Y1f_cm << X2f_cm; blas::axpy(2.0, X1f_cm, Y1f_cm);
	BCS_CHECK( array_view_equal(Y1f_cm, axpy_rf, m, n) );

	// dot

	double dot_rd = 38.0;
	float  dot_rf = 38.0f;

	BCS_CHECK_APPROX_( blas::dot(x1d, x2d), dot_rd, td );
	BCS_CHECK_APPROX_( blas::dot(x1f, x2f), dot_rf, tf );
	BCS_CHECK_APPROX_( blas::dot(X1d_rm, X2d_rm), dot_rd, td );
	BCS_CHECK_APPROX_( blas::dot(X1f_rm, X2f_rm), dot_rf, tf );
	BCS_CHECK_APPROX_( blas::dot(X1d_cm, X2d_cm), dot_rd, td );
	BCS_CHECK_APPROX_( blas::dot(X1f_cm, X2f_cm), dot_rf, tf );

	// rot

	double cd = 2, sd = 3;
	float  cf = 2, sf = 3;

	double rot_ad[N] = {9, -2, 19, 0, 31, 22};
	float  rot_af[N] = {9, -2, 19, 0, 31, 22};
	double rot_bd[N] = {-7, 16, -9, 26, -14, 6};
	float  rot_bf[N] = {-7, 16, -9, 26, -14, 6};

	y1d << x1d; y2d << x2d; blas::rot(y1d, y2d, cd, sd);
	BCS_CHECK( array_view_equal(y1d, rot_ad, N) );
	BCS_CHECK( array_view_equal(y2d, rot_bd, N) );

	y1f << x1f; y2f << x2f; blas::rot(y1f, y2f, cf, sf);
	BCS_CHECK( array_view_equal(y1f, rot_af, N) );
	BCS_CHECK( array_view_equal(y2f, rot_bf, N) );

	Y1d_rm << X1d_rm; Y2d_rm << X2d_rm; blas::rot(Y1d_rm, Y2d_rm, cd, sd);
	BCS_CHECK( array_view_equal(Y1d_rm, rot_ad, m, n) );
	BCS_CHECK( array_view_equal(Y2d_rm, rot_bd, m, n) );

	Y1f_rm << X1f_rm; Y2f_rm << X2f_rm; blas::rot(Y1f_rm, Y2f_rm, cf, sf);
	BCS_CHECK( array_view_equal(Y1f_rm, rot_af, m, n) );
	BCS_CHECK( array_view_equal(Y2f_rm, rot_bf, m, n) );

	Y1d_cm << X1d_cm; Y2d_cm << X2d_cm; blas::rot(Y1d_cm, Y2d_cm, cd, sd);
	BCS_CHECK( array_view_equal(Y1d_cm, rot_ad, m, n) );
	BCS_CHECK( array_view_equal(Y2d_cm, rot_bd, m, n) );

	Y1f_cm << X1f_cm; Y2f_cm << X2f_cm; blas::rot(Y1f_cm, Y2f_cm, cf, sf);
	BCS_CHECK( array_view_equal(Y1f_cm, rot_af, m, n) );
	BCS_CHECK( array_view_equal(Y2f_cm, rot_bf, m, n) );

}

test_suite* test_array_blas_suite()
{
	test_suite *tsuite = new test_suite( "test_array_blas" );

	tsuite->add( new test_blas_level1() );

	return tsuite;
}


