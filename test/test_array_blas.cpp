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

	double ad = 2.0;
	float  af = 2.0f;

	y1d << x2d; blas::axpy(x1d, y1d, ad);
	BCS_CHECK( array_view_equal(y1d, axpy_rd, N) );

	y1f << x2f; blas::axpy(x1f, y1f, af);
	BCS_CHECK( array_view_equal(y1f, axpy_rf, N) );

	Y1d_rm << X2d_rm; blas::axpy(X1d_rm, Y1d_rm, ad);
	BCS_CHECK( array_view_equal(Y1d_rm, axpy_rd, m, n) );

	Y1f_rm << X2f_rm; blas::axpy(X1f_rm, Y1f_rm, af);
	BCS_CHECK( array_view_equal(Y1f_rm, axpy_rf, m, n) );

	Y1d_cm << X2d_cm; blas::axpy(X1d_cm, Y1d_cm, ad);
	BCS_CHECK( array_view_equal(Y1d_cm, axpy_rd, m, n) );

	Y1f_cm << X2f_cm; blas::axpy(X1f_cm, Y1f_cm, af);
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


// Auxiliary functions for BLAS Level 2

template<typename T, typename TOrd>
array2d<T, TOrd> _make_symmetric(const caview2d<T, TOrd>& A)
{
	check_arg(A.dim0() == A.dim1());
	index_t n = A.dim0();

	array2d<T, TOrd> S((size_t)n, (size_t)n);
	for (index_t i = 0; i < n; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			S(i, j) = (A(i, j) + A(j, i)) / T(2);
		}
	}

	return S;
}

template<typename T, typename TOrd>
array1d<T> _compute_mv(const caview2d<T, TOrd>& A, const caview1d<T>& x, const caview1d<T>& y0, T alpha, T beta, char trans)
{
	index_t m = A.dim0();
	index_t n = A.dim1();
	index_t nr = (trans == 'N' || trans == 'n') ? m : n;

	array1d<T> r((size_t)nr);
	if (trans == 'N' || trans == 'n')
	{
		for (index_t i = 0; i < m; ++i)
		{
			T s = 0;
			for (index_t j = 0; j < n; ++j) s += A(i, j) * x(j);
			r(i) = alpha * s;
			if (beta != 0) r(i) += beta * y0(i);
		}
	}
	else
	{
		for (index_t j = 0; j < n; ++j)
		{
			T s = 0;
			for (index_t i = 0; i < m; ++i) s += A(i, j) * x(i);
			r(j) = alpha * s;
			if (beta != 0) r(j) += beta * y0(j);
		}
	}

	return r;
}

template<typename T, typename TOrd>
array2d<T, TOrd> _compute_rank1_update(const caview2d<T, TOrd>& A, const caview1d<T>& x, const caview1d<T>& y, T alpha)
{
	index_t m = A.dim0();
	index_t n = A.dim1();

	array2d<T, TOrd> U((size_t)m, (size_t)n);
	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			U(i, j) = A(i, j) + alpha * (x(i) * y(j));
		}
	}

	return U;
}


template<typename T, typename TOrd>
bool verify_gemv(const caview2d<T, TOrd>& A, const caview1d<T>& x, const caview1d<T>& y0, T alpha, T beta, char trans)
{
	// compute ground-truth in a slow but safe way

	array1d<T> r = _compute_mv(A, x, y0, alpha, beta, trans);

	// use gemv to compute

	array1d<T> y(y0.nelems(), y0.pbase());
	blas::gemv(A, x, y, alpha, beta, trans);

	// test

	return array_view_approx(y, r);
}


template<typename T, typename TOrd>
bool verify_ger(const caview2d<T, TOrd>& A0, const caview1d<T>& x, const caview1d<T>& y, T alpha)
{
	// compute ground-truth

	array2d<T, TOrd> R = _compute_rank1_update(A0, x, y, alpha);

	// use ger to compute

	array2d<T, TOrd> A = clone_array(A0);
	blas::ger(A, x, y, alpha);

	// test

	return array_view_approx(A, R);
}


template<typename T, typename TOrd>
bool verify_symv(const caview2d<T, TOrd>& A, const caview1d<T>& x, const caview1d<T>& y0, T alpha, T beta)
{
	// compute ground-truth in a slow but safe way

	array2d<T, TOrd> As = _make_symmetric(A);
	array1d<T> r = _compute_mv(As, x, y0, alpha, beta, 'N');

	// use symv to compute

	array1d<T> y(y0.nelems(), y0.pbase());
	blas::symv(As, x, y, alpha, beta);

	// test

	return array_view_approx(y, r);
}


BCS_TEST_CASE( test_blas_level2 )
{
	// prepare data

	const size_t N = 12;
	const size_t Nv = 4;
	const size_t m1 = 3;
	const size_t n1 = 4;
	const size_t m2 = 4;
	const size_t n2 = 3;

	double asrc_d[N] = {12, 2, 5, 8, 7, 1, 3, 4, 6, 5, 3, 10};
	double xsrc_d[Nv] = {2, 4, 3, 5};
	double ysrc_d[Nv] = {1, 7, 4, 2};

	float asrc_f[N] = {12, 2, 5, 8, 7, 1, 3, 4, 6, 5, 3, 10};
	float xsrc_f[Nv] = {2, 4, 3, 5};
	float ysrc_f[Nv] = {1, 7, 4, 2};

	caview2d<double, row_major_t> A1_d_rm = get_aview2d_rm(asrc_d, m1, n1);
	caview2d<float,  row_major_t> A1_f_rm = get_aview2d_rm(asrc_f, m1, n1);
	caview2d<double, row_major_t> A2_d_rm = get_aview2d_rm(asrc_d, m2, n2);
	caview2d<float,  row_major_t> A2_f_rm = get_aview2d_rm(asrc_f, m2, n2);
	caview2d<double, row_major_t> As_d_rm = get_aview2d_rm(asrc_d, m1, m1);
	caview2d<float,  row_major_t> As_f_rm = get_aview2d_rm(asrc_f, m1, m1);

	caview2d<double, column_major_t> A1_d_cm = get_aview2d_cm(asrc_d, m1, n1);
	caview2d<float,  column_major_t> A1_f_cm = get_aview2d_cm(asrc_f, m1, n1);
	caview2d<double, column_major_t> A2_d_cm = get_aview2d_cm(asrc_d, m2, n2);
	caview2d<float,  column_major_t> A2_f_cm = get_aview2d_cm(asrc_f, m2, n2);
	caview2d<double, column_major_t> As_d_cm = get_aview2d_cm(asrc_d, m1, m1);
	caview2d<float,  column_major_t> As_f_cm = get_aview2d_cm(asrc_f, m1, m1);

	caview1d<double> x1_d = get_aview1d(xsrc_d, n1);
	caview1d<float>  x1_f = get_aview1d(xsrc_f, n1);
	caview1d<double> x2_d = get_aview1d(xsrc_d, n2);
	caview1d<float>  x2_f = get_aview1d(xsrc_f, n2);

	caview1d<double> y1_d = get_aview1d(ysrc_d, m1);
	caview1d<float>  y1_f = get_aview1d(ysrc_f, m1);
	caview1d<double> y2_d = get_aview1d(ysrc_d, m2);
	caview1d<float>  y2_f = get_aview1d(ysrc_f, m2);

	double alpha_d, beta_d;
	float  alpha_f, beta_f;

	// gemv

	alpha_d = 1; beta_d = 0;
	alpha_f = 1; beta_f = 0;

	BCS_CHECK( verify_gemv(A1_d_cm, x1_d, y1_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A1_d_cm, x2_d, y2_d, alpha_d, beta_d, 'T') );
	BCS_CHECK( verify_gemv(A2_d_cm, x2_d, y2_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A2_d_cm, x1_d, y1_d, alpha_d, beta_d, 'T') );

	BCS_CHECK( verify_gemv(A1_d_rm, x1_d, y1_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A1_d_rm, x2_d, y2_d, alpha_d, beta_d, 'T') );
	BCS_CHECK( verify_gemv(A2_d_rm, x2_d, y2_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A2_d_rm, x1_d, y1_d, alpha_d, beta_d, 'T') );

	BCS_CHECK( verify_gemv(A1_f_cm, x1_f, y1_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A1_f_cm, x2_f, y2_f, alpha_f, beta_f, 'T') );
	BCS_CHECK( verify_gemv(A2_f_cm, x2_f, y2_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A2_f_cm, x1_f, y1_f, alpha_f, beta_f, 'T') );

	BCS_CHECK( verify_gemv(A1_f_rm, x1_f, y1_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A1_f_rm, x2_f, y2_f, alpha_f, beta_f, 'T') );
	BCS_CHECK( verify_gemv(A2_f_rm, x2_f, y2_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A2_f_rm, x1_f, y1_f, alpha_f, beta_f, 'T') );

	alpha_d = 2; beta_d = 0;
	alpha_f = 2; beta_f = 0;

	BCS_CHECK( verify_gemv(A1_d_cm, x1_d, y1_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A1_d_cm, x2_d, y2_d, alpha_d, beta_d, 'T') );
	BCS_CHECK( verify_gemv(A2_d_cm, x2_d, y2_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A2_d_cm, x1_d, y1_d, alpha_d, beta_d, 'T') );

	BCS_CHECK( verify_gemv(A1_d_rm, x1_d, y1_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A1_d_rm, x2_d, y2_d, alpha_d, beta_d, 'T') );
	BCS_CHECK( verify_gemv(A2_d_rm, x2_d, y2_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A2_d_rm, x1_d, y1_d, alpha_d, beta_d, 'T') );

	BCS_CHECK( verify_gemv(A1_f_cm, x1_f, y1_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A1_f_cm, x2_f, y2_f, alpha_f, beta_f, 'T') );
	BCS_CHECK( verify_gemv(A2_f_cm, x2_f, y2_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A2_f_cm, x1_f, y1_f, alpha_f, beta_f, 'T') );

	BCS_CHECK( verify_gemv(A1_f_rm, x1_f, y1_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A1_f_rm, x2_f, y2_f, alpha_f, beta_f, 'T') );
	BCS_CHECK( verify_gemv(A2_f_rm, x2_f, y2_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A2_f_rm, x1_f, y1_f, alpha_f, beta_f, 'T') );

	alpha_d = 2; beta_d = 0.5;
	alpha_f = 2; beta_f = 0.5;

	BCS_CHECK( verify_gemv(A1_d_cm, x1_d, y1_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A1_d_cm, x2_d, y2_d, alpha_d, beta_d, 'T') );
	BCS_CHECK( verify_gemv(A2_d_cm, x2_d, y2_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A2_d_cm, x1_d, y1_d, alpha_d, beta_d, 'T') );

	BCS_CHECK( verify_gemv(A1_d_rm, x1_d, y1_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A1_d_rm, x2_d, y2_d, alpha_d, beta_d, 'T') );
	BCS_CHECK( verify_gemv(A2_d_rm, x2_d, y2_d, alpha_d, beta_d, 'N') );
	BCS_CHECK( verify_gemv(A2_d_rm, x1_d, y1_d, alpha_d, beta_d, 'T') );

	BCS_CHECK( verify_gemv(A1_f_cm, x1_f, y1_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A1_f_cm, x2_f, y2_f, alpha_f, beta_f, 'T') );
	BCS_CHECK( verify_gemv(A2_f_cm, x2_f, y2_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A2_f_cm, x1_f, y1_f, alpha_f, beta_f, 'T') );

	BCS_CHECK( verify_gemv(A1_f_rm, x1_f, y1_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A1_f_rm, x2_f, y2_f, alpha_f, beta_f, 'T') );
	BCS_CHECK( verify_gemv(A2_f_rm, x2_f, y2_f, alpha_f, beta_f, 'N') );
	BCS_CHECK( verify_gemv(A2_f_rm, x1_f, y1_f, alpha_f, beta_f, 'T') );

	// ger

	alpha_d = 1;
	alpha_f = 1;

	BCS_CHECK( verify_ger(A1_d_cm, y1_d, x1_d, alpha_d) );
	BCS_CHECK( verify_ger(A2_d_cm, y2_d, x2_d, alpha_d) );

	BCS_CHECK( verify_ger(A1_d_rm, y1_d, x1_d, alpha_d) );
	BCS_CHECK( verify_ger(A2_d_rm, y2_d, x2_d, alpha_d) );

	BCS_CHECK( verify_ger(A1_f_cm, y1_f, x1_f, alpha_f) );
	BCS_CHECK( verify_ger(A2_f_cm, y2_f, x2_f, alpha_f) );

	BCS_CHECK( verify_ger(A1_f_rm, y1_f, x1_f, alpha_f) );
	BCS_CHECK( verify_ger(A2_f_rm, y2_f, x2_f, alpha_f) );

	alpha_d = 2;
	alpha_f = 2;

	BCS_CHECK( verify_ger(A1_d_cm, y1_d, x1_d, alpha_d) );
	BCS_CHECK( verify_ger(A2_d_cm, y2_d, x2_d, alpha_d) );

	BCS_CHECK( verify_ger(A1_d_rm, y1_d, x1_d, alpha_d) );
	BCS_CHECK( verify_ger(A2_d_rm, y2_d, x2_d, alpha_d) );

	BCS_CHECK( verify_ger(A1_f_cm, y1_f, x1_f, alpha_f) );
	BCS_CHECK( verify_ger(A2_f_cm, y2_f, x2_f, alpha_f) );

	BCS_CHECK( verify_ger(A1_f_rm, y1_f, x1_f, alpha_f) );
	BCS_CHECK( verify_ger(A2_f_rm, y2_f, x2_f, alpha_f) );

	// symv

	alpha_d = 1; beta_d = 0;
	alpha_f = 1; beta_f = 0;

	BCS_CHECK( verify_symv(As_d_cm, x2_d, y1_d, alpha_d, beta_d) );
	BCS_CHECK( verify_symv(As_d_rm, x2_d, y1_d, alpha_d, beta_d) );

	BCS_CHECK( verify_symv(As_f_cm, x2_f, y1_f, alpha_f, beta_f) );
	BCS_CHECK( verify_symv(As_f_rm, x2_f, y1_f, alpha_f, beta_f) );

	alpha_d = 2; beta_d = 0;
	alpha_f = 2; beta_f = 0;

	BCS_CHECK( verify_symv(As_d_cm, x2_d, y1_d, alpha_d, beta_d) );
	BCS_CHECK( verify_symv(As_d_rm, x2_d, y1_d, alpha_d, beta_d) );

	BCS_CHECK( verify_symv(As_f_cm, x2_f, y1_f, alpha_f, beta_f) );
	BCS_CHECK( verify_symv(As_f_rm, x2_f, y1_f, alpha_f, beta_f) );

	alpha_d = 2; beta_d = 0.5;
	alpha_f = 2; beta_f = 0.5;

	BCS_CHECK( verify_symv(As_d_cm, x2_d, y1_d, alpha_d, beta_d) );
	BCS_CHECK( verify_symv(As_d_rm, x2_d, y1_d, alpha_d, beta_d) );

	BCS_CHECK( verify_symv(As_f_cm, x2_f, y1_f, alpha_f, beta_f) );
	BCS_CHECK( verify_symv(As_f_rm, x2_f, y1_f, alpha_f, beta_f) );

}


// Auxiliary functions for BLAS level 3

template<typename T, typename TOrd>
array2d<T, TOrd> _compute_mm(
		const caview2d<T, TOrd>& A,
		const caview2d<T, TOrd>& B,
		const caview2d<T, TOrd>& C, T alpha, T beta, char transa, char transb)
{

	std::unique_ptr<array2d<T, TOrd> > pOpA( transa == 'N' || transa == 'n' ?
			new array2d<T, TOrd>(clone_array(A)) : new array2d<T, TOrd>(transpose(A)) );

	std::unique_ptr<array2d<T, TOrd> > pOpB( transb == 'N' || transb == 'n' ?
			new array2d<T, TOrd>(clone_array(B)) : new array2d<T, TOrd>(transpose(B)) );

	const array2d<T, TOrd>& OpA = *pOpA;
	const array2d<T, TOrd>& OpB = *pOpB;

	check_arg(OpA.dim1() == OpB.dim0() && OpA.dim0() == C.dim0() && OpB.dim1() == C.dim1(),
			"_compute_mm: mismatched dimensions");

	index_t m = OpA.dim0();
	index_t k = OpA.dim1();
	index_t n = OpB.dim1();

	array2d<T, TOrd> R((size_t)m, (size_t)n);

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			T cv = 0;
			for (index_t l = 0; l < k; ++l)
			{
				cv += OpA(i, l) * OpB(l, j);
			}
			R(i, j) = alpha * cv + beta * C(i, j);
		}
	}

	return R;
}


template<typename T, typename TOrd>
bool verify_gemm(
		const caview2d<T, TOrd>& A,
		const caview2d<T, TOrd>& B,
		const caview2d<T, TOrd>& C0, T alpha, T beta, char transa, char transb)
{
	// compute ground-truth

	array2d<T, TOrd> R = _compute_mm(A, B, C0, alpha, beta, transa, transb);

	// use gemm to compute

	array2d<T, TOrd> C = clone_array(C0);
	blas::gemm(A, B, C, transa, transb, alpha, beta);

	// test
	return array_view_approx(C, R);
}


template<typename T, typename TOrd>
bool verify_gemm_(TOrd, const T *src_a, const T *src_b, const T *src_c,
		size_t m, size_t n, size_t k, T alpha, T beta, char transa, char transb)
{
	size_t ma, na, mb, nb;

	if (transa == 'N' || transa == 'n') { ma = m; na = k; } else { ma = k; na = m; }
	if (transb == 'N' || transb == 'n') { mb = k; nb = n; } else { mb = n; nb = k; }

	caview2d<T, TOrd> A(src_a, (index_t)ma, (index_t)na, ma, na);
	caview2d<T, TOrd> B(src_b, (index_t)mb, (index_t)nb, mb, nb);
	caview2d<T, TOrd> C(src_c, (index_t)m, (index_t)n, m, n);

	return verify_gemm(A, B, C, alpha, beta, transa, transb);
}



BCS_TEST_CASE( test_blas_level3 )
{
	// gemm

	const size_t N = 20;
	double src_a_d[N] = {7, 4, 9, 3, 9, 3, 5, 8, 10, 1, 12, 10, 6, 5, 6, 4, 7, 7, 10, 9};
	double src_b_d[N] = {8, 5, 10, 7, 5, 12, 11, 7, 8, 8, 3, 4, 6, 3, 11, 3, 3, 2, 3, 6};
	double src_c_d[N] = {2, 4, 10, 1, 12, 9, 6, 7, 3, 6, 12, 7, 7, 3, 6, 8, 9, 5, 5, 12};

	float src_a_f[N] = {7, 4, 9, 3, 9, 3, 5, 8, 10, 1, 12, 10, 6, 5, 6, 4, 7, 7, 10, 9};
	float src_b_f[N] = {8, 5, 10, 7, 5, 12, 11, 7, 8, 8, 3, 4, 6, 3, 11, 3, 3, 2, 3, 6};
	float src_c_f[N] = {2, 4, 10, 1, 12, 9, 6, 7, 3, 6, 12, 7, 7, 3, 6, 8, 9, 5, 5, 12};

	double alpha_d, beta_d;
	float  alpha_f, beta_f;

	// alpha = 1, beta = 0

	alpha_d = 1; beta_d = 0;
	alpha_f = 1; beta_f = 0;

	// cm double

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'T', 'T') );

	// cm float

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'T', 'T') );

	// rm double

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'T', 'T') );

	// rm float

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'T', 'T') );

	// alpha = 2, beta = 0.5

	alpha_d = 2.0;  beta_d = 0.5;
	alpha_f = 2.0f; beta_f = 0.5f;

	// cm double

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'T', 'T') );

	// cm float

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(column_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'T', 'T') );

	// rm double

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 4, 5, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 3, 5, 4, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 3, 5, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 4, 5, 3, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 3, 4, alpha_d, beta_d, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_d, src_b_d, src_c_d, 5, 4, 3, alpha_d, beta_d, 'T', 'T') );

	// rm float

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 4, 5, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 3, 5, 4, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 3, 5, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 4, 5, 3, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 3, 4, alpha_f, beta_f, 'T', 'T') );

	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'N', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'N', 'T') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'T', 'N') );
	BCS_CHECK( verify_gemm_(row_major_t(), src_a_f, src_b_f, src_c_f, 5, 4, 3, alpha_f, beta_f, 'T', 'T') );

}


test_suite* test_array_blas_suite()
{
	test_suite *tsuite = new test_suite( "test_array_blas" );

	tsuite->add( new test_blas_level1() );
	tsuite->add( new test_blas_level2() );
	tsuite->add( new test_blas_level3() );

	return tsuite;
}


