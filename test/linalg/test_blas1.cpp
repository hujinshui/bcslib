/**
 * @file test_blas1.cpp
 *
 * Unit testing of BLAS level 1 functions on matrix
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/linalg.h>

using namespace bcs;

template<typename T, class MatX>
T my_asum(const IDenseMatrix<MatX, T>& X)
{
	T s = 0;
	const T *x = X.ptr_data();
	for (index_t i = 0; i < X.nelems(); ++i) s += math::abs(x[i]);
	return s;
}


TEST( MatrixBlasL1, Asum )
{
	const index_t n = 8;

	dense_col<double> xd(n);
	for (index_t i = 0; i < n; ++i) xd[i] = double(i + 1);

	ASSERT_NEAR( my_asum(xd), blas::asum(xd), 1.0e-12 );

	dense_col<float> xf(n);
	for (index_t i = 0; i < n; ++i) xf[i] = float(i + 1);

	ASSERT_NEAR( my_asum(xf), blas::asum(xf), 1.0e-6f );
}


template<typename T, class MatX, class MatY>
void my_axpy(const T a, const IDenseMatrix<MatX, T>& X, IDenseMatrix<MatY, T>& Y)
{
	const T *x = X.ptr_data();
	T *y = Y.ptr_data();

	for (index_t i = 0; i < X.nelems(); ++i) y[i] += a * x[i];
}

TEST( MatrixBlasL1, Axpy )
{
	const index_t n = 8;

	double ad = 3.2;

	dense_col<double> xd(n);
	for (index_t i = 0; i < n; ++i) xd[i] = double(i + 1);

	dense_col<double> yd0(n, 2.5);
	dense_col<double> yd(n, 2.5);

	my_axpy(ad, xd, yd0);
	blas::axpy(ad, xd, yd);

	ASSERT_TRUE( is_approx(yd, yd0, 1.0e-12) );

	float af = 3.2f;

	dense_col<float> xf(n);
	for (index_t i = 0; i < n; ++i) xf[i] = float(i + 1);

	dense_col<float> yf0(n, 2.5);
	dense_col<float> yf(n, 2.5);

	my_axpy(af, xf, yf0);
	blas::axpy(af, xf, yf);

	ASSERT_TRUE( is_approx(yf, yf0, 1.0e-6f) );

}


template<typename T, class MatX, class MatY>
T my_dot(const IDenseMatrix<MatX, T>& X, const IDenseMatrix<MatY, T>& Y)
{
	T s = 0;
	const T *x = X.ptr_data();
	const T *y = Y.ptr_data();
	for (index_t i = 0; i < X.nelems(); ++i) s += x[i] * y[i];
	return s;
}


TEST( MatrixBlasL1, Dot )
{
	const index_t n = 8;

	dense_col<double> xd(n);
	dense_col<double> yd(n);
	for (index_t i = 0; i < n; ++i) xd[i] = double(i + 1);
	for (index_t i = 0; i < n; ++i) yd[i] = double(i + 2);

	ASSERT_NEAR( my_dot(xd, yd), blas::dot(xd, yd), 1.0e-12 );

	dense_col<float> xf(n);
	dense_col<float> yf(n);
	for (index_t i = 0; i < n; ++i) xf[i] = float(i + 1);
	for (index_t i = 0; i < n; ++i) yf[i] = float(i + 2);

	ASSERT_NEAR( my_dot(xf, yf), blas::dot(xf, yf), 1.0e-6f );
}

template<typename T, class MatX>
T my_nrm2(const IDenseMatrix<MatX, T>& X)
{
	T s = 0;
	const T *x = X.ptr_data();
	for (index_t i = 0; i < X.nelems(); ++i) s += math::sqr(x[i]);
	return math::sqrt(s);
}


TEST( MatrixBlasL1, Nrm2 )
{
	const index_t n = 8;

	dense_col<double> xd(n);
	for (index_t i = 0; i < n; ++i) xd[i] = double(i + 1);

	ASSERT_NEAR( my_nrm2(xd), blas::nrm2(xd), 1.0e-12 );

	dense_col<float> xf(n);
	for (index_t i = 0; i < n; ++i) xf[i] = float(i + 1);

	ASSERT_NEAR( my_nrm2(xf), blas::nrm2(xf), 1.0e-6f );
}



template<typename T, class MatX, class MatY>
void my_rot(IDenseMatrix<MatX, T>& X, IDenseMatrix<MatY, T>& Y, const T c, const T s)
{
	T *x = X.ptr_data();
	T *y = Y.ptr_data();

	for (index_t i = 0; i < X.nelems(); ++i)
	{
		T xi = x[i];
		T yi = y[i];

		x[i] = c * xi + s * yi;
		y[i] = c * yi - s * xi;
	}
}

TEST( MatrixBlasL1, Rot )
{
	const index_t n = 8;

	// double

	double cd = 2.5;
	double sd = 3.2;

	dense_col<double> xd0(n);
	dense_col<double> yd0(n);
	dense_col<double> xd(n);
	dense_col<double> yd(n);

	for (index_t i = 0; i < n; ++i)
	{
		xd0[i] = double(i + 1);
		xd[i]  = double(i + 1);
		yd0[i] = double(i + 2);
		yd[i]  = double(i + 2);
	}

	my_rot(xd0, yd0, cd, sd);
	blas::rot(xd, yd, cd, sd);

	ASSERT_TRUE( is_approx(xd, xd0, 1.0e-12) );
	ASSERT_TRUE( is_approx(yd, yd0, 1.0e-12) );

	// float

	float cf = 2.5f;
	float sf = 3.2f;

	dense_col<float> xf0(n);
	dense_col<float> yf0(n);
	dense_col<float> xf(n);
	dense_col<float> yf(n);

	for (index_t i = 0; i < n; ++i)
	{
		xf0[i] = float(i + 1);
		xf[i]  = float(i + 1);
		yf0[i] = float(i + 2);
		yf[i]  = float(i + 2);
	}

	my_rot(xf0, yf0, cf, sf);
	blas::rot(xf, yf, cf, sf);

	ASSERT_TRUE( is_approx(xf, xf0, 1.0e-6f) );
	ASSERT_TRUE( is_approx(yf, yf0, 1.0e-6f) );

}







