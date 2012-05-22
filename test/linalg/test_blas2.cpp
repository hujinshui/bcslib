/*
 * @file test_gemv.cpp
 *
 * Unit testing for BLAS Level 2
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/linalg.h>

#include "test_blas_aux.h"

using namespace bcs;


/************************************************
 *
 *  GEMV
 *
 ************************************************/

template<typename T>
void test_gemv_n()
{
	const index_t m = 5;
	const index_t n = 6;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(m, n);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_col<T> x(n);
	for (index_t i = 0; i < x.nelems(); ++i) x[i] = T(2 * i - 5);

	dense_col<T> y0(m);
	for (index_t i = 0; i < y0.nelems(); ++i) y0[i] = T(i+2);

	dense_col<T> y(y0);
	ASSERT_TRUE( y0.ptr_data() != y.ptr_data() );

	my_mv(alpha, a, x, beta, y0);
	blas::gemv_n(alpha, a, x, beta, y);

	ASSERT_TRUE( is_equal(y, y0) );
}

template<typename T>
void test_gemv_t()
{
	const index_t m = 5;
	const index_t n = 6;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(m, n);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_col<T> x(m);
	for (index_t i = 0; i < x.nelems(); ++i) x[i] = T(2 * i - 5);

	dense_col<T> y0(n);
	for (index_t i = 0; i < y0.nelems(); ++i) y0[i] = T(i+2);

	dense_col<T> y(y0);
	ASSERT_TRUE( y0.ptr_data() != y.ptr_data() );

	dense_matrix<T> at = a.trans();
	my_mv(alpha, at, x, beta, y0);
	blas::gemv_t(alpha, a, x, beta, y);

	ASSERT_TRUE( is_equal(y, y0) );
}


/************************************************
 *
 *  GEVM
 *
 ************************************************/


template<typename T>
void test_gevm_n()
{
	const index_t m = 5;
	const index_t n = 6;
	const index_t ld = 3;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(m, n);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_matrix<T> x_bk(ld, m);
	for (index_t i = 0; i < x_bk.nelems(); ++i) x_bk[i] = T(2 * i - 5);

	ref_matrix_ex<T, 1, DynamicDim> x( x_bk.row(0) );

	dense_matrix<T> y0_bk(ld, n);
	for (index_t i = 0; i < y0_bk.nelems(); ++i) y0_bk[i] = T(i+2);

	dense_matrix<T> y_bk(y0_bk);

	ref_matrix_ex<T, 1, DynamicDim> y0( y0_bk.row(0) );
	ref_matrix_ex<T, 1, DynamicDim> y( y_bk.row(0) );

	ASSERT_TRUE( y0.ptr_data() != y.ptr_data() );

	my_vm(alpha, x, a, beta, y0);
	blas::gevm_n(alpha, x, a, beta, y);

	ASSERT_TRUE( is_equal(y, y0) );
}

template<typename T>
void test_gevm_t()
{
	const index_t m = 5;
	const index_t n = 6;
	const index_t ld = 3;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(m, n);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_matrix<T> x_bk(ld, n);
	for (index_t i = 0; i < x_bk.nelems(); ++i) x_bk[i] = T(2 * i - 5);

	ref_matrix_ex<T, 1, DynamicDim> x( x_bk.row(0) );

	dense_matrix<T> y0_bk(ld, m);
	for (index_t i = 0; i < y0_bk.nelems(); ++i) y0_bk[i] = T(i+2);

	dense_matrix<T> y_bk(y0_bk);

	ref_matrix_ex<T, 1, DynamicDim> y0( y0_bk.row(0) );
	ref_matrix_ex<T, 1, DynamicDim> y( y_bk.row(0) );

	ASSERT_TRUE( y0.ptr_data() != y.ptr_data() );

	dense_matrix<T> at = a.trans();

	my_vm(alpha, x, at, beta, y0);
	blas::gevm_t(alpha, x, a, beta, y);

	ASSERT_TRUE( is_equal(y, y0) );
}


TEST( MatrixBlasL2, GemvN_DDd )
{
	test_gemv_n<double>();
}

TEST( MatrixBlasL2, GemvN_DDs )
{
	test_gemv_n<float>();
}

TEST( MatrixBlasL2, GemvT_DDd )
{
	test_gemv_t<double>();
}

TEST( MatrixBlasL2, GemvT_DDs )
{
	test_gemv_t<float>();
}

TEST( MatrixBlasL2, GevmN_DDd )
{
	test_gevm_n<double>();
}

TEST( MatrixBlasL2, GevmN_DDs )
{
	test_gevm_n<float>();
}

TEST( MatrixBlasL2, GevmT_DDd )
{
	test_gevm_t<double>();
}

TEST( MatrixBlasL2, GevmT_DDs )
{
	test_gevm_t<float>();
}


/************************************************
 *
 *  GER
 *
 ************************************************/

template<typename T>
void test_ger()
{
	const T alpha = T(0.5);

	const index_t m = 5;
	const index_t n = 6;

	dense_matrix<T> a0(m, n);
	for (index_t i = 0; i < a0.nelems(); ++i) a0[i] = T(i+1);
	dense_matrix<T> a(a0);

	ASSERT_TRUE( a0.ptr_data() != a.ptr_data() );

	dense_col<T> x(m);
	for (index_t i = 0; i < m; ++i) x[i] = T(2 * i + 1);

	dense_col<T> y(n);
	for (index_t i = 0; i < n; ++i) y[i] = T(3 * i - 5);


	for (index_t j = 0; j < n; ++j)
		for (index_t i = 0; i < m; ++i)
			a0(i, j) += alpha * (x[i] * y[j]);

	blas::ger(alpha, x, y, a);

	ASSERT_TRUE( is_equal(a, a0) );
}

TEST( MatrixBlasL2, Ger_DDd )
{
	test_ger<double>();
}

TEST( MatrixBlasL2, Ger_DDs )
{
	test_ger<float>();
}



/************************************************
 *
 *  SYMV
 *
 ************************************************/

template<typename T>
void test_symv()
{
	const index_t n = 6;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(n, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < n; ++i)
		{
			a(i, j) = T(i + j + 1);
		}
	}

	dense_col<T> x(n);
	for (index_t i = 0; i < x.nelems(); ++i) x[i] = T(2 * i - 5);

	dense_col<T> y0(n);
	for (index_t i = 0; i < y0.nelems(); ++i) y0[i] = T(i+2);

	dense_col<T> y(y0);
	ASSERT_TRUE( y0.ptr_data() != y.ptr_data() );

	my_mv(alpha, a, x, beta, y0);
	blas::symv(alpha, a, x, beta, y);

	ASSERT_TRUE( is_equal(y, y0) );
}


TEST( MatrixBlasL2, Symv_DDd )
{
	test_symv<double>();
}

TEST( MatrixBlasL2, Symv_DDs )
{
	test_symv<float>();
}


/************************************************
 *
 *  SYVM
 *
 ************************************************/

template<typename T>
void test_syvm()
{
	const index_t n = 6;
	const index_t ld = 3;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(n, n);
	for (index_t j = 0; j < n; ++j)
		for (index_t i = 0; i < n; ++i) a(i, j) = T(i + j + 1);

	dense_matrix<T> x_bk(ld, n);
	for (index_t i = 0; i < x_bk.nelems(); ++i) x_bk[i] = T(2 * i - 5);

	ref_matrix_ex<T, 1, DynamicDim> x( x_bk.row(0) );

	dense_matrix<T> y0_bk(ld, n);
	for (index_t i = 0; i < y0_bk.nelems(); ++i) y0_bk[i] = T(i+2);

	dense_matrix<T> y_bk(y0_bk);

	ref_matrix_ex<T, 1, DynamicDim> y0( y0_bk.row(0) );
	ref_matrix_ex<T, 1, DynamicDim> y( y_bk.row(0) );

	ASSERT_TRUE( y0.ptr_data() != y.ptr_data() );

	my_vm(alpha, x, a, beta, y0);
	blas::syvm(alpha, x, a, beta, y);

	ASSERT_TRUE( is_equal(y, y0) );
}

TEST( MatrixBlasL2, Syvm_DDd )
{
	test_syvm<double>();
}

TEST( MatrixBlasL2, Syvm_DDs )
{
	test_syvm<float>();
}




