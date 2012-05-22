/*
 * @file test_gemm.cpp
 *
 * Unit testing for BLAS Level 3
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/linalg.h>

#include "test_blas_aux.h"

using namespace bcs;


/************************************************
 *
 *  GEMM
 *
 ************************************************/

template<typename T>
void test_gemm_nn()
{
	const index_t m = 4;
	const index_t n = 5;
	const index_t k = 6;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(m, k);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_matrix<T> b(k, n);
	for (index_t i = 0; i < b.nelems(); ++i) b[i] = T(2 * i - 10);

	dense_matrix<T> c0(m, n);
	for (index_t i = 0; i < c0.nelems(); ++i) c0[i] = T(i+2);

	dense_matrix<T> c(c0);
	ASSERT_TRUE( c0.ptr_data() != c.ptr_data() );

	my_mm(alpha, a, b, beta, c0);
	blas::gemm_nn(alpha, a, b, beta, c);

	ASSERT_TRUE( is_equal(c, c0) );
}

template<typename T>
void test_gemm_nt()
{
	const index_t m = 4;
	const index_t n = 5;
	const index_t k = 6;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(m, k);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_matrix<T> b(n, k);
	for (index_t i = 0; i < b.nelems(); ++i) b[i] = T(2 * i - 10);

	dense_matrix<T> c0(m, n);
	for (index_t i = 0; i < c0.nelems(); ++i) c0[i] = T(i+2);

	dense_matrix<T> c(c0);
	ASSERT_TRUE( c0.ptr_data() != c.ptr_data() );

	dense_matrix<T> bt = b.trans();

	my_mm(alpha, a, bt, beta, c0);
	blas::gemm_nt(alpha, a, b, beta, c);

	ASSERT_TRUE( is_equal(c, c0) );
}

template<typename T>
void test_gemm_tn()
{
	const index_t m = 4;
	const index_t n = 5;
	const index_t k = 6;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(k, m);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_matrix<T> b(k, n);
	for (index_t i = 0; i < b.nelems(); ++i) b[i] = T(2 * i - 10);

	dense_matrix<T> c0(m, n);
	for (index_t i = 0; i < c0.nelems(); ++i) c0[i] = T(i+2);

	dense_matrix<T> c(c0);
	ASSERT_TRUE( c0.ptr_data() != c.ptr_data() );

	dense_matrix<T> at = a.trans();

	my_mm(alpha, at, b, beta, c0);
	blas::gemm_tn(alpha, a, b, beta, c);

	ASSERT_TRUE( is_equal(c, c0) );
}

template<typename T>
void test_gemm_tt()
{
	const index_t m = 4;
	const index_t n = 5;
	const index_t k = 6;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(k, m);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_matrix<T> b(n, k);
	for (index_t i = 0; i < b.nelems(); ++i) b[i] = T(2 * i - 10);

	dense_matrix<T> c0(m, n);
	for (index_t i = 0; i < c0.nelems(); ++i) c0[i] = T(i+2);

	dense_matrix<T> c(c0);
	ASSERT_TRUE( c0.ptr_data() != c.ptr_data() );

	dense_matrix<T> at = a.trans();
	dense_matrix<T> bt = b.trans();

	my_mm(alpha, at, bt, beta, c0);
	blas::gemm_tt(alpha, a, b, beta, c);

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixBlasL3, GemmNN_DDd )
{
	test_gemm_nn<double>();
}

TEST( MatrixBlasL3, GemmNN_DDs )
{
	test_gemm_nn<float>();
}

TEST( MatrixBlasL3, GemmNT_DDd )
{
	test_gemm_nt<double>();
}

TEST( MatrixBlasL3, GemmNT_DDs )
{
	test_gemm_nt<float>();
}

TEST( MatrixBlasL3, GemmTN_DDd )
{
	test_gemm_tn<double>();
}

TEST( MatrixBlasL3, GemmTN_DDs )
{
	test_gemm_tn<float>();
}

TEST( MatrixBlasL3, GemmTT_DDd )
{
	test_gemm_tt<double>();
}

TEST( MatrixBlasL3, GemmTT_DDs )
{
	test_gemm_tt<float>();
}


/************************************************
 *
 *  SYMM
 *
 ************************************************/

template<typename T>
void test_symm_l()
{
	const index_t m = 5;
	const index_t n = 6;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(m, m);
	for (index_t j = 0; j < m; ++j)
		for (index_t i = 0; i < m; ++i) a(i, j) = T(i + j + 1);

	dense_matrix<T> b(m, n);
	for (index_t i = 0; i < b.nelems(); ++i) b[i] = T(2 * i - 10);

	dense_matrix<T> c0(m, n);
	for (index_t i = 0; i < c0.nelems(); ++i) c0[i] = T(i + 2);

	dense_matrix<T> c(c0);
	ASSERT_TRUE( c0.ptr_data() != c.ptr_data() );

	my_mm(alpha, a, b, beta, c0);
	blas::symm_l(alpha, a, b, beta, c);

	ASSERT_TRUE( is_equal(c, c0) );
}

template<typename T>
void test_symm_r()
{
	const index_t m = 5;
	const index_t n = 6;

	T alpha = T(1.5);
	T beta = T(0.5);

	dense_matrix<T> a(n, n);
	for (index_t j = 0; j < n; ++j)
		for (index_t i = 0; i < n; ++i) a(i, j) = T(i + j + 1);

	dense_matrix<T> b(m, n);
	for (index_t i = 0; i < b.nelems(); ++i) b[i] = T(2 * i - 10);

	dense_matrix<T> c0(m, n);
	for (index_t i = 0; i < c0.nelems(); ++i) c0[i] = T(i + 2);

	dense_matrix<T> c(c0);
	ASSERT_TRUE( c0.ptr_data() != c.ptr_data() );

	my_mm(alpha, b, a, beta, c0);
	blas::symm_r(alpha, a, b, beta, c);

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( MatrixBlasL3, SymmL_DDd )
{
	test_symm_l<double>();
}

TEST( MatrixBlasL3, SymmL_DDs )
{
	test_symm_l<float>();
}

TEST( MatrixBlasL3, SymmR_DDd )
{
	test_symm_r<double>();
}

TEST( MatrixBlasL3, SymmR_DDs )
{
	test_symm_r<float>();
}



