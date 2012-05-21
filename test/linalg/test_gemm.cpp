/*
 * @file test_gemm.cpp
 *
 * Unit testing for GEMM
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/linalg.h>

using namespace bcs;

template<typename T, class MatA, class MatB, class MatC>
void my_mm(const T alpha, const IDenseMatrix<MatA, T>& a, IDenseMatrix<MatB, T>& b,
		const T beta, IDenseMatrix<MatC, T>& c)
{
	const index_t m = a.nrows();
	const index_t k = a.ncolumns();
	const index_t n = b.ncolumns();

	check_arg(b.nrows() == k, "The size of b is incorrect.");
	check_arg(c.nrows() == m && c.ncolumns() == n, "The size of c is incorrect.");

	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i)
		{
			T s(0);
			for (index_t u = 0; u < k; ++u) s += a(i, u) * b(u, j);
			c(i, j) = alpha * s + beta * c(i, j);
		}
	}
}


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





