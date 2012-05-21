/*
 * @file test_matrix_blas2.cpp
 *
 * Unit testing for BLAS Level 2
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/linalg.h>

using namespace bcs;

template<typename T, class MatA, class MatB, class MatC>
void my_mv(const T alpha, const IDenseMatrix<MatA, T>& a, IDenseMatrix<MatB, T>& b,
		const T beta, IDenseMatrix<MatC, T>& c)
{
	const index_t m = a.nrows();
	const index_t n = a.ncolumns();

	check_arg(b.nrows() == n, "The size of b is incorrect.");
	check_arg(c.nrows() == m && c.ncolumns() == 1, "The size of c is incorrect.");

	for (index_t i = 0; i < m; ++i)
	{
		T s(0);
		for (index_t j = 0; j < n; ++j)
		{
			s += a(i, j) * b(j, 0);
		}
		c(i, 0) = alpha * s + beta * c(i, 0);
	}

}


template<typename T>
void test_gemv_n()
{
	const index_t m = 4;
	const index_t n = 5;

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

TEST( MatrixBlasL2, GemvNDDdouble )
{
	test_gemv_n<double>();
}

TEST( MatrixBlasL2, GemvNDDsingle )
{
	test_gemv_n<float>();
}

template<typename T>
void test_gemv_t()
{
	const index_t m = 4;
	const index_t n = 5;

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

TEST( MatrixBlasL2, GemvTDDdouble )
{
	test_gemv_t<double>();
}

TEST( MatrixBlasL2, GemvTDDsingle )
{
	test_gemv_t<float>();
}




