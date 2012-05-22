/**
 * @file test_blas_aux.h
 *
 * Some helper functions for BLAS testing
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_TEST_BLAS_AUX_H_
#define BCSLIB_TEST_BLAS_AUX_H_

#include <bcslib/matrix.h>


template<typename T, class MatA, class MatB, class MatC>
void my_mv(const T alpha, const bcs::IDenseMatrix<MatA, T>& a, bcs::IDenseMatrix<MatB, T>& x,
		const T beta, bcs::IDenseMatrix<MatC, T>& y)
{
	using bcs::index_t;

	const index_t m = a.nrows();
	const index_t n = a.ncolumns();

	bcs::check_arg(x.nrows() == n, "The size of x is incorrect.");
	bcs::check_arg(y.nrows() == m && y.ncolumns() == 1, "The size of y is incorrect.");

	for (index_t i = 0; i < m; ++i)
	{
		T s(0);
		for (index_t j = 0; j < n; ++j)
		{
			s += a(i, j) * x(j, 0);
		}
		y(i, 0) = alpha * s + beta * y(i, 0);
	}
}

template<typename T, class MatA, class MatB, class MatC>
void my_vm(const T alpha, const bcs::IDenseMatrix<MatA, T>& x, bcs::IDenseMatrix<MatB, T>& a,
		const T beta, bcs::IDenseMatrix<MatC, T>& y)
{
	using bcs::index_t;

	const index_t m = a.nrows();
	const index_t n = a.ncolumns();

	bcs::check_arg(x.ncolumns() == m, "The size of x is incorrect.");
	bcs::check_arg(y.ncolumns() == n && y.nrows() == 1, "The size of y is incorrect.");

	for (index_t j = 0; j < n; ++j)
	{
		T s(0);
		for (index_t i = 0; i < m; ++i)
		{
			s += x(0, i) * a(i, j);
		}
		y(0, j) = alpha * s + beta * y(0, j);
	}
}


template<typename T, class MatA, class MatB, class MatC>
void my_mm(const T alpha, const bcs::IDenseMatrix<MatA, T>& a, bcs::IDenseMatrix<MatB, T>& b,
		const T beta, bcs::IDenseMatrix<MatC, T>& c)
{
	using bcs::index_t;

	const index_t m = a.nrows();
	const index_t k = a.ncolumns();
	const index_t n = b.ncolumns();

	bcs::check_arg(b.nrows() == k, "The size of b is incorrect.");
	bcs::check_arg(c.nrows() == m && c.ncolumns() == n, "The size of c is incorrect.");

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



#endif /* TEST_BLAS_AUX_H_ */
