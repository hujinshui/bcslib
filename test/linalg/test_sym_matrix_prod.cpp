/**
 * @file test_sym_matrix_prod.cpp
 *
 * Unit testing for symmetric matrix product
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/linalg.h>

#include "test_blas_aux.h"

using namespace bcs;


template<typename T>
void test_sy_smat_col(const index_t n)
{
	dense_matrix<T> a(n, n);
	for (index_t j = 0; j < n; ++j)
		for (index_t i = 0; i < n; ++i) a(i, j) = T(i + j + 1);

	dense_col<T> x(n);
	for (index_t i = 0; i < x.nelems(); ++i) x[i] = T(2 * i - n);

	dense_col<T> y0(n, T(0));
	my_mv(T(1), a, x, T(0), y0);

	dense_matrix<T> y = mm(as_sym(a), x);

	ASSERT_EQ(n, y.nrows());
	ASSERT_EQ(1, y.ncolumns());

	ASSERT_TRUE( is_equal(y, y0) );

	for (index_t i = 0; i < y0.nelems(); ++i) y0[i] = T(i+1);
	for (index_t i = 0; i < y.nelems(); ++i) y[i] = T(i+1);

	my_mv(T(1), a, x, T(1), y0);
	y += mm(as_sym(a), x);

	ASSERT_TRUE( is_equal(y, y0) );
}


TEST( SymMatrixProd, SMatCol_DDd )
{
	test_sy_smat_col<double>(6);
}

TEST( SymMatrixProd, SMatCol_DDs )
{
	test_sy_smat_col<float>(6);
}



template<typename T>
void test_sy_row_smat(const index_t n)
{
	dense_matrix<T> a(n, n);
	for (index_t j = 0; j < n; ++j)
		for (index_t i = 0; i < n; ++i) a(i, j) = T(i + j + 1);

	dense_row<T> x(n);
	for (index_t i = 0; i < x.nelems(); ++i) x[i] = T(2 * i - n);

	dense_row<T> y0(n, T(0));
	my_vm(T(1), x, a, T(0), y0);

	dense_matrix<T> y = mm(x, as_sym(a));

	ASSERT_EQ(1, y.nrows());
	ASSERT_EQ(n, y.ncolumns());

	ASSERT_TRUE( is_equal(y, y0) );

	for (index_t i = 0; i < y0.nelems(); ++i) y0[i] = T(i+1);
	for (index_t i = 0; i < y.nelems(); ++i) y[i] = T(i+1);

	my_vm(T(1), x, a, T(1), y0);
	y += mm(x, as_sym(a));

	ASSERT_TRUE( is_equal(y, y0) );
}


TEST( SymMatrixProd, RowSMat_DDd )
{
	test_sy_row_smat<double>(6);
}

TEST( SymMatrixProd, RowSMat_DDs )
{
	test_sy_row_smat<float>(6);
}



template<typename T>
void test_sy_smat_mat(const index_t m, const index_t n)
{
	dense_matrix<T> a(m, m);
	for (index_t j = 0; j < m; ++j)
		for (index_t i = 0; i < m; ++i) a(i, j) = T(i + j + 1);

	dense_matrix<T> b(m, n);
	for (index_t i = 0; i < b.nelems(); ++i) b[i] = T(2 * i - n);

	dense_matrix<T> c0(m, n, T(0));
	my_mm(T(1), a, b, T(0), c0);

	dense_matrix<T> c = mm(as_sym(a), b);

	ASSERT_EQ(m, c.nrows());
	ASSERT_EQ(n, c.ncolumns());

	ASSERT_TRUE( is_equal(c, c0) );

	for (index_t i = 0; i < c0.nelems(); ++i) c0[i] = T(i+1);
	for (index_t i = 0; i < c.nelems(); ++i) c[i] = T(i+1);

	my_mm(T(1), a, b, T(1), c0);
	c += mm(as_sym(a), b);

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( SymMatrixProd, SMatMat_DDd )
{
	test_sy_smat_mat<double>(5, 6);
}

TEST( SymMatrixProd, SMatMat_DDs )
{
	test_sy_smat_mat<float>(5, 6);
}



