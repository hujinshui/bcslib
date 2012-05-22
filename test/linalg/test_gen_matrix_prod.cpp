/*
 * @file test_gen_matrix_prod.cpp
 *
 * Test general matrix product
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/linalg.h>

#include "test_blas_aux.h"

using namespace bcs;


template<typename T>
void test_ge_mat_col(const index_t m, const index_t n)
{
	dense_matrix<T> a(m, n);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_col<T> x(n);
	for (index_t i = 0; i < x.nelems(); ++i) x[i] = T(2 * i - n);

	dense_col<T> y0(m, T(0));
	my_mv(T(1), a, x, T(0), y0);

	dense_matrix<T> y = mm(a, x);

	ASSERT_EQ(m, y.nrows());
	ASSERT_EQ(1, y.ncolumns());

	ASSERT_TRUE( is_equal(y, y0) );

	for (index_t i = 0; i < y0.nelems(); ++i) y0[i] = T(i+1);
	for (index_t i = 0; i < y.nelems(); ++i) y[i] = T(i+1);

	my_mv(T(1), a, x, T(1), y0);
	y += mm(a, x);

	ASSERT_TRUE( is_equal(y, y0) );
}


TEST( GeneralMatrixProd, MatCol_DDd )
{
	test_ge_mat_col<double>(5, 6);
}

TEST( GeneralMatrixProd, MatCol_DDs )
{
	test_ge_mat_col<float>(5, 6);
}


template<typename T>
void test_ge_tmat_col(const index_t m, const index_t n)
{
	dense_matrix<T> a(m, n);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_col<T> x(m);
	for (index_t i = 0; i < x.nelems(); ++i) x[i] = T(2 * i - n);

	dense_col<T> y0(n, T(0));
	dense_matrix<T> at = a.trans();
	my_mv(T(1), at, x, T(0), y0);

	dense_matrix<T> y = mm(a.trans(), x);

	ASSERT_EQ(n, y.nrows());
	ASSERT_EQ(1, y.ncolumns());

	ASSERT_TRUE( is_equal(y, y0) );

	for (index_t i = 0; i < y0.nelems(); ++i) y0[i] = T(i+1);
	for (index_t i = 0; i < y.nelems(); ++i) y[i] = T(i+1);

	my_mv(T(1), at, x, T(1), y0);
	y += mm(a.trans(), x);

	ASSERT_TRUE( is_equal(y, y0) );
}

TEST( GeneralMatrixProd, TMatCol_DDd )
{
	test_ge_tmat_col<double>(5, 6);
}

TEST( GeneralMatrixProd, TMatCol_DDs )
{
	test_ge_tmat_col<float>(5, 6);
}


template<typename T>
void test_ge_row_mat(const index_t m, const index_t n)
{
	dense_matrix<T> a(m, n);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_row<T> x(m);
	for (index_t i = 0; i < x.nelems(); ++i) x[i] = T(2 * i - n);

	dense_row<T> y0(n, T(0));
	my_vm(T(1), x, a, T(0), y0);

	dense_matrix<T> y = mm(x, a);

	ASSERT_EQ(1, y.nrows());
	ASSERT_EQ(n, y.ncolumns());

	ASSERT_TRUE( is_equal(y, y0) );

	for (index_t i = 0; i < y0.nelems(); ++i) y0[i] = T(i+1);
	for (index_t i = 0; i < y.nelems(); ++i) y[i] = T(i+1);

	my_vm(T(1), x, a, T(1), y0);
	y += mm(x, a);

	ASSERT_TRUE( is_equal(y, y0) );
}

TEST( GeneralMatrixProd, RowMat_DDd )
{
	test_ge_row_mat<double>(5, 6);
}

TEST( GeneralMatrixProd, RowMat_DDs )
{
	test_ge_row_mat<float>(5, 6);
}

template<typename T>
void test_ge_row_tmat(const index_t m, const index_t n)
{
	dense_matrix<T> a(m, n);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_row<T> x(n);
	for (index_t i = 0; i < x.nelems(); ++i) x[i] = T(2 * i - n);

	dense_row<T> y0(m, T(0));

	dense_matrix<T> at = a.trans();
	my_vm(T(1), x, at, T(0), y0);

	dense_matrix<T> y = mm(x, a.trans());

	ASSERT_EQ(1, y.nrows());
	ASSERT_EQ(m, y.ncolumns());

	ASSERT_TRUE( is_equal(y, y0) );

	for (index_t i = 0; i < y0.nelems(); ++i) y0[i] = T(i+1);
	for (index_t i = 0; i < y.nelems(); ++i) y[i] = T(i+1);

	my_vm(T(1), x, at, T(1), y0);
	y += mm(x, a.trans());

	ASSERT_TRUE( is_equal(y, y0) );
}

TEST( GeneralMatrixProd, RowTMat_DDd )
{
	test_ge_row_tmat<double>(5, 6);
}

TEST( GeneralMatrixProd, RowTMat_DDs )
{
	test_ge_row_tmat<float>(5, 6);
}




template<typename T>
void test_ge_mat_mat(const index_t m, const index_t n, const index_t k)
{
	dense_matrix<T> a(m, k);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_matrix<T> b(k, n);
	for (index_t i = 0; i < b.nelems(); ++i) b[i] = T(2 * i - n);

	dense_matrix<T> c0(m, n, T(0));
	my_mm(T(1), a, b, T(0), c0);

	dense_matrix<T> c = mm(a, b);

	ASSERT_EQ(m, c.nrows());
	ASSERT_EQ(n, c.ncolumns());

	ASSERT_TRUE( is_equal(c, c0) );

	for (index_t i = 0; i < c0.nelems(); ++i) c0[i] = T(i+1);
	for (index_t i = 0; i < c.nelems(); ++i) c[i] = T(i+1);

	my_mm(T(1), a, b, T(1), c0);
	c += mm(a, b);

	ASSERT_TRUE( is_equal(c, c0) );
}

TEST( GeneralMatrixProd, MatMat_DDd )
{
	test_ge_mat_mat<double>(4, 5, 6);
}

TEST( GeneralMatrixProd, MatMat_DDs )
{
	test_ge_mat_mat<float>(4, 5, 6);
}


template<typename T>
void test_ge_mat_tmat(const index_t m, const index_t n, const index_t k)
{
	dense_matrix<T> a(m, k);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_matrix<T> b(n, k);
	for (index_t i = 0; i < b.nelems(); ++i) b[i] = T(2 * i - n);

	dense_matrix<T> c0(m, n, T(0));

	dense_matrix<T> bt = b.trans();
	my_mm(T(1), a, bt, T(0), c0);

	dense_matrix<T> c = mm(a, b.trans());

	ASSERT_EQ(m, c.nrows());
	ASSERT_EQ(n, c.ncolumns());

	ASSERT_TRUE( is_equal(c, c0) );

	for (index_t i = 0; i < c0.nelems(); ++i) c0[i] = T(i+1);
	for (index_t i = 0; i < c.nelems(); ++i) c[i] = T(i+1);

	my_mm(T(1), a, bt, T(1), c0);
	c += mm(a, b.trans());

	ASSERT_TRUE( is_equal(c, c0) );
}

TEST( GeneralMatrixProd, MatTMat_DDd )
{
	test_ge_mat_tmat<double>(4, 5, 6);
}

TEST( GeneralMatrixProd, MatTMat_DDs )
{
	test_ge_mat_tmat<float>(4, 5, 6);
}


template<typename T>
void test_ge_tmat_mat(const index_t m, const index_t n, const index_t k)
{
	dense_matrix<T> a(k, m);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_matrix<T> b(k, n);
	for (index_t i = 0; i < b.nelems(); ++i) b[i] = T(2 * i - n);

	dense_matrix<T> c0(m, n, T(0));

	dense_matrix<T> at = a.trans();
	my_mm(T(1), at, b, T(0), c0);

	dense_matrix<T> c = mm(a.trans(), b);

	ASSERT_EQ(m, c.nrows());
	ASSERT_EQ(n, c.ncolumns());

	ASSERT_TRUE( is_equal(c, c0) );

	for (index_t i = 0; i < c0.nelems(); ++i) c0[i] = T(i+1);
	for (index_t i = 0; i < c.nelems(); ++i) c[i] = T(i+1);

	my_mm(T(1), at, b, T(1), c0);
	c += mm(a.trans(), b);

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( GeneralMatrixProd, TMatMat_DDd )
{
	test_ge_tmat_mat<double>(4, 5, 6);
}

TEST( GeneralMatrixProd, TMatMat_DDs )
{
	test_ge_tmat_mat<float>(4, 5, 6);
}


template<typename T>
void test_ge_tmat_tmat(const index_t m, const index_t n, const index_t k)
{
	dense_matrix<T> a(k, m);
	for (index_t i = 0; i < a.nelems(); ++i) a[i] = T(i+1);

	dense_matrix<T> b(n, k);
	for (index_t i = 0; i < b.nelems(); ++i) b[i] = T(2 * i - n);

	dense_matrix<T> c0(m, n, T(0));

	dense_matrix<T> at = a.trans();
	dense_matrix<T> bt = b.trans();
	my_mm(T(1), at, bt, T(0), c0);

	dense_matrix<T> c = mm(a.trans(), b.trans());

	ASSERT_EQ(m, c.nrows());
	ASSERT_EQ(n, c.ncolumns());

	ASSERT_TRUE( is_equal(c, c0) );

	for (index_t i = 0; i < c0.nelems(); ++i) c0[i] = T(i+1);
	for (index_t i = 0; i < c.nelems(); ++i) c[i] = T(i+1);

	my_mm(T(1), at, bt, T(1), c0);
	c += mm(a.trans(), b.trans());

	ASSERT_TRUE( is_equal(c, c0) );
}


TEST( GeneralMatrixProd, TMatTMat_DDd )
{
	test_ge_tmat_tmat<double>(4, 5, 6);
}

TEST( GeneralMatrixProd, TMatTMat_DDs )
{
	test_ge_tmat_tmat<float>(4, 5, 6);
}



