/**
 * @file test_ref_matrix.cpp
 *
 * Unit Testing of Ref/CRef Matrices
 *
 * @author Dahua Lin
 */

#include "bcs_test_basics.h"
#include <bcslib/matrix/matrix.h>

using namespace bcs;
using namespace bcs::test;

// explicit template for syntax check

template class bcs::RefMatrix<double, DynamicDim, DynamicDim>;
template class bcs::RefMatrix<double, DynamicDim, 1>;
template class bcs::RefMatrix<double, 1, DynamicDim>;
template class bcs::RefMatrix<double, 2, 2>;

template class bcs::CRefMatrix<double, DynamicDim, DynamicDim>;
template class bcs::CRefMatrix<double, DynamicDim, 1>;
template class bcs::CRefMatrix<double, 1, DynamicDim>;
template class bcs::CRefMatrix<double, 2, 2>;

template class bcs::RefCol<double, DynamicDim>;
template class bcs::RefCol<double, 3>;
template class bcs::RefRow<double, DynamicDim>;
template class bcs::RefRow<double, 3>;

template class bcs::CRefCol<double, DynamicDim>;
template class bcs::CRefCol<double, 3>;
template class bcs::CRefRow<double, DynamicDim>;
template class bcs::CRefRow<double, 3>;


// auxiliary functions

template<class Derived, typename T>
static bool verify_dense_matrix(const IDenseMatrix<Derived, T>& A, index_t m, index_t n)
{
	if (!(A.nrows() == m)) return false;
	if (!(A.ncolumns() == n)) return false;
	if (!(A.nelems() == m * n)) return false;
	if (!(A.size() == size_t(m * n))) return false;
	if (!(A.is_empty() == (m == 0 || n == 0))) return false;
	if (!(A.is_vector() == (m == 1 || n == 1))) return false;
	if (!(A.lead_dim() == m)) return false;

	if (A.is_empty())
	{
		if (A.ptr_base() != BCS_NULL) return false;
	}
	else
	{
		if (A.ptr_base() == BCS_NULL) return false;
		const T *p = A.ptr_base();

		for (index_t i = 0; i < m; ++i)
		{
			for (index_t j = 0; j < n; ++j)
			{
				if (A.elem(i, j) != p[i + j * m]) return false;
				if (A(i, j) != p[i + j * m]) return false;
				if (A[i + j * m] != p[i + j * m]) return false;
			}
		}
	}

	return true;
}


// test cases

TEST( RefClasses, RefMatrix)
{
	const index_t m = 2;
	const index_t n = 3;

	double src[m * n] = {1, 2, 3, 4, 5, 6};
	double dst[m * n] = {0, 0, 0, 0, 0, 0};

	CRefMatrix<double> S(src, m, n);
	RefMatrix<double> D(dst, m, n);

	ASSERT_TRUE( verify_dense_matrix(S, m, n) );
	ASSERT_TRUE( verify_dense_matrix(D, m, n) );

	ASSERT_TRUE( S.ptr_base() == src );
	ASSERT_TRUE( D.ptr_base() == dst );

	D.fill(7.0);
	ASSERT_TRUE( elems_equal(6, dst, 7.0) );

	D.copy_from(src);
	ASSERT_TRUE( elems_equal(6, dst, src) );

	CRefMatrix<double> S2(D);

	ASSERT_TRUE( verify_dense_matrix(S2, m, n) );
	ASSERT_TRUE( S2.ptr_base() == D.ptr_base() );

	double tar[m * n] = {10, 20, 30, 40, 50, 60};
	RefMatrix<double> S3(tar, m, n);

	ASSERT_TRUE( verify_dense_matrix(S3, m, n) );
	ASSERT_TRUE( S3.ptr_base() == tar );

	S3 = S2;

	ASSERT_TRUE( S3.ptr_base() == tar );
	ASSERT_TRUE( S2.ptr_base() == D.ptr_base() );
	ASSERT_FALSE( S3.ptr_base() == S2.ptr_base() );

	ASSERT_TRUE( is_equal(S2, S3) );
	ASSERT_TRUE( elems_equal(6, dst, tar) );
}


TEST( RefClasses, RefVector )
{
	const index_t len = 6;

	double src[len] = {1, 2, 3, 4, 5, 6};
	double dst[len] = {-1, -1, -1, -1, -1};

	CRefRow<double> crow(src, len);
	RefRow<double> row(dst, len);

	ASSERT_TRUE( verify_dense_matrix(crow, 1, len) );
	ASSERT_TRUE( verify_dense_matrix(row, 1, len) );

	ASSERT_TRUE( crow.ptr_base() == src );
	ASSERT_TRUE( row.ptr_base() == dst );

	row.zero();

	ASSERT_TRUE( elems_equal(len, dst, 0.0) );

	row.copy_from(src);

	ASSERT_TRUE( elems_equal(len, dst, src) );
	ASSERT_TRUE( is_equal(row, crow) );

	CRefCol<double> ccol(src, len);
	RefCol<double> col(dst, len);

	ASSERT_TRUE( verify_dense_matrix(ccol, len, 1) );
	ASSERT_TRUE( verify_dense_matrix(col, len, 1) );

	ASSERT_TRUE( is_equal(ccol, col) );

	col.fill(2.0);

	ASSERT_TRUE( elems_equal(len, dst, 2.0) );
}







