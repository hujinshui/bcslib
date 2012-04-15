/**
 * @file test_dense_matrix.cpp
 *
 * Unit testing for dense matrices
 *
 * @author Dahua Lin
 */

#include "bcs_test_basics.h"

#include <bcslib/matrix/dense_matrix.h>

using namespace bcs;
using namespace bcs::test;

// explicit template for syntax check

template class bcs::DenseMatrix<double>;
template class bcs::DenseMatrix<double, bcs::DynamicDim, 1>;
template class bcs::DenseMatrix<double, 1, bcs::DynamicDim>;
template class bcs::DenseMatrix<double, 2, 2>;


template<typename T, int RowDim, int ColDim>
bool verify_dense_matrix(const DenseMatrix<T, RowDim, ColDim>& A, index_t m, index_t n)
{
	typedef DenseMatrix<T, RowDim, ColDim> mat_t;

#ifdef BCS_USE_STATIC_ASSERT
	static_assert(mat_t::RowDimension == RowDim, "RowDim mismatch");
	static_assert(mat_t::ColDimension == ColDim, "ColDim mismatch");
#endif

	if (!(A.nrows() == m)) return false;
	if (!(A.ncolumns() == n)) return false;
	if (!(A.nelems() == m * n)) return false;
	if (!(A.size() == size_t(m * n))) return false;
	if (!(A.is_empty() == (m == 0 || n == 0))) return false;
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
			}
		}
	}

	return true;
}


TEST( MatrixBasics, MatrixXX )
{
	mat_f64 A;
	//ASSERT_TRUE( verify_dense_matrix(A, (index_t)0, (index_t)0) );

}




