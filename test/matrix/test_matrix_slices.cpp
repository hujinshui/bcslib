/**
 * @file test_matrix_slices.cpp
 *
 * Unit testing of slice (rows & columns) views
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/matrix.h>

#include <algorithm>

using namespace bcs;

template<class Mat>
void test_matrix_single_slices(IDenseMatrix<Mat, double>& mat)
{
	index_t m = mat.nrows();
	index_t n = mat.ncolumns();

	const IDenseMatrix<Mat, double>& cmat = mat;

	dense_col<double> tcol(m);

	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) tcol[i] = mat(i, j);
		ASSERT_TRUE( is_equal(tcol, cmat.column(j)) );
		ASSERT_TRUE( is_equal(tcol,  mat.column(j)) );
	}

	dense_row<double> trow(n);

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j) trow[j] = mat(i, j);
		ASSERT_TRUE( is_equal(trow, cmat.row(i)) );
		ASSERT_TRUE( is_equal(trow,  mat.row(i)) );
	}

	if (m >= 5)
	{
		range rgn = colon(1, m-1);
		dense_col<double> tcol1(m-2);

		for (index_t j = 0; j < n; ++j)
		{
			for (index_t i = 0; i < m-2; ++i) tcol1[i] = mat(i+1, j);

			ASSERT_TRUE( is_equal(tcol1, cmat.V(rgn, j)) );
			ASSERT_TRUE( is_equal(tcol1,  mat.V(rgn, j)) );
		}
	}

	if (n >= 5)
	{
		range rgn = colon(1, n-1);
		dense_row<double> trow1(n-2);

		for (index_t i = 0; i < m; ++i)
		{
			for (index_t j = 0; j < n-2; ++j) trow1[j] = mat(i, j+1);

			ASSERT_TRUE( is_equal(trow1, cmat.V(i, rgn)) );
			ASSERT_TRUE( is_equal(trow1,  mat.V(i, rgn)) );
		}
	}
}

template<int CTRows, int CTCols>
void test_densemat_single_slices(const index_t m, const index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+1);

	test_matrix_single_slices(A);
}

template<int CTRows, int CTCols>
void test_crefmatex_single_slices(const index_t m, const index_t n)
{
	const index_t ldim = m + 2;
	scoped_block<double> blk(ldim * n);
	for (index_t i = 0; i < ldim * n; ++i) blk[i] = double(i+2);

	cref_matrix_ex<double, CTRows, CTCols> A(blk.ptr_begin(), m, n, ldim);
	test_matrix_single_slices(A);
}


TEST( MatrixSlices, DenseMatDD )
{
	test_densemat_single_slices<DynamicDim, DynamicDim>(7, 8);
}

TEST( MatrixSlices, DenseMatDS )
{
	test_densemat_single_slices<DynamicDim, 8>(7, 8);
}

TEST( MatrixSlices, DenseMatD1 )
{
	test_densemat_single_slices<DynamicDim, 1>(7, 1);
}

TEST( MatrixSlices, DenseMatSD )
{
	test_densemat_single_slices<7, DynamicDim>(7, 8);
}

TEST( MatrixSlices, DenseMatSS )
{
	test_densemat_single_slices<7, 8>(7, 8);
}

TEST( MatrixSlices, DenseMatS1 )
{
	test_densemat_single_slices<7, 1>(7, 1);
}

TEST( MatrixSlices, DenseMat1D )
{
	test_densemat_single_slices<1, DynamicDim>(1, 8);
}

TEST( MatrixSlices, DenseMat1S )
{
	test_densemat_single_slices<1, 8>(1, 8);
}

TEST( MatrixSlices, DenseMat11 )
{
	test_densemat_single_slices<1, 1>(1, 1);
}


TEST( MatrixSlices, CRefMatExDD )
{
	test_crefmatex_single_slices<DynamicDim, DynamicDim>(7, 8);
}

TEST( MatrixSlices, CRefMatExDS )
{
	test_crefmatex_single_slices<DynamicDim, 8>(7, 8);
}

TEST( MatrixSlices, CRefMatExD1 )
{
	test_crefmatex_single_slices<DynamicDim, 1>(7, 1);
}

TEST( MatrixSlices, CRefMatExSD )
{
	test_crefmatex_single_slices<7, DynamicDim>(7, 8);
}

TEST( MatrixSlices, CRefMatExSS )
{
	test_crefmatex_single_slices<7, 8>(7, 8);
}

TEST( MatrixSlices, CRefMatExS1 )
{
	test_crefmatex_single_slices<7, 1>(7, 1);
}

TEST( MatrixSlices, CRefMatEx1D )
{
	test_crefmatex_single_slices<1, DynamicDim>(1, 8);
}

TEST( MatrixSlices, CRefMatEx1S )
{
	test_crefmatex_single_slices<1, 8>(1, 8);
}

TEST( MatrixSlices, CRefMatEx11 )
{
	test_crefmatex_single_slices<1, 1>(1, 1);
}




