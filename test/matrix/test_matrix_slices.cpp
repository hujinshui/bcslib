/**
 * @file test_matrix_slices.cpp
 *
 * Unit testing of slice (rows & columns) views
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/matrix.h>

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
		ASSERT_TRUE( is_equal(tcol, mat.column(j)) );
	}

	dense_row<double> trow(n);

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j) trow[j] = mat(i, j);

		ASSERT_TRUE( is_equal(trow, cmat.row(i)) );
		ASSERT_TRUE( is_equal(trow, mat.row(i)) );
	}
}

template<int CTRows, int CTCols>
void test_densemat_single_slices(const index_t m, const index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	for (index_t i = 0; i < m * n; ++i) A[i] = (i+1);

	test_matrix_single_slices(A);
}

template<int CTRows, int CTCols>
void test_refmatex_single_slices(const index_t m, const index_t n)
{
	const index_t ldim = m + 2;
	scoped_block<double> blk(ldim * n);

	ref_matrix_ex<double, CTRows, CTCols> A(blk.ptr_begin(), m, n, ldim);
	for (index_t j = 0; j < n; ++j)
		for (index_t i = 0; i < m; ++i)
			A(i, j) = (i + j * m + 1);

	test_matrix_single_slices(A);
}


TEST( MatrixSlices, DenseMatSingleSlicesDD )
{
	test_densemat_single_slices<DynamicDim, DynamicDim>(5, 6);
}

TEST( MatrixSlices, DenseMatSingleSlicesDS )
{
	test_densemat_single_slices<DynamicDim, 6>(5, 6);
}

TEST( MatrixSlices, DenseMatSingleSlicesD1 )
{
	test_densemat_single_slices<DynamicDim, 1>(5, 1);
}

TEST( MatrixSlices, DenseMatSingleSlicesSD )
{
	test_densemat_single_slices<5, DynamicDim>(5, 6);
}

TEST( MatrixSlices, DenseMatSingleSlicesSS )
{
	test_densemat_single_slices<5, 6>(5, 6);
}

TEST( MatrixSlices, DenseMatSingleSlicesS1 )
{
	test_densemat_single_slices<5, 1>(5, 1);
}

TEST( MatrixSlices, DenseMatSingleSlices1D )
{
	test_densemat_single_slices<1, DynamicDim>(1, 6);
}

TEST( MatrixSlices, DenseMatSingleSlices1S )
{
	test_densemat_single_slices<1, 6>(1, 6);
}

TEST( MatrixSlices, DenseMatSingleSlices11 )
{
	test_densemat_single_slices<1, 1>(1, 1);
}


TEST( MatrixSlices, RefMatExSingleSlicesDD )
{
	test_refmatex_single_slices<DynamicDim, DynamicDim>(5, 6);
}

TEST( MatrixSlices, RefMatExSingleSlicesDS )
{
	test_refmatex_single_slices<DynamicDim, 6>(5, 6);
}

TEST( MatrixSlices, RefMatExSingleSlicesD1 )
{
	test_refmatex_single_slices<DynamicDim, 1>(5, 1);
}

TEST( MatrixSlices, RefMatExSingleSlicesSD )
{
	test_refmatex_single_slices<5, DynamicDim>(5, 6);
}

TEST( MatrixSlices, RefMatExSingleSlicesSS )
{
	test_refmatex_single_slices<5, 6>(5, 6);
}

TEST( MatrixSlices, RefMatExSingleSlicesS1 )
{
	test_refmatex_single_slices<5, 1>(5, 1);
}

TEST( MatrixSlices, RefMatExSingleSlices1D )
{
	test_refmatex_single_slices<1, DynamicDim>(1, 6);
}

TEST( MatrixSlices, RefMatExSingleSlices1S )
{
	test_refmatex_single_slices<1, 6>(1, 6);
}

TEST( MatrixSlices, RefMatExSingleSlices11 )
{
	test_refmatex_single_slices<1, 1>(1, 1);
}




