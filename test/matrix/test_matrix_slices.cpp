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
	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+1);

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
			A(i, j) = double(i + j * m + 1);

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



template<class Mat>
void test_matrix_multi_slices(IDenseMatrix<Mat, double>& mat)
{
	index_t m = mat.nrows();
	index_t n = mat.ncolumns();

	index_t sm = std::max(m/2, index_t(1));
	index_t sn = std::max(n/2, index_t(1));

	const IDenseMatrix<Mat, double>& cmat = mat;

	dense_matrix<double> tcols(m, sn);

	for (index_t j = 0; j < n - sn + 1; ++j)
	{
		for (index_t j2 = 0; j2 < sn; ++j2)
			for (index_t i = 0; i < m; ++i) tcols(i, j2) = mat(i, j+j2);

		ASSERT_TRUE( is_equal(cmat.columns(colon(j, j+sn)), tcols) );
		ASSERT_TRUE( is_equal( mat.columns(colon(j, j+sn)), tcols) );
	}

	dense_matrix<double> trows(sm, n);

	for (index_t i = 0; i < m - sm + 1; ++i)
	{
		for (index_t j = 0; j < n; ++j)
			for (index_t i2 = 0; i2 < sm; ++i2) trows(i2, j) = mat(i+i2, j);

		ASSERT_TRUE( is_equal(cmat.rows(colon(i, i+sm)), trows) );
		ASSERT_TRUE( is_equal( mat.rows(colon(i, i+sm)), trows) );
	}
}

template<int CTRows, int CTCols>
void test_densemat_multi_slices(const index_t m, const index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+1);

	test_matrix_multi_slices(A);
}

template<int CTRows, int CTCols>
void test_refmatex_multi_slices(const index_t m, const index_t n)
{
	const index_t ldim = m + 2;
	scoped_block<double> blk(ldim * n);

	ref_matrix_ex<double, CTRows, CTCols> A(blk.ptr_begin(), m, n, ldim);
	for (index_t j = 0; j < n; ++j)
		for (index_t i = 0; i < m; ++i)
			A(i, j) = double(i + j * m + 1);

	test_matrix_multi_slices(A);
}



TEST( MatrixMultiSlices, DenseMatMultiSlicesDD )
{
	test_densemat_multi_slices<DynamicDim, DynamicDim>(5, 6);
}

TEST( MatrixMultiSlices, DenseMatMultiSlicesDS )
{
	test_densemat_multi_slices<DynamicDim, 6>(5, 6);
}

TEST( MatrixMultiSlices, DenseMatMultiSlicesD1 )
{
	test_densemat_multi_slices<DynamicDim, 1>(5, 1);
}

TEST( MatrixMultiSlices, DenseMatMultiSlicesSD )
{
	test_densemat_multi_slices<5, DynamicDim>(5, 6);
}

TEST( MatrixMultiSlices, DenseMatMultiSlicesSS )
{
	test_densemat_multi_slices<5, 6>(5, 6);
}

TEST( MatrixMultiSlices, DenseMatMultiSlicesS1 )
{
	test_densemat_multi_slices<5, 1>(5, 1);
}

TEST( MatrixMultiSlices, DenseMatMultiSlices1D )
{
	test_densemat_multi_slices<1, DynamicDim>(1, 6);
}

TEST( MatrixMultiSlices, DenseMatMultiSlices1S )
{
	test_densemat_multi_slices<1, 6>(1, 6);
}

TEST( MatrixMultiSlices, DenseMatMultiSlices11 )
{
	test_densemat_multi_slices<1, 1>(1, 1);
}


TEST( MatrixMultiSlices, RefMatExMultiSlicesDD )
{
	test_refmatex_multi_slices<DynamicDim, DynamicDim>(5, 6);
}

TEST( MatrixMultiSlices, RefMatExMultiSlicesDS )
{
	test_refmatex_multi_slices<DynamicDim, 6>(5, 6);
}

TEST( MatrixMultiSlices, RefMatExMultiSlicesD1 )
{
	test_refmatex_multi_slices<DynamicDim, 1>(5, 1);
}

TEST( MatrixMultiSlices, RefMatExMultiSlicesSD )
{
	test_refmatex_multi_slices<5, DynamicDim>(5, 6);
}

TEST( MatrixMultiSlices, RefMatExMultiSlicesSS )
{
	test_refmatex_multi_slices<5, 6>(5, 6);
}

TEST( MatrixMultiSlices, RefMatExMultiSlicesS1 )
{
	test_refmatex_multi_slices<5, 1>(5, 1);
}

TEST( MatrixMultiSlices, RefMatExMultiSlices1D )
{
	test_refmatex_multi_slices<1, DynamicDim>(1, 6);
}

TEST( MatrixMultiSlices, RefMatExMultiSlices1S )
{
	test_refmatex_multi_slices<1, 6>(1, 6);
}

TEST( MatrixMultiSlices, RefMatExMultiSlices11 )
{
	test_refmatex_multi_slices<1, 1>(1, 1);
}




