/**
 * @file test_matrix_blocks.cpp
 *
 * Unit testing of block views
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/matrix.h>

#include <algorithm>

using namespace bcs;

template<class Mat>
void test_matrix_block_views(IDenseMatrix<Mat, double>& mat)
{
	index_t m = mat.nrows();
	index_t n = mat.ncolumns();

	const IDenseMatrix<Mat, double>& cmat = mat;

	// whole x whole

	ASSERT_TRUE( is_equal(cmat, cmat.V(whole(), whole())) );
	ASSERT_TRUE( is_equal( mat,  mat.V(whole(), whole())) );

	// whole x range

	if (n >= 5)
	{
		range crg = colon(1, n-1);
		dense_matrix<double> smat(m, n-2);

		for (index_t j = 0; j < n-2; ++j)
			for (index_t i = 0; i < m; ++i) smat(i, j) = mat(i, j+1);

		ASSERT_TRUE( is_equal(smat, cmat.V(whole(), crg)) );
		ASSERT_TRUE( is_equal(smat,  mat.V(whole(), crg)) );
	}

	// range x whole

	if (m >= 5)
	{
		range rrg = colon(1, m-1);
		dense_matrix<double> smat(m-2, n);

		for (index_t j = 0; j < n; ++j)
			for (index_t i = 0; i < m-2; ++i) smat(i, j) = mat(i+1, j);

		ASSERT_TRUE( is_equal(smat, cmat.V(rrg, whole())) );
		ASSERT_TRUE( is_equal(smat,  mat.V(rrg, whole())) );
	}

	// range x range

	if (m >= 5 && n >= 5)
	{
		range rrg = colon(1, m-1);
		range crg = colon(1, n-1);
		dense_matrix<double> smat(m-2, n-2);

		for (index_t j = 0; j < n-2; ++j)
			for (index_t i = 0; i < m-2; ++i) smat(i, j) = mat(i+1, j+1);

		ASSERT_TRUE( is_equal(smat, cmat.V(rrg, crg)) );
		ASSERT_TRUE( is_equal(smat,  mat.V(rrg, crg)) );
	}

}


template<int CTRows, int CTCols>
void test_densemat_block_views(const index_t m, const index_t n)
{
	dense_matrix<double, CTRows, CTCols> mat(m, n);
	for (index_t i = 0; i < m * n; ++i) mat[i] = double(i+2);
	test_matrix_block_views(mat);
}



TEST( MatrixBlocks, DenseMatDD )
{
	test_densemat_block_views<DynamicDim, DynamicDim>(7, 8);
}

TEST( MatrixBlocks, DenseMatDS )
{
	test_densemat_block_views<DynamicDim, 8>(7, 8);
}

TEST( MatrixBlocks, DenseMatD1 )
{
	test_densemat_block_views<DynamicDim, 1>(7, 1);
}

TEST( MatrixBlocks, DenseMatSD )
{
	test_densemat_block_views<7, DynamicDim>(7, 8);
}

TEST( MatrixBlocks, DenseMatSS )
{
	test_densemat_block_views<7, 8>(7, 8);
}

TEST( MatrixBlocks, DenseMatS1 )
{
	test_densemat_block_views<7, 1>(7, 1);
}

TEST( MatrixBlocks, DenseMat1D )
{
	test_densemat_block_views<1, DynamicDim>(1, 8);
}

TEST( MatrixBlocks, DenseMat1S )
{
	test_densemat_block_views<1, 8>(1, 8);
}

TEST( MatrixBlocks, DenseMat11 )
{
	test_densemat_block_views<1, 1>(1, 1);
}


template<int CTRows, int CTCols>
void test_crefex_block_views(const index_t m, const index_t n)
{
	const index_t ldim = m + 2;
	dense_matrix<double> mat(ldim, n);
	for (index_t i = 0; i < ldim * n; ++i) mat[i] = double(i+2);

	cref_matrix_ex<double, CTRows, CTCols> rmat(mat.ptr_data(), m, n, ldim);
	test_matrix_block_views(rmat);
}


TEST( MatrixBlocks, CRefMatExDD )
{
	test_crefex_block_views<DynamicDim, DynamicDim>(7, 8);
}

TEST( MatrixBlocks, CRefMatExDS )
{
	test_crefex_block_views<DynamicDim, 8>(7, 8);
}

TEST( MatrixBlocks, CRefMatExD1 )
{
	test_crefex_block_views<DynamicDim, 1>(7, 1);
}

TEST( MatrixBlocks, CRefMatExSD )
{
	test_crefex_block_views<7, DynamicDim>(7, 8);
}

TEST( MatrixBlocks, CRefMatExSS )
{
	test_crefex_block_views<7, 8>(7, 8);
}

TEST( MatrixBlocks, CRefMatExS1 )
{
	test_crefex_block_views<7, 1>(7, 1);
}

TEST( MatrixBlocks, CRefMatEx1D )
{
	test_crefex_block_views<1, DynamicDim>(1, 8);
}

TEST( MatrixBlocks, CRefMatEx1S )
{
	test_crefex_block_views<1, 8>(1, 8);
}

TEST( MatrixBlocks, CRefMatEx11 )
{
	test_crefex_block_views<1, 1>(1, 1);
}





