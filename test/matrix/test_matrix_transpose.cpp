/**
 * @file test_matrix_transpose.cpp
 *
 * Test matrix transposition
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;


template<class Mat>
void test_matrix_tranpose(const Mat& mat)
{
	const index_t m = mat.nrows();
	const index_t n = mat.ncolumns();

	dense_matrix<double> tmat0(n, m);

	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) tmat0(j, i) = mat(i, j);
	}


	dense_matrix<double, ct_cols<Mat>::value, ct_rows<Mat>::value> tmat1 = mat.trans();
	dense_matrix<double, ct_cols<Mat>::value, ct_rows<Mat>::value> tmat2 = transpose(mat);

	ASSERT_EQ(n, tmat1.nrows());
	ASSERT_EQ(m, tmat1.ncolumns());
	ASSERT_EQ(m * n, tmat1.nelems());

	ASSERT_EQ(n, tmat2.nrows());
	ASSERT_EQ(m, tmat2.ncolumns());
	ASSERT_EQ(m * n, tmat2.nelems());

	ASSERT_TRUE( is_equal(tmat0, tmat1) );
	ASSERT_TRUE( is_equal(tmat0, tmat2) );
}


template<int CTRows, int CTCols>
void test_transpose_on_densemat(const index_t m, const index_t n)
{
	dense_matrix<double, CTRows, CTCols> mat(m, n);
	for (index_t i = 0; i < m * n; ++i) mat[i] = double(i+2);
	test_matrix_tranpose(mat);
}


TEST( DenseTranspose, DD11 )
{
	test_transpose_on_densemat<0, 0>(1, 1);
}

TEST( DenseTranspose, DD14 )
{
	test_transpose_on_densemat<0, 0>(1, 4);
}

TEST( DenseTranspose, DD15 )
{
	test_transpose_on_densemat<0, 0>(1, 5);
}

TEST( DenseTranspose, DD41 )
{
	test_transpose_on_densemat<0, 0>(4, 1);
}

TEST( DenseTranspose, DD44 )
{
	test_transpose_on_densemat<0, 0>(4, 4);
}

TEST( DenseTranspose, DD45 )
{
	test_transpose_on_densemat<0, 0>(4, 5);
}

TEST( DenseTranspose, DD51 )
{
	test_transpose_on_densemat<0, 0>(5, 1);
}

TEST( DenseTranspose, DD54 )
{
	test_transpose_on_densemat<0, 0>(5, 4);
}

TEST( DenseTranspose, DD55 )
{
	test_transpose_on_densemat<0, 0>(5, 5);
}


TEST( DenseTranspose, DS11 )
{
	test_transpose_on_densemat<0, 1>(1, 1);
}

TEST( DenseTranspose, DS14 )
{
	test_transpose_on_densemat<0, 4>(1, 4);
}

TEST( DenseTranspose, DS15 )
{
	test_transpose_on_densemat<0, 5>(1, 5);
}

TEST( DenseTranspose, DS41 )
{
	test_transpose_on_densemat<0, 1>(4, 1);
}

TEST( DenseTranspose, DS44 )
{
	test_transpose_on_densemat<0, 4>(4, 4);
}

TEST( DenseTranspose, DS45 )
{
	test_transpose_on_densemat<0, 5>(4, 5);
}

TEST( DenseTranspose, DS51 )
{
	test_transpose_on_densemat<0, 1>(5, 1);
}

TEST( DenseTranspose, DS54 )
{
	test_transpose_on_densemat<0, 4>(5, 4);
}

TEST( DenseTranspose, DS55 )
{
	test_transpose_on_densemat<0, 5>(5, 5);
}

TEST( DenseTranspose, SD11 )
{
	test_transpose_on_densemat<1, 0>(1, 1);
}

TEST( DenseTranspose, SD14 )
{
	test_transpose_on_densemat<1, 0>(1, 4);
}

TEST( DenseTranspose, SD15 )
{
	test_transpose_on_densemat<1, 0>(1, 5);
}

TEST( DenseTranspose, SD41 )
{
	test_transpose_on_densemat<4, 0>(4, 1);
}

TEST( DenseTranspose, SD44 )
{
	test_transpose_on_densemat<4, 0>(4, 4);
}

TEST( DenseTranspose, SD45 )
{
	test_transpose_on_densemat<4, 0>(4, 5);
}

TEST( DenseTranspose, SD51 )
{
	test_transpose_on_densemat<5, 0>(5, 1);
}

TEST( DenseTranspose, SD54 )
{
	test_transpose_on_densemat<5, 0>(5, 4);
}

TEST( DenseTranspose, SD55 )
{
	test_transpose_on_densemat<5, 0>(5, 5);
}

TEST( DenseTranspose, SS11 )
{
	test_transpose_on_densemat<1, 1>(1, 1);
}

TEST( DenseTranspose, SS14 )
{
	test_transpose_on_densemat<1, 4>(1, 4);
}

TEST( DenseTranspose, SS15 )
{
	test_transpose_on_densemat<1, 5>(1, 5);
}

TEST( DenseTranspose, SS41 )
{
	test_transpose_on_densemat<4, 1>(4, 1);
}

TEST( DenseTranspose, SS44 )
{
	test_transpose_on_densemat<4, 4>(4, 4);
}

TEST( DenseTranspose, SS45 )
{
	test_transpose_on_densemat<4, 5>(4, 5);
}

TEST( DenseTranspose, SS51 )
{
	test_transpose_on_densemat<5, 1>(5, 1);
}

TEST( DenseTranspose, SS54 )
{
	test_transpose_on_densemat<5, 4>(5, 4);
}

TEST( DenseTranspose, SS55 )
{
	test_transpose_on_densemat<5, 5>(5, 5);
}


template<int CTRows, int CTCols>
void test_transpose_on_refexmat(const index_t m, const index_t n)
{
	const index_t ldim = m + 3;

	dense_matrix<double> mat0(ldim, n);
	for (index_t i = 0; i < ldim * n; ++i) mat0[i] = double(i+2);

	ref_matrix_ex<double, CTRows, CTCols> mat(mat0.ptr_data(), m, n, ldim);
	test_matrix_tranpose(mat);
}

TEST( RefExTranspose, DD11 )
{
	test_transpose_on_refexmat<0, 0>(1, 1);
}

TEST( RefExTranspose, DD14 )
{
	test_transpose_on_refexmat<0, 0>(1, 4);
}

TEST( RefExTranspose, DD15 )
{
	test_transpose_on_refexmat<0, 0>(1, 5);
}

TEST( RefExTranspose, DD41 )
{
	test_transpose_on_refexmat<0, 0>(4, 1);
}

TEST( RefExTranspose, DD44 )
{
	test_transpose_on_refexmat<0, 0>(4, 4);
}

TEST( RefExTranspose, DD45 )
{
	test_transpose_on_refexmat<0, 0>(4, 5);
}

TEST( RefExTranspose, DD51 )
{
	test_transpose_on_refexmat<0, 0>(5, 1);
}

TEST( RefExTranspose, DD54 )
{
	test_transpose_on_refexmat<0, 0>(5, 4);
}

TEST( RefExTranspose, DD55 )
{
	test_transpose_on_refexmat<0, 0>(5, 5);
}


TEST( RefExTranspose, DS11 )
{
	test_transpose_on_refexmat<0, 1>(1, 1);
}

TEST( RefExTranspose, DS14 )
{
	test_transpose_on_refexmat<0, 4>(1, 4);
}

TEST( RefExTranspose, DS15 )
{
	test_transpose_on_refexmat<0, 5>(1, 5);
}

TEST( RefExTranspose, DS41 )
{
	test_transpose_on_refexmat<0, 1>(4, 1);
}

TEST( RefExTranspose, DS44 )
{
	test_transpose_on_refexmat<0, 4>(4, 4);
}

TEST( RefExTranspose, DS45 )
{
	test_transpose_on_refexmat<0, 5>(4, 5);
}

TEST( RefExTranspose, DS51 )
{
	test_transpose_on_refexmat<0, 1>(5, 1);
}

TEST( RefExTranspose, DS54 )
{
	test_transpose_on_refexmat<0, 4>(5, 4);
}

TEST( RefExTranspose, DS55 )
{
	test_transpose_on_refexmat<0, 5>(5, 5);
}

TEST( RefExTranspose, SD11 )
{
	test_transpose_on_refexmat<1, 0>(1, 1);
}

TEST( RefExTranspose, SD14 )
{
	test_transpose_on_refexmat<1, 0>(1, 4);
}

TEST( RefExTranspose, SD15 )
{
	test_transpose_on_refexmat<1, 0>(1, 5);
}

TEST( RefExTranspose, SD41 )
{
	test_transpose_on_refexmat<4, 0>(4, 1);
}

TEST( RefExTranspose, SD44 )
{
	test_transpose_on_refexmat<4, 0>(4, 4);
}

TEST( RefExTranspose, SD45 )
{
	test_transpose_on_refexmat<4, 0>(4, 5);
}

TEST( RefExTranspose, SD51 )
{
	test_transpose_on_refexmat<5, 0>(5, 1);
}

TEST( RefExTranspose, SD54 )
{
	test_transpose_on_refexmat<5, 0>(5, 4);
}

TEST( RefExTranspose, SD55 )
{
	test_transpose_on_refexmat<5, 0>(5, 5);
}

TEST( RefExTranspose, SS11 )
{
	test_transpose_on_refexmat<1, 1>(1, 1);
}

TEST( RefExTranspose, SS14 )
{
	test_transpose_on_refexmat<1, 4>(1, 4);
}

TEST( RefExTranspose, SS15 )
{
	test_transpose_on_refexmat<1, 5>(1, 5);
}

TEST( RefExTranspose, SS41 )
{
	test_transpose_on_refexmat<4, 1>(4, 1);
}

TEST( RefExTranspose, SS44 )
{
	test_transpose_on_refexmat<4, 4>(4, 4);
}

TEST( RefExTranspose, SS45 )
{
	test_transpose_on_refexmat<4, 5>(4, 5);
}

TEST( RefExTranspose, SS51 )
{
	test_transpose_on_refexmat<5, 1>(5, 1);
}

TEST( RefExTranspose, SS54 )
{
	test_transpose_on_refexmat<5, 4>(5, 4);
}

TEST( RefExTranspose, SS55 )
{
	test_transpose_on_refexmat<5, 5>(5, 5);
}


