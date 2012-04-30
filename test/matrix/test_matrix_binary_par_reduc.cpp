/**
 * @file test_matrix_binary_par_reduc.cpp
 *
 * Unit testing for binary partial reduction
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;


template<typename T, class LMat, class RMat>
void test_slicewise_dot(const IMatrixXpr<LMat, T>& A, const IMatrixXpr<RMat, T>& B)
{
	dense_matrix<T> Amat(A);
	dense_matrix<T> Bmat(B);
	index_t m = A.nrows();
	index_t n = A.ncolumns();

	dense_matrix<T> cw = dot(colwise(A), colwise(B));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<T> cw0(1, n);
	for (index_t j = 0; j < n; ++j)
	{
		T s(0);
		for (index_t i = 0; i < m; ++i) s += Amat(i, j) * Bmat(i, j);
		cw0[j] = s;
	}

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<T> rw = dot(rowwise(A), rowwise(B));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<T> rw0(m, 1);
	for (index_t i = 0; i < m; ++i)
	{
		T s(0);
		for (index_t j = 0; j < n; ++j) s += Amat(i, j) * Bmat(i, j);
		rw0[i] = s;
	}

	ASSERT_TRUE( is_equal(rw, rw0) );
}


template<int CTRows, int CTCols>
void test_slicewise_dot_on_dense(index_t m, index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	dense_matrix<double, CTRows, CTCols> B(m, n);

	for (index_t i = 0; i < m * n; ++i)
	{
		A[i] = (3 * i + 2);
		B[i] = i + 1;
	}

	test_slicewise_dot(A, B);
}


template<int CTRows, int CTCols>
void test_slicewise_dot_on_refex(index_t m, index_t n)
{
	index_t ldim = m + 2;
	dense_matrix<double> Amat(ldim, n);
	dense_matrix<double> Bmat(ldim, n);

	for (index_t i = 0; i < ldim * n; ++i)
	{
		Amat[i] = (3 * i + 2);
		Bmat[i] = i + 1;
	}

	ref_matrix_ex<double, CTRows, CTCols> A(Amat.ptr_data(), m, n, ldim);
	ref_matrix_ex<double, CTRows, CTCols> B(Bmat.ptr_data(), m, n, ldim);

	test_slicewise_dot(A, B);
}


TEST( MatrixBinaryParReduc, SliceWiseDotOnDenseDD )
{
	test_slicewise_dot_on_dense<DynamicDim, DynamicDim>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnDenseDS )
{
	test_slicewise_dot_on_dense<DynamicDim, 9>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnDenseD1 )
{
	test_slicewise_dot_on_dense<DynamicDim, 1>(6, 1);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnDenseSD )
{
	test_slicewise_dot_on_dense<6, DynamicDim>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnDenseSS )
{
	test_slicewise_dot_on_dense<6, 9>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnDenseS1 )
{
	test_slicewise_dot_on_dense<6, 1>(6, 1);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnDense1D )
{
	test_slicewise_dot_on_dense<1, DynamicDim>(1, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnDense1S )
{
	test_slicewise_dot_on_dense<1, 9>(1, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnDense11 )
{
	test_slicewise_dot_on_dense<1, 1>(1, 1);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnRefExDD )
{
	test_slicewise_dot_on_refex<DynamicDim, DynamicDim>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnRefExDS )
{
	test_slicewise_dot_on_refex<DynamicDim, 9>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnRefExD1 )
{
	test_slicewise_dot_on_refex<DynamicDim, 1>(6, 1);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnRefExSD )
{
	test_slicewise_dot_on_refex<6, DynamicDim>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnRefExSS )
{
	test_slicewise_dot_on_refex<6, 9>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnRefExS1 )
{
	test_slicewise_dot_on_refex<6, 1>(6, 1);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnRefEx1D )
{
	test_slicewise_dot_on_refex<1, DynamicDim>(1, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnRefEx1S )
{
	test_slicewise_dot_on_refex<1, 9>(1, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseDotOnRefEx11 )
{
	test_slicewise_dot_on_refex<1, 1>(1, 1);
}


TEST( MatrixBinaryParReduc, SliceWiseL1NormDiff )
{
	const index_t m = 6;
	const index_t n = 9;

	dense_matrix<double> Amat(m, n);
	dense_matrix<double> Bmat(m, n);

	for (index_t i = 0; i < m * n; ++i)
	{
		Amat[i] = (3 * i + 2);
		Bmat[i] = 60 - i;
	}

	const IMatrixXpr<dense_matrix<double>, double>& A = Amat;
	const IMatrixXpr<dense_matrix<double>, double>& B = Bmat;

	dense_matrix<double> cw = L1norm_diff(colwise(A), colwise(B));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<double> cw0(1, n);
	for (index_t j = 0; j < n; ++j)
		cw0[j] = L1norm_diff(Amat.column(j), Bmat.column(j));

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<double> rw = L1norm_diff(rowwise(A), rowwise(B));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<double> rw0(m, 1);
	for (index_t i = 0; i < m; ++i)
		rw0[i] = L1norm_diff(Amat.row(i), Bmat.row(i));

	ASSERT_TRUE( is_equal(rw, rw0) );
}


TEST( MatrixBinaryParReduc, SliceWiseSqL2NormDiff )
{
	const index_t m = 6;
	const index_t n = 9;

	dense_matrix<double> Amat(m, n);
	dense_matrix<double> Bmat(m, n);

	for (index_t i = 0; i < m * n; ++i)
	{
		Amat[i] = (3 * i + 2);
		Bmat[i] = 60 - i;
	}

	const IMatrixXpr<dense_matrix<double>, double>& A = Amat;
	const IMatrixXpr<dense_matrix<double>, double>& B = Bmat;

	dense_matrix<double> cw = sqL2norm_diff(colwise(A), colwise(B));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<double> cw0(1, n);
	for (index_t j = 0; j < n; ++j)
		cw0[j] = sqL2norm_diff(Amat.column(j), Bmat.column(j));

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<double> rw = sqL2norm_diff(rowwise(A), rowwise(B));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<double> rw0(m, 1);
	for (index_t i = 0; i < m; ++i)
		rw0[i] = sqL2norm_diff(Amat.row(i), Bmat.row(i));

	ASSERT_TRUE( is_equal(rw, rw0) );
}


template<typename T, class LMat, class RMat>
void test_slicewise_L2norm_diff(const IMatrixXpr<LMat, T>& A, const IMatrixXpr<RMat, T>& B)
{
	const index_t m = A.nrows();
	const index_t n = A.ncolumns();

	dense_matrix<T> Amat(A);
	dense_matrix<T> Bmat(B);

	dense_matrix<T> cw = L2norm_diff(colwise(A), colwise(B));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<T> cw0(1, n);
	for (index_t j = 0; j < n; ++j)
		cw0[j] = L2norm_diff(Amat.column(j), Bmat.column(j));

	ASSERT_TRUE( is_approx(cw, cw0, 1.0e-13) );

	dense_matrix<T> rw = L2norm_diff(rowwise(A), rowwise(B));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<T> rw0(m, 1);
	for (index_t i = 0; i < m; ++i)
		rw0[i] = L2norm_diff(Amat.row(i), Bmat.row(i));

	ASSERT_TRUE( is_approx(rw, rw0, 1.0e-13) );
}

template<int CTRows, int CTCols>
void test_slicewise_L2norm_diff_on_dense(index_t m, index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	dense_matrix<double, CTRows, CTCols> B(m, n);

	for (index_t i = 0; i < m * n; ++i)
	{
		A[i] = (3 * i + 2);
		B[i] = 60 - i;
	}

	test_slicewise_L2norm_diff(A, B);
}


template<int CTRows, int CTCols>
void test_slicewise_L2norm_diff_on_refex(index_t m, index_t n)
{
	index_t ldim = m + 2;
	dense_matrix<double> Amat(ldim, n);
	dense_matrix<double> Bmat(ldim, n);
	for (index_t i = 0; i < ldim * n; ++i) Amat[i] = (3 * i + 2);
	for (index_t i = 0; i < ldim * n; ++i) Bmat[i] = (60 - i);

	ref_matrix_ex<double, CTRows, CTCols> A(Amat.ptr_data(), m, n, ldim);
	ref_matrix_ex<double, CTRows, CTCols> B(Bmat.ptr_data(), m, n, ldim);
	test_slicewise_L2norm_diff(A, B);
}


TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnDenseDD )
{
	test_slicewise_L2norm_diff_on_dense<DynamicDim, DynamicDim>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnDenseDS )
{
	test_slicewise_L2norm_diff_on_dense<DynamicDim, 9>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnDenseD1 )
{
	test_slicewise_L2norm_diff_on_dense<DynamicDim, 1>(6, 1);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnDenseSD )
{
	test_slicewise_L2norm_diff_on_dense<6, DynamicDim>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnDenseSS )
{
	test_slicewise_L2norm_diff_on_dense<6, 9>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnDenseS1 )
{
	test_slicewise_L2norm_diff_on_dense<6, 1>(6, 1);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnDense1D )
{
	test_slicewise_L2norm_diff_on_dense<1, DynamicDim>(1, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnDense1S )
{
	test_slicewise_L2norm_diff_on_dense<1, 9>(1, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnDense11 )
{
	test_slicewise_L2norm_diff_on_dense<1, 1>(1, 1);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnRefExDD )
{
	test_slicewise_L2norm_diff_on_refex<DynamicDim, DynamicDim>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnRefExDS )
{
	test_slicewise_L2norm_diff_on_refex<DynamicDim, 9>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnRefExD1 )
{
	test_slicewise_L2norm_diff_on_refex<DynamicDim, 1>(6, 1);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnRefExSD )
{
	test_slicewise_L2norm_diff_on_refex<6, DynamicDim>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnRefExSS )
{
	test_slicewise_L2norm_diff_on_refex<6, 9>(6, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnRefExS1 )
{
	test_slicewise_L2norm_diff_on_refex<6, 1>(6, 1);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnRefEx1D )
{
	test_slicewise_L2norm_diff_on_refex<1, DynamicDim>(1, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnRefEx1S )
{
	test_slicewise_L2norm_diff_on_refex<1, 9>(1, 9);
}

TEST( MatrixBinaryParReduc, SliceWiseL2NormDiffOnRefEx11 )
{
	test_slicewise_L2norm_diff_on_refex<1, 1>(1, 1);
}


TEST( MatrixBinaryParReduc, SliceWiseLinfNormDiff )
{
	const index_t m = 6;
	const index_t n = 9;

	dense_matrix<double> Amat(m, n);
	dense_matrix<double> Bmat(m, n);

	for (index_t i = 0; i < m * n; ++i)
	{
		Amat[i] = (3 * i + 2);
		Bmat[i] = 60 - i;
	}

	const IMatrixXpr<dense_matrix<double>, double>& A = Amat;
	const IMatrixXpr<dense_matrix<double>, double>& B = Bmat;

	dense_matrix<double> cw = Linfnorm_diff(colwise(A), colwise(B));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<double> cw0(1, n);
	for (index_t j = 0; j < n; ++j)
		cw0[j] = Linfnorm_diff(Amat.column(j), Bmat.column(j));

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<double> rw = Linfnorm_diff(rowwise(A), rowwise(B));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<double> rw0(m, 1);
	for (index_t i = 0; i < m; ++i)
		rw0[i] = Linfnorm_diff(Amat.row(i), Bmat.row(i));

	ASSERT_TRUE( is_equal(rw, rw0) );
}




