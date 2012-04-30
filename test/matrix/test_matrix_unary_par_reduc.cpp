/**
 * @file test_matrix_unary_par_reduc.cpp
 *
 * Unit testing for unary partial reduction on matrix
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;


template<typename T, class Mat>
void test_slicewise_sum(const IMatrixXpr<Mat, T>& A)
{
	dense_matrix<T> Amat(A);
	index_t m = A.nrows();
	index_t n = A.ncolumns();

	dense_matrix<T> cw = sum(colwise(A));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<T> cw0(1, n);
	for (index_t j = 0; j < n; ++j)
	{
		T s(0);
		for (index_t i = 0; i < m; ++i) s += Amat(i, j);
		cw0[j] = s;
	}

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<T> rw = sum(rowwise(A));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<T> rw0(m, 1);
	for (index_t i = 0; i < m; ++i)
	{
		T s(0);
		for (index_t j = 0; j < n; ++j) s += Amat(i, j);
		rw0[i] = s;
	}

	ASSERT_TRUE( is_equal(rw, rw0) );
}


template<int CTRows, int CTCols>
void test_slicewise_sum_on_dense(index_t m, index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	for (index_t i = 0; i < m * n; ++i) A[i] = double(3 * i + 2);

	test_slicewise_sum(A);
}


template<int CTRows, int CTCols>
void test_slicewise_sum_on_refex(index_t m, index_t n)
{
	index_t ldim = m + 2;
	dense_matrix<double> Amat(ldim, n);
	for (index_t i = 0; i < ldim * n; ++i) Amat[i] = double(3 * i + 2);

	ref_matrix_ex<double, CTRows, CTCols> A(Amat.ptr_data(), m, n, ldim);
	test_slicewise_sum(A);
}


TEST( MatrixUnaryParReduc, SliceWiseSumOnDenseDD )
{
	test_slicewise_sum_on_dense<DynamicDim, DynamicDim>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnDenseDS )
{
	test_slicewise_sum_on_dense<DynamicDim, 9>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnDenseD1 )
{
	test_slicewise_sum_on_dense<DynamicDim, 1>(6, 1);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnDenseSD )
{
	test_slicewise_sum_on_dense<6, DynamicDim>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnDenseSS )
{
	test_slicewise_sum_on_dense<6, 9>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnDenseS1 )
{
	test_slicewise_sum_on_dense<6, 1>(6, 1);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnDense1D )
{
	test_slicewise_sum_on_dense<1, DynamicDim>(1, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnDense1S )
{
	test_slicewise_sum_on_dense<1, 9>(1, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnDense11 )
{
	test_slicewise_sum_on_dense<1, 1>(1, 1);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnRefExDD )
{
	test_slicewise_sum_on_refex<DynamicDim, DynamicDim>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnRefExDS )
{
	test_slicewise_sum_on_refex<DynamicDim, 9>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnRefExD1 )
{
	test_slicewise_sum_on_refex<DynamicDim, 1>(6, 1);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnRefExSD )
{
	test_slicewise_sum_on_refex<6, DynamicDim>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnRefExSS )
{
	test_slicewise_sum_on_refex<6, 9>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnRefExS1 )
{
	test_slicewise_sum_on_refex<6, 1>(6, 1);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnRefEx1D )
{
	test_slicewise_sum_on_refex<1, DynamicDim>(1, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnRefEx1S )
{
	test_slicewise_sum_on_refex<1, 9>(1, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseSumOnRefEx11 )
{
	test_slicewise_sum_on_refex<1, 1>(1, 1);
}


template<typename T, class Mat>
void test_slicewise_mean(const IMatrixXpr<Mat, T>& A)
{
	dense_matrix<T> Amat(A);
	index_t m = A.nrows();
	index_t n = A.ncolumns();

	dense_matrix<T> cw = mean(colwise(A));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<T> cw0(1, n);
	for (index_t j = 0; j < n; ++j)
	{
		T s(0);
		for (index_t i = 0; i < m; ++i) s += Amat(i, j);
		cw0[j] = s / T(m);
	}

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<T> rw = mean(rowwise(A));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<T> rw0(m, 1);
	for (index_t i = 0; i < m; ++i)
	{
		T s(0);
		for (index_t j = 0; j < n; ++j) s += Amat(i, j);
		rw0[i] = s / T(n);
	}

	ASSERT_TRUE( is_equal(rw, rw0) );
}


template<int CTRows, int CTCols>
void test_slicewise_mean_on_dense(index_t m, index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	for (index_t i = 0; i < m * n; ++i) A[i] = double(3 * i + 2);

	test_slicewise_mean(A);
}


TEST( MatrixUnaryParReduc, SliceWiseMeanOnDenseDD )
{
	test_slicewise_mean_on_dense<DynamicDim, DynamicDim>(4, 8);
}

TEST( MatrixUnaryParReduc, SliceWiseMeanOnDenseDS )
{
	test_slicewise_mean_on_dense<DynamicDim, 8>(4, 8);
}

TEST( MatrixUnaryParReduc, SliceWiseMeanOnDenseD1 )
{
	test_slicewise_mean_on_dense<DynamicDim, 1>(4, 1);
}

TEST( MatrixUnaryParReduc, SliceWiseMeanOnDenseSD )
{
	test_slicewise_mean_on_dense<4, DynamicDim>(4, 8);
}

TEST( MatrixUnaryParReduc, SliceWiseMeanOnDenseSS )
{
	test_slicewise_mean_on_dense<4, 8>(4, 8);
}

TEST( MatrixUnaryParReduc, SliceWiseMeanOnDenseS1 )
{
	test_slicewise_mean_on_dense<4, 1>(4, 1);
}

TEST( MatrixUnaryParReduc, SliceWiseMeanOnDense1D )
{
	test_slicewise_mean_on_dense<1, DynamicDim>(1, 8);
}

TEST( MatrixUnaryParReduc, SliceWiseMeanOnDense1S )
{
	test_slicewise_mean_on_dense<1, 8>(1, 8);
}

TEST( MatrixUnaryParReduc, SliceWiseMeanOnDense11 )
{
	test_slicewise_mean_on_dense<1, 1>(1, 1);
}



TEST( MatrixUnaryParReduc, SliceWiseMin )
{
	const index_t m = 6;
	const index_t n = 9;

	dense_matrix<double> Amat(m, n);
	for (index_t i = 0; i < m * n; ++i) Amat[i] = double(3 * i + 2);

	const IMatrixXpr<dense_matrix<double>, double>& A = Amat;

	dense_matrix<double> cw = min_val(colwise(A));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<double> cw0(1, n);
	for (index_t j = 0; j < n; ++j) cw0[j] = min_val(Amat.column(j));

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<double> rw = min_val(rowwise(A));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<double> rw0(m, 1);
	for (index_t i = 0; i < m; ++i) rw0[i] = min_val(Amat.row(i));

	ASSERT_TRUE( is_equal(rw, rw0) );
}


TEST( MatrixUnaryParReduc, SliceWiseMax )
{
	const index_t m = 6;
	const index_t n = 9;

	dense_matrix<double> Amat(m, n);
	for (index_t i = 0; i < m * n; ++i) Amat[i] = double(3 * i + 2);

	const IMatrixXpr<dense_matrix<double>, double>& A = Amat;

	dense_matrix<double> cw = max_val(colwise(A));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<double> cw0(1, n);
	for (index_t j = 0; j < n; ++j) cw0[j] = max_val(Amat.column(j));

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<double> rw = max_val(rowwise(A));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<double> rw0(m, 1);
	for (index_t i = 0; i < m; ++i) rw0[i] = max_val(Amat.row(i));

	ASSERT_TRUE( is_equal(rw, rw0) );
}


TEST( MatrixUnaryParReduc, SliceWiseL1Norm )
{
	const index_t m = 6;
	const index_t n = 9;

	dense_matrix<double> Amat(m, n);
	for (index_t i = 0; i < m * n; ++i) Amat[i] = double(3 * i + 2);

	const IMatrixXpr<dense_matrix<double>, double>& A = Amat;

	dense_matrix<double> cw = L1norm(colwise(A));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<double> cw0(1, n);
	for (index_t j = 0; j < n; ++j) cw0[j] = L1norm(Amat.column(j));

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<double> rw = L1norm(rowwise(A));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<double> rw0(m, 1);
	for (index_t i = 0; i < m; ++i) rw0[i] = L1norm(Amat.row(i));

	ASSERT_TRUE( is_equal(rw, rw0) );
}


TEST( MatrixUnaryParReduc, SliceWiseSqL2Norm )
{
	const index_t m = 6;
	const index_t n = 9;

	dense_matrix<double> Amat(m, n);
	for (index_t i = 0; i < m * n; ++i) Amat[i] = double(3 * i + 2);

	const IMatrixXpr<dense_matrix<double>, double>& A = Amat;

	dense_matrix<double> cw = sqL2norm(colwise(A));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<double> cw0(1, n);
	for (index_t j = 0; j < n; ++j) cw0[j] = sqL2norm(Amat.column(j));

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<double> rw = sqL2norm(rowwise(A));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<double> rw0(m, 1);
	for (index_t i = 0; i < m; ++i) rw0[i] = sqL2norm(Amat.row(i));

	ASSERT_TRUE( is_equal(rw, rw0) );
}


template<typename T, class Mat>
void test_slicewise_L2norm(const IMatrixXpr<Mat, T>& A)
{
	const index_t m = A.nrows();
	const index_t n = A.ncolumns();

	dense_matrix<T> Amat(A);

	dense_matrix<T> cw = L2norm(colwise(A));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<T> cw0(1, n);
	for (index_t j = 0; j < n; ++j) cw0[j] = L2norm(Amat.column(j));

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<T> rw = L2norm(rowwise(A));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<T> rw0(m, 1);
	for (index_t i = 0; i < m; ++i) rw0[i] = L2norm(Amat.row(i));

	ASSERT_TRUE( is_approx(rw, rw0, 1.0e-13) );
}

template<int CTRows, int CTCols>
void test_slicewise_L2norm_on_dense(index_t m, index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	for (index_t i = 0; i < m * n; ++i) A[i] = double(3 * i + 2);

	test_slicewise_L2norm(A);
}


template<int CTRows, int CTCols>
void test_slicewise_L2norm_on_refex(index_t m, index_t n)
{
	index_t ldim = m + 2;
	dense_matrix<double> Amat(ldim, n);
	for (index_t i = 0; i < ldim * n; ++i) Amat[i] = double(3 * i + 2);

	ref_matrix_ex<double, CTRows, CTCols> A(Amat.ptr_data(), m, n, ldim);
	test_slicewise_L2norm(A);
}


TEST( MatrixUnaryParReduc, SliceWiseL2NormOnDenseDD )
{
	test_slicewise_L2norm_on_dense<DynamicDim, DynamicDim>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnDenseDS )
{
	test_slicewise_L2norm_on_dense<DynamicDim, 9>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnDenseD1 )
{
	test_slicewise_L2norm_on_dense<DynamicDim, 1>(6, 1);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnDenseSD )
{
	test_slicewise_L2norm_on_dense<6, DynamicDim>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnDenseSS )
{
	test_slicewise_L2norm_on_dense<6, 9>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnDenseS1 )
{
	test_slicewise_L2norm_on_dense<6, 1>(6, 1);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnDense1D )
{
	test_slicewise_L2norm_on_dense<1, DynamicDim>(1, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnDense1S )
{
	test_slicewise_L2norm_on_dense<1, 9>(1, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnDense11 )
{
	test_slicewise_L2norm_on_dense<1, 1>(1, 1);
}


TEST( MatrixUnaryParReduc, SliceWiseL2NormOnRefExDD )
{
	test_slicewise_L2norm_on_refex<DynamicDim, DynamicDim>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnRefExDS )
{
	test_slicewise_L2norm_on_refex<DynamicDim, 9>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnRefExD1 )
{
	test_slicewise_L2norm_on_refex<DynamicDim, 1>(6, 1);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnRefExSD )
{
	test_slicewise_L2norm_on_refex<6, DynamicDim>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnRefExSS )
{
	test_slicewise_L2norm_on_refex<6, 9>(6, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnRefExS1 )
{
	test_slicewise_L2norm_on_refex<6, 1>(6, 1);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnRefEx1D )
{
	test_slicewise_L2norm_on_refex<1, DynamicDim>(1, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnRefEx1S )
{
	test_slicewise_L2norm_on_refex<1, 9>(1, 9);
}

TEST( MatrixUnaryParReduc, SliceWiseL2NormOnRefEx11 )
{
	test_slicewise_L2norm_on_refex<1, 1>(1, 1);
}


TEST( MatrixUnaryParReduc, SliceWiseLinfNorm )
{
	const index_t m = 6;
	const index_t n = 9;

	dense_matrix<double> Amat(m, n);
	for (index_t i = 0; i < m * n; ++i)
		Amat[i] = double(3 * i + 2) * (i % 2 == 0 ? 1.0 : -1.0);

	const IMatrixXpr<dense_matrix<double>, double>& A = Amat;

	dense_matrix<double> cw = Linfnorm(colwise(A));

	ASSERT_EQ( 1, cw.nrows() );
	ASSERT_EQ( n, cw.ncolumns() );

	dense_matrix<double> cw0(1, n);
	for (index_t j = 0; j < n; ++j) cw0[j] = Linfnorm(Amat.column(j));

	ASSERT_TRUE( is_equal(cw, cw0) );

	dense_matrix<double> rw = Linfnorm(rowwise(A));

	ASSERT_EQ( m, rw.nrows() );
	ASSERT_EQ( 1, rw.ncolumns() );

	dense_matrix<double> rw0(m, 1);
	for (index_t i = 0; i < m; ++i) rw0[i] = Linfnorm(Amat.row(i));

	ASSERT_TRUE( is_equal(rw, rw0) );
}




