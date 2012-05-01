/*
 * @file test_broadcast.cpp
 *
 * Test broadcasting element-wise evaluation
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;


template<int CTRows, int CTCols>
void test_bsx_addcols(index_t m, index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	dense_col<double, CTRows> b(m);

	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+2);
	for (index_t i = 0; i < m; ++i) b[i] = double(3 * i + 5);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = A(i, j) + b(i, 0);
	}

	dense_matrix<double> R = colwise(A) + b;

	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );
	ASSERT_TRUE( is_equal(R, R0) );

	dense_matrix<double> R2(A);
	colwise(R2) += b;

	ASSERT_EQ( m, R2.nrows() );
	ASSERT_EQ( n, R2.ncolumns() );
	ASSERT_TRUE( is_equal(R2, R0) );
}


TEST( MatrixBroadcast, AddColsDD  )
{
	test_bsx_addcols<DynamicDim, DynamicDim>(6, 8);
}

TEST( MatrixBroadcast, AddColsDS  )
{
	test_bsx_addcols<DynamicDim, 8>(6, 8);
}

TEST( MatrixBroadcast, AddColsD1  )
{
	test_bsx_addcols<DynamicDim, 1>(6, 1);
}

TEST( MatrixBroadcast, AddColsSD  )
{
	test_bsx_addcols<6, DynamicDim>(6, 8);
}

TEST( MatrixBroadcast, AddColsSS  )
{
	test_bsx_addcols<6, 8>(6, 8);
}

TEST( MatrixBroadcast, AddColsS1  )
{
	test_bsx_addcols<6, 1>(6, 1);
}

TEST( MatrixBroadcast, AddCols1D  )
{
	test_bsx_addcols<1, DynamicDim>(1, 8);
}

TEST( MatrixBroadcast, AddCols1S  )
{
	test_bsx_addcols<1, 8>(1, 8);
}

TEST( MatrixBroadcast, AddCols11  )
{
	test_bsx_addcols<1, 1>(1, 1);
}



template<int CTRows, int CTCols>
void test_bsx_addrows(index_t m, index_t n)
{
	dense_matrix<double, CTRows, CTCols> A(m, n);
	dense_row<double, CTCols> b(n);

	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+2);
	for (index_t i = 0; i < n; ++i) b[i] = double(3 * i + 5);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = A(i, j) + b(0, j);
	}

	dense_matrix<double> R = rowwise(A) + b;

	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );
	ASSERT_TRUE( is_equal(R, R0) );

	dense_matrix<double> R2(A);
	rowwise(R2) += b;

	ASSERT_EQ( m, R2.nrows() );
	ASSERT_EQ( n, R2.ncolumns() );
	ASSERT_TRUE( is_equal(R2, R0) );
}


TEST( MatrixBroadcast, AddRowsDD  )
{
	test_bsx_addrows<DynamicDim, DynamicDim>(6, 8);
}

TEST( MatrixBroadcast, AddRowsDS  )
{
	test_bsx_addrows<DynamicDim, 8>(6, 8);
}


TEST( MatrixBroadcast, AddRowsD1  )
{
	test_bsx_addrows<DynamicDim, 1>(6, 1);
}


TEST( MatrixBroadcast, AddRowsSD  )
{
	test_bsx_addrows<6, DynamicDim>(6, 8);
}

TEST( MatrixBroadcast, AddRowsSS  )
{
	test_bsx_addrows<6, 8>(6, 8);
}

TEST( MatrixBroadcast, AddRowsS1  )
{
	test_bsx_addrows<6, 1>(6, 1);
}

TEST( MatrixBroadcast, AddRows1D  )
{
	test_bsx_addrows<1, DynamicDim>(1, 8);
}

TEST( MatrixBroadcast, AddRows1S  )
{
	test_bsx_addrows<1, 8>(1, 8);
}

TEST( MatrixBroadcast, AddRows11  )
{
	test_bsx_addrows<1, 1>(1, 1);
}


TEST( MatrixBroadcast, SubtractCols )
{
	const index_t m = 6;
	const index_t n = 8;

	dense_matrix<double> A(m, n);
	dense_col<double> b(m);

	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+2);
	for (index_t i = 0; i < m; ++i) b[i] = double(3 * i + 5);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = A(i, j) - b(i, 0);
	}

	dense_matrix<double> R = colwise(A) - b;

	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );
	ASSERT_TRUE( is_equal(R, R0) );

	dense_matrix<double> R2(A);
	colwise(R2) -= b;

	ASSERT_EQ( m, R2.nrows() );
	ASSERT_EQ( n, R2.ncolumns() );
	ASSERT_TRUE( is_equal(R2, R0) );
}

TEST( MatrixBroadcast, SubtractRows )
{
	const index_t m = 6;
	const index_t n = 8;

	dense_matrix<double> A(m, n);
	dense_row<double> b(n);

	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+2);
	for (index_t i = 0; i < n; ++i) b[i] = double(3 * i + 5);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = A(i, j) - b(0, j);
	}

	dense_matrix<double> R = rowwise(A) - b;

	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );
	ASSERT_TRUE( is_equal(R, R0) );

	dense_matrix<double> R2(A);
	rowwise(R2) -= b;

	ASSERT_EQ( m, R2.nrows() );
	ASSERT_EQ( n, R2.ncolumns() );
	ASSERT_TRUE( is_equal(R2, R0) );
}


TEST( MatrixBroadcast, MultiplyCols )
{
	const index_t m = 6;
	const index_t n = 8;

	dense_matrix<double> A(m, n);
	dense_col<double> b(m);

	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+2);
	for (index_t i = 0; i < m; ++i) b[i] = double(3 * i + 5);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = A(i, j) * b(i, 0);
	}

	dense_matrix<double> R = colwise(A) * b;

	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );
	ASSERT_TRUE( is_equal(R, R0) );

	dense_matrix<double> R2(A);
	colwise(R2) *= b;

	ASSERT_EQ( m, R2.nrows() );
	ASSERT_EQ( n, R2.ncolumns() );
	ASSERT_TRUE( is_equal(R2, R0) );
}

TEST( MatrixBroadcast, MultiplyRows )
{
	const index_t m = 6;
	const index_t n = 8;

	dense_matrix<double> A(m, n);
	dense_row<double> b(n);

	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+2);
	for (index_t i = 0; i < n; ++i) b[i] = double(3 * i + 5);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = A(i, j) * b(0, j);
	}

	dense_matrix<double> R = rowwise(A) * b;

	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );
	ASSERT_TRUE( is_equal(R, R0) );

	dense_matrix<double> R2(A);
	rowwise(R2) *= b;

	ASSERT_EQ( m, R2.nrows() );
	ASSERT_EQ( n, R2.ncolumns() );
	ASSERT_TRUE( is_equal(R2, R0) );
}


TEST( MatrixBroadcast, DivideCols )
{
	const index_t m = 6;
	const index_t n = 8;

	dense_matrix<double> A(m, n);
	dense_col<double> b(m);

	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+2);
	for (index_t i = 0; i < m; ++i) b[i] = double(3 * i + 5);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = A(i, j) / b(i, 0);
	}

	dense_matrix<double> R = colwise(A) / b;

	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );
	ASSERT_TRUE( is_approx(R, R0, 1.0e-14) );

	dense_matrix<double> R2(A);
	colwise(R2) /= b;

	ASSERT_EQ( m, R2.nrows() );
	ASSERT_EQ( n, R2.ncolumns() );
	ASSERT_TRUE( is_approx(R2, R0, 1.0e-14) );
}

TEST( MatrixBroadcast, DivideRows )
{
	const index_t m = 6;
	const index_t n = 8;

	dense_matrix<double> A(m, n);
	dense_row<double> b(n);

	for (index_t i = 0; i < m * n; ++i) A[i] = double(i+2);
	for (index_t i = 0; i < n; ++i) b[i] = double(3 * i + 5);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = A(i, j) / b(0, j);
	}

	dense_matrix<double> R = rowwise(A) / b;

	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );
	ASSERT_TRUE( is_approx(R, R0, 1.0e-14) );

	dense_matrix<double> R2(A);
	rowwise(R2) /= b;

	ASSERT_EQ( m, R2.nrows() );
	ASSERT_EQ( n, R2.ncolumns() );
	ASSERT_TRUE( is_approx(R2, R0, 1.0e-14) );
}





