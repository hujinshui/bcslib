/*
 * @file test_repeat_vecs.cpp
 *
 * Test of repeat_cols and repeat_rows
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;

template<int CTRows, int CTCols>
void test_repeat_dense_cols(index_t m, index_t n)
{
	dense_col<double, CTRows> col(m);
	for (index_t i = 0; i < m; ++i) col[i] = double(i+2);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = col(i, 0);
	}

	dense_matrix<double, CTRows, CTCols> R;

	if (CTCols == DynamicDim)
	{
		R = repeat_cols(col, n);
	}
	else
	{
		R = repcols<CTCols>::of(col);
	}

	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );

	ASSERT_TRUE( is_equal(R, R0) );
}


TEST( RepeatVectors, RepDenseColsDD )
{
	test_repeat_dense_cols<DynamicDim, DynamicDim>(6, 5);
}

TEST( RepeatVectors, RepDenseColsDS )
{
	test_repeat_dense_cols<DynamicDim, 5>(6, 5);
}

TEST( RepeatVectors, RepDenseColsD1 )
{
	test_repeat_dense_cols<DynamicDim, 1>(6, 1);
}

TEST( RepeatVectors, RepDenseColsSD )
{
	test_repeat_dense_cols<6, DynamicDim>(6, 5);
}

TEST( RepeatVectors, RepDenseColsSS )
{
	test_repeat_dense_cols<6, 5>(6, 5);
}

TEST( RepeatVectors, RepDenseColsS1 )
{
	test_repeat_dense_cols<6, 1>(6, 1);
}

TEST( RepeatVectors, RepDenseCols1D )
{
	test_repeat_dense_cols<1, DynamicDim>(1, 5);
}

TEST( RepeatVectors, RepDenseCols1S )
{
	test_repeat_dense_cols<1, 5>(1, 5);
}

TEST( RepeatVectors, RepDenseCols11 )
{
	test_repeat_dense_cols<1, 1>(1, 1);
}


template<int CTRows, int CTCols>
void test_repeat_dense_rows(index_t m, index_t n)
{
	dense_row<double, CTCols> row(n);
	for (index_t j = 0; j < n; ++j) row[j] = double(j+2);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = row(0, j);
	}

	dense_matrix<double, CTRows, CTCols> R;

	if (CTRows == DynamicDim)
	{
		R = repeat_rows(row, m);
	}
	else
	{
		R = reprows<CTRows>::of(row);
	}


	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );

	ASSERT_TRUE( is_equal(R, R0) );
}


TEST( RepeatVectors, RepDenseRowsDD )
{
	test_repeat_dense_rows<DynamicDim, DynamicDim>(6, 5);
}

TEST( RepeatVectors, RepDenseRowsDS )
{
	test_repeat_dense_rows<DynamicDim, 5>(6, 5);
}

TEST( RepeatVectors, RepDenseRowsD1 )
{
	test_repeat_dense_rows<DynamicDim, 1>(6, 1);
}

TEST( RepeatVectors, RepDenseRowsSD )
{
	test_repeat_dense_rows<6, DynamicDim>(6, 5);
}

TEST( RepeatVectors, RepDenseRowsSS )
{
	test_repeat_dense_rows<6, 5>(6, 5);
}

TEST( RepeatVectors, RepDenseRowsS1 )
{
	test_repeat_dense_rows<6, 1>(6, 1);
}

TEST( RepeatVectors, RepDenseRows1D )
{
	test_repeat_dense_rows<1, DynamicDim>(1, 5);
}

TEST( RepeatVectors, RepDenseRows1S )
{
	test_repeat_dense_rows<1, 5>(1, 5);
}

TEST( RepeatVectors, RepDenseRows11 )
{
	test_repeat_dense_rows<1, 1>(1, 1);
}


template<int CTRows, int CTCols>
void test_repeat_refex_cols(index_t m, index_t n)
{
	const index_t ldim = m + 3;

	dense_col<double> col_(ldim);
	for (index_t i = 0; i < ldim; ++i) col_[i] = double(i+2);
	ref_matrix_ex<double, CTRows, 1> col(col_.ptr_data(), m, 1, ldim);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = col(i, 0);
	}

	dense_matrix<double> R_(ldim, n, 0.0);
	ref_matrix_ex<double, CTRows, CTCols> R(R_.ptr_data(), m, n, ldim);

	if (CTCols == DynamicDim)
	{
		R = repeat_cols(col, n);
	}
	else
	{
		R = repcols<CTCols>::of(col);
	}

	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );

	ASSERT_TRUE( is_equal(R, R0) );
}


TEST( RepeatVectors, RepRefExColsDD )
{
	test_repeat_refex_cols<DynamicDim, DynamicDim>(6, 5);
}

TEST( RepeatVectors, RepRefExColsDS )
{
	test_repeat_refex_cols<DynamicDim, 5>(6, 5);
}

TEST( RepeatVectors, RepRefExColsD1 )
{
	test_repeat_refex_cols<DynamicDim, 1>(6, 1);
}

TEST( RepeatVectors, RepRefExColsSD )
{
	test_repeat_refex_cols<6, DynamicDim>(6, 5);
}

TEST( RepeatVectors, RepRefExColsSS )
{
	test_repeat_refex_cols<6, 5>(6, 5);
}

TEST( RepeatVectors, RepRefExColsS1 )
{
	test_repeat_refex_cols<6, 1>(6, 1);
}

TEST( RepeatVectors, RepRefExCols1D )
{
	test_repeat_refex_cols<1, DynamicDim>(1, 5);
}

TEST( RepeatVectors, RepRefExCols1S )
{
	test_repeat_refex_cols<1, 5>(1, 5);
}

TEST( RepeatVectors, RepRefExCols11 )
{
	test_repeat_refex_cols<1, 1>(1, 1);
}


template<int CTRows, int CTCols>
void test_repeat_refex_rows(index_t m, index_t n)
{
	const index_t ldim = m + 3;

	dense_matrix<double> rmat_(ldim, n);
	for (index_t i = 0; i < ldim * n; ++i) rmat_[i] = double(i+2);
	ref_matrix_ex<double, 1, CTCols> row(rmat_.ptr_data(), 1, n, ldim);

	dense_matrix<double> R0(m, n);
	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i) R0(i, j) = row(0, j);
	}

	dense_matrix<double> R_(ldim, n, 0.0);
	ref_matrix_ex<double, CTRows, CTCols> R(R_.ptr_data(), m, n, ldim);

	if (CTRows == DynamicDim)
	{
		R = repeat_rows(row, m);
	}
	else
	{
		R = reprows<CTRows>::of(row);
	}

	ASSERT_EQ( m, R.nrows() );
	ASSERT_EQ( n, R.ncolumns() );

	ASSERT_TRUE( is_equal(R, R0) );
}


TEST( RepeatVectors, RepRefExRowsDD )
{
	test_repeat_refex_rows<DynamicDim, DynamicDim>(6, 5);
}

TEST( RepeatVectors, RepRefExRowsDS )
{
	test_repeat_refex_rows<DynamicDim, 5>(6, 5);
}

TEST( RepeatVectors, RepRefExRowsD1 )
{
	test_repeat_refex_rows<DynamicDim, 1>(6, 1);
}

TEST( RepeatVectors, RepRefExRowsSD )
{
	test_repeat_refex_rows<6, DynamicDim>(6, 5);
}

TEST( RepeatVectors, RepRefExRowsSS )
{
	test_repeat_refex_rows<6, 5>(6, 5);
}

TEST( RepeatVectors, RepRefExRowsS1 )
{
	test_repeat_refex_rows<6, 1>(6, 1);
}

TEST( RepeatVectors, RepRefExRows1D )
{
	test_repeat_refex_rows<1, DynamicDim>(1, 5);
}

TEST( RepeatVectors, RepRefExRows1S )
{
	test_repeat_refex_rows<1, 5>(1, 5);
}

TEST( RepeatVectors, RepRefExRows11 )
{
	test_repeat_refex_rows<1, 1>(1, 1);
}


