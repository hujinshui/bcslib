/**
 * @file test_dense_matrix.cpp
 *
 * Unit testing for dense matrices
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;

// explicit template for syntax check

template class bcs::cref_matrix<double, DynamicDim, DynamicDim>;
template class bcs::cref_matrix<double, DynamicDim, 1>;
template class bcs::cref_matrix<double, 1, DynamicDim>;
template class bcs::cref_matrix<double, 2, 2>;

template class bcs::ref_matrix<double, DynamicDim, DynamicDim>;
template class bcs::ref_matrix<double, DynamicDim, 1>;
template class bcs::ref_matrix<double, 1, DynamicDim>;
template class bcs::ref_matrix<double, 2, 2>;

#ifdef BCS_USE_STATIC_ASSERT
static_assert(bcs::is_base_of<
		bcs::IMatrixXpr<bcs::cref_matrix<double>, double>,
		bcs::cref_matrix<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IMatrixView<bcs::cref_matrix<double>, double>,
		bcs::cref_matrix<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IDenseMatrix<bcs::cref_matrix<double>, double>,
		bcs::cref_matrix<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IMatrixXpr<bcs::ref_matrix<double>, double>,
		bcs::ref_matrix<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IMatrixView<bcs::ref_matrix<double>, double>,
		bcs::ref_matrix<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IDenseMatrix<bcs::ref_matrix<double>, double>,
		bcs::ref_matrix<double> >::value, "Base verification failed.");
#endif


template<class Mat, typename T>
void do_test_ref_matrix(Mat& a1, index_t m, index_t n, const T *origin)
{
	ASSERT_EQ(m, a1.nrows());
	ASSERT_EQ(n, a1.ncolumns());
	ASSERT_EQ(m * n, a1.nelems());
	ASSERT_EQ(m, a1.lead_dim());
	ASSERT_TRUE( a1.ptr_data() == origin );

	ASSERT_EQ( (size_t)a1.nelems(), a1.size() );
	ASSERT_EQ(a1.nelems() == 0, is_empty(a1));
	ASSERT_EQ(a1.nrows() == 1, is_row(a1));
	ASSERT_EQ(a1.ncolumns() == 1, is_column(a1));
	ASSERT_EQ(a1.nelems() == 1, is_scalar(a1));
	ASSERT_EQ(a1.nrows() == 1 || a1.ncolumns() == 1, is_vector(a1));
	ASSERT_EQ(a1.nrows() == 1 && a1.ncolumns() == 1, is_scalar(a1));

	// test element access

	bool elem_access_ok = true;

	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i)
		{
			if (a1(i, j) != origin[i + j * m]) elem_access_ok = false;
			if (a1[i + j * m] != origin[i + j * m]) elem_access_ok = false;
		}
	}

	ASSERT_TRUE( elem_access_ok );

	// test slice pointers

	bool colptr_ok = true;
	for (index_t j = 0; j < n; ++j)
	{
		if (col_ptr(a1, j) != origin + j * m) colptr_ok = false;
	}
	ASSERT_TRUE( colptr_ok );

	bool rowptr_ok = true;
	for (index_t i = 0; i < m; ++i)
	{
		if (row_ptr(a1, i) != origin + i) rowptr_ok = false;
	}
	ASSERT_TRUE( rowptr_ok );

	// test copy construction

	Mat a2(a1);

	ASSERT_EQ(m, a2.nrows());
	ASSERT_EQ(n, a2.ncolumns());
	ASSERT_EQ(m * n, a2.nelems());
	ASSERT_EQ(m, a2.lead_dim());
	ASSERT_TRUE( a2.ptr_data() == a1.ptr_data() );
}



template<class Mat>
static void test_ref_matrix(index_t m, index_t n)
{
	scoped_block<double> origin_blk(m * n);
	for (index_t i = 0; i < m * n; ++i) origin_blk[i] = double(i+1);
	double *origin = origin_blk.ptr_begin();

	// test construction

	Mat a1(origin, m, n);
	do_test_ref_matrix(a1, m, n, origin);
}


template<class Vec>
static void test_ref_vector(index_t m, index_t n)
{
	scoped_block<double> origin_blk(m * n);
	for (index_t i = 0; i < m * n; ++i) origin_blk[i] = double(i+1);
	double *origin = origin_blk.ptr_begin();

	// test construction

	index_t len = (m > 1 ? m : n);

	Vec a1(origin, len);
	do_test_ref_matrix(a1, m, n, origin);
}



template<class Mat>
static void test_ref_matrix_ex(index_t m, index_t n, index_t ldim)
{
	const int CTRows = matrix_traits<Mat>::compile_time_num_rows;
	const int CTCols = matrix_traits<Mat>::compile_time_num_cols;


	scoped_block<double> origin_blk(ldim * n);
	for (index_t i = 0; i < ldim * n; ++i) origin_blk[i] = double(i+1);
	double *origin = origin_blk.ptr_begin();

	// test construction

	Mat a1(origin, m, n, ldim);

	ASSERT_EQ(m, a1.nrows());
	ASSERT_EQ(n, a1.ncolumns());
	ASSERT_EQ(m * n, a1.nelems());
	ASSERT_EQ(ldim, a1.lead_dim());
	ASSERT_TRUE( a1.ptr_data() == origin );

	ASSERT_EQ( (size_t)a1.nelems(), a1.size() );
	ASSERT_EQ(a1.nelems() == 0, is_empty(a1));
	ASSERT_EQ(a1.nrows() == 1, is_row(a1));
	ASSERT_EQ(a1.ncolumns() == 1, is_column(a1));
	ASSERT_EQ(a1.nelems() == 1, is_scalar(a1));
	ASSERT_EQ(a1.nrows() == 1 || a1.ncolumns() == 1, is_vector(a1));
	ASSERT_EQ(a1.nrows() == 1 && a1.ncolumns() == 1, is_scalar(a1));

	// test element access

	bool elem_access_ok = true;

	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i)
		{
			if (a1(i, j) != origin[i + j * ldim]) elem_access_ok = false;
		}
	}

	ASSERT_TRUE( elem_access_ok );

	if (CTCols == 1)
	{
		bool linear_indexing_ok = true;

		for (index_t i = 0; i < m; ++i)
		{
			if (a1[i] != origin[i]) linear_indexing_ok = false;
		}

		ASSERT_TRUE(linear_indexing_ok);
	}

	if (CTRows == 1)
	{
		bool linear_indexing_ok = true;

		for (index_t i = 0; i < n; ++i)
		{
			if (a1[i] != origin[i * ldim]) linear_indexing_ok = false;
		}

		ASSERT_TRUE(linear_indexing_ok);
	}

	// test slice pointers

	bool colptr_ok = true;
	for (index_t j = 0; j < n; ++j)
	{
		if (col_ptr(a1, j) != origin + j * ldim) colptr_ok = false;
	}
	ASSERT_TRUE( colptr_ok );

	bool rowptr_ok = true;
	for (index_t i = 0; i < m; ++i)
	{
		if (row_ptr(a1, i) != origin + i) rowptr_ok = false;
	}
	ASSERT_TRUE( rowptr_ok );

	// test copy construction

	Mat a2(a1);

	ASSERT_EQ(m, a2.nrows());
	ASSERT_EQ(n, a2.ncolumns());
	ASSERT_EQ(m * n, a2.nelems());
	ASSERT_EQ(ldim, a2.lead_dim());
	ASSERT_TRUE( a2.ptr_data() == a1.ptr_data() );
}




TEST( RefMatrix, CDD )
{
	test_ref_matrix<cref_matrix<double, DynamicDim, DynamicDim> >(3, 4);
}

TEST( RefMatrix, WDD )
{
	test_ref_matrix<ref_matrix<double, DynamicDim, DynamicDim> >(3, 4);
}

TEST( RefMatrix, CDS )
{
	test_ref_matrix<cref_matrix<double, DynamicDim, 4> >(3, 4);
}

TEST( RefMatrix, WDS )
{
	test_ref_matrix<ref_matrix<double, DynamicDim, 4> >(3, 4);
}

TEST( RefMatrix, CD1 )
{
	test_ref_matrix<cref_matrix<double, DynamicDim, 1> >(3, 1);
}

TEST( RefMatrix, WD1 )
{
	test_ref_matrix<ref_matrix<double, DynamicDim, 1> >(3, 1);
}


TEST( RefMatrix, CSD )
{
	test_ref_matrix<cref_matrix<double, 3, DynamicDim> >(3, 4);
}

TEST( RefMatrix, WSD )
{
	test_ref_matrix<ref_matrix<double, 3, DynamicDim> >(3, 4);
}

TEST( RefMatrix, CSS )
{
	test_ref_matrix<cref_matrix<double, 3, 4> >(3, 4);
}

TEST( RefMatrix, WSS )
{
	test_ref_matrix<ref_matrix<double, 3, 4> >(3, 4);
}

TEST( RefMatrix, CS1 )
{
	test_ref_matrix<cref_matrix<double, 3, 1> >(3, 1);
}

TEST( RefMatrix, WS1 )
{
	test_ref_matrix<ref_matrix<double, 3, 1> >(3, 1);
}

TEST( RefMatrix, C1D )
{
	test_ref_matrix<cref_matrix<double, 1, DynamicDim> >(1, 4);
}

TEST( RefMatrix, W1D )
{
	test_ref_matrix<ref_matrix<double, 1, DynamicDim> >(1, 4);
}

TEST( RefMatrix, C1S )
{
	test_ref_matrix<cref_matrix<double, 1, 4> >(1, 4);
}

TEST( RefMatrix, W1S )
{
	test_ref_matrix<ref_matrix<double, 1, 4> >(1, 4);
}

TEST( RefMatrix, C11 )
{
	test_ref_matrix<cref_matrix<double, 1, 1> >(1, 1);
}

TEST( RefMatrix, W11 )
{
	test_ref_matrix<ref_matrix<double, 1, 1> >(1, 1);
}





TEST( RefVector, ConstColDyn )
{
	test_ref_vector<cref_col<double, DynamicDim> >(5, 1);
}

TEST( RefVector, ColDyn )
{
	test_ref_vector<ref_col<double, DynamicDim> >(5, 1);
}

TEST( RefVector, ConstColSta )
{
	test_ref_vector<cref_col<double, 5> >(5, 1);
}

TEST( RefVector, ColSta )
{
	test_ref_vector<ref_col<double, 5> >(5, 1);
}

TEST( RefVector, ConstRowDyn )
{
	test_ref_vector<cref_row<double, DynamicDim> >(1, 5);
}

TEST( RefVector, RowDyn )
{
	test_ref_vector<ref_row<double, DynamicDim> >(1, 5);
}

TEST( RefVector, ConstRowSta )
{
	test_ref_vector<cref_row<double, 5> >(1, 5);
}

TEST( RefVector, RowSta )
{
	test_ref_vector<ref_row<double, 5> >(1, 5);
}




TEST( RefMatrixEx, CDD )
{
	test_ref_matrix_ex<cref_matrix_ex<double, DynamicDim, DynamicDim> >(3, 4, 7);
}

TEST( RefMatrixEx, WDD )
{
	test_ref_matrix_ex<ref_matrix_ex<double, DynamicDim, DynamicDim> >(3, 4, 7);
}

TEST( RefMatrixEx, CDS )
{
	test_ref_matrix_ex<cref_matrix_ex<double, DynamicDim, 4> >(3, 4, 7);
}

TEST( RefMatrixEx, WDS )
{
	test_ref_matrix_ex<ref_matrix_ex<double, DynamicDim, 4> >(3, 4, 7);
}

TEST( RefMatrixEx, CD1 )
{
	test_ref_matrix_ex<cref_matrix_ex<double, DynamicDim, 1> >(3, 1, 7);
}

TEST( RefMatrixEx, WD1 )
{
	test_ref_matrix_ex<ref_matrix_ex<double, DynamicDim, 1> >(3, 1, 7);
}

TEST( RefMatrixEx, CSD )
{
	test_ref_matrix_ex<cref_matrix_ex<double, 3, DynamicDim> >(3, 4, 7);
}

TEST( RefMatrixEx, WSD )
{
	test_ref_matrix_ex<ref_matrix_ex<double, 3, DynamicDim> >(3, 4, 7);
}

TEST( RefMatrixEx, CSS )
{
	test_ref_matrix_ex<cref_matrix_ex<double, 3, 4> >(3, 4, 7);
}

TEST( RefMatrixEx, WSS )
{
	test_ref_matrix_ex<ref_matrix_ex<double, 3, 4> >(3, 4, 7);
}

TEST( RefMatrixEx, CS1 )
{
	test_ref_matrix_ex<cref_matrix_ex<double, 3, 1> >(3, 1, 7);
}

TEST( RefMatrixEx, WS1 )
{
	test_ref_matrix_ex<ref_matrix_ex<double, 3, 1> >(3, 1, 7);
}

TEST( RefMatrixEx, C1D )
{
	test_ref_matrix_ex<cref_matrix_ex<double, 1, DynamicDim> >(1, 4, 7);
}

TEST( RefMatrixEx, W1D )
{
	test_ref_matrix_ex<ref_matrix_ex<double, 1, DynamicDim> >(1, 4, 7);
}

TEST( RefMatrixEx, C1S )
{
	test_ref_matrix_ex<cref_matrix_ex<double, 1, 4> >(1, 4, 7);
}

TEST( RefMatrixEx, W1S )
{
	test_ref_matrix_ex<ref_matrix_ex<double, 1, 4> >(1, 4, 7);
}

TEST( RefMatrixEx, C11 )
{
	test_ref_matrix_ex<cref_matrix_ex<double, 1, 1> >(1, 1, 7);
}

TEST( RefMatrixEx, W11 )
{
	test_ref_matrix_ex<ref_matrix_ex<double, 1, 1> >(1, 1, 7);
}





