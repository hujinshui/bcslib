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

template class bcs::ref_matrix<double, DynamicDim, DynamicDim>;
template class bcs::ref_matrix<double, DynamicDim, 1>;
template class bcs::ref_matrix<double, 1, DynamicDim>;
template class bcs::ref_matrix<double, 2, 2>;

#ifdef BCS_USE_STATIC_ASSERT
static_assert(bcs::is_base_of<
		bcs::IMatrixXpr<bcs::ref_matrix<double>, double>,
		bcs::ref_matrix<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IMatrixView<bcs::ref_matrix<double>, double>,
		bcs::ref_matrix<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IRegularMatrix<bcs::ref_matrix<double>, double>,
		bcs::ref_matrix<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IDenseMatrix<bcs::ref_matrix<double>, double>,
		bcs::ref_matrix<double> >::value, "Base verification failed.");
#endif


template<class Mat>
static void test_ref_matrix(index_t m, index_t n)
{
	scoped_block<double> origin_blk(m * n);
	for (index_t i = 0; i < m * n; ++i) origin_blk[i] = (i+1);
	double *origin = origin_blk.ptr_begin();

	// test construction

	Mat a1(origin, m, n);

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

	// test assignment

	Mat a3(BCS_NULL, matrix_traits<Mat>::compile_time_num_rows, matrix_traits<Mat>::compile_time_num_cols);

	a3 = a1;

	ASSERT_EQ(m, a3.nrows());
	ASSERT_EQ(n, a3.ncolumns());
	ASSERT_EQ(m * n, a3.nelems());
	ASSERT_EQ(m, a3.lead_dim());
	ASSERT_TRUE( a3.ptr_data() == a1.ptr_data() );
}


TEST( RefMatrix, ConstDRowDCol )
{
	test_ref_matrix<cref_matrix<double, DynamicDim, DynamicDim> >(3, 4);
}

TEST( RefMatrix, DRowDCol )
{
	test_ref_matrix<ref_matrix<double, DynamicDim, DynamicDim> >(3, 4);
}

TEST( RefMatrix, ConstDRowSCol )
{
	test_ref_matrix<cref_matrix<double, DynamicDim, 4> >(3, 4);
}

TEST( RefMatrix, DRowSCol )
{
	test_ref_matrix<ref_matrix<double, DynamicDim, 4> >(3, 4);
}

TEST( RefMatrix, ConstSRowDCol )
{
	test_ref_matrix<cref_matrix<double, 3, DynamicDim> >(3, 4);
}

TEST( RefMatrix, SRowDCol )
{
	test_ref_matrix<ref_matrix<double, 3, DynamicDim> >(3, 4);
}

TEST( RefMatrix, ConstSRowSCol )
{
	test_ref_matrix<cref_matrix<double, 3, 4> >(3, 4);
}

TEST( RefMatrix, SRowSCol )
{
	test_ref_matrix<ref_matrix<double, 3, 4> >(3, 4);
}


