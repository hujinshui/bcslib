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

template class bcs::dense_matrix<double, DynamicDim, DynamicDim>;
template class bcs::dense_matrix<double, DynamicDim, 1>;
template class bcs::dense_matrix<double, 1, DynamicDim>;
template class bcs::dense_matrix<double, 2, 2>;

template class bcs::dense_col<double, DynamicDim>;
template class bcs::dense_col<double, 3>;
template class bcs::dense_row<double, DynamicDim>;
template class bcs::dense_row<double, 3>;

#ifdef BCS_USE_STATIC_ASSERT
static_assert(sizeof(bcs::mat22_f64) == sizeof(double) * 4, "Incorrect size for fixed-size matrices");
static_assert(sizeof(bcs::mat23_f64) == sizeof(double) * 6, "Incorrect size for fixed-size matrices");
static_assert(sizeof(bcs::mat32_f64) == sizeof(double) * 6, "Incorrect size for fixed-size matrices");
static_assert(sizeof(bcs::mat33_f64) == sizeof(double) * 9, "Incorrect size for fixed-size matrices");
static_assert(sizeof(bcs::col2_f64) == sizeof(double) * 2, "Incorrect size for fixed-size matrices");
static_assert(sizeof(bcs::col3_f64) == sizeof(double) * 3, "Incorrect size for fixed-size matrices");
static_assert(sizeof(bcs::row2_f64) == sizeof(double) * 2, "Incorrect size for fixed-size matrices");
static_assert(sizeof(bcs::row3_f64) == sizeof(double) * 3, "Incorrect size for fixed-size matrices");

static_assert(bcs::is_base_of<
		bcs::IMatrixXpr<bcs::dense_matrix<double>, double>,
		bcs::dense_matrix<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IMatrixView<bcs::dense_matrix<double>, double>,
		bcs::dense_matrix<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IRegularMatrix<bcs::dense_matrix<double>, double>,
		bcs::dense_matrix<double> >::value, "Base verification failed.");

static_assert(bcs::is_base_of<
		bcs::IDenseMatrix<bcs::dense_matrix<double>, double>,
		bcs::dense_matrix<double> >::value, "Base verification failed.");
#endif


template<int CTRows, int CTCols>
static void test_dense_matrix(index_t m, index_t n)
{
	const bool IsStatic = (CTRows > 0 && CTCols > 0);

	// test default construction

	dense_matrix<double, CTRows, CTCols> a0;

	ASSERT_EQ(CTRows, a0.nrows());
	ASSERT_EQ(CTCols, a0.ncolumns());
	ASSERT_EQ(CTRows * CTCols, a0.nelems());
	ASSERT_EQ(CTRows, a0.lead_dim());

	ASSERT_EQ( (size_t)a0.nelems(), a0.size() );
	ASSERT_EQ(a0.nelems() == 0, is_empty(a0));
	ASSERT_EQ(a0.nrows() == 1, is_row(a0));
	ASSERT_EQ(a0.ncolumns() == 1, is_column(a0));
	ASSERT_EQ(a0.nelems() == 1, is_scalar(a0));
	ASSERT_EQ(a0.nrows() == 1 || a0.ncolumns() == 1, is_vector(a0));

	if (IsStatic)
	{
		ASSERT_TRUE(a0.ptr_data() != 0);
	}
	else
	{
		ASSERT_TRUE(a0.ptr_data() == 0);
	}

	// test size-given construction

	dense_matrix<double, CTRows, CTCols> a1(m, n);

	ASSERT_EQ(m, a1.nrows());
	ASSERT_EQ(n, a1.ncolumns());
	ASSERT_EQ(m * n, a1.nelems());
	ASSERT_EQ(m, a1.lead_dim());

	ASSERT_EQ( (size_t)a1.nelems(), a1.size() );
	ASSERT_EQ(a1.nelems() == 0, is_empty(a1));
	ASSERT_EQ(a1.nrows() == 1, is_row(a1));
	ASSERT_EQ(a1.ncolumns() == 1, is_column(a1));
	ASSERT_EQ(a1.nelems() == 1, is_scalar(a1));
	ASSERT_EQ(a1.nrows() == 1 || a1.ncolumns() == 1, is_vector(a1));

	if (a1.nelems() > 0)
	{
		ASSERT_TRUE(a1.ptr_data() != 0);
	}
	else
	{
		ASSERT_TRUE(a1.ptr_data() == 0);
	}

	// test element access

	double* p1 = a1.ptr_data();

	for (index_t j = 0; j < n; ++j)
		for (index_t i = 0; i < m; ++i) p1[i + j * m] = double(i + j * 1000);

	bool elem_access_ok = true;

	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i)
		{
			if (a1(i, j) != p1[i + j * m]) elem_access_ok = false;
			if (a1[i + j * m] != p1[i + j * m]) elem_access_ok = false;
		}
	}

	ASSERT_TRUE( elem_access_ok );

	// test slice pointers

	bool colptr_ok = true;
	for (index_t j = 0; j < n; ++j)
	{
		if (col_ptr(a1, j) != p1 + j * m) colptr_ok = false;
	}
	ASSERT_TRUE( colptr_ok );

	bool rowptr_ok = true;
	for (index_t i = 0; i < m; ++i)
	{
		if (row_ptr(a1, i) != p1 + i) rowptr_ok = false;
	}
	ASSERT_TRUE( rowptr_ok );

	// test copy construction

	dense_matrix<double, CTRows, CTCols> a2(a1);

	ASSERT_EQ(m, a2.nrows());
	ASSERT_EQ(n, a2.ncolumns());
	ASSERT_EQ(m * n, a2.nelems());
	ASSERT_EQ(m, a2.lead_dim());

	if (m * n > 0)
	{
		ASSERT_TRUE( a1.ptr_data() != a2.ptr_data() );
		ASSERT_TRUE( elems_equal(a1.nelems(), a1.ptr_data(), a2.ptr_data()) );
	}

	// test constant construction

	const double cval = 12.5;
	dense_matrix<double, CTRows, CTCols> a3(m, n, cval);

	ASSERT_EQ(m, a3.nrows());
	ASSERT_EQ(n, a3.ncolumns());
	ASSERT_EQ(m * n, a3.nelems());
	ASSERT_EQ(m, a3.lead_dim());

	ASSERT_TRUE( elems_equal(a3.nelems(), a3.ptr_data(), cval ) );

	// test source construction

	dense_matrix<double, CTRows, CTCols> a4(m, n, p1);

	ASSERT_EQ(m, a4.nrows());
	ASSERT_EQ(n, a4.ncolumns());
	ASSERT_EQ(m * n, a4.nelems());
	ASSERT_EQ(m, a4.lead_dim());

	ASSERT_TRUE(a4.ptr_data() != p1);

	ASSERT_TRUE( elems_equal(a4.nelems(), a4.ptr_data(), p1 ) );

}

template<int CTRows, int CTCols>
static void test_dense_matrix_swap_and_resize(index_t m1, index_t n1, index_t m2, index_t n2)
{
	const bool IsStatic = (CTRows > 0 && CTCols > 0);

	scoped_block<double> src1_blk(m1 * n1);
	scoped_block<double> src2_blk(m2 * n2);

	for (index_t i = 0; i < m1 * n1; ++i) src1_blk[i] = double(i+1);
	for (index_t i = 0; i < m2 * n2; ++i) src2_blk[i] = double(2 * (i+1));

	const double *src1 = src1_blk.ptr_begin();
	const double *src2 = src2_blk.ptr_begin();

	// construction

	dense_matrix<double, CTRows, CTCols> a1(m1, n1, src1);

	ASSERT_EQ(m1, 		a1.nrows());
	ASSERT_EQ(n1, 		a1.ncolumns());
	ASSERT_EQ(m1 * n1, 	a1.nelems());
	ASSERT_EQ(m1, 		a1.lead_dim());
	ASSERT_TRUE( elems_equal(a1.nelems(), a1.ptr_data(), src1) );

	const double *p1 = a1.ptr_data();

	dense_matrix<double, CTRows, CTCols> a2(m2, n2, src2);

	ASSERT_EQ(m2, 		a2.nrows());
	ASSERT_EQ(n2, 		a2.ncolumns());
	ASSERT_EQ(m2 * n2, 	a2.nelems());
	ASSERT_EQ(m2, 		a2.lead_dim());
	ASSERT_TRUE( elems_equal(a2.nelems(), a2.ptr_data(), src2) );

	const double *p2 = a2.ptr_data();

	// test swap

	swap(a1, a2);

	ASSERT_EQ(m1, a2.nrows());
	ASSERT_EQ(n1, a2.ncolumns());
	ASSERT_TRUE( elems_equal(a2.nelems(), a2.ptr_data(), src1) );

	ASSERT_EQ(m2, a1.nrows());
	ASSERT_EQ(n2, a1.ncolumns());
	ASSERT_TRUE( elems_equal(a1.nelems(), a1.ptr_data(), src2) );

	if (!IsStatic)
	{
		ASSERT_TRUE(p1 == a2.ptr_data());
		ASSERT_TRUE(p2 == a1.ptr_data());
	}

	swap(a1, a2);

	ASSERT_EQ(m1, a1.nrows());
	ASSERT_EQ(n1, a1.ncolumns());
	ASSERT_TRUE( elems_equal(a1.nelems(), a1.ptr_data(), src1) );

	ASSERT_EQ(m2, a2.nrows());
	ASSERT_EQ(n2, a2.ncolumns());
	ASSERT_TRUE( elems_equal(a2.nelems(), a2.ptr_data(), src2) );

	if (!IsStatic)
	{
		ASSERT_TRUE(p1 == a1.ptr_data());
		ASSERT_TRUE(p2 == a2.ptr_data());
	}

	// test resize

	p2 = a2.ptr_data();

	a2.resize(m2, n2);

	ASSERT_EQ(m2, a2.nrows());
	ASSERT_EQ(n2, a2.ncolumns());
	ASSERT_TRUE( a2.ptr_data() == p2 );

	a2.resize(m1, n1);

	ASSERT_EQ(m1, a2.nrows());
	ASSERT_EQ(n1, a2.ncolumns());
	if (m1 * n1 != m2 * n2)
	{
		ASSERT_TRUE( a2.ptr_data() != p2 );
	}

}


template<int CTLen>
void test_dense_col(index_t len)
{
	scoped_block<double> src_blk(len);
	for (index_t i = 0; i < len; ++i) src_blk[i] = double(i+1);
	const double *src = src_blk.ptr_begin();

	dense_col<double, CTLen> v0;

	ASSERT_EQ(CTLen, v0.nrows());
	ASSERT_EQ(1, v0.ncolumns());
	ASSERT_EQ(CTLen, v0.nelems());

	dense_col<double, CTLen> v1(len);

	ASSERT_EQ(len, v1.nrows());
	ASSERT_EQ(1, v1.ncolumns());
	ASSERT_EQ(len, v1.nelems());

	const double cval = 12.5;
	dense_col<double, CTLen> v2(len, cval);

	ASSERT_EQ(len, v2.nrows());
	ASSERT_EQ(1, v2.ncolumns());
	ASSERT_EQ(len, v2.nelems());

	ASSERT_TRUE( elems_equal(len, v2.ptr_data(), cval) );

	dense_col<double, CTLen> v3(len, src);

	ASSERT_EQ(len, v3.nrows());
	ASSERT_EQ(1, v3.ncolumns());
	ASSERT_EQ(len, v3.nelems());

	ASSERT_TRUE( elems_equal(len, v3.ptr_data(), src) );

	dense_matrix<double, CTLen, 1> a3(v3);
	dense_col<double, CTLen> v4(a3);

	ASSERT_EQ(len, v4.nrows());
	ASSERT_EQ(1, v4.ncolumns());
	ASSERT_EQ(len, v4.nelems());

	ASSERT_TRUE( elems_equal(len, v4.ptr_data(), src) );
}


template<int CTLen>
void test_dense_row(index_t len)
{
	scoped_block<double> src_blk(len);
	for (index_t i = 0; i < len; ++i) src_blk[i] = double(i+1);
	const double *src = src_blk.ptr_begin();

	dense_row<double, CTLen> v0;

	ASSERT_EQ(CTLen, v0.ncolumns());
	ASSERT_EQ(1, v0.nrows());
	ASSERT_EQ(CTLen, v0.nelems());

	dense_row<double, CTLen> v1(len);

	ASSERT_EQ(len, v1.ncolumns());
	ASSERT_EQ(1, v1.nrows());
	ASSERT_EQ(len, v1.nelems());

	const double cval = 12.5;
	dense_row<double, CTLen> v2(len, cval);

	ASSERT_EQ(len, v2.ncolumns());
	ASSERT_EQ(1, v2.nrows());
	ASSERT_EQ(len, v2.nelems());

	ASSERT_TRUE( elems_equal(len, v2.ptr_data(), cval) );

	dense_row<double, CTLen> v3(len, src);

	ASSERT_EQ(len, v3.ncolumns());
	ASSERT_EQ(1, v3.nrows());
	ASSERT_EQ(len, v3.nelems());

	ASSERT_TRUE( elems_equal(len, v3.ptr_data(), src) );

	dense_matrix<double, 1, CTLen> a3(v3);
	dense_row<double, CTLen> v4(a3);

	ASSERT_EQ(len, v4.ncolumns());
	ASSERT_EQ(1, v4.nrows());
	ASSERT_EQ(len, v4.nelems());

	ASSERT_TRUE( elems_equal(len, v4.ptr_data(), src) );
}



TEST( DenseMatrix, DRowDCol )
{
	test_dense_matrix<DynamicDim, DynamicDim>(3, 4);
	test_dense_matrix_swap_and_resize<DynamicDim, DynamicDim>(3, 4, 5, 6);
}

TEST( DenseMatrix, DRowSCol )
{
	test_dense_matrix<DynamicDim, 4>(3, 4);
	test_dense_matrix_swap_and_resize<DynamicDim, 4>(3, 4, 5, 4);
}

TEST( DenseMatrix, SRowDCol )
{
	test_dense_matrix<3, DynamicDim>(3, 4);
	test_dense_matrix_swap_and_resize<3, DynamicDim>(3, 4, 3, 6);
}

TEST( DenseMatrix, SRowSCol )
{
	test_dense_matrix<3, 4>(3, 4);
	test_dense_matrix_swap_and_resize<3, 4>(3, 4, 3, 4);
}

TEST( DenseMatrix, DColVec )
{
	test_dense_matrix<DynamicDim, 1>(5, 1);
	test_dense_col<DynamicDim>(5);
	test_dense_matrix_swap_and_resize<DynamicDim, 1>(5, 1, 6, 1);
}

TEST( DenseMatrix, SColVec )
{
	test_dense_matrix<5, 1>(5, 1);
	test_dense_col<5>(5);
	test_dense_matrix_swap_and_resize<5, 1>(5, 1, 5, 1);
}

TEST( DenseMatrix, DRowVec )
{
	test_dense_matrix<1, DynamicDim>(1, 5);
	test_dense_row<DynamicDim>(5);
	test_dense_matrix_swap_and_resize<1, DynamicDim>(1, 5, 1, 6);
}

TEST( DenseMatrix, SRowVec )
{
	test_dense_matrix<1, 5>(1, 5);
	test_dense_row<5>(5);
	test_dense_matrix_swap_and_resize<1, 5>(1, 5, 1, 5);
}







