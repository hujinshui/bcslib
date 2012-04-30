/**
 * @file test_matrix_manip.cpp
 *
 * Unit testing for matrix manipulation
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;

template<int CTRows, int CTCols>
void test_matrix_equal(index_t m, index_t n, index_t ldim)
{
	scoped_block<double> blk_a(ldim * n);
	scoped_block<double> blk_b(ldim * n);

	for (index_t i = 0; i < ldim * n; ++i) blk_a[i] = double(i+1);
	for (index_t i = 0; i < ldim * n; ++i) blk_b[i] = double(i+1);

	double *pa = blk_a.ptr_begin();
	double *pb = blk_b.ptr_begin();

	ref_matrix_ex<double, CTRows, CTCols> a(pa, m, n, ldim);
	ref_matrix_ex<double, CTRows, CTCols> b(pb, m, n, ldim);

	ASSERT_TRUE( is_equal(a, b) );

	b(m-1, n-1) = 0;

	ASSERT_FALSE( is_equal(a, b) );
}


TEST( MatrixManip, EqualRDCD )
{
	test_matrix_equal<DynamicDim, DynamicDim>(3, 4, 5);
}

TEST( MatrixManip, EqualRDCS )
{
	test_matrix_equal<DynamicDim, 4>(3, 4, 5);
}

TEST( MatrixManip, EqualRDC1 )
{
	test_matrix_equal<DynamicDim, 1>(3, 1, 5);
}

TEST( MatrixManip, EqualRSCD )
{
	test_matrix_equal<3, DynamicDim>(3, 4, 5);
}

TEST( MatrixManip, EqualRSCS )
{
	test_matrix_equal<3, 4>(3, 4, 5);
}

TEST( MatrixManip, EqualRSC1 )
{
	test_matrix_equal<3, 1>(3, 1, 5);
}

TEST( MatrixManip, EqualR1CD )
{
	test_matrix_equal<1, DynamicDim>(1, 4, 5);
}

TEST( MatrixManip, EqualR1CS )
{
	test_matrix_equal<1, 4>(1, 4, 5);
}

TEST( MatrixManip, EqualR1C1 )
{
	test_matrix_equal<1, 1>(1, 1, 5);
}


template<int CTRows, int CTCols>
void test_matrix_copy(index_t m, index_t n, index_t ldim)
{
	scoped_block<double> blk_a(ldim * n);
	scoped_block<double> blk_b(ldim * n);
	scoped_block<double> blk_r(ldim * n);

	for (index_t i = 0; i < ldim * n; ++i) blk_a[i] = double(i+1);
	for (index_t i = 0; i < ldim * n; ++i) blk_b[i] = 0;

	for (index_t i = 0; i < ldim * n; ++i) blk_r[i] = 0;
	for (index_t j = 0; j < n; ++j)
		for(index_t i = 0; i < m; ++i) blk_r[i + ldim * j] = blk_a[i + ldim * j];

	double *pa = blk_a.ptr_begin();
	double *pb = blk_b.ptr_begin();
	double *pr = blk_r.ptr_begin();

	ref_matrix_ex<double, CTRows, CTCols> a(pa, m, n, ldim);
	ref_matrix_ex<double, CTRows, CTCols> b(pb, m, n, ldim);

	copy(a, b);

	ASSERT_TRUE( is_equal(a, b) );
	ASSERT_TRUE( elems_equal(ldim * n, pb, pr) );
}

TEST( MatrixManip, CopyRDCD )
{
	test_matrix_copy<DynamicDim, DynamicDim>(3, 4, 5);
}

TEST( MatrixManip, CopyRDCS )
{
	test_matrix_copy<DynamicDim, 4>(3, 4, 5);
}

TEST( MatrixManip, CopyRDC1 )
{
	test_matrix_copy<DynamicDim, 1>(3, 1, 5);
}

TEST( MatrixManip, CopyRSCD )
{
	test_matrix_copy<3, DynamicDim>(3, 4, 5);
}

TEST( MatrixManip, CopyRSCS )
{
	test_matrix_copy<3, 4>(3, 4, 5);
}

TEST( MatrixManip, CopyRSC1 )
{
	test_matrix_copy<3, 1>(3, 1, 5);
}

TEST( MatrixManip, CopyR1CD )
{
	test_matrix_copy<1, DynamicDim>(1, 4, 5);
}

TEST( MatrixManip, CopyR1CS )
{
	test_matrix_copy<1, 4>(1, 4, 5);
}

TEST( MatrixManip, CopyR1C1 )
{
	test_matrix_copy<1, 1>(1, 1, 5);
}


template<int CTRows, int CTCols>
void test_matrix_fill(index_t m, index_t n, index_t ldim)
{
	scoped_block<double> blk_a(ldim * n);
	scoped_block<double> blk_r(ldim * n);

	const double fv = 12.5;

	for (index_t i = 0; i < ldim * n; ++i) blk_a[i] = 0;
	for (index_t i = 0; i < ldim * n; ++i) blk_r[i] = 0;
	for (index_t j = 0; j < n; ++j)
		for(index_t i = 0; i < m; ++i) blk_r[i + ldim * j] = fv;

	double *pa = blk_a.ptr_begin();
	double *pr = blk_r.ptr_begin();

	ref_matrix_ex<double, CTRows, CTCols> a(pa, m, n, ldim);

	fill(a, fv);

	ASSERT_TRUE( elems_equal(ldim * n, pa, pr) );
}

TEST( MatrixManip, FillRDCD )
{
	test_matrix_fill<DynamicDim, DynamicDim>(3, 4, 5);
}

TEST( MatrixManip, FillRDCS )
{
	test_matrix_fill<DynamicDim, 4>(3, 4, 5);
}

TEST( MatrixManip, FillRDC1 )
{
	test_matrix_fill<DynamicDim, 1>(3, 1, 5);
}

TEST( MatrixManip, FillRSCD )
{
	test_matrix_fill<3, DynamicDim>(3, 4, 5);
}

TEST( MatrixManip, FillRSCS )
{
	test_matrix_fill<3, 4>(3, 4, 5);
}

TEST( MatrixManip, FillRSC1 )
{
	test_matrix_fill<3, 1>(3, 1, 5);
}

TEST( MatrixManip, FillR1CD )
{
	test_matrix_fill<1, DynamicDim>(1, 4, 5);
}

TEST( MatrixManip, FillR1CS )
{
	test_matrix_fill<1, 4>(1, 4, 5);
}

TEST( MatrixManip, FillR1C1 )
{
	test_matrix_fill<1, 1>(1, 1, 5);
}




template<int CTRows, int CTCols>
void test_matrix_zero(index_t m, index_t n, index_t ldim)
{
	scoped_block<double> blk_a(ldim * n);
	scoped_block<double> blk_r(ldim * n);

	for (index_t i = 0; i < ldim * n; ++i) blk_a[i] = -1;
	for (index_t i = 0; i < ldim * n; ++i) blk_r[i] = -1;
	for (index_t j = 0; j < n; ++j)
		for(index_t i = 0; i < m; ++i) blk_r[i + ldim * j] = 0;

	double *pa = blk_a.ptr_begin();
	double *pr = blk_r.ptr_begin();

	ref_matrix_ex<double, CTRows, CTCols> a(pa, m, n, ldim);

	zero(a);

	ASSERT_TRUE( elems_equal(ldim * n, pa, pr) );
}


TEST( MatrixManip, ZeroRDCD )
{
	test_matrix_zero<DynamicDim, DynamicDim>(3, 4, 5);
}

TEST( MatrixManip, ZeroRDCS )
{
	test_matrix_zero<DynamicDim, 4>(3, 4, 5);
}

TEST( MatrixManip, ZeroRDC1 )
{
	test_matrix_zero<DynamicDim, 1>(3, 1, 5);
}

TEST( MatrixManip, ZeroRSCD )
{
	test_matrix_zero<3, DynamicDim>(3, 4, 5);
}

TEST( MatrixManip, ZeroRSCS )
{
	test_matrix_zero<3, 4>(3, 4, 5);
}

TEST( MatrixManip, ZeroRSC1 )
{
	test_matrix_zero<3, 1>(3, 1, 5);
}

TEST( MatrixManip, ZeroR1CD )
{
	test_matrix_zero<1, DynamicDim>(1, 4, 5);
}

TEST( MatrixManip, ZeroR1CS )
{
	test_matrix_zero<1, 4>(1, 4, 5);
}

TEST( MatrixManip, ZeroR1C1 )
{
	test_matrix_zero<1, 1>(1, 1, 5);
}





