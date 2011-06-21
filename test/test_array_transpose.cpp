/**
 * @file test_array_transpose.cpp
 *
 * Unit testing of array transposition
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/test/test_array_aux.h>
#include <bcslib/array/array2d.h>

using namespace bcs;
using namespace bcs::test;


template<typename T>
bool verify_blkwise_transpose(const T *src, T *dst, T *r, size_t bdim, size_t m, size_t n, T *cache)
{
	size_t ne = m * n;

	for (size_t i = 0; i < ne; ++i) dst[i] = 0;

	// establish ground-truth
	direct_transpose_matrix(src, r, m, n);

	// do block-wise transposition
	blockwise_transpose_matrix(src, dst, m, n, bdim, cache);

	return elements_equal(dst, r, m * n);
}


template<typename T, typename TOrd, typename TIndexer0, typename TIndexer1>
bool verify_array2d_transpose(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
{
	size_t m = a.nrows();
	size_t n = a.ncolumns();

	array2d<T, TOrd> r(n, m);

	for (index_t i = 0; i < (index_t)m; ++i)
	{
		for (index_t j = 0; j < (index_t)n; ++j)
		{
			r(j, i) = a(i, j);
		}
	}

	array2d<T, TOrd> t = a.Tp();

	return t == r;
}



BCS_TEST_CASE( test_direct_transpose )
{
	size_t Nmax = 16;
	double src[16];
	for (size_t i = 0; i < Nmax; ++i) src[i] = i + 1;

	// 1 x 2

	const size_t m0 = 1, n0 = 2;
	double t0[m0 * n0] = {0, 0};
	double r0[m0 * n0] = {1, 2};

	direct_transpose_matrix(src, t0, m0, n0);
	BCS_CHECK( elements_equal(t0, r0, m0 * n0) );

	// 2 x 2

	const size_t m1 = 2, n1 = 2;
	double t1[m1 * n1] = {0, 0, 0, 0};
	double r1[m1 * n1] = {1, 3, 2, 4};

	direct_transpose_matrix(src, t1, m1, n1);
	BCS_CHECK( elements_equal(t1, r1, m1 * n1) );

	// 2 x 3

	const size_t m2 = 2, n2 = 3;
	double t2[m2 * n2] = {0, 0, 0, 0, 0, 0};
	double r2[m2 * n2] = {1, 4, 2, 5, 3, 6};

	direct_transpose_matrix(src, t2, m2, n2);
	BCS_CHECK( elements_equal(t2, r2, m2 * n2) );

	// 3 x 2

	const size_t m3 = 3, n3 = 2;
	double t3[m3 * n3] = {0, 0, 0, 0, 0, 0};
	double r3[m3 * n3] = {1, 3, 5, 2, 4, 6};

	direct_transpose_matrix(src, t3, m3, n3);
	BCS_CHECK( elements_equal(t3, r3, m3 * n3) );

	// 3 x 3

	const size_t m4 = 3, n4 = 3;
	double t4[m4 * n4] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	double r4[m4 * n4] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

	direct_transpose_matrix(src, t4, m4, n4);
	BCS_CHECK( elements_equal(t4, r4, m4 * n4) );
}


BCS_TEST_CASE( test_sqmatrix_inplace_transpose )
{
	// 1 x 1

	double t1[1] = {1};
	double r1[1] = {1};

	direct_transpose_sqmatrix_inplace(t1, 1);
	BCS_CHECK( elements_equal(t1, r1, 1) );

	// 2 x 2

	double t2[4] = {1, 2, 3, 4};
	double r2[4] = {1, 3, 2, 4};

	direct_transpose_sqmatrix_inplace(t2, 2);
	BCS_CHECK( elements_equal(t2, r2, 4) );

	// 3 x 3

	double t3[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	double r3[9] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

	direct_transpose_sqmatrix_inplace(t3, 3);
	BCS_CHECK( elements_equal(t3, r3, 9) );

	// 4 x 4

	double t4[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	double r4[16] = {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16};

	direct_transpose_sqmatrix_inplace(t4, 4);
	BCS_CHECK( elements_equal(t4, r4, 16) );
}


BCS_TEST_CASE( test_blockwise_transpose )
{
	const size_t Nmax = 200;
	double src[Nmax];
	for (size_t i = 0; i < Nmax; ++i) src[i] = i + 1;

	double dst[Nmax];
	double r[Nmax];
	double cache[Nmax];

	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 6, 6, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 6, 7, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 6, 8, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 6, 9, cache) );

	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 7, 6, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 7, 7, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 7, 8, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 7, 9, cache) );

	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 8, 6, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 8, 7, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 8, 8, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 8, 9, cache) );

	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 9, 6, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 9, 7, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 9, 8, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 3, 9, 9, cache) );

	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 6, 6, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 6, 7, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 6, 8, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 6, 9, cache) );

	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 7, 6, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 7, 7, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 7, 8, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 7, 9, cache) );

	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 8, 6, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 8, 7, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 8, 8, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 8, 9, cache) );

	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 9, 6, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 9, 7, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 9, 8, cache) );
	BCS_CHECK( verify_blkwise_transpose(src, dst, r, 4, 9, 9, cache) );
}


BCS_TEST_CASE( test_array2d_transpose )
{
	const size_t Nmax = 40000;

	scoped_buffer<double> buf(Nmax);
	for (size_t i = 0; i < Nmax; ++i) buf[i] = i + 1;
	const double *src = buf.pbase();

	size_t ds[6] = {2, 5, 11, 20, 55, 83};

	// row-major

	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[0], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[0], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[0], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[0], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[0], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[0], ds[5]) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[1], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[1], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[1], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[1], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[1], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[1], ds[5]) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[2], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[2], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[2], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[2], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[2], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[2], ds[5]) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[3], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[3], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[3], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[3], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[3], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[3], ds[5]) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[4], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[4], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[4], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[4], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[4], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[4], ds[5]) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[5], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[5], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[5], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[5], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[5], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm(src, ds[5], ds[5]) ) );

	// column-major

	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[0], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[0], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[0], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[0], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[0], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[0], ds[5]) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[1], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[1], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[1], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[1], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[1], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[1], ds[5]) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[2], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[2], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[2], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[2], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[2], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[2], ds[5]) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[3], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[3], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[3], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[3], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[3], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[3], ds[5]) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[4], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[4], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[4], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[4], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[4], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[4], ds[5]) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[5], ds[0]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[5], ds[1]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[5], ds[2]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[5], ds[3]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[5], ds[4]) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm(src, ds[5], ds[5]) ) );

	// non-dense views

	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm_ex(src, 200, 200, step_ind(ds[2], 2), step_ind(ds[2], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm_ex(src, 200, 200, step_ind(ds[2], 2), step_ind(ds[3], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm_ex(src, 200, 200, step_ind(ds[2], 2), step_ind(ds[4], 2)) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm_ex(src, 200, 200, step_ind(ds[3], 2), step_ind(ds[2], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm_ex(src, 200, 200, step_ind(ds[3], 2), step_ind(ds[3], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm_ex(src, 200, 200, step_ind(ds[3], 2), step_ind(ds[4], 2)) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm_ex(src, 200, 200, step_ind(ds[4], 2), step_ind(ds[2], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm_ex(src, 200, 200, step_ind(ds[4], 2), step_ind(ds[3], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_rm_ex(src, 200, 200, step_ind(ds[4], 2), step_ind(ds[4], 2)) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm_ex(src, 200, 200, step_ind(ds[2], 2), step_ind(ds[2], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm_ex(src, 200, 200, step_ind(ds[2], 2), step_ind(ds[3], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm_ex(src, 200, 200, step_ind(ds[2], 2), step_ind(ds[4], 2)) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm_ex(src, 200, 200, step_ind(ds[3], 2), step_ind(ds[2], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm_ex(src, 200, 200, step_ind(ds[3], 2), step_ind(ds[3], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm_ex(src, 200, 200, step_ind(ds[3], 2), step_ind(ds[4], 2)) ) );

	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm_ex(src, 200, 200, step_ind(ds[4], 2), step_ind(ds[2], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm_ex(src, 200, 200, step_ind(ds[4], 2), step_ind(ds[3], 2)) ) );
	BCS_CHECK( verify_array2d_transpose( get_aview2d_cm_ex(src, 200, 200, step_ind(ds[4], 2), step_ind(ds[4], 2)) ) );

}


test_suite* test_array_transposition_suite()
{
	test_suite *tsuite = new test_suite("test_transposition");

	tsuite->add( new test_direct_transpose() );
	tsuite->add( new test_sqmatrix_inplace_transpose() );
	tsuite->add( new test_blockwise_transpose() );
	tsuite->add( new test_array2d_transpose() );

	return tsuite;
}


