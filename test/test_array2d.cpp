/**
 * @file test_array2d.cpp
 *
 * The Unit Testing for array2d
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/test_units.h>
#include <bcslib/array/array2d.h>

#include <iostream>

using namespace bcs;
using namespace bcs::test;

// Explicit instantiation for syntax checking

template class bcs::array2d<double, row_major_t>;
template class bcs::array2d<double, column_major_t>;

template class bcs::aview2d<double, row_major_t, step_ind, step_ind>;
template class bcs::aview2d<double, column_major_t, step_ind, step_ind>;

template class bcs::aview2d<double, row_major_t, indices, indices>;
template class bcs::aview2d<double, column_major_t, indices, indices>;



template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
void print_array(const bcs::const_aview2d<T, TOrd, TIndexer0, TIndexer1>& view)
{
	index_t m = (index_t)view.nrows();
	index_t n = (index_t)view.ncolumns();

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			std::cout << view(i, j) << ' ';
		}
		std::cout << std::endl;
	}
}


template<typename FwdIter>
void print_collection(FwdIter first, FwdIter last)
{
	for (FwdIter it = first; it != last; ++it)
	{
		std::cout << *it << ' ';
	}
	std::cout << std::endl;
}


template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
bool array_integrity_test(const bcs::const_aview2d<T, TOrd, TIndexer0, TIndexer1>& view)
{
	size_t m = view.dim0();
	size_t n = view.dim1();

	if (view.ndims() != 2) return false;
	if (view.shape() != arr_shape(m, n)) return false;
	if (view.nelems() != m * n) return false;

	for (index_t i = 0; i < (index_t)m; ++i)
	{
		for (index_t j = 0; j < (index_t)n; ++j)
		{
			if (view.ptr(i, j) != &(view(i, j))) return false;
		}
	}

	array2d<T, TOrd> acopy = make_copy(view);
	if (!(acopy == view)) return false;

	return true;
}


template<typename T, class TIndexer0, class TIndexer1>
bool array_iteration_test(const bcs::const_aview2d<T, row_major_t, TIndexer0, TIndexer1>& view)
{
	size_t m = view.dim0();
	size_t n = view.dim1();

	bcs::block<T> buffer(m * n);
	T *p = buffer.pbase();

	for (index_t i = 0; i < (index_t)m; ++i)
	{
		for (index_t j = 0; j < (index_t)n; ++j)
		{
			*p++ = view(i, j);
		}
	}

	return collection_equal(view.begin(), view.end(), buffer.pbase(), m * n);
}

template<typename T, class TIndexer0, class TIndexer1>
bool array_iteration_test(const bcs::const_aview2d<T, column_major_t, TIndexer0, TIndexer1>& view)
{
	size_t m = view.dim0();
	size_t n = view.dim1();

	bcs::block<T> buffer(m * n);
	T *p = buffer.pbase();

	for (index_t j = 0; j < (index_t)n; ++j)
	{
		for (index_t i = 0; i < (index_t)m; ++i)
		{
			*p++ = view(i, j);
		}
	}

	return collection_equal(view.begin(), view.end(), buffer.pbase(), m * n);
}



BCS_TEST_CASE( test_dense_array2d  )
{
	double src[24];
	for (int i = 0; i < 24; ++i) src[i] = i+1;

	double r0[] = {0, 0, 0, 0, 0, 0};
	double r1[] = {1, 2, 3, 4, 5, 6};

	// row major

	array2d<double, row_major_t> a0_rm(0, 0);
	BCS_CHECK( is_dense_view(a0_rm) );
	BCS_CHECK( array_integrity_test(a0_rm) );
	BCS_CHECK( array_iteration_test(a0_rm) );

	array2d<double, row_major_t> a1_rm(2, 3, src);
	BCS_CHECK( is_dense_view(a1_rm) );
	BCS_CHECK( array_integrity_test(a1_rm) );
	BCS_CHECK( array_view_equal(a1_rm, r1, 2, 3) );
	BCS_CHECK( array_iteration_test(a1_rm) );

	block<double> a1_rm_buf(6);
	export_to(a1_rm, a1_rm_buf.pbase());
	BCS_CHECK( collection_equal(a1_rm_buf.pbase(), a1_rm_buf.pend(), r1, 6) );

	fill(a1_rm, 0.0);
	BCS_CHECK( array_view_equal(a1_rm, r0, 2, 3) );

	// column major

	array2d<double, column_major_t> a0_cm(0, 0);
	BCS_CHECK( is_dense_view(a0_cm) );
	BCS_CHECK( array_integrity_test(a0_cm) );
	BCS_CHECK( array_iteration_test(a0_cm) );

	array2d<double, column_major_t> a1_cm(2, 3, src);
	BCS_CHECK( is_dense_view(a1_cm) );
	BCS_CHECK( array_integrity_test(a1_cm) );
	BCS_CHECK( array_view_equal(a1_cm, r1, 2, 3) );
	BCS_CHECK( array_iteration_test(a1_cm) );

	block<double> a1_cm_buf(6);
	export_to(a1_cm, a1_cm_buf.pbase());
	BCS_CHECK( collection_equal(a1_cm_buf.pbase(), a1_cm_buf.pend(), r1, 6) );

	fill(a1_cm, 0.0);
	BCS_CHECK( array_view_equal(a1_cm, r0, 2, 3) );

}


BCS_TEST_CASE( test_gen_array2d )
{
	double src[36];
	for (int i = 0; i < 36; ++i) src[i] = i+1;

	// (id_ind, step_ind(2) )

	aview2d<double, row_major_t, id_ind, step_ind> a1_rm(src, 5, 6, id_ind(2), step_ind(3, 2));
	double r1_rm[] = {1, 3, 5, 7, 9, 11};

	BCS_CHECK( !is_dense_view(a1_rm) );
	BCS_CHECK( array_integrity_test(a1_rm) );
	BCS_CHECK( array_view_equal(a1_rm, r1_rm, 2, 3) );
	BCS_CHECK( array_iteration_test(a1_rm) );

	aview2d<double, column_major_t, id_ind, step_ind> a1_cm(src, 5, 6, id_ind(2), step_ind(3, 2));
	double r1_cm[] = {1, 2, 11, 12, 21, 22};

	BCS_CHECK( !is_dense_view(a1_cm) );
	BCS_CHECK( array_integrity_test(a1_cm) );
	BCS_CHECK( array_view_equal(a1_cm, r1_cm, 2, 3) );
	BCS_CHECK( array_iteration_test(a1_cm) );


	// (step_ind(2), step_ind(2) )

	aview2d<double, row_major_t, step_ind, step_ind> a2_rm(src, 5, 6, step_ind(2, 2), step_ind(3, 2));
	double r2_rm[] = {1, 3, 5, 13, 15, 17};

	BCS_CHECK( !is_dense_view(a2_rm) );
	BCS_CHECK( array_integrity_test(a2_rm) );
	BCS_CHECK( array_view_equal(a2_rm, r2_rm, 2, 3) );
	BCS_CHECK( array_iteration_test(a2_rm) );

	aview2d<double, column_major_t, step_ind, step_ind> a2_cm(src, 5, 6, step_ind(2, 2), step_ind(3, 2));
	double r2_cm[] = {1, 3, 11, 13, 21, 23};

	BCS_CHECK( !is_dense_view(a2_cm) );
	BCS_CHECK( array_integrity_test(a2_cm) );
	BCS_CHECK( array_view_equal(a2_cm, r2_cm, 2, 3) );
	BCS_CHECK( array_iteration_test(a2_cm) );


	// (rep_ind, rep_ind)

	aview2d<double, row_major_t, rep_ind, rep_ind> a3_rm(src, 5, 6, rep_ind(2), rep_ind(3));
	double r3_rm[] = {1, 1, 1, 1, 1, 1};

	BCS_CHECK( !is_dense_view(a3_rm) );
	BCS_CHECK( array_integrity_test(a3_rm) );
	BCS_CHECK( array_view_equal(a3_rm, r3_rm, 2, 3) );
	BCS_CHECK( array_iteration_test(a3_rm) );

	aview2d<double, column_major_t, rep_ind, rep_ind> a3_cm(src, 5, 6, rep_ind(2), rep_ind(3));
	double r3_cm[] = {1, 1, 1, 1, 1, 1};

	BCS_CHECK( !is_dense_view(a3_cm) );
	BCS_CHECK( array_integrity_test(a3_cm) );
	BCS_CHECK( array_view_equal(a3_cm, r3_cm, 2, 3) );
	BCS_CHECK( array_iteration_test(a3_cm) );


	// (indices, id_ind)

	index_t inds[] = {0, 2, 3};

	aview2d<double, row_major_t, indices, id_ind> a4_rm(src, 5, 6, indices(ref_arr(inds, 3)), id_ind(4));
	double r4_rm[] = {1, 2, 3, 4, 13, 14, 15, 16, 19, 20, 21, 22};

	BCS_CHECK( !is_dense_view(a4_rm) );
	BCS_CHECK( array_integrity_test(a4_rm) );
	BCS_CHECK( array_view_equal(a4_rm, r4_rm, 3, 4) );
	BCS_CHECK( array_iteration_test(a4_rm) );

	aview2d<double, column_major_t, indices, id_ind> a4_cm(src, 5, 6, indices(ref_arr(inds, 3)), id_ind(4));
	double r4_cm[] = {1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18, 19};

	BCS_CHECK( !is_dense_view(a4_cm) );
	BCS_CHECK( array_integrity_test(a4_cm) );
	BCS_CHECK( array_view_equal(a4_cm, r4_cm, 3, 4) );
	BCS_CHECK( array_iteration_test(a4_cm) );

	// (indices, step_ind(2))

	aview2d<double, row_major_t, indices, step_ind> a5_rm(src, 5, 6, indices(ref_arr(inds, 3)), step_ind(3, 2));
	double r5_rm[] = {1, 3, 5, 13, 15, 17, 19, 21, 23};

	BCS_CHECK( !is_dense_view(a5_rm) );
	BCS_CHECK( array_integrity_test(a5_rm) );
	BCS_CHECK( array_view_equal(a5_rm, r5_rm, 3, 3) );
	BCS_CHECK( array_iteration_test(a5_rm) );

	aview2d<double, column_major_t, indices, step_ind> a5_cm(src, 5, 6, indices(ref_arr(inds, 3)), step_ind(3, 2));
	double r5_cm[] = {1, 3, 4, 11, 13, 14, 21, 23, 24};

	BCS_CHECK( !is_dense_view(a5_cm) );
	BCS_CHECK( array_integrity_test(a5_cm) );
	BCS_CHECK( array_view_equal(a5_cm, r5_cm, 3, 3) );
	BCS_CHECK( array_iteration_test(a5_cm) );

	// (indices, indices)

	aview2d<double, row_major_t, indices, indices> a6_rm(src, 5, 6, indices(ref_arr(inds, 2)), indices(ref_arr(inds, 3)));
	double r6_rm[] = {1, 3, 4, 13, 15, 16};

	BCS_CHECK( !is_dense_view(a6_rm) );
	BCS_CHECK( array_integrity_test(a6_rm) );
	BCS_CHECK( array_view_equal(a6_rm, r6_rm, 2, 3) );
	BCS_CHECK( array_iteration_test(a6_rm) );

	aview2d<double, column_major_t, indices, indices> a6_cm(src, 5, 6, indices(ref_arr(inds, 2)), indices(ref_arr(inds, 3)));
	double r6_cm[] = {1, 3, 11, 13, 16, 18};

	BCS_CHECK( !is_dense_view(a6_cm) );
	BCS_CHECK( array_integrity_test(a6_cm) );
	BCS_CHECK( array_view_equal(a6_cm, r6_cm, 2, 3) );
	BCS_CHECK( array_iteration_test(a6_cm) );

	// (rep_ind, indices)

	aview2d<double, row_major_t, rep_ind, indices> a7_rm(src, 5, 6, rep_ind(2), indices(ref_arr(inds, 3)));
	double r7_rm[] = {1, 3, 4, 1, 3, 4};

	BCS_CHECK( !is_dense_view(a7_rm) );
	BCS_CHECK( array_integrity_test(a7_rm) );
	BCS_CHECK( array_view_equal(a7_rm, r7_rm, 2, 3) );
	BCS_CHECK( array_iteration_test(a7_rm) );

	aview2d<double, column_major_t, rep_ind, indices> a7_cm(src, 5, 6, rep_ind(2), indices(ref_arr(inds, 3)));
	double r7_cm[] = {1, 1, 11, 11, 16, 16};

	BCS_CHECK( !is_dense_view(a7_cm) );
	BCS_CHECK( array_integrity_test(a7_cm) );
	BCS_CHECK( array_view_equal(a7_cm, r7_cm, 2, 3) );
	BCS_CHECK( array_iteration_test(a7_cm) );

}


BCS_TEST_CASE( test_array2d_slices )
{
	double src[36];
	for (int i = 0; i < 36; ++i) src[i] = i+1;

	// (id_ind, id_ind)

	aview2d<double, row_major_t, id_ind, id_ind> a1_rm(src, 5, 6, id_ind(3), id_ind(4));

	double a1_rm_r1[] = {1, 2, 3, 4};
	double a1_rm_r2[] = {7, 8, 9, 10};
	BCS_CHECK_EQUAL( a1_rm.row(0), const_aview1d<double>(a1_rm_r1, 4) );
	BCS_CHECK_EQUAL( a1_rm.row(1), const_aview1d<double>(a1_rm_r2, 4) );

	double a1_rm_c1[] = {1, 7, 13};
	double a1_rm_c2[] = {2, 8, 14};
	BCS_CHECK_EQUAL( a1_rm.column(0), const_aview1d<double>(a1_rm_c1, 3) );
	BCS_CHECK_EQUAL( a1_rm.column(1), const_aview1d<double>(a1_rm_c2, 3) );

	aview2d<double, column_major_t, id_ind, id_ind> a1_cm(src, 5, 6, id_ind(3), id_ind(4));

	double a1_cm_r1[] = {1, 6, 11, 16};
	double a1_cm_r2[] = {2, 7, 12, 17};
	BCS_CHECK_EQUAL( a1_cm.row(0), const_aview1d<double>(a1_cm_r1, 4) );
	BCS_CHECK_EQUAL( a1_cm.row(1), const_aview1d<double>(a1_cm_r2, 4) );

	double a1_cm_c1[] = {1, 2, 3};
	double a1_cm_c2[] = {6, 7, 8};
	BCS_CHECK_EQUAL( a1_cm.column(0), const_aview1d<double>(a1_cm_c1, 3) );
	BCS_CHECK_EQUAL( a1_cm.column(1), const_aview1d<double>(a1_cm_c2, 3) );

	// (step_ind, step_ind)

	aview2d<double, row_major_t, step_ind, step_ind> a2_rm(src, 5, 6, step_ind(2, 2), step_ind(3, 2));

	double a2_rm_r1[] = {1, 3, 5};
	double a2_rm_r2[] = {13, 15, 17};
	BCS_CHECK_EQUAL( a2_rm.row(0), const_aview1d<double>(a2_rm_r1, 3) );
	BCS_CHECK_EQUAL( a2_rm.row(1), const_aview1d<double>(a2_rm_r2, 3) );

	double a2_rm_c1[] = {1, 13};
	double a2_rm_c2[] = {3, 15};
	BCS_CHECK_EQUAL( a2_rm.column(0), const_aview1d<double>(a2_rm_c1, 2) );
	BCS_CHECK_EQUAL( a2_rm.column(1), const_aview1d<double>(a2_rm_c2, 2) );

	aview2d<double, column_major_t, step_ind, step_ind> a2_cm(src, 5, 6, step_ind(2, 2), step_ind(3, 2));

	double a2_cm_r1[] = {1, 11, 21};
	double a2_cm_r2[] = {3, 13, 23};
	BCS_CHECK_EQUAL( a2_cm.row(0), const_aview1d<double>(a2_cm_r1, 3) );
	BCS_CHECK_EQUAL( a2_cm.row(1), const_aview1d<double>(a2_cm_r2, 3) );

	double a2_cm_c1[] = {1, 3};
	double a2_cm_c2[] = {11, 13};
	BCS_CHECK_EQUAL( a2_cm.column(0), const_aview1d<double>(a2_cm_c1, 2) );
	BCS_CHECK_EQUAL( a2_cm.column(1), const_aview1d<double>(a2_cm_c2, 2) );


	// (rep_ind, step_ind)

	aview2d<double, row_major_t, rep_ind, step_ind> a3_rm(src, 5, 6, rep_ind(4), step_ind(3, 2));

	double a3_rm_r1[] = {1, 3, 5};
	double a3_rm_r2[] = {1, 3, 5};
	BCS_CHECK_EQUAL( a3_rm.row(0), const_aview1d<double>(a3_rm_r1, 3) );
	BCS_CHECK_EQUAL( a3_rm.row(1), const_aview1d<double>(a3_rm_r2, 3) );

	double a3_rm_c1[] = {1, 1, 1, 1};
	double a3_rm_c2[] = {3, 3, 3, 3};
	BCS_CHECK_EQUAL( a3_rm.column(0), const_aview1d<double>(a3_rm_c1, 4) );
	BCS_CHECK_EQUAL( a3_rm.column(1), const_aview1d<double>(a3_rm_c2, 4) );

	aview2d<double, column_major_t, rep_ind, step_ind> a3_cm(src, 5, 6, rep_ind(4), step_ind(3, 2));

	double a3_cm_r1[] = {1, 11, 21};
	double a3_cm_r2[] = {1, 11, 21};
	BCS_CHECK_EQUAL( a3_cm.row(0), const_aview1d<double>(a3_cm_r1, 3) );
	BCS_CHECK_EQUAL( a3_cm.row(1), const_aview1d<double>(a3_cm_r2, 3) );

	double a3_cm_c1[] = {1, 1, 1, 1};
	double a3_cm_c2[] = {11, 11, 11, 11};
	BCS_CHECK_EQUAL( a3_cm.column(0), const_aview1d<double>(a3_cm_c1, 4) );
	BCS_CHECK_EQUAL( a3_cm.column(1), const_aview1d<double>(a3_cm_c2, 4) );


	// (step_ind, indices)

	index_t rinds[] = {0, 2, 4};
	index_t cinds[] = {0, 2, 3, 5};

	aview2d<double, row_major_t, step_ind, indices> a4_rm(src, 5, 6, step_ind(3, 2), indices(ref_arr(cinds, 4)));

	double a4_rm_r1[] = {1, 3, 4, 6};
	double a4_rm_r2[] = {13, 15, 16, 18};
	BCS_CHECK_EQUAL( a4_rm.row(0), const_aview1d<double>(a4_rm_r1, 4) );
	BCS_CHECK_EQUAL( a4_rm.row(1), const_aview1d<double>(a4_rm_r2, 4) );

	double a4_rm_c1[] = {1, 13, 25};
	double a4_rm_c2[] = {3, 15, 27};
	BCS_CHECK_EQUAL( a4_rm.column(0), const_aview1d<double>(a4_rm_c1, 3) );
	BCS_CHECK_EQUAL( a4_rm.column(1), const_aview1d<double>(a4_rm_c2, 3) );

	aview2d<double, column_major_t, step_ind, indices> a4_cm(src, 5, 6, step_ind(3, 2), indices(ref_arr(cinds, 4)));

	double a4_cm_r1[] = {1, 11, 16, 26};
	double a4_cm_r2[] = {3, 13, 18, 28};
	BCS_CHECK_EQUAL( a4_cm.row(0), const_aview1d<double>(a4_cm_r1, 4) );
	BCS_CHECK_EQUAL( a4_cm.row(1), const_aview1d<double>(a4_cm_r2, 4) );

	double a4_cm_c1[] = {1, 3, 5};
	double a4_cm_c2[] = {11, 13, 15};
	BCS_CHECK_EQUAL( a4_cm.column(0), const_aview1d<double>(a4_cm_c1, 3) );
	BCS_CHECK_EQUAL( a4_cm.column(1), const_aview1d<double>(a4_cm_c2, 3) );

	// (indices, indices)

	aview2d<double, row_major_t, indices, indices> a5_rm(src, 5, 6, indices(ref_arr(rinds, 3)), indices(ref_arr(cinds, 4)));

	double a5_rm_r1[] = {1, 3, 4, 6};
	double a5_rm_r2[] = {13, 15, 16, 18};
	BCS_CHECK_EQUAL( a5_rm.row(0), const_aview1d<double>(a5_rm_r1, 4) );
	BCS_CHECK_EQUAL( a5_rm.row(1), const_aview1d<double>(a5_rm_r2, 4) );

	double a5_rm_c1[] = {1, 13, 25};
	double a5_rm_c2[] = {3, 15, 27};
	BCS_CHECK_EQUAL( a5_rm.column(0), const_aview1d<double>(a5_rm_c1, 3) );
	BCS_CHECK_EQUAL( a5_rm.column(1), const_aview1d<double>(a5_rm_c2, 3) );

	aview2d<double, column_major_t, indices, indices> a5_cm(src, 5, 6, indices(ref_arr(rinds, 3)), indices(ref_arr(cinds, 4)));

	double a5_cm_r1[] = {1, 11, 16, 26};
	double a5_cm_r2[] = {3, 13, 18, 28};
	BCS_CHECK_EQUAL( a5_cm.row(0), const_aview1d<double>(a5_cm_r1, 4) );
	BCS_CHECK_EQUAL( a5_cm.row(1), const_aview1d<double>(a5_cm_r2, 4) );

	double a5_cm_c1[] = {1, 3, 5};
	double a5_cm_c2[] = {11, 13, 15};
	BCS_CHECK_EQUAL( a5_cm.column(0), const_aview1d<double>(a5_cm_c1, 3) );
	BCS_CHECK_EQUAL( a5_cm.column(1), const_aview1d<double>(a5_cm_c2, 3) );
}


BCS_TEST_CASE( test_array2d_subviews )
{

	double src0[144];
	for (int i = 0; i < 144; ++i) src0[i] = i+1;


	// dense base

	aview2d<double, row_major_t> a0_rm = dense_aview2d(src0, 6, 6, row_major_t());
	aview2d<double, column_major_t> a0_cm = dense_aview2d(src0, 6, 6, column_major_t());

	// dense => (whole, whole)

	BCS_CHECK_EQUAL( a0_rm.V(whole(), whole()),  dense_aview2d(src0, 6, 6, row_major_t()) );
	BCS_CHECK_EQUAL( a0_cm.V(whole(), whole()),  dense_aview2d(src0, 6, 6, column_major_t()) );

	// dense => (i, whole)

	double a0_rm_v1[] = {13, 14, 15, 16, 17, 18};
	double a0_cm_v1[] = {3, 9, 15, 21, 27, 33};

	BCS_CHECK_EQUAL( a0_rm.V(2, whole()), aview1d<double>(a0_rm_v1, 6) );
	BCS_CHECK_EQUAL( a0_cm.V(2, whole()), aview1d<double>(a0_cm_v1, 6) );

	// dense => (whole, j)

	double a0_rm_v2[] = {4, 10, 16, 22, 28, 34};
	double a0_cm_v2[] = {19, 20, 21, 22, 23, 24};

	BCS_CHECK_EQUAL( a0_rm.V(whole(), 3), aview1d<double>(a0_rm_v2, 6) );
	BCS_CHECK_EQUAL( a0_cm.V(whole(), 3), aview1d<double>(a0_cm_v2, 6) );

	// dense => (i, range)

	double a0_rm_v3[] = {8, 9, 10, 11};
	double a0_cm_v3[] = {8, 14, 20, 26};

	BCS_CHECK_EQUAL( a0_rm.V(1, rgn(1, 5)), aview1d<double>(a0_rm_v3, 4) );
	BCS_CHECK_EQUAL( a0_cm.V(1, rgn(1, 5)), aview1d<double>(a0_cm_v3, 4) );

	// dense => (step_range, i)

	double a0_rm_v4[] = {3, 15, 27};
	double a0_cm_v4[] = {13, 15, 17};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(0, aend(), 2), 2), aview1d<double>(a0_rm_v4, 3) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(0, aend(), 2), 2), aview1d<double>(a0_cm_v4, 3) );


	// dense => (range, whole)

	double a0_rm_s1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
	double a0_cm_s1[] = {1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21, 25, 26, 27, 31, 32, 33};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(0, 3), whole()),  dense_aview2d(a0_rm_s1, 3, 6, row_major_t()) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(0, 3), whole()),  dense_aview2d(a0_cm_s1, 3, 6, column_major_t()) );

	// dense => (whole, range)

	double a0_rm_s2[] = {2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22, 26, 27, 28, 32, 33, 34};
	double a0_cm_s2[] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

	BCS_CHECK_EQUAL( a0_rm.V(whole(), rgn(1, 4)),  dense_aview2d(a0_rm_s2, 6, 3, row_major_t()) );
	BCS_CHECK_EQUAL( a0_cm.V(whole(), rgn(1, 4)),  dense_aview2d(a0_cm_s2, 6, 3, column_major_t()) );


	// dense => (range, range)

	double a0_rm_s3[] = {14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29};
	double a0_cm_s3[] = {9, 10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(2, 5), rgn(1, 5)),  dense_aview2d(a0_rm_s3, 3, 4, row_major_t()) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(2, 5), rgn(1, 5)),  dense_aview2d(a0_cm_s3, 3, 4, column_major_t()) );

	// dense => (range, step_range)

	double a0_rm_s4[] = {14, 16, 18, 20, 22, 24, 26, 28, 30};
	double a0_cm_s4[] = {9, 10, 11, 21, 22, 23, 33, 34, 35};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(2, 5), rgn(1, 6, 2)),  dense_aview2d(a0_rm_s4, 3, 3, row_major_t()) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(2, 5), rgn(1, 6, 2)),  dense_aview2d(a0_cm_s4, 3, 3, column_major_t()) );


	// dense => (step_range, step_range)

	double a0_rm_s5[] = {2, 4, 6, 14, 16, 18, 26, 28, 30};
	double a0_cm_s5[] = {7, 9, 11, 19, 21, 23, 31, 33, 35};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(0, 5, 2), rgn(1, 6, 2)),  dense_aview2d(a0_rm_s5, 3, 3, row_major_t()) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(0, 5, 2), rgn(1, 6, 2)),  dense_aview2d(a0_cm_s5, 3, 3, column_major_t()) );


	// dense => (step_range, indices)

	double a0_rm_s6[] = {1, 3, 4, 5, 13, 15, 16, 17, 25, 27, 28, 29};
	double a0_cm_s6[] = {1, 3, 5, 13, 15, 17, 19, 21, 23, 25, 27, 29};

	index_t a0_cinds_s6[] = {0, 2, 3, 4};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(0, aend(), 2), indices(ref_arr(a0_cinds_s6, 4))),  dense_aview2d(a0_rm_s6, 3, 4, row_major_t()) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(0, aend(), 2), indices(ref_arr(a0_cinds_s6, 4))),  dense_aview2d(a0_cm_s6, 3, 4, column_major_t()) );

	// dense => (rep_range, indices)

	index_t a0_cinds_s7[] = {0, 2, 3, 4};

	double a0_rm_s7[] = {7, 9, 10, 11, 7, 9, 10, 11, 7, 9, 10, 11};
	double a0_cm_s7[] = {2, 2, 2, 14, 14, 14, 20, 20, 20, 26, 26, 26};

	BCS_CHECK_EQUAL( a0_rm.V(rep(1, 3), indices(ref_arr(a0_cinds_s7, 4))),  dense_aview2d(a0_rm_s7, 3, 4, row_major_t()) );
	BCS_CHECK_EQUAL( a0_cm.V(rep(1, 3), indices(ref_arr(a0_cinds_s7, 4))),  dense_aview2d(a0_cm_s7, 3, 4, column_major_t()) );

	// dense => (indices, indices)

	index_t a0_rinds_s8[] = {1, 3, 4};
	index_t a0_cinds_s8[] = {0, 2, 3, 5};

	double a0_rm_s8[] = {7, 9, 10, 12, 19, 21, 22, 24, 25, 27, 28, 30};
	double a0_cm_s8[] = {2, 4, 5, 14, 16, 17, 20, 22, 23, 32, 34, 35};

	BCS_CHECK_EQUAL( a0_rm.V(indices(ref_arr(a0_rinds_s8, 3)), indices(ref_arr(a0_cinds_s8, 4))),  dense_aview2d(a0_rm_s8, 3, 4, row_major_t()) );
	BCS_CHECK_EQUAL( a0_cm.V(indices(ref_arr(a0_rinds_s8, 3)), indices(ref_arr(a0_cinds_s8, 4))),  dense_aview2d(a0_cm_s8, 3, 4, column_major_t()) );


	// step base

	aview2d<double, row_major_t> Arm = dense_aview2d(src0, 12, 12, row_major_t());
	aview2d<double, column_major_t> Acm = dense_aview2d(src0, 12, 12, column_major_t());

	aview2d<double, row_major_t, step_ind, step_ind> a1_rm = Arm.V(rgn(0, aend(), 2), rgn(0, aend(), 2));
	aview2d<double, column_major_t, step_ind, step_ind> a1_cm = Acm.V(rgn(0, aend(), 2), rgn(0, aend(), 2));

	// step => (whole, whole)

	double a1_all[36] = {
			1, 3, 5, 7, 9, 11,
			25, 27, 29, 31, 33, 35,
			49, 51, 53, 55, 57, 59,
			73, 75, 77, 79, 81, 83,
			97, 99, 101, 103, 105, 107,
			121, 123, 125, 127, 129, 131
	};

	BCS_CHECK_EQUAL( a1_rm.V(whole(), whole()), dense_aview2d(a1_all, 6, 6, row_major_t()) );
	BCS_CHECK_EQUAL( a1_cm.V(whole(), whole()), dense_aview2d(a1_all, 6, 6, column_major_t()) );

	// step => (i, whole)

	double a1_rm_v1[] = {49, 51, 53, 55, 57, 59};
	double a1_cm_v1[] = {5, 29, 53, 77, 101, 125};

	BCS_CHECK_EQUAL( a1_rm.V(2, whole()), aview1d<double>(a1_rm_v1, 6) );
	BCS_CHECK_EQUAL( a1_cm.V(2, whole()), aview1d<double>(a1_cm_v1, 6) );

	// step => (whole, j)

	double a1_rm_v2[] = {7, 31, 55, 79, 103, 127};
	double a1_cm_v2[] = {73, 75, 77, 79, 81, 83};

	BCS_CHECK_EQUAL( a1_rm.V(whole(), 3), aview1d<double>(a1_rm_v2, 6) );
	BCS_CHECK_EQUAL( a1_cm.V(whole(), 3), aview1d<double>(a1_cm_v2, 6) );

	// step => (i, range)

	double a1_rm_v3[] = {27, 29, 31, 33};
	double a1_cm_v3[] = {27, 51, 75, 99};

	BCS_CHECK_EQUAL( a1_rm.V(1, rgn(1, 5)), aview1d<double>(a1_rm_v3, 4) );
	BCS_CHECK_EQUAL( a1_cm.V(1, rgn(1, 5)), aview1d<double>(a1_cm_v3, 4) );

	// step => (step_range, i)

	double a1_rm_v4[] = {5, 53, 101};
	double a1_cm_v4[] = {49, 53, 57};

	BCS_CHECK_EQUAL( a1_rm.V(rgn(0, aend(), 2), 2), aview1d<double>(a1_rm_v4, 3) );
	BCS_CHECK_EQUAL( a1_cm.V(rgn(0, aend(), 2), 2), aview1d<double>(a1_cm_v4, 3) );



	// step => (range, step_range)

	double a1_rm_s1[] = {25, 29, 33, 49, 53, 57, 73, 77, 81, 97, 101, 105};
	double a1_cm_s1[] = {3, 5, 7, 9, 51, 53, 55, 57, 99, 101, 103, 105};

	BCS_CHECK_EQUAL( a1_rm.V(rgn(1, 5), rgn(0, aend(), 2)),  dense_aview2d(a1_rm_s1, 4, 3, row_major_t()) );
	BCS_CHECK_EQUAL( a1_cm.V(rgn(1, 5), rgn(0, aend(), 2)),  dense_aview2d(a1_cm_s1, 4, 3, column_major_t()) );

	// step => (rep_range, range)

	double a1_rm_s2[] = {51, 53, 55, 57, 51, 53, 55, 57, 51, 53, 55, 57};
	double a1_cm_s2[] = {29, 29, 29, 53, 53, 53, 77, 77, 77, 101, 101, 101};

	BCS_CHECK_EQUAL( a1_rm.V(rep(2, 3), rgn(1, 5)),  dense_aview2d(a1_rm_s2, 3, 4, row_major_t()) );
	BCS_CHECK_EQUAL( a1_cm.V(rep(2, 3), rgn(1, 5)),  dense_aview2d(a1_cm_s2, 3, 4, column_major_t()) );

	// step => (step_range, -step_range)

	double a1_rm_s3[] = {59, 55, 51, 107, 103, 99};
	double a1_cm_s3[] = {125, 129, 77, 81, 29, 33};

	BCS_CHECK_EQUAL( a1_rm.V(rgn(2, 5, 2), rgn(5, 0, -2)), dense_aview2d(a1_rm_s3, 2, 3, row_major_t()) );
	BCS_CHECK_EQUAL( a1_cm.V(rgn(2, 5, 2), rgn(5, 0, -2)), dense_aview2d(a1_cm_s3, 2, 3, column_major_t()) );

	// step => (indices, indices)

	index_t a1_s4_rinds[3] = {2, 0, 3};
	index_t a1_s4_cinds[4] = {4, 1, 5, 2};

	double a1_rm_s4[] = {57, 51, 59, 53, 9, 3, 11, 5, 81, 75, 83, 77};
	double a1_cm_s4[] = {101, 97, 103, 29, 25, 31, 125, 121, 127, 53, 49, 55};

	BCS_CHECK_EQUAL( a1_rm.V(indices(ref_arr(a1_s4_rinds, 3)), indices(ref_arr(a1_s4_cinds, 4))), dense_aview2d(a1_rm_s4, 3, 4, row_major_t()) );
	BCS_CHECK_EQUAL( a1_cm.V(indices(ref_arr(a1_s4_rinds, 3)), indices(ref_arr(a1_s4_cinds, 4))), dense_aview2d(a1_cm_s4, 3, 4, column_major_t()) );


	// rep x -step base

	aview2d<double, row_major_t, rep_ind, step_ind> a2_rm = Arm.V(rep(3, 5), rgn(11, 0, -2));
	aview2d<double, column_major_t, rep_ind, step_ind> a2_cm = Acm.V(rep(3, 5), rgn(11, 0, -2));

	// rep x -step => (whole, whole)

	double a2_rm_all[30] = {
			48, 46, 44, 42, 40, 38,
			48, 46, 44, 42, 40, 38,
			48, 46, 44, 42, 40, 38,
			48, 46, 44, 42, 40, 38,
			48, 46, 44, 42, 40, 38
	};
	double a2_cm_all[30] = {
			136, 136, 136, 136, 136,
			112, 112, 112, 112, 112,
			88, 88, 88, 88, 88,
			64, 64, 64, 64, 64,
			40, 40, 40, 40, 40,
			16, 16, 16, 16, 16
	};

	BCS_CHECK_EQUAL( a2_rm.V(whole(), whole()), dense_aview2d(a2_rm_all, 5, 6, row_major_t()) );
	BCS_CHECK_EQUAL( a2_cm.V(whole(), whole()), dense_aview2d(a2_cm_all, 5, 6, column_major_t()) );


	// rep x -step => (i, whole)

	double a2_rm_v1[] = {48, 46, 44, 42, 40, 38};
	double a2_cm_v1[] = {136, 112, 88, 64, 40, 16};

	BCS_CHECK_EQUAL( a2_rm.V(2, whole()), aview1d<double>(a2_rm_v1, 6) );
	BCS_CHECK_EQUAL( a2_cm.V(2, whole()), aview1d<double>(a2_cm_v1, 6) );

	// rep x -step => (whole, j)

	double a2_rm_v2[] = {42, 42, 42, 42, 42};
	double a2_cm_v2[] = {64, 64, 64, 64, 64};

	BCS_CHECK_EQUAL( a2_rm.V(whole(), 3), aview1d<double>(a2_rm_v2, 5) );
	BCS_CHECK_EQUAL( a2_cm.V(whole(), 3), aview1d<double>(a2_cm_v2, 5) );

	// rep x -step => (i, range)

	double a2_rm_v3[] = {46, 44, 42, 40};
	double a2_cm_v3[] = {112, 88, 64, 40};

	BCS_CHECK_EQUAL( a2_rm.V(1, rgn(1, 5)), aview1d<double>(a2_rm_v3, 4) );
	BCS_CHECK_EQUAL( a2_cm.V(1, rgn(1, 5)), aview1d<double>(a2_cm_v3, 4) );

	// rep x -step => (step_range, i)

	double a2_rm_v4[] = {44, 44, 44};
	double a2_cm_v4[] = {88, 88, 88};

	BCS_CHECK_EQUAL( a2_rm.V(rgn(0, aend(), 2), 2), aview1d<double>(a2_rm_v4, 3) );
	BCS_CHECK_EQUAL( a2_cm.V(rgn(0, aend(), 2), 2), aview1d<double>(a2_cm_v4, 3) );



	// rep x -step => (range, range)

	double a2_rm_s1[] = {46, 44, 42, 40, 46, 44, 42, 40};
	double a2_cm_s1[] = {112, 112, 88, 88, 64, 64, 40, 40};

	BCS_CHECK_EQUAL( a2_rm.V(rgn(1, 3), rgn(1, 5)), dense_aview2d(a2_rm_s1, 2, 4, row_major_t()) );
	BCS_CHECK_EQUAL( a2_cm.V(rgn(1, 3), rgn(1, 5)), dense_aview2d(a2_cm_s1, 2, 4, column_major_t()) );

	// rep x -step => (step_range, -step_range)

	double a2_rm_s2[] = {38, 40, 42, 44, 46, 48, 38, 40, 42, 44, 46, 48};
	double a2_cm_s2[] = {16, 16, 40, 40, 64, 64, 88, 88, 112, 112, 136, 136};

	BCS_CHECK_EQUAL( a2_rm.V(rgn(0, 3, 2), rev_whole()), dense_aview2d(a2_rm_s2, 2, 6, row_major_t()) );
	BCS_CHECK_EQUAL( a2_cm.V(rgn(0, 3, 2), rev_whole()), dense_aview2d(a2_cm_s2, 2, 6, column_major_t()) );

	// rep x -step => (range, indices)

	index_t a2_s3_cinds[4] = {0, 2, 3, 5};

	double a2_rm_s3[] = {48, 44, 42, 38, 48, 44, 42, 38, 48, 44, 42, 38};
	double a2_cm_s3[] = {136, 136, 136, 88, 88, 88, 64, 64, 64, 16, 16, 16};

	BCS_CHECK_EQUAL( a2_rm.V(rgn(0, 3), indices(ref_arr(a2_s3_cinds, 4))), dense_aview2d(a2_rm_s3, 3, 4, row_major_t()) );
	BCS_CHECK_EQUAL( a2_cm.V(rgn(0, 3), indices(ref_arr(a2_s3_cinds, 4))), dense_aview2d(a2_cm_s3, 3, 4, column_major_t()) );


	// rep x -step => (indices, indices)

	index_t a2_s4_rinds[3] = {0, 2, 3};
	index_t a2_s4_cinds[4] = {0, 2, 3, 5};

	double a2_rm_s4[] = {48, 44, 42, 38, 48, 44, 42, 38, 48, 44, 42, 38};
	double a2_cm_s4[] = {136, 136, 136, 88, 88, 88, 64, 64, 64, 16, 16, 16};

	BCS_CHECK_EQUAL( a2_rm.V(indices(ref_arr(a2_s4_rinds, 3)), indices(ref_arr(a2_s4_cinds, 4))), dense_aview2d(a2_rm_s4, 3, 4, row_major_t()) );
	BCS_CHECK_EQUAL( a2_cm.V(indices(ref_arr(a2_s4_rinds, 3)), indices(ref_arr(a2_s4_cinds, 4))), dense_aview2d(a2_cm_s4, 3, 4, column_major_t()) );


	// indices x step base

	index_t a3_rinds[5] = {1, 3, 4, 7, 9};

	aview2d<double, row_major_t, indices, step_ind> a3_rm = Arm.V(indices(ref_arr(a3_rinds, 5)), rgn(1, aend(), 2));
	aview2d<double, column_major_t, indices, step_ind> a3_cm = Acm.V(indices(ref_arr(a3_rinds, 5)), rgn(1, aend(), 2));

	// indices x step => (whole, whole)

	double a3_rm_all[30] = {
			14, 16, 18, 20, 22, 24,
			38, 40, 42, 44, 46, 48,
			50, 52, 54, 56, 58, 60,
			86, 88, 90, 92, 94, 96,
			110, 112, 114, 116, 118, 120
	};

	double a3_cm_all[30] = {
			14, 16, 17, 20, 22,
			38, 40, 41, 44, 46,
			62, 64, 65, 68, 70,
			86, 88, 89, 92, 94,
			110, 112, 113, 116, 118,
			134, 136, 137, 140, 142
	};

	BCS_CHECK_EQUAL( a3_rm.V(whole(), whole()), dense_aview2d(a3_rm_all, 5, 6, row_major_t()) );
	BCS_CHECK_EQUAL( a3_cm.V(whole(), whole()), dense_aview2d(a3_cm_all, 5, 6, column_major_t()) );


	// indices x step => (i, whole)

	double a3_rm_v1[] = {50, 52, 54, 56, 58, 60};
	double a3_cm_v1[] = {17, 41, 65, 89, 113, 137};

	BCS_CHECK_EQUAL( a3_rm.V(2, whole()), aview1d<double>(a3_rm_v1, 6) );
	BCS_CHECK_EQUAL( a3_cm.V(2, whole()), aview1d<double>(a3_cm_v1, 6) );

	// indices x step => (whole, j)

	double a3_rm_v2[] = {20, 44, 56, 92, 116};
	double a3_cm_v2[] = {86, 88, 89, 92, 94};

	BCS_CHECK_EQUAL( a3_rm.V(whole(), 3), aview1d<double>(a3_rm_v2, 5) );
	BCS_CHECK_EQUAL( a3_cm.V(whole(), 3), aview1d<double>(a3_cm_v2, 5) );

	// indices x step => (i, range)

	double a3_rm_v3[] = {40, 42, 44, 46};
	double a3_cm_v3[] = {40, 64, 88, 112};

	BCS_CHECK_EQUAL( a3_rm.V(1, rgn(1, 5)), aview1d<double>(a3_rm_v3, 4) );
	BCS_CHECK_EQUAL( a3_cm.V(1, rgn(1, 5)), aview1d<double>(a3_cm_v3, 4) );

	// indices x step => (step_range, i)

	double a3_rm_v4[] = {18, 54, 114};
	double a3_cm_v4[] = {62, 65, 70};

	BCS_CHECK_EQUAL( a3_rm.V(rgn(0, aend(), 2), 2), aview1d<double>(a3_rm_v4, 3) );
	BCS_CHECK_EQUAL( a3_cm.V(rgn(0, aend(), 2), 2), aview1d<double>(a3_cm_v4, 3) );


	// indices x step => (-step_range, range)

	double a3_rm_s1[15] = {112, 114, 116, 88, 90, 92, 52, 54, 56, 40, 42, 44, 16, 18, 20};
	double a3_cm_s1[15] = {46, 44, 41, 40, 38, 70, 68, 65, 64, 62, 94, 92, 89, 88, 86};

	BCS_CHECK_EQUAL( a3_rm.V(rev_whole(), rgn(1, 4)), dense_aview2d(a3_rm_s1, 5, 3, row_major_t()) );
	BCS_CHECK_EQUAL( a3_cm.V(rev_whole(), rgn(1, 4)), dense_aview2d(a3_cm_s1, 5, 3, column_major_t()) );

	// indices x step => (indices, indices)

	index_t a3_s2_rinds[3] = {0, 2, 3};
	index_t a3_s2_cinds[3] = {1, 3, 4};

	double a3_rm_s2[9] = {16, 20, 22, 52, 56, 58, 88, 92, 94};
	double a3_cm_s2[9] = {38, 41, 44, 86, 89, 92, 110, 113, 116};

	BCS_CHECK_EQUAL( a3_rm.V(indices(ref_arr(a3_s2_rinds, 3)), indices(ref_arr(a3_s2_cinds, 3))), dense_aview2d(a3_rm_s2, 3, 3, row_major_t()) );
	BCS_CHECK_EQUAL( a3_cm.V(indices(ref_arr(a3_s2_rinds, 3)), indices(ref_arr(a3_s2_cinds, 3))), dense_aview2d(a3_cm_s2, 3, 3, column_major_t()) );

	// indices x step => (step_range, rep)

	double a3_rm_s3[6] = {18, 18, 54, 54, 114, 114};
	double a3_cm_s3[6] = {62, 65, 70, 62, 65, 70};

	BCS_CHECK_EQUAL( a3_rm.V(rgn(0, aend(), 2), rep(2, 2)), dense_aview2d(a3_rm_s3, 3, 2, row_major_t()) );
	BCS_CHECK_EQUAL( a3_cm.V(rgn(0, aend(), 2), rep(2, 2)), dense_aview2d(a3_cm_s3, 3, 2, column_major_t()) );

}





test_suite *test_array2d_suite()
{
	test_suite *suite = new test_suite( "test_array2d" );

	suite->add( new test_dense_array2d() );
	suite->add( new test_gen_array2d() );
	suite->add( new test_array2d_slices() );
	suite->add( new test_array2d_subviews() );

	return suite;
}





