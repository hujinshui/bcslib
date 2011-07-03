/**
 * @file test_array2d.cpp
 *
 * The Unit Testing for array2d
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/test_units.h>
#include <bcslib/test/test_array_aux.h>
#include <bcslib/array/array2d.h>
#include <vector>

using namespace bcs;
using namespace bcs::test;

// Explicit instantiation for syntax checking

template class bcs::caview2d<double, row_major_t>;
template class bcs::caview2d<double, column_major_t>;

template class bcs::aview2d<double, row_major_t>;
template class bcs::aview2d<double, column_major_t>;

template class bcs::caview2d_ex<double, row_major_t, step_range, step_range>;
template class bcs::caview2d_ex<double, column_major_t, step_range, step_range>;
template class bcs::aview2d_ex<double, row_major_t, step_range, step_range>;
template class bcs::aview2d_ex<double, column_major_t, step_range, step_range>;

template class bcs::array2d<double, row_major_t>;
template class bcs::array2d<double, column_major_t>;


// Auxiliary test functions

template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
bool array_integrity_test(const bcs::caview2d_ex<T, TOrd, TIndexer0, TIndexer1>& view)
{
	index_t m = view.dim0();
	index_t n = view.dim1();

	if (view.ndims() != 2) return false;
	if (view.nrows() != (size_t)m) return false;
	if (view.ncolumns() != (size_t)n) return false;
	if (view.shape() != arr_shape(m, n)) return false;
	if (view.nelems() != (size_t)(m * n)) return false;

	return true;
}

template<typename T, typename TOrd>
bool array_integrity_test(const bcs::caview2d<T, TOrd>& view)
{
	index_t m = view.dim0();
	index_t n = view.dim1();

	if (view.ndims() != 2) return false;
	if (view.nrows() != (size_t)m) return false;
	if (view.ncolumns() != (size_t)n) return false;
	if (view.shape() != arr_shape(m, n)) return false;
	if (view.nelems() != (size_t)(m * n)) return false;

	if (std::is_same<TOrd, row_major_t>::value)
	{
		slice2d_info sli = view.slice_info();
		if (sli.nslices != view.nrows()) return false;
		if (sli.len != view.ncolumns()) return false;
		if (sli.stride != view.base_dim1()) return false;

		for (index_t i = 0; i < m; ++i)
		{
			for (index_t j = 0; j < n; ++j)
			{
				if (view.ptr(i, j) != &(view(i, j))) return false;
			}
		}
	}
	else
	{
		slice2d_info sli = view.slice_info();
		if (sli.nslices != view.ncolumns()) return false;
		if (sli.len != view.nrows()) return false;
		if (sli.stride != view.base_dim0()) return false;

		for (index_t j = 0; j < n; ++j)
		{
			for (index_t i = 0; i < m; ++i)
			{
				if (view.ptr(i, j) != &(view(i, j))) return false;
			}
		}
	}

	return true;
}


template<typename T, typename TOrd>
bool array_iteration_test(const bcs::caview2d<T, TOrd>& view)
{
	if (begin(view) != view.begin()) return false;
	if (end(view) != view.end()) return false;

	index_t m = view.dim0();
	index_t n = view.dim1();

	bcs::scoped_buffer<T> buf((size_t)(m * n));
	T *p = buf.pbase();

	if (std::is_same<TOrd, row_major_t>::value)
	{
		for (index_t i = 0; i < m; ++i)
		{
			for (index_t j = 0; j < n; ++j)
			{
				*p++ = view(i, j);
			}
		}
	}
	else
	{
		for (index_t j = 0; j < n; ++j)
		{
			for (index_t i = 0; i < m; ++i)
			{
				*p++ = view(i, j);
			}
		}
	}

	return collection_equal(view.begin(), view.end(), buf.pbase(), (size_t)(m * n));
}


template<typename T, typename TOrd>
bool test_generic_operations(bcs::aview2d<T, TOrd>& view, const double *src)
{
	block<T> blk(view.nelems());
	export_to(view, blk.pbase());

	if (!collection_equal(blk.pbase(), blk.pend(), src, view.nelems())) return false;

	index_t d0 = view.dim0();
	index_t d1 = view.dim1();
	set_zeros(view);
	for (index_t i = 0; i < d0; ++i)
	{
		for (index_t j = 0; j < d1; ++j)
		{
			if (view(i, j) != T(0)) return false;
		}
	}

	fill(view, T(2));
	for (index_t i = 0; i < d0; ++i)
	{
		for (index_t j = 0; j < d1; ++j)
		{
			if (view(i, j) != T(2)) return false;
		}
	}

	import_from(view, src);
	if (!array_view_equal(view, src, view.nrows(), view.ncolumns())) return false;

	return true;
}


template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
bool test_generic_operations(bcs::aview2d_ex<T, TOrd, TIndexer0, TIndexer1>& view, const double *src)
{
	block<T> blk(view.nelems());
	export_to(view, blk.pbase());

	if (!collection_equal(blk.pbase(), blk.pend(), src, view.nelems())) return false;

	index_t d0 = view.dim0();
	index_t d1 = view.dim1();
	for (index_t i = 0; i < d0; ++i)
	{
		for (index_t j = 0; j < d1; ++j)
		{
			view(i, j) = T(0);
		}
	}

	fill(view, T(2));
	for (index_t i = 0; i < d0; ++i)
	{
		for (index_t j = 0; j < d1; ++j)
		{
			if (view(i, j) != T(2)) return false;
		}
	}

	import_from(view, src);
	if (!array_view_equal(view, src, view.nrows(), view.ncolumns())) return false;

	return true;
}



// test cases


BCS_TEST_CASE( test_dense_array2d  )
{
	double src[24];
	for (int i = 0; i < 24; ++i) src[i] = i+1;
	index_t k = 0;

	size_t m = 2;
	size_t n = 3;

	double v2 = 7.0;

	// row major

	array2d<double, row_major_t> a0_rm(0, 0);

	BCS_CHECK( array_integrity_test(a0_rm) );
	BCS_CHECK( array_iteration_test(a0_rm) );

	array2d<double, row_major_t> a1_rm(m, n);
	k = 0;
	for (index_t i = 0; i < (index_t)m; ++i)
	{
		for (index_t j = 0; j < (index_t)n; ++j)
		{
			a1_rm(i, j) = src[k++];
		}
	}

	BCS_CHECK( array_integrity_test(a1_rm) );
	BCS_CHECK( array_view_equal(a1_rm, src, m, n) );
	BCS_CHECK( array_iteration_test(a1_rm) );
	BCS_CHECK( test_generic_operations(a1_rm, src) );

	array2d<double, row_major_t> a2_rm(m, n, v2);

	BCS_CHECK( array_integrity_test(a2_rm) );
	BCS_CHECK( array_view_equal(a2_rm, v2, m, n) );
	BCS_CHECK( array_iteration_test(a2_rm) );

	array2d<double, row_major_t> a3_rm(m, n, src);

	BCS_CHECK( array_integrity_test(a3_rm) );
	BCS_CHECK( array_view_equal(a3_rm, src, m, n) );
	BCS_CHECK( array_iteration_test(a3_rm) );
	BCS_CHECK( test_generic_operations(a3_rm, src) );

	array2d<double, row_major_t> a4_rm(a3_rm);

	BCS_CHECK( array_integrity_test(a3_rm) );
	BCS_CHECK( array_view_equal(a3_rm, src, m, n) );
	BCS_CHECK( array_iteration_test(a3_rm) );

	BCS_CHECK( a4_rm.pbase() != a3_rm.pbase() );
	BCS_CHECK( array_integrity_test(a4_rm) );
	BCS_CHECK( array_view_equal(a4_rm, src, m, n) );
	BCS_CHECK( array_iteration_test(a4_rm) );
	BCS_CHECK( test_generic_operations(a4_rm, src) );

	const double *p4_rm = a4_rm.pbase();
	array2d<double, row_major_t> a5_rm(std::move(a4_rm));

	BCS_CHECK( a4_rm.pbase() == BCS_NULL );
	BCS_CHECK( a4_rm.nelems() == 0 );

	BCS_CHECK( a5_rm.pbase() == p4_rm );
	BCS_CHECK( array_integrity_test(a5_rm) );
	BCS_CHECK( array_view_equal(a5_rm, src, m, n) );
	BCS_CHECK( array_iteration_test(a5_rm) );
	BCS_CHECK( test_generic_operations(a5_rm, src) );

	BCS_CHECK( a1_rm == a1_rm );
	array2d<double, row_major_t> a6_rm(a1_rm);
	BCS_CHECK( a1_rm == a6_rm );
	a6_rm(1, 1) += 1;
	BCS_CHECK( a1_rm != a6_rm );

	array2d<double, row_major_t> a7_rm = a1_rm.shared_copy();

	BCS_CHECK( a7_rm.pbase() == a1_rm.pbase() );
	BCS_CHECK( array_integrity_test(a7_rm) );
	BCS_CHECK( array_view_equal(a7_rm, src, m, n) );
	BCS_CHECK( array_iteration_test(a7_rm) );
	BCS_CHECK( test_generic_operations(a7_rm, src) );

	// column major

	array2d<double, column_major_t> a0_cm(0, 0);

	BCS_CHECK( array_integrity_test(a0_cm) );
	BCS_CHECK( array_iteration_test(a0_cm) );

	array2d<double, column_major_t> a1_cm(m, n);
	k = 0;
	for (index_t j = 0; j < (index_t)n; ++j)
	{
		for (index_t i = 0; i < (index_t)m; ++i)
		{
			a1_cm(i, j) = src[k++];
		}
	}

	BCS_CHECK( array_integrity_test(a1_cm) );
	BCS_CHECK( array_view_equal(a1_cm, src, m, n) );
	BCS_CHECK( array_iteration_test(a1_cm) );
	BCS_CHECK( test_generic_operations(a1_cm, src) );

	array2d<double, column_major_t> a2_cm(m, n, v2);

	BCS_CHECK( array_integrity_test(a2_cm) );
	BCS_CHECK( array_view_equal(a2_cm, v2, m, n) );
	BCS_CHECK( array_iteration_test(a2_cm) );

	array2d<double, column_major_t> a3_cm(m, n, src);

	BCS_CHECK( array_integrity_test(a3_cm) );
	BCS_CHECK( array_view_equal(a3_cm, src, m, n) );
	BCS_CHECK( array_iteration_test(a3_cm) );
	BCS_CHECK( test_generic_operations(a3_cm, src) );

	array2d<double, column_major_t> a4_cm(a3_cm);

	BCS_CHECK( array_integrity_test(a3_cm) );
	BCS_CHECK( array_view_equal(a3_cm, src, m, n) );
	BCS_CHECK( array_iteration_test(a3_cm) );

	BCS_CHECK( a4_cm.pbase() != a3_cm.pbase() );
	BCS_CHECK( array_integrity_test(a4_cm) );
	BCS_CHECK( array_view_equal(a4_cm, src, m, n) );
	BCS_CHECK( array_iteration_test(a4_cm) );
	BCS_CHECK( test_generic_operations(a4_cm, src) );

	const double *p4_cm = a4_cm.pbase();
	array2d<double, column_major_t> a5_cm(std::move(a4_cm));

	BCS_CHECK( a4_cm.pbase() == BCS_NULL );
	BCS_CHECK( a4_cm.nelems() == 0 );

	BCS_CHECK( a5_cm.pbase() == p4_cm );
	BCS_CHECK( array_integrity_test(a5_cm) );
	BCS_CHECK( array_view_equal(a5_cm, src, m, n) );
	BCS_CHECK( array_iteration_test(a5_cm) );
	BCS_CHECK( test_generic_operations(a5_cm, src) );

	BCS_CHECK( a1_cm == a1_cm );
	array2d<double, column_major_t> a6_cm(a1_cm);
	BCS_CHECK( a1_cm == a6_cm );
	a6_cm(1, 1) += 1;
	BCS_CHECK( a1_cm != a6_cm );

	array2d<double, column_major_t> a7_cm = a1_cm.shared_copy();

	BCS_CHECK( a7_cm.pbase() == a1_cm.pbase() );
	BCS_CHECK( array_integrity_test(a7_cm) );
	BCS_CHECK( array_view_equal(a7_cm, src, m, n) );
	BCS_CHECK( array_iteration_test(a7_cm) );
	BCS_CHECK( test_generic_operations(a7_cm, src) );

}


BCS_TEST_CASE( test_aview2d_ex )
{
	double src[36];
	for (int i = 0; i < 36; ++i) src[i] = i+1;

	// rgn x rgn

	aview2d<double, row_major_t> a0_rm(src, 2, 3, 5, 6);
	double r0_rm[] = {1, 2, 3, 7, 8, 9};

	BCS_CHECK( array_integrity_test(a0_rm) );
	BCS_CHECK( array_view_equal(a0_rm, r0_rm, 2, 3) );
	BCS_CHECK( test_generic_operations(a0_rm, r0_rm) );

	aview2d<double, column_major_t> a0_cm(src, 2, 3, 5, 6);
	double r0_cm[] = {1, 2, 6, 7, 11, 12};

	BCS_CHECK( array_integrity_test(a0_cm) );
	BCS_CHECK( array_view_equal(a0_cm, r0_cm, 2, 3) );
	BCS_CHECK( test_generic_operations(a0_cm, r0_cm) );


	// rgn x step_rgn

	aview2d_ex<double, row_major_t, range, step_range> a1_rm(src, 5, 6, rgn_n(0, 2), rgn_n(0, 3, 2));
	double r1_rm[] = {1, 3, 5, 7, 9, 11};

	BCS_CHECK( array_integrity_test(a1_rm) );
	BCS_CHECK( array_view_equal(a1_rm, r1_rm, 2, 3) );
	BCS_CHECK( test_generic_operations(a1_rm, r1_rm) );

	aview2d_ex<double, column_major_t, range, step_range> a1_cm(src, 5, 6, rgn_n(0, 2), rgn_n(0, 3, 2));
	double r1_cm[] = {1, 2, 11, 12, 21, 22};

	BCS_CHECK( array_integrity_test(a1_cm) );
	BCS_CHECK( array_view_equal(a1_cm, r1_cm, 2, 3) );
	BCS_CHECK( test_generic_operations(a1_cm, r1_cm) );

	// step_rgn x step_rgn

	aview2d_ex<double, row_major_t, step_range, step_range> a2_rm(src, 5, 6, rgn_n(0, 2, 2), rgn_n(0, 3, 2));
	double r2_rm[] = {1, 3, 5, 13, 15, 17};

	BCS_CHECK( array_integrity_test(a2_rm) );
	BCS_CHECK( array_view_equal(a2_rm, r2_rm, 2, 3) );
	BCS_CHECK( test_generic_operations(a2_rm, r2_rm) );

	aview2d_ex<double, column_major_t, step_range, step_range> a2_cm(src, 5, 6, rgn_n(0, 2, 2), rgn_n(0, 3, 2));
	double r2_cm[] = {1, 3, 11, 13, 21, 23};

	BCS_CHECK( array_integrity_test(a2_cm) );
	BCS_CHECK( array_view_equal(a2_cm, r2_cm, 2, 3) );
	BCS_CHECK( test_generic_operations(a2_cm, r2_cm) );

	// rep_rgn x rep_rgn

	aview2d_ex<double, row_major_t, rep_range, rep_range> a3_rm(src, 5, 6, rep(1, 2), rep(2, 3));
	double r3_rm[] = {9, 9, 9, 9, 9, 9};

	BCS_CHECK( array_integrity_test(a3_rm) );
	BCS_CHECK( array_view_equal(a3_rm, r3_rm, 2, 3) );
	BCS_CHECK( test_generic_operations(a3_rm, r3_rm) );

	aview2d_ex<double, column_major_t, rep_range, rep_range> a3_cm(src, 5, 6, rep(1, 2), rep(2, 3));
	double r3_cm[] = {12, 12, 12, 12, 12, 12};

	BCS_CHECK( array_integrity_test(a3_cm) );
	BCS_CHECK( array_view_equal(a3_cm, r3_cm, 2, 3) );
	BCS_CHECK( test_generic_operations(a3_cm, r3_cm) );
}



BCS_TEST_CASE( test_array2d_slices )
{
	double src[60];
	for (int i = 0; i < 60; ++i) src[i] = i+1;

	aview2d<double, row_major_t> Arm(src, 5, 6, 7, 8);
	aview2d<double, column_major_t> Acm(src, 5, 6, 7, 8);

	// row

	double r0_rm[6] = {9, 10, 11, 12, 13, 14};
	double r0_cm[6] = {2, 9, 16, 23, 30, 37};

	BCS_CHECK( array_view_equal( Arm.row(1), r0_rm, 6 ) );
	BCS_CHECK( array_view_equal( Acm.row(1), r0_cm, 6 ) );

	// row range

	double r1_rm[4] = {10, 11, 12, 13};
	double r1_cm[4] = {9, 16, 23, 30};

	BCS_CHECK( array_view_equal( Arm.row(1, rgn_n(1, 4)), r1_rm, 4 ) );
	BCS_CHECK( array_view_equal( Acm.row(1, rgn_n(1, 4)), r1_cm, 4 ) );

	double r2_rm[3] = {9, 11, 13};
	double r2_cm[3] = {2, 16, 30};

	BCS_CHECK( array_view_equal( Arm.row(1, rgn_n(0, 3, 2)), r2_rm, 3 ) );
	BCS_CHECK( array_view_equal( Acm.row(1, rgn_n(0, 3, 2)), r2_cm, 3 ) );

	double r3_rm[3] = {11, 11, 11};
	double r3_cm[3] = {16, 16, 16};

	BCS_CHECK( array_view_equal( Arm.row(1, rep(2, 3)), r3_rm, 3 ) );
	BCS_CHECK( array_view_equal( Acm.row(1, rep(2, 3)), r3_cm, 3 ) );

	// column

	double c0_rm[5] = {2, 10, 18, 26, 34};
	double c0_cm[5] = {8, 9, 10, 11, 12};

	BCS_CHECK( array_view_equal( Arm.column(1), c0_rm, 5 ) );
	BCS_CHECK( array_view_equal( Acm.column(1), c0_cm, 5 ) );

	// column range

	double c1_rm[4] = {10, 18, 26, 34};
	double c1_cm[4] = {9, 10, 11, 12};

	BCS_CHECK( array_view_equal( Arm.column(1, rgn_n(1, 4)), c1_rm, 4 ) );
	BCS_CHECK( array_view_equal( Acm.column(1, rgn_n(1, 4)), c1_cm, 4 ) );

	double c2_rm[3] = {2, 18, 34};
	double c2_cm[3] = {8, 10, 12};

	BCS_CHECK( array_view_equal( Arm.column(1, rgn_n(0, 3, 2)), c2_rm, 3 ) );
	BCS_CHECK( array_view_equal( Acm.column(1, rgn_n(0, 3, 2)), c2_cm, 3 ) );

	double c3_rm[3] = {18, 18, 18};
	double c3_cm[3] = {10, 10, 10};

	BCS_CHECK( array_view_equal( Arm.column(1, rep(2, 3)), c3_rm, 3 ) );
	BCS_CHECK( array_view_equal( Acm.column(1, rep(2, 3)), c3_cm, 3 ) );

}


BCS_TEST_CASE( test_array2d_subviews )
{
	double src0[60];
	for (int i = 0; i < 60; ++i) src0[i] = i+1;

	const aview2d<double, row_major_t> a0_rm(src0, 5, 6, 7, 8);
	const aview2d<double, column_major_t> a0_cm(src0, 5, 6, 7, 8);

	// (whole, whole)

	double a0_rm_s0[] = {
			1, 2, 3, 4, 5, 6,
			9, 10, 11, 12, 13, 14,
			17, 18, 19, 20, 21, 22,
			25, 26, 27, 28, 29, 30,
			33, 34, 35, 36, 37, 38
	};

	double a0_cm_s0[] = {
			1, 2, 3, 4, 5,
			8, 9, 10, 11, 12,
			15, 16, 17, 18, 19,
			22, 23, 24, 25, 26,
			29, 30, 31, 32, 33,
			36, 37, 38, 39, 40
	};

	BCS_CHECK_EQUAL( a0_rm.V(whole(), whole()),  get_aview2d_rm(a0_rm_s0, 5, 6) );
	BCS_CHECK_EQUAL( a0_cm.V(whole(), whole()),  get_aview2d_cm(a0_cm_s0, 5, 6) );

	// (range, whole)

	double a0_rm_s1[] = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22};
	double a0_cm_s1[] = {1, 2, 3, 8, 9, 10, 15, 16, 17, 22, 23, 24, 29, 30, 31, 36, 37, 38};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(0, 3), whole()),  get_aview2d_rm(a0_rm_s1, 3, 6) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(0, 3), whole()),  get_aview2d_cm(a0_cm_s1, 3, 6) );

	// (whole, range)

	double a0_rm_s2[] = {2, 3, 4, 10, 11, 12, 18, 19, 20, 26, 27, 28, 34, 35, 36};
	double a0_cm_s2[] = {8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26};

	BCS_CHECK_EQUAL( a0_rm.V(whole(), rgn(1, 4)),  get_aview2d_rm(a0_rm_s2, 5, 3) );
	BCS_CHECK_EQUAL( a0_cm.V(whole(), rgn(1, 4)),  get_aview2d_cm(a0_cm_s2, 5, 3) );

	// (range, range)

	double a0_rm_s3[] = {18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37};
	double a0_cm_s3[] = {10, 11, 12, 17, 18, 19, 24, 25, 26, 31, 32, 33};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(2, 5), rgn(1, 5)),  get_aview2d_rm(a0_rm_s3, 3, 4) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(2, 5), rgn(1, 5)),  get_aview2d_cm(a0_cm_s3, 3, 4) );

	// (range, step_range)

	double a0_rm_s4[] = {18, 20, 22, 26, 28, 30, 34, 36, 38};
	double a0_cm_s4[] = {10, 11, 12, 24, 25, 26, 38, 39, 40};

	BCS_CHECK( array_view_equal(a0_rm.V(rgn(2, 5), rgn(1, 6, 2)),  a0_rm_s4, 3, 3) );
	BCS_CHECK( array_view_equal(a0_cm.V(rgn(2, 5), rgn(1, 6, 2)),  a0_cm_s4, 3, 3) );

	// (step_range, step_range)

	double a0_rm_s5[] = {2, 4, 6, 18, 20, 22, 34, 36, 38};
	double a0_cm_s5[] = {8, 10, 12, 22, 24, 26, 36, 38, 40};

	BCS_CHECK( array_view_equal(a0_rm.V(rgn(0, 5, 2), rgn(1, 6, 2)), a0_rm_s5, 3, 3) );
	BCS_CHECK( array_view_equal(a0_cm.V(rgn(0, 5, 2), rgn(1, 6, 2)), a0_cm_s5, 3, 3) );

	// (step_range, rev_whole)

	double a0_rm_s6[] = {6, 5, 4, 3, 2, 1, 22, 21, 20, 19, 18, 17, 38, 37, 36, 35, 34, 33};
	double a0_cm_s6[] = {36, 38, 40, 29, 31, 33, 22, 24, 26, 15, 17, 19, 8, 10, 12, 1, 3, 5};

	BCS_CHECK( array_view_equal(a0_rm.V(rgn(0, 5, 2), rev_whole()), a0_rm_s6, 3, 6) );
	BCS_CHECK( array_view_equal(a0_cm.V(rgn(0, 5, 2), rev_whole()), a0_cm_s6, 3, 6) );
}


BCS_TEST_CASE( test_aview2d_clone )
{
	double src[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

	aview2d<double, row_major_t> view1_rm(src, 3, 5);
	array2d<double, row_major_t> a1_rm = clone_array(view1_rm);

	BCS_CHECK( array_integrity_test(a1_rm) );
	BCS_CHECK( array_iteration_test(a1_rm) );
	BCS_CHECK( a1_rm.pbase() != view1_rm.pbase() );
	BCS_CHECK_EQUAL( a1_rm, view1_rm );

	aview2d_ex<double, row_major_t, step_range, step_range> view2_rm(src, 4, 5, rgn_n(0, 2, 2), rgn_n(0, 3, 2));
	array2d<double, row_major_t> a2_rm = clone_array(view2_rm);

	double a2_rm_s[6] = {1, 3, 5, 11, 13, 15};
	BCS_CHECK( array_integrity_test(a2_rm) );
	BCS_CHECK( array_iteration_test(a2_rm) );
	BCS_CHECK( array_view_equal(a2_rm, a2_rm_s, 2, 3) );

	aview2d<double, column_major_t> view1_cm(src, 3, 5);
	array2d<double, column_major_t> a1_cm = clone_array(view1_cm);

	BCS_CHECK( array_integrity_test(a1_cm) );
	BCS_CHECK( array_iteration_test(a1_cm) );
	BCS_CHECK( a1_cm.pbase() != view1_cm.pbase() );
	BCS_CHECK_EQUAL( a1_cm, view1_cm );

	aview2d_ex<double, column_major_t, step_range, step_range> view2_cm(src, 4, 5, rgn_n(0, 2, 2), rgn_n(0, 3, 2));
	array2d<double, column_major_t> a2_cm = clone_array(view2_cm);

	double a2_cm_s[6] = {1, 3, 9, 11, 17, 19};
	BCS_CHECK( array_integrity_test(a2_cm) );
	BCS_CHECK( array_iteration_test(a2_cm) );
	BCS_CHECK( array_view_equal(a2_cm, a2_cm_s, 2, 3) );
}


BCS_TEST_CASE( test_subarr_selection2d )
{
	const size_t m0 = 6;
	const size_t n0 = 6;
	double src[m0 * n0];
	for (size_t i = 0; i < m0 * n0; ++i) src[i] = (double)(i + 1);

	caview2d<double, row_major_t>    Arm = get_aview2d_rm(src, m0, n0);
	caview2d<double, column_major_t> Acm = get_aview2d_cm(src, m0, n0);

	// select_elems

	const size_t sn0 = 6;
	index_t Is[sn0] = {1, 3, 4, 5, 2, 0};
	index_t Js[sn0] = {2, 2, 4, 5, 0, 0};
	std::vector<index_t> vIs(Is, Is+sn0);
	std::vector<index_t> vJs(Js, Js+sn0);

	array1d<double> s0_rm = select_elems(Arm, sn0, vIs.begin(), vJs.begin());
	array1d<double> s0_cm = select_elems(Acm, sn0, vIs.begin(), vJs.begin());

	double s0_rm_r[sn0] = {9, 21, 29, 36, 13, 1};
	double s0_cm_r[sn0] = {14, 16, 29, 36, 3, 1};

	BCS_CHECK( array_view_equal(s0_rm, s0_rm_r, sn0) );
	BCS_CHECK( array_view_equal(s0_cm, s0_cm_r, sn0) );

	// select_rows

	const size_t sn1 = 3;
	index_t rs[sn1] = {1, 4, 5};

	std::vector<index_t> rows(rs, rs+sn1);

	array2d<double, row_major_t>    s1_rm = select_rows(Arm, sn1, rows.begin());
	array2d<double, column_major_t> s1_cm = select_rows(Acm, sn1, rows.begin());

	double s1_rm_r[sn1 * n0] = {7, 8, 9, 10, 11, 12, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
	double s1_cm_r[sn1 * n0] = {2, 5, 6, 8, 11, 12, 14, 17, 18, 20, 23, 24, 26, 29, 30, 32, 35, 36};

	BCS_CHECK( array_view_equal(s1_rm, s1_rm_r, sn1, n0) );
	BCS_CHECK( array_view_equal(s1_cm, s1_cm_r, sn1, n0) );

	// select_columns

	const size_t sn2 = 3;
	index_t cs[sn2] = {2, 3, 5};

	std::vector<index_t> cols(cs, cs+sn2);

	array2d<double, row_major_t>    s2_rm = select_columns(Arm, sn2, cols.begin());
	array2d<double, column_major_t> s2_cm = select_columns(Acm, sn2, cols.begin());

	double s2_rm_r[m0 * sn2] = {3, 4, 6, 9, 10, 12, 15, 16, 18, 21, 22, 24, 27, 28, 30, 33, 34, 36};
	double s2_cm_r[m0 * sn2] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36};

	BCS_CHECK( array_view_equal(s2_rm, s2_rm_r, m0, sn2) );
	BCS_CHECK( array_view_equal(s2_cm, s2_cm_r, m0, sn2) );

	// select_rows_and_cols

	const size_t sm3 = 2;  index_t rs3[sm3] = {2, 4};
	const size_t sn3 = 3;  index_t cs3[sn3] = {1, 3, 5};

	std::vector<index_t> rows3(rs3, rs3+sm3);
	std::vector<index_t> cols3(cs3, cs3+sn3);

	array2d<double, row_major_t>    s3_rm = select_rows_and_cols(Arm, sm3, rows3.begin(), sn3, cols3.begin());
	array2d<double, column_major_t> s3_cm = select_rows_and_cols(Acm, sm3, rows3.begin(), sn3, cols3.begin());

	double s3_rm_r[sm3 * sn3] = {14, 16, 18, 26, 28, 30};
	double s3_cm_r[sm3 * sn3] = {9, 11, 21, 23, 33, 35};

	BCS_CHECK( array_view_equal(s3_rm, s3_rm_r, sm3, sn3) );
	BCS_CHECK( array_view_equal(s3_cm, s3_cm_r, sm3, sn3) );

}



test_suite *test_array2d_suite()
{
	test_suite *suite = new test_suite( "test_array2d" );

	suite->add( new test_dense_array2d() );
	suite->add( new test_aview2d_ex() );
	suite->add( new test_array2d_slices() );
	suite->add( new test_array2d_subviews() );
	suite->add( new test_aview2d_clone() );
	suite->add( new test_subarr_selection2d() );

	return suite;
}





