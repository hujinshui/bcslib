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

template class bcs::aview2d<double, row_major_t,    id_ind, id_ind>;
template class bcs::aview2d<double, column_major_t, id_ind, id_ind>;
template class bcs::aview2d<double, row_major_t,    id_ind, step_ind>;
template class bcs::aview2d<double, column_major_t, id_ind, step_ind>;
template class bcs::aview2d<double, row_major_t,    id_ind, arr_ind>;
template class bcs::aview2d<double, column_major_t, id_ind, arr_ind>;

template class bcs::aview2d<double, row_major_t,    step_ind, id_ind>;
template class bcs::aview2d<double, column_major_t, step_ind, id_ind>;
template class bcs::aview2d<double, row_major_t,    step_ind, step_ind>;
template class bcs::aview2d<double, column_major_t, step_ind, step_ind>;
template class bcs::aview2d<double, row_major_t,    step_ind, arr_ind>;
template class bcs::aview2d<double, column_major_t, step_ind, arr_ind>;

template class bcs::aview2d<double, row_major_t,    arr_ind, id_ind>;
template class bcs::aview2d<double, column_major_t, arr_ind, id_ind>;
template class bcs::aview2d<double, row_major_t,    arr_ind, step_ind>;
template class bcs::aview2d<double, column_major_t, arr_ind, step_ind>;
template class bcs::aview2d<double, row_major_t,    arr_ind, arr_ind>;
template class bcs::aview2d<double, column_major_t, arr_ind, arr_ind>;

template class bcs::array2d<double, row_major_t>;
template class bcs::array2d<double, column_major_t>;


// A class for concept checked

template<class Arr>
class array_view2d_concept_check
{
	BCS_STATIC_ASSERT_V(is_array_view<Arr>);
	static_assert(is_array_view_ndim<Arr, 2>::value, "is_array_view_ndim<Arr, 2>");

	typedef typename array_view_traits<Arr>::value_type value_type;
	typedef std::array<index_t, 2> shape_type;

	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::size_type, size_t);
	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::index_type, index_t);
	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::const_reference, const value_type&);
	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::reference, value_type&);
	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::const_pointer, const value_type*);
	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::shape_type, shape_type);
	BCS_STATIC_ASSERT(array_view_traits<Arr>::num_dims == 2);

	void check_const(const Arr& a)
	{
		BCS_ASSERT_SAME_TYPE(decltype(get_num_elems(a)), size_t);
		BCS_ASSERT_SAME_TYPE(decltype(get_array_shape(a)), shape_type);

		BCS_ASSERT_SAME_TYPE(decltype(begin(a)), typename array_view_traits<Arr>::const_iterator);
		BCS_ASSERT_SAME_TYPE(decltype(end(a)), typename array_view_traits<Arr>::const_iterator);

		BCS_ASSERT_SAME_TYPE(decltype(is_dense_view(a)), bool);
		BCS_ASSERT_SAME_TYPE(decltype(ptr_base(a)), const value_type*);
	}

	void check_non_const(Arr& a)
	{
		BCS_ASSERT_SAME_TYPE(decltype(get_num_elems(a)), size_t);
		BCS_ASSERT_SAME_TYPE(decltype(get_array_shape(a)), shape_type);

		BCS_ASSERT_SAME_TYPE(decltype(begin(a)), typename array_view_traits<Arr>::iterator);
		BCS_ASSERT_SAME_TYPE(decltype(end(a)), typename array_view_traits<Arr>::iterator);

		BCS_ASSERT_SAME_TYPE(decltype(is_dense_view(a)), bool);
		BCS_ASSERT_SAME_TYPE(decltype(ptr_base(a)), value_type*);
	}
};

template class array_view2d_concept_check<bcs::aview2d<double, row_major_t,    id_ind,   id_ind> >;
template class array_view2d_concept_check<bcs::aview2d<double, column_major_t, id_ind,   id_ind> >;
template class array_view2d_concept_check<bcs::aview2d<double, row_major_t,    step_ind, id_ind> >;
template class array_view2d_concept_check<bcs::aview2d<double, column_major_t, step_ind, id_ind> >;

template class array_view2d_concept_check<bcs::array2d<double, row_major_t> >;
template class array_view2d_concept_check<bcs::array2d<double, column_major_t> >;


// Auxiliary test functions




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
bool array_integrity_test(const bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& view)
{
	index_t m = view.dim0();
	index_t n = view.dim1();

	if (view.ndims() != 2) return false;
	if (view.nrows() != (size_t)m) return false;
	if (view.ncolumns() != (size_t)n) return false;
	if (view.shape() != arr_shape(m, n)) return false;
	if (view.nelems() != (size_t)(m * n)) return false;

	if (get_num_elems(view) != view.nelems()) return false;
	if (get_array_shape(view) != view.shape()) return false;
	if (ptr_base(view) != view.pbase()) return false;

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			if (view.ptr(i, j) != &(view(i, j))) return false;
		}
	}

	return true;
}


template<typename T, class TIndexer0, class TIndexer1>
bool array_iteration_test(const bcs::aview2d<T, row_major_t, TIndexer0, TIndexer1>& view)
{
	if (begin(view) != view.begin()) return false;
	if (end(view) != view.end()) return false;

	index_t m = view.dim0();
	index_t n = view.dim1();

	bcs::block<T> buffer((size_t)(m * n));
	T *p = buffer.pbase();

	for (index_t i = 0; i < m; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			*p++ = view(i, j);
		}
	}

	return collection_equal(view.begin(), view.end(), buffer.pbase(), (size_t)(m * n));
}


template<typename T, class TIndexer0, class TIndexer1>
bool array_iteration_test(const bcs::aview2d<T, column_major_t, TIndexer0, TIndexer1>& view)
{
	if (begin(view) != view.begin()) return false;
	if (end(view) != view.end()) return false;

	index_t m = view.dim0();
	index_t n = view.dim1();

	bcs::block<T> buffer((size_t)(m * n));
	T *p = buffer.pbase();

	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i)
		{
			*p++ = view(i, j);
		}
	}

	return collection_equal(view.begin(), view.end(), buffer.pbase(), (size_t)(m * n));
}

template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
bool test_generic_operations(bcs::aview2d<T, TOrd, TIndexer0, TIndexer1>& view, const double *src)
{
	block<T> blk(view.nelems());
	export_to(view, blk.pbase());

	if (!collection_equal(blk.pbase(), blk.pend(), src, view.nelems())) return false;

	index_t d0 = view.dim0();
	index_t d1 = view.dim1();
	if (is_dense_view(view))
	{
		set_zeros(view);
		for (index_t i = 0; i < d0; ++i)
		{
			for (index_t j = 0; j < d1; ++j)
			{
				if (view(i, j) != T(0)) return false;
			}
		}
	}

	fill(view, T(1));
	for (index_t i = 0; i < d0; ++i)
	{
		for (index_t j = 0; j < d1; ++j)
		{
			if (view(i, j) != T(1)) return false;
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

	array2d<double, row_major_t> a2_rm(m, n, v2);

	BCS_CHECK( array_integrity_test(a2_rm) );
	BCS_CHECK( array_view_equal(a2_rm, v2, m, n) );
	BCS_CHECK( array_iteration_test(a2_rm) );

	array2d<double, row_major_t> a3_rm(m, n, src);

	BCS_CHECK( array_integrity_test(a3_rm) );
	BCS_CHECK( array_view_equal(a3_rm, src, m, n) );
	BCS_CHECK( array_iteration_test(a3_rm) );

	array2d<double, row_major_t> a4_rm(a3_rm);

	BCS_CHECK( array_integrity_test(a3_rm) );
	BCS_CHECK( array_view_equal(a3_rm, src, m, n) );
	BCS_CHECK( array_iteration_test(a3_rm) );

	BCS_CHECK( a4_rm.pbase() != a3_rm.pbase() );
	BCS_CHECK( array_integrity_test(a4_rm) );
	BCS_CHECK( array_view_equal(a4_rm, src, m, n) );
	BCS_CHECK( array_iteration_test(a4_rm) );

	const double *p4_rm = a4_rm.pbase();
	array2d<double, row_major_t> a5_rm(std::move(a4_rm));

	BCS_CHECK( a4_rm.pbase() == BCS_NULL );
	BCS_CHECK( a4_rm.nelems() == 0 );

	BCS_CHECK( a5_rm.pbase() == p4_rm );
	BCS_CHECK( array_integrity_test(a5_rm) );
	BCS_CHECK( array_view_equal(a5_rm, src, m, n) );
	BCS_CHECK( array_iteration_test(a5_rm) );

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

	BCS_CHECK( test_generic_operations(a1_rm, src) );

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

	array2d<double, column_major_t> a2_cm(m, n, v2);

	BCS_CHECK( array_integrity_test(a2_cm) );
	BCS_CHECK( array_view_equal(a2_cm, v2, m, n) );
	BCS_CHECK( array_iteration_test(a2_cm) );

	array2d<double, column_major_t> a3_cm(m, n, src);

	BCS_CHECK( array_integrity_test(a3_cm) );
	BCS_CHECK( array_view_equal(a3_cm, src, m, n) );
	BCS_CHECK( array_iteration_test(a3_cm) );

	array2d<double, column_major_t> a4_cm(a3_cm);

	BCS_CHECK( array_integrity_test(a3_cm) );
	BCS_CHECK( array_view_equal(a3_cm, src, m, n) );
	BCS_CHECK( array_iteration_test(a3_cm) );

	BCS_CHECK( a4_cm.pbase() != a3_cm.pbase() );
	BCS_CHECK( array_integrity_test(a4_cm) );
	BCS_CHECK( array_view_equal(a4_cm, src, m, n) );
	BCS_CHECK( array_iteration_test(a4_cm) );

	const double *p4_cm = a4_cm.pbase();
	array2d<double, column_major_t> a5_cm(std::move(a4_cm));

	BCS_CHECK( a4_cm.pbase() == BCS_NULL );
	BCS_CHECK( a4_cm.nelems() == 0 );

	BCS_CHECK( a5_cm.pbase() == p4_cm );
	BCS_CHECK( array_integrity_test(a5_cm) );
	BCS_CHECK( array_view_equal(a5_cm, src, m, n) );
	BCS_CHECK( array_iteration_test(a5_cm) );

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

	BCS_CHECK( test_generic_operations(a1_cm, src) );

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

	aview2d<double, row_major_t, arr_ind, id_ind> a4_rm(src, 5, 6, arr_ind(3, inds), id_ind(4));
	double r4_rm[] = {1, 2, 3, 4, 13, 14, 15, 16, 19, 20, 21, 22};

	BCS_CHECK( !is_dense_view(a4_rm) );
	BCS_CHECK( array_integrity_test(a4_rm) );
	BCS_CHECK( array_view_equal(a4_rm, r4_rm, 3, 4) );
	BCS_CHECK( array_iteration_test(a4_rm) );

	aview2d<double, column_major_t, arr_ind, id_ind> a4_cm(src, 5, 6, arr_ind(3, inds), id_ind(4));
	double r4_cm[] = {1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18, 19};

	BCS_CHECK( !is_dense_view(a4_cm) );
	BCS_CHECK( array_integrity_test(a4_cm) );
	BCS_CHECK( array_view_equal(a4_cm, r4_cm, 3, 4) );
	BCS_CHECK( array_iteration_test(a4_cm) );

	// (indices, step_ind(2))

	aview2d<double, row_major_t, arr_ind, step_ind> a5_rm(src, 5, 6, arr_ind(3, inds), step_ind(3, 2));
	double r5_rm[] = {1, 3, 5, 13, 15, 17, 19, 21, 23};

	BCS_CHECK( !is_dense_view(a5_rm) );
	BCS_CHECK( array_integrity_test(a5_rm) );
	BCS_CHECK( array_view_equal(a5_rm, r5_rm, 3, 3) );
	BCS_CHECK( array_iteration_test(a5_rm) );

	aview2d<double, column_major_t, arr_ind, step_ind> a5_cm(src, 5, 6, arr_ind(3, inds), step_ind(3, 2));
	double r5_cm[] = {1, 3, 4, 11, 13, 14, 21, 23, 24};

	BCS_CHECK( !is_dense_view(a5_cm) );
	BCS_CHECK( array_integrity_test(a5_cm) );
	BCS_CHECK( array_view_equal(a5_cm, r5_cm, 3, 3) );
	BCS_CHECK( array_iteration_test(a5_cm) );

	// (indices, indices)

	aview2d<double, row_major_t, arr_ind, arr_ind> a6_rm(src, 5, 6, arr_ind(2, inds), arr_ind(3, inds));
	double r6_rm[] = {1, 3, 4, 13, 15, 16};

	BCS_CHECK( !is_dense_view(a6_rm) );
	BCS_CHECK( array_integrity_test(a6_rm) );
	BCS_CHECK( array_view_equal(a6_rm, r6_rm, 2, 3) );
	BCS_CHECK( array_iteration_test(a6_rm) );

	aview2d<double, column_major_t, arr_ind, arr_ind> a6_cm(src, 5, 6, arr_ind(2, inds), arr_ind(3, inds));
	double r6_cm[] = {1, 3, 11, 13, 16, 18};

	BCS_CHECK( !is_dense_view(a6_cm) );
	BCS_CHECK( array_integrity_test(a6_cm) );
	BCS_CHECK( array_view_equal(a6_cm, r6_cm, 2, 3) );
	BCS_CHECK( array_iteration_test(a6_cm) );

	// (rep_ind, indices)

	aview2d<double, row_major_t, rep_ind, arr_ind> a7_rm(src, 5, 6, rep_ind(2), arr_ind(3, inds));
	double r7_rm[] = {1, 3, 4, 1, 3, 4};

	BCS_CHECK( !is_dense_view(a7_rm) );
	BCS_CHECK( array_integrity_test(a7_rm) );
	BCS_CHECK( array_view_equal(a7_rm, r7_rm, 2, 3) );
	BCS_CHECK( array_iteration_test(a7_rm) );

	aview2d<double, column_major_t, rep_ind, arr_ind> a7_cm(src, 5, 6, rep_ind(2), arr_ind(3, inds));
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
	BCS_CHECK_EQUAL( a1_rm.row(0), aview1d<double>(a1_rm_r1, 4) );
	BCS_CHECK_EQUAL( a1_rm.row(1), aview1d<double>(a1_rm_r2, 4) );

	double a1_rm_c1[] = {1, 7, 13};
	double a1_rm_c2[] = {2, 8, 14};
	BCS_CHECK_EQUAL( a1_rm.column(0), aview1d<double>(a1_rm_c1, 3) );
	BCS_CHECK_EQUAL( a1_rm.column(1), aview1d<double>(a1_rm_c2, 3) );

	aview2d<double, column_major_t, id_ind, id_ind> a1_cm(src, 5, 6, id_ind(3), id_ind(4));

	double a1_cm_r1[] = {1, 6, 11, 16};
	double a1_cm_r2[] = {2, 7, 12, 17};
	BCS_CHECK_EQUAL( a1_cm.row(0), aview1d<double>(a1_cm_r1, 4) );
	BCS_CHECK_EQUAL( a1_cm.row(1), aview1d<double>(a1_cm_r2, 4) );

	double a1_cm_c1[] = {1, 2, 3};
	double a1_cm_c2[] = {6, 7, 8};
	BCS_CHECK_EQUAL( a1_cm.column(0), aview1d<double>(a1_cm_c1, 3) );
	BCS_CHECK_EQUAL( a1_cm.column(1), aview1d<double>(a1_cm_c2, 3) );

	// (step_ind, step_ind)

	aview2d<double, row_major_t, step_ind, step_ind> a2_rm(src, 5, 6, step_ind(2, 2), step_ind(3, 2));

	double a2_rm_r1[] = {1, 3, 5};
	double a2_rm_r2[] = {13, 15, 17};
	BCS_CHECK_EQUAL( a2_rm.row(0), aview1d<double>(a2_rm_r1, 3) );
	BCS_CHECK_EQUAL( a2_rm.row(1), aview1d<double>(a2_rm_r2, 3) );

	double a2_rm_c1[] = {1, 13};
	double a2_rm_c2[] = {3, 15};
	BCS_CHECK_EQUAL( a2_rm.column(0), aview1d<double>(a2_rm_c1, 2) );
	BCS_CHECK_EQUAL( a2_rm.column(1), aview1d<double>(a2_rm_c2, 2) );

	aview2d<double, column_major_t, step_ind, step_ind> a2_cm(src, 5, 6, step_ind(2, 2), step_ind(3, 2));

	double a2_cm_r1[] = {1, 11, 21};
	double a2_cm_r2[] = {3, 13, 23};
	BCS_CHECK_EQUAL( a2_cm.row(0), aview1d<double>(a2_cm_r1, 3) );
	BCS_CHECK_EQUAL( a2_cm.row(1), aview1d<double>(a2_cm_r2, 3) );

	double a2_cm_c1[] = {1, 3};
	double a2_cm_c2[] = {11, 13};
	BCS_CHECK_EQUAL( a2_cm.column(0), aview1d<double>(a2_cm_c1, 2) );
	BCS_CHECK_EQUAL( a2_cm.column(1), aview1d<double>(a2_cm_c2, 2) );


	// (rep_ind, step_ind)

	aview2d<double, row_major_t, rep_ind, step_ind> a3_rm(src, 5, 6, rep_ind(4), step_ind(3, 2));

	double a3_rm_r1[] = {1, 3, 5};
	double a3_rm_r2[] = {1, 3, 5};
	BCS_CHECK_EQUAL( a3_rm.row(0), aview1d<double>(a3_rm_r1, 3) );
	BCS_CHECK_EQUAL( a3_rm.row(1), aview1d<double>(a3_rm_r2, 3) );

	double a3_rm_c1[] = {1, 1, 1, 1};
	double a3_rm_c2[] = {3, 3, 3, 3};
	BCS_CHECK_EQUAL( a3_rm.column(0), aview1d<double>(a3_rm_c1, 4) );
	BCS_CHECK_EQUAL( a3_rm.column(1), aview1d<double>(a3_rm_c2, 4) );

	aview2d<double, column_major_t, rep_ind, step_ind> a3_cm(src, 5, 6, rep_ind(4), step_ind(3, 2));

	double a3_cm_r1[] = {1, 11, 21};
	double a3_cm_r2[] = {1, 11, 21};
	BCS_CHECK_EQUAL( a3_cm.row(0), aview1d<double>(a3_cm_r1, 3) );
	BCS_CHECK_EQUAL( a3_cm.row(1), aview1d<double>(a3_cm_r2, 3) );

	double a3_cm_c1[] = {1, 1, 1, 1};
	double a3_cm_c2[] = {11, 11, 11, 11};
	BCS_CHECK_EQUAL( a3_cm.column(0), aview1d<double>(a3_cm_c1, 4) );
	BCS_CHECK_EQUAL( a3_cm.column(1), aview1d<double>(a3_cm_c2, 4) );


	// (step_ind, indices)

	index_t rinds[] = {0, 2, 4};
	index_t cinds[] = {0, 2, 3, 5};

	aview2d<double, row_major_t, step_ind, arr_ind> a4_rm(src, 5, 6, step_ind(3, 2), arr_ind(4, cinds));

	double a4_rm_r1[] = {1, 3, 4, 6};
	double a4_rm_r2[] = {13, 15, 16, 18};
	BCS_CHECK_EQUAL( a4_rm.row(0), aview1d<double>(a4_rm_r1, 4) );
	BCS_CHECK_EQUAL( a4_rm.row(1), aview1d<double>(a4_rm_r2, 4) );

	double a4_rm_c1[] = {1, 13, 25};
	double a4_rm_c2[] = {3, 15, 27};
	BCS_CHECK_EQUAL( a4_rm.column(0), aview1d<double>(a4_rm_c1, 3) );
	BCS_CHECK_EQUAL( a4_rm.column(1), aview1d<double>(a4_rm_c2, 3) );

	aview2d<double, column_major_t, step_ind, arr_ind> a4_cm(src, 5, 6, step_ind(3, 2), arr_ind(4, cinds));

	double a4_cm_r1[] = {1, 11, 16, 26};
	double a4_cm_r2[] = {3, 13, 18, 28};
	BCS_CHECK_EQUAL( a4_cm.row(0), aview1d<double>(a4_cm_r1, 4) );
	BCS_CHECK_EQUAL( a4_cm.row(1), aview1d<double>(a4_cm_r2, 4) );

	double a4_cm_c1[] = {1, 3, 5};
	double a4_cm_c2[] = {11, 13, 15};
	BCS_CHECK_EQUAL( a4_cm.column(0), aview1d<double>(a4_cm_c1, 3) );
	BCS_CHECK_EQUAL( a4_cm.column(1), aview1d<double>(a4_cm_c2, 3) );

	// (indices, indices)

	aview2d<double, row_major_t, arr_ind, arr_ind> a5_rm(src, 5, 6, arr_ind(3, rinds), arr_ind(4, cinds));

	double a5_rm_r1[] = {1, 3, 4, 6};
	double a5_rm_r2[] = {13, 15, 16, 18};
	BCS_CHECK_EQUAL( a5_rm.row(0), aview1d<double>(a5_rm_r1, 4) );
	BCS_CHECK_EQUAL( a5_rm.row(1), aview1d<double>(a5_rm_r2, 4) );

	double a5_rm_c1[] = {1, 13, 25};
	double a5_rm_c2[] = {3, 15, 27};
	BCS_CHECK_EQUAL( a5_rm.column(0), aview1d<double>(a5_rm_c1, 3) );
	BCS_CHECK_EQUAL( a5_rm.column(1), aview1d<double>(a5_rm_c2, 3) );

	aview2d<double, column_major_t, arr_ind, arr_ind> a5_cm(src, 5, 6, arr_ind(3, rinds), arr_ind(4, cinds));

	double a5_cm_r1[] = {1, 11, 16, 26};
	double a5_cm_r2[] = {3, 13, 18, 28};
	BCS_CHECK_EQUAL( a5_cm.row(0), aview1d<double>(a5_cm_r1, 4) );
	BCS_CHECK_EQUAL( a5_cm.row(1), aview1d<double>(a5_cm_r2, 4) );

	double a5_cm_c1[] = {1, 3, 5};
	double a5_cm_c2[] = {11, 13, 15};
	BCS_CHECK_EQUAL( a5_cm.column(0), aview1d<double>(a5_cm_c1, 3) );
	BCS_CHECK_EQUAL( a5_cm.column(1), aview1d<double>(a5_cm_c2, 3) );
}



BCS_TEST_CASE( test_array2d_subviews )
{

	double src0[144];
	for (int i = 0; i < 144; ++i) src0[i] = i+1;


	// dense base

	const aview2d<double, row_major_t> a0_rm = get_aview2d_rm(src0, 6, 6);
	const aview2d<double, column_major_t> a0_cm = get_aview2d_cm(src0, 6, 6);

	// dense => (whole, whole)

	BCS_CHECK_EQUAL( a0_rm.V(whole(), whole()),  get_aview2d_rm(src0, 6, 6) );
	BCS_CHECK_EQUAL( a0_cm.V(whole(), whole()),  get_aview2d_cm(src0, 6, 6) );

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

	BCS_CHECK_EQUAL( a0_rm.V(rgn(0, a0_rm.dim0(), 2), 2), aview1d<double>(a0_rm_v4, 3) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(0, a0_cm.dim0(), 2), 2), aview1d<double>(a0_cm_v4, 3) );


	// dense => (range, whole)

	double a0_rm_s1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
	double a0_cm_s1[] = {1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21, 25, 26, 27, 31, 32, 33};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(0, 3), whole()),  get_aview2d_rm(a0_rm_s1, 3, 6) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(0, 3), whole()),  get_aview2d_cm(a0_cm_s1, 3, 6) );

	// dense => (whole, range)

	double a0_rm_s2[] = {2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22, 26, 27, 28, 32, 33, 34};
	double a0_cm_s2[] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

	BCS_CHECK_EQUAL( a0_rm.V(whole(), rgn(1, 4)),  get_aview2d_rm(a0_rm_s2, 6, 3) );
	BCS_CHECK_EQUAL( a0_cm.V(whole(), rgn(1, 4)),  get_aview2d_cm(a0_cm_s2, 6, 3) );


	// dense => (range, range)

	double a0_rm_s3[] = {14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29};
	double a0_cm_s3[] = {9, 10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(2, 5), rgn(1, 5)),  get_aview2d_rm(a0_rm_s3, 3, 4) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(2, 5), rgn(1, 5)),  get_aview2d_cm(a0_cm_s3, 3, 4) );

	// dense => (range, step_range)

	double a0_rm_s4[] = {14, 16, 18, 20, 22, 24, 26, 28, 30};
	double a0_cm_s4[] = {9, 10, 11, 21, 22, 23, 33, 34, 35};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(2, 5), rgn(1, 6, 2)),  get_aview2d_rm(a0_rm_s4, 3, 3) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(2, 5), rgn(1, 6, 2)),  get_aview2d_cm(a0_cm_s4, 3, 3) );


	// dense => (step_range, step_range)

	double a0_rm_s5[] = {2, 4, 6, 14, 16, 18, 26, 28, 30};
	double a0_cm_s5[] = {7, 9, 11, 19, 21, 23, 31, 33, 35};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(0, 5, 2), rgn(1, 6, 2)),  get_aview2d_rm(a0_rm_s5, 3, 3) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(0, 5, 2), rgn(1, 6, 2)),  get_aview2d_cm(a0_cm_s5, 3, 3) );


	// dense => (step_range, indices)

	double a0_rm_s6[] = {1, 3, 4, 5, 13, 15, 16, 17, 25, 27, 28, 29};
	double a0_cm_s6[] = {1, 3, 5, 13, 15, 17, 19, 21, 23, 25, 27, 29};

	index_t a0_cinds_s6[] = {0, 2, 3, 4};

	BCS_CHECK_EQUAL( a0_rm.V(rgn(0, a0_rm.dim0(), 2), arr_ind(4, a0_cinds_s6)),  get_aview2d_rm(a0_rm_s6, 3, 4) );
	BCS_CHECK_EQUAL( a0_cm.V(rgn(0, a0_cm.dim0(), 2), arr_ind(4, a0_cinds_s6)),  get_aview2d_cm(a0_cm_s6, 3, 4) );

	// dense => (rep_range, indices)

	index_t a0_cinds_s7[] = {0, 2, 3, 4};

	double a0_rm_s7[] = {7, 9, 10, 11, 7, 9, 10, 11, 7, 9, 10, 11};
	double a0_cm_s7[] = {2, 2, 2, 14, 14, 14, 20, 20, 20, 26, 26, 26};

	BCS_CHECK_EQUAL( a0_rm.V(rep(1, 3), arr_ind(4, a0_cinds_s7)),  get_aview2d_rm(a0_rm_s7, 3, 4) );
	BCS_CHECK_EQUAL( a0_cm.V(rep(1, 3), arr_ind(4, a0_cinds_s7)),  get_aview2d_cm(a0_cm_s7, 3, 4) );

	// dense => (indices, indices)

	index_t a0_rinds_s8[] = {1, 3, 4};
	index_t a0_cinds_s8[] = {0, 2, 3, 5};

	double a0_rm_s8[] = {7, 9, 10, 12, 19, 21, 22, 24, 25, 27, 28, 30};
	double a0_cm_s8[] = {2, 4, 5, 14, 16, 17, 20, 22, 23, 32, 34, 35};

	BCS_CHECK_EQUAL( a0_rm.V(arr_ind(3, a0_rinds_s8), arr_ind(4, a0_cinds_s8)),  get_aview2d_rm(a0_rm_s8, 3, 4) );
	BCS_CHECK_EQUAL( a0_cm.V(arr_ind(3, a0_rinds_s8), arr_ind(4, a0_cinds_s8)),  get_aview2d_cm(a0_cm_s8, 3, 4) );


	// step base

	aview2d<double, row_major_t> Arm = get_aview2d_rm(src0, 12, 12);
	aview2d<double, column_major_t> Acm = get_aview2d_cm(src0, 12, 12);

	aview2d<double, row_major_t, step_ind, step_ind> a1_rm = Arm.V(rgn(0, Arm.dim0(), 2), rgn(0, Arm.dim1(), 2));
	aview2d<double, column_major_t, step_ind, step_ind> a1_cm = Acm.V(rgn(0, Acm.dim0(), 2), rgn(0, Acm.dim1(), 2));

	// step => (whole, whole)

	double a1_all[36] = {
			1, 3, 5, 7, 9, 11,
			25, 27, 29, 31, 33, 35,
			49, 51, 53, 55, 57, 59,
			73, 75, 77, 79, 81, 83,
			97, 99, 101, 103, 105, 107,
			121, 123, 125, 127, 129, 131
	};

	BCS_CHECK_EQUAL( a1_rm.V(whole(), whole()), get_aview2d_rm(a1_all, 6, 6) );
	BCS_CHECK_EQUAL( a1_cm.V(whole(), whole()), get_aview2d_cm(a1_all, 6, 6) );

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

	BCS_CHECK_EQUAL( a1_rm.V(rgn(0, a1_rm.dim0(), 2), 2), aview1d<double>(a1_rm_v4, 3) );
	BCS_CHECK_EQUAL( a1_cm.V(rgn(0, a1_cm.dim0(), 2), 2), aview1d<double>(a1_cm_v4, 3) );



	// step => (range, step_range)

	double a1_rm_s1[] = {25, 29, 33, 49, 53, 57, 73, 77, 81, 97, 101, 105};
	double a1_cm_s1[] = {3, 5, 7, 9, 51, 53, 55, 57, 99, 101, 103, 105};

	BCS_CHECK_EQUAL( a1_rm.V(rgn(1, 5), rgn(0, a1_rm.dim1(), 2)),  get_aview2d_rm(a1_rm_s1, 4, 3) );
	BCS_CHECK_EQUAL( a1_cm.V(rgn(1, 5), rgn(0, a1_cm.dim1(), 2)),  get_aview2d_cm(a1_cm_s1, 4, 3) );

	// step => (rep_range, range)

	double a1_rm_s2[] = {51, 53, 55, 57, 51, 53, 55, 57, 51, 53, 55, 57};
	double a1_cm_s2[] = {29, 29, 29, 53, 53, 53, 77, 77, 77, 101, 101, 101};

	BCS_CHECK_EQUAL( a1_rm.V(rep(2, 3), rgn(1, 5)),  get_aview2d_rm(a1_rm_s2, 3, 4) );
	BCS_CHECK_EQUAL( a1_cm.V(rep(2, 3), rgn(1, 5)),  get_aview2d_cm(a1_cm_s2, 3, 4) );

	// step => (step_range, -step_range)

	double a1_rm_s3[] = {59, 55, 51, 107, 103, 99};
	double a1_cm_s3[] = {125, 129, 77, 81, 29, 33};

	BCS_CHECK_EQUAL( a1_rm.V(rgn(2, 5, 2), rgn(5, 0, -2)), get_aview2d_rm(a1_rm_s3, 2, 3) );
	BCS_CHECK_EQUAL( a1_cm.V(rgn(2, 5, 2), rgn(5, 0, -2)), get_aview2d_cm(a1_cm_s3, 2, 3) );

	// step => (indices, indices)

	index_t a1_s4_rinds[3] = {2, 0, 3};
	index_t a1_s4_cinds[4] = {4, 1, 5, 2};

	double a1_rm_s4[] = {57, 51, 59, 53, 9, 3, 11, 5, 81, 75, 83, 77};
	double a1_cm_s4[] = {101, 97, 103, 29, 25, 31, 125, 121, 127, 53, 49, 55};

	BCS_CHECK_EQUAL( a1_rm.V(arr_ind(3, a1_s4_rinds), arr_ind(4, a1_s4_cinds)), get_aview2d_rm(a1_rm_s4, 3, 4) );
	BCS_CHECK_EQUAL( a1_cm.V(arr_ind(3, a1_s4_rinds), arr_ind(4, a1_s4_cinds)), get_aview2d_cm(a1_cm_s4, 3, 4) );


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

	BCS_CHECK_EQUAL( a2_rm.V(whole(), whole()), get_aview2d_rm(a2_rm_all, 5, 6) );
	BCS_CHECK_EQUAL( a2_cm.V(whole(), whole()), get_aview2d_cm(a2_cm_all, 5, 6) );


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

	BCS_CHECK_EQUAL( a2_rm.V(rgn(0, a2_rm.dim0(), 2), 2), aview1d<double>(a2_rm_v4, 3) );
	BCS_CHECK_EQUAL( a2_cm.V(rgn(0, a2_cm.dim0(), 2), 2), aview1d<double>(a2_cm_v4, 3) );



	// rep x -step => (range, range)

	double a2_rm_s1[] = {46, 44, 42, 40, 46, 44, 42, 40};
	double a2_cm_s1[] = {112, 112, 88, 88, 64, 64, 40, 40};

	BCS_CHECK_EQUAL( a2_rm.V(rgn(1, 3), rgn(1, 5)), get_aview2d_rm(a2_rm_s1, 2, 4) );
	BCS_CHECK_EQUAL( a2_cm.V(rgn(1, 3), rgn(1, 5)), get_aview2d_cm(a2_cm_s1, 2, 4) );

	// rep x -step => (step_range, -step_range)

	double a2_rm_s2[] = {38, 40, 42, 44, 46, 48, 38, 40, 42, 44, 46, 48};
	double a2_cm_s2[] = {16, 16, 40, 40, 64, 64, 88, 88, 112, 112, 136, 136};

	BCS_CHECK_EQUAL( a2_rm.V(rgn(0, 3, 2), rev_whole()), get_aview2d_rm(a2_rm_s2, 2, 6) );
	BCS_CHECK_EQUAL( a2_cm.V(rgn(0, 3, 2), rev_whole()), get_aview2d_cm(a2_cm_s2, 2, 6) );

	// rep x -step => (range, indices)

	index_t a2_s3_cinds[4] = {0, 2, 3, 5};

	double a2_rm_s3[] = {48, 44, 42, 38, 48, 44, 42, 38, 48, 44, 42, 38};
	double a2_cm_s3[] = {136, 136, 136, 88, 88, 88, 64, 64, 64, 16, 16, 16};

	BCS_CHECK_EQUAL( a2_rm.V(rgn(0, 3), arr_ind(4, a2_s3_cinds)), get_aview2d_rm(a2_rm_s3, 3, 4) );
	BCS_CHECK_EQUAL( a2_cm.V(rgn(0, 3), arr_ind(4, a2_s3_cinds)), get_aview2d_cm(a2_cm_s3, 3, 4) );


	// rep x -step => (indices, indices)

	index_t a2_s4_rinds[3] = {0, 2, 3};
	index_t a2_s4_cinds[4] = {0, 2, 3, 5};

	double a2_rm_s4[] = {48, 44, 42, 38, 48, 44, 42, 38, 48, 44, 42, 38};
	double a2_cm_s4[] = {136, 136, 136, 88, 88, 88, 64, 64, 64, 16, 16, 16};

	BCS_CHECK_EQUAL( a2_rm.V(arr_ind(3, a2_s4_rinds), arr_ind(4, a2_s4_cinds)), get_aview2d_rm(a2_rm_s4, 3, 4) );
	BCS_CHECK_EQUAL( a2_cm.V(arr_ind(3, a2_s4_rinds), arr_ind(4, a2_s4_cinds)), get_aview2d_cm(a2_cm_s4, 3, 4) );


	// indices x step base

	index_t a3_rinds[5] = {1, 3, 4, 7, 9};

	aview2d<double, row_major_t, arr_ind, step_ind> a3_rm = Arm.V(arr_ind(5, a3_rinds), rgn(1, Arm.dim1(), 2));
	aview2d<double, column_major_t, arr_ind, step_ind> a3_cm = Acm.V(arr_ind(5, a3_rinds), rgn(1, Acm.dim1(), 2));

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

	BCS_CHECK_EQUAL( a3_rm.V(whole(), whole()), get_aview2d_rm(a3_rm_all, 5, 6) );
	BCS_CHECK_EQUAL( a3_cm.V(whole(), whole()), get_aview2d_cm(a3_cm_all, 5, 6) );


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

	BCS_CHECK_EQUAL( a3_rm.V(rgn(0, a3_rm.dim0(), 2), 2), aview1d<double>(a3_rm_v4, 3) );
	BCS_CHECK_EQUAL( a3_cm.V(rgn(0, a3_cm.dim0(), 2), 2), aview1d<double>(a3_cm_v4, 3) );


	// indices x step => (-step_range, range)

	double a3_rm_s1[15] = {112, 114, 116, 88, 90, 92, 52, 54, 56, 40, 42, 44, 16, 18, 20};
	double a3_cm_s1[15] = {46, 44, 41, 40, 38, 70, 68, 65, 64, 62, 94, 92, 89, 88, 86};

	BCS_CHECK_EQUAL( a3_rm.V(rev_whole(), rgn(1, 4)), get_aview2d_rm(a3_rm_s1, 5, 3) );
	BCS_CHECK_EQUAL( a3_cm.V(rev_whole(), rgn(1, 4)), get_aview2d_cm(a3_cm_s1, 5, 3) );

	// indices x step => (indices, indices)

	index_t a3_s2_rinds[3] = {0, 2, 3};
	index_t a3_s2_cinds[3] = {1, 3, 4};

	double a3_rm_s2[9] = {16, 20, 22, 52, 56, 58, 88, 92, 94};
	double a3_cm_s2[9] = {38, 41, 44, 86, 89, 92, 110, 113, 116};

	BCS_CHECK_EQUAL( a3_rm.V(arr_ind(3, a3_s2_rinds), arr_ind(3, a3_s2_cinds)), get_aview2d_rm(a3_rm_s2, 3, 3) );
	BCS_CHECK_EQUAL( a3_cm.V(arr_ind(3, a3_s2_rinds), arr_ind(3, a3_s2_cinds)), get_aview2d_cm(a3_cm_s2, 3, 3) );

	// indices x step => (step_range, rep)

	double a3_rm_s3[6] = {18, 18, 54, 54, 114, 114};
	double a3_cm_s3[6] = {62, 65, 70, 62, 65, 70};

	BCS_CHECK_EQUAL( a3_rm.V(rgn(0, a3_rm.dim0(), 2), rep(2, 2)), get_aview2d_rm(a3_rm_s3, 3, 2) );
	BCS_CHECK_EQUAL( a3_cm.V(rgn(0, a3_cm.dim0(), 2), rep(2, 2)), get_aview2d_cm(a3_cm_s3, 3, 2) );

}


BCS_TEST_CASE( test_denseview_judge_2d )
{
	double src[60];

	// row_major

	typedef aview2d<double, row_major_t, step_ind, step_ind> arr_rm_t;

	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(5, 1), step_ind(6, 1)) ), true );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(5, 1), step_ind(3, 2)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(5, 1), step_ind(4, 1)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(5, 1), step_ind(1, 2)) ), false );

	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(3, 2), step_ind(6, 1)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(3, 2), step_ind(3, 2)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(3, 2), step_ind(4, 1)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(3, 2), step_ind(1, 2)) ), false );

	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(4, 1), step_ind(6, 1)) ), true );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(4, 1), step_ind(3, 2)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(4, 1), step_ind(4, 1)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(4, 1), step_ind(1, 2)) ), false );

	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(1, 2), step_ind(6, 1)) ), true );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(1, 2), step_ind(3, 2)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(1, 2), step_ind(4, 1)) ), true );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(1, 2), step_ind(1, 2)) ), true );

	// column_major

	typedef aview2d<double, column_major_t, step_ind, step_ind> arr_cm_t;

	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(5, 1), step_ind(6, 1)) ), true );
	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(5, 1), step_ind(3, 2)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(5, 1), step_ind(4, 1)) ), true );
	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(5, 1), step_ind(1, 2)) ), true );

	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(3, 2), step_ind(6, 1)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(3, 2), step_ind(3, 2)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(3, 2), step_ind(4, 1)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(3, 2), step_ind(1, 2)) ), false );

	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(4, 1), step_ind(6, 1)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(4, 1), step_ind(3, 2)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(4, 1), step_ind(4, 1)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(4, 1), step_ind(1, 2)) ), true );

	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(1, 2), step_ind(6, 1)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(1, 2), step_ind(3, 2)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_cm_t(src, 5, 6, step_ind(1, 2), step_ind(4, 1)) ), false );
	BCS_CHECK_EQUAL( is_dense_view( arr_rm_t(src, 5, 6, step_ind(1, 2), step_ind(1, 2)) ), true );
}



BCS_TEST_CASE( test_aview2d_clone )
{
	double src[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

	aview2d<double, row_major_t, id_ind, id_ind> view1_rm(src, 3, 5, id_ind(3), id_ind(5));
	array2d<double, row_major_t> a1_rm = clone_array(view1_rm);

	BCS_CHECK( array_integrity_test(a1_rm) );
	BCS_CHECK( array_iteration_test(a1_rm) );
	BCS_CHECK( a1_rm.pbase() != view1_rm.pbase() );
	BCS_CHECK_EQUAL( a1_rm, view1_rm );

	aview2d<double, row_major_t, step_ind, step_ind> view2_rm(src, 4, 5, step_ind(2, 2), step_ind(3, 2));
	array2d<double, row_major_t> a2_rm = clone_array(view2_rm);

	BCS_CHECK( array_integrity_test(a2_rm) );
	BCS_CHECK( array_iteration_test(a2_rm) );
	BCS_CHECK( a2_rm.pbase() != view2_rm.pbase() );
	BCS_CHECK_EQUAL( a2_rm, view2_rm );


	aview2d<double, column_major_t, id_ind, id_ind> view1_cm(src, 3, 5, id_ind(3), id_ind(5));
	array2d<double, column_major_t> a1_cm = clone_array(view1_cm);

	BCS_CHECK( array_integrity_test(a1_cm) );
	BCS_CHECK( array_iteration_test(a1_cm) );
	BCS_CHECK( a1_cm.pbase() != view1_cm.pbase() );
	BCS_CHECK_EQUAL( a1_cm, view1_cm );

	aview2d<double, column_major_t, step_ind, step_ind> view2_cm(src, 4, 5, step_ind(2, 2), step_ind(3, 2));
	array2d<double, column_major_t> a2_cm = clone_array(view2_cm);

	BCS_CHECK( array_integrity_test(a2_rm) );
	BCS_CHECK( array_iteration_test(a2_rm) );
	BCS_CHECK( a2_cm.pbase() != view2_cm.pbase() );
	BCS_CHECK_EQUAL( a2_cm, view2_cm );
}


BCS_TEST_CASE( test_subarr_selection2d )
{
	const size_t m0 = 6;
	const size_t n0 = 6;
	double src[m0 * n0];
	for (size_t i = 0; i < m0 * n0; ++i) src[i] = i + 1;

	caview2d<double, row_major_t>    Arm = get_aview2d_rm(src, m0, n0);
	caview2d<double, column_major_t> Acm = get_aview2d_cm(src, m0, n0);

	// select_elems

	const size_t sn0 = 6;
	index_t Is[sn0] = {1, 3, 4, 5, 2, 0};
	index_t Js[sn0] = {2, 2, 4, 5, 0, 0};
	std::vector<index_t> vIs(Is, Is+sn0);
	std::vector<index_t> vJs(Js, Js+sn0);

	array1d<double> s0_rm = select_elems(Arm, vIs, vJs);
	array1d<double> s0_cm = select_elems(Acm, vIs, vJs);

	double s0_rm_r[sn0] = {9, 21, 29, 36, 13, 1};
	double s0_cm_r[sn0] = {14, 16, 29, 36, 3, 1};

	BCS_CHECK( array_view_equal(s0_rm, s0_rm_r, sn0) );
	BCS_CHECK( array_view_equal(s0_cm, s0_cm_r, sn0) );

	// select_rows

	const size_t sn1 = 3;
	index_t rs[sn1] = {1, 4, 5};

	std::vector<index_t> rows(rs, rs+sn1);

	array2d<double, row_major_t>    s1_rm = select_rows(Arm, rows);
	array2d<double, column_major_t> s1_cm = select_rows(Acm, rows);

	double s1_rm_r[sn1 * n0] = {7, 8, 9, 10, 11, 12, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
	double s1_cm_r[sn1 * n0] = {2, 5, 6, 8, 11, 12, 14, 17, 18, 20, 23, 24, 26, 29, 30, 32, 35, 36};

	BCS_CHECK( array_view_equal(s1_rm, s1_rm_r, sn1, n0) );
	BCS_CHECK( array_view_equal(s1_cm, s1_cm_r, sn1, n0) );

	// select_columns

	const size_t sn2 = 3;
	index_t cs[sn2] = {2, 3, 5};

	std::vector<index_t> cols(cs, cs+sn2);

	array2d<double, row_major_t>    s2_rm = select_columns(Arm, cols);
	array2d<double, column_major_t> s2_cm = select_columns(Acm, cols);

	double s2_rm_r[m0 * sn2] = {3, 4, 6, 9, 10, 12, 15, 16, 18, 21, 22, 24, 27, 28, 30, 33, 34, 36};
	double s2_cm_r[m0 * sn2] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36};

	BCS_CHECK( array_view_equal(s2_rm, s2_rm_r, m0, sn2) );
	BCS_CHECK( array_view_equal(s2_cm, s2_cm_r, m0, sn2) );

	// select_rows_and_cols

	const size_t sm3 = 2;  index_t rs3[sm3] = {2, 4};
	const size_t sn3 = 3;  index_t cs3[sn3] = {1, 3, 5};

	std::vector<index_t> rows3(rs3, rs3+sm3);
	std::vector<index_t> cols3(cs3, cs3+sn3);

	array2d<double, row_major_t>    s3_rm = select_rows_and_cols(Arm, rows3, cols3);
	array2d<double, column_major_t> s3_cm = select_rows_and_cols(Acm, rows3, cols3);

	double s3_rm_r[sm3 * sn3] = {14, 16, 18, 26, 28, 30};
	double s3_cm_r[sm3 * sn3] = {9, 11, 21, 23, 33, 35};

	BCS_CHECK( array_view_equal(s3_rm, s3_rm_r, sm3, sn3) );
	BCS_CHECK( array_view_equal(s3_cm, s3_cm_r, sm3, sn3) );

}


test_suite *test_array2d_suite()
{
	test_suite *suite = new test_suite( "test_array2d" );

	suite->add( new test_dense_array2d() );
	suite->add( new test_gen_array2d() );
	suite->add( new test_array2d_slices() );
	suite->add( new test_array2d_subviews() );
	suite->add( new test_denseview_judge_2d() );
	suite->add( new test_aview2d_clone() );
	suite->add( new test_subarr_selection2d() );

	return suite;
}





