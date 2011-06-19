/**
 * @file test_array1d.cpp
 *
 * The Unit Testing for array1d
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/test_units.h>
#include <bcslib/test/test_array_aux.h>
#include <bcslib/array/array1d.h>


using namespace bcs;
using namespace bcs::test;

// Explicit instantiation for syntax checking

template class bcs::aview1d<double, id_ind>;
template class bcs::aview1d<double, step_ind>;
template class bcs::aview1d<double, rep_ind>;
template class bcs::aview1d<double, arr_ind >;

template class bcs::array1d<double>;

// A class for concept checked

template<class Arr>
class array_view1d_concept_check
{
	BCS_STATIC_ASSERT_V(is_array_view<Arr>);
	static_assert(is_array_view_ndim<Arr, 1>::value, "is_array_view_ndim<Arr, 1>");

	typedef typename array_view_traits<Arr>::value_type value_type;
	typedef std::array<index_t, 1> shape_type;

	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::size_type, size_t);
	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::index_type, index_t);
	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::const_reference, const value_type&);
	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::reference, value_type&);
	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::const_pointer, const value_type*);
	BCS_ASSERT_SAME_TYPE(typename array_view_traits<Arr>::shape_type, shape_type);
	BCS_STATIC_ASSERT(array_view_traits<Arr>::num_dims == 1);

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

template class array_view1d_concept_check<bcs::aview1d<double> >;
template class array_view1d_concept_check<bcs::array1d<double> >;


template<typename T, class TIndexer>
bool array_integrity_test(const bcs::aview1d<T, TIndexer>& view)
{
	index_t n = view.dim0();

	if (n != (index_t)view.nelems()) return false;
	if (view.ndims() != 1) return false;
	if (view.shape() != arr_shape(n)) return false;
	if (get_num_elems(view) != view.nelems()) return false;
	if (get_array_shape(view) != view.shape()) return false;
	if (ptr_base(view) != view.pbase()) return false;

	for (index_t i = 0; i < n; ++i)
	{
		if (view.ptr(i) != &(view(i))) return false;
		if (view.ptr(i) != &(view[i])) return false;
	}

	return true;
}

template<typename T, class TIndexer>
bool array_iteration_test(const bcs::aview1d<T, TIndexer>& view)
{
	if (begin(view) != view.begin()) return false;
	if (end(view) != view.end()) return false;

	index_t n = view.dim0();
	bcs::block<T> buffer(view.nelems());
	T *b = buffer.pbase();
	for (index_t i = 0; i < n; ++i) b[i] = view(i);

	return collection_equal(view.begin(), view.end(), buffer.pbase(), (size_t)n);
}


template<typename T, class TIndexer>
bool test_generic_operations(bcs::aview1d<T, TIndexer>& view, const double *src)
{
	block<T> blk(view.nelems());
	export_to(view, blk.pbase());

	if (!collection_equal(blk.pbase(), blk.pend(), src, view.nelems())) return false;

	index_t d = view.dim0();
	if (is_dense_view(view))
	{
		set_zeros(view);
		for (index_t i = 0; i < d; ++i)
		{
			if (view(i) != T(0)) return false;
		}
	}

	fill(view, T(1));
	for (index_t i = 0; i < d; ++i)
	{
		if (view(i) != T(1)) return false;
	}

	import_from(view, src);
	if (!array_view_equal(view, src, view.nelems())) return false;

	return true;
}



// Test cases


BCS_TEST_CASE( test_dense_array1d )
{
	double src1[5] = {3, 4, 5, 1, 2};
	size_t n1 = 5;

	double v2 = 7;
	double src2[6] = {7, 7, 7, 7, 7, 7};
	size_t n2 = 6;

	array1d<double> a0(0);

	BCS_CHECK( array_integrity_test(a0) );
	BCS_CHECK( array_iteration_test(a0) );

	array1d<double> a1(5);

	for (int i = 0; i < 5; ++i) a1(i) = src1[i];

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_view_equal(a1, src1, n1) );
	BCS_CHECK( array_iteration_test(a1) );

	array1d<double> a2(n2, v2);

	BCS_CHECK( array_integrity_test(a2) );
	BCS_CHECK( array_view_equal(a2, src2, n2) );
	BCS_CHECK( array_iteration_test(a2) );

	array1d<double> a3(n1, src1);

	BCS_CHECK( array_integrity_test(a3) );
	BCS_CHECK( array_view_equal(a3, src1, n1) );
	BCS_CHECK( array_iteration_test(a3) );

	array1d<double> a4(a3);

	BCS_CHECK( array_integrity_test(a3) );
	BCS_CHECK( array_view_equal(a3, src1, n1) );
	BCS_CHECK( array_iteration_test(a3) );

	BCS_CHECK( a4.pbase() != a3.pbase() );
	BCS_CHECK( array_integrity_test(a4) );
	BCS_CHECK( array_view_equal(a4, src1, n1) );
	BCS_CHECK( array_iteration_test(a4) );

	const double *p4 = a4.pbase();
	array1d<double> a5(std::move(a4));

	BCS_CHECK( a4.pbase() == BCS_NULL );
	BCS_CHECK( a4.nelems() == 0 );

	BCS_CHECK( a5.pbase() == p4 );
	BCS_CHECK( array_integrity_test(a5) );
	BCS_CHECK( array_view_equal(a5, src1, n1) );
	BCS_CHECK( array_iteration_test(a5) );

	BCS_CHECK( a1 == a1 );
	array1d<double> a6(a1);
	BCS_CHECK( a1 == a6 );
	a6[2] += 1;
	BCS_CHECK( a1 != a6 );

	array1d<double> a7 = a1.shared_copy();

	BCS_CHECK( a7.pbase() == a1.pbase() );
	BCS_CHECK( array_integrity_test(a7) );
	BCS_CHECK( array_view_equal(a7, src1, n1) );
	BCS_CHECK( array_iteration_test(a7) );

	BCS_CHECK( test_generic_operations(a1, src1) );
}


BCS_TEST_CASE( test_step_array1d )
{
	double src0[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	double src1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

	aview1d<double, step_ind> a1(src1, step_ind(3, 1));
	double r1[] = {1, 2, 3};
	size_t n1 = 3;

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_view_equal(a1, r1, n1) );
	BCS_CHECK( array_iteration_test(a1) );

	aview1d<double, step_ind> a2(src1, step_ind(4, 2));
	double r2[] = {1, 3, 5, 7};
	size_t n2 = 4;

	BCS_CHECK( array_integrity_test(a2) );
	BCS_CHECK( array_view_equal(a2, r2, n2) );
	BCS_CHECK( array_iteration_test(a2) );

	aview1d<double, step_ind> a3(src1 + 7, step_ind(3, -2));
	double r3[] = {8, 6, 4};
	size_t n3 = 3;

	BCS_CHECK( array_integrity_test(a3) );
	BCS_CHECK( array_view_equal(a3, r3, n3) );
	BCS_CHECK( array_iteration_test(a3) );

	aview1d<double, step_ind> a0(src0, step_ind(4, 2));

	BCS_CHECK( test_generic_operations(a2, r2) );
}


BCS_TEST_CASE( test_rep_array1d )
{
	double v = 2;

	aview1d<double, rep_ind> a0(&v, rep_ind(0));

	BCS_CHECK( a0.nelems() == 0 );
	BCS_CHECK( array_integrity_test(a0) );
	BCS_CHECK( array_iteration_test(a0) );

	aview1d<double, rep_ind> a1(&v, rep_ind(5));
	double r1[5] = {2, 2, 2, 2, 2};
	size_t n1 = 5;

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_view_equal(a1, r1, n1) );
	BCS_CHECK( array_iteration_test(a1) );

	BCS_CHECK( test_generic_operations(a1, r1) );
}


BCS_TEST_CASE( test_arrind_array1d )
{
	double src0[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	double src1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

	index_t inds1[4] = {1, 3, 6, 7};
	aview1d<double, arr_ind> a1(src1, arr_ind(4, inds1));
	double r1[] = {2, 4, 7, 8};

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_view_equal(a1, r1, 4) );
	BCS_CHECK( array_iteration_test(a1) );

	index_t inds2[6] = {5, 1, 4, 2, 2, 3};
	arr_ind ai2(6, inds2);
	aview1d<double, arr_ind> a2(src1, ai2);
	double r2[] = {6, 2, 5, 3, 3, 4};

	BCS_CHECK( array_integrity_test(a2) );
	BCS_CHECK( array_view_equal(a2, r2, 6) );
	BCS_CHECK( array_iteration_test(a2) );

	index_t inds0[5] = {0, 2, 3, 5, 7};
	aview1d<double, arr_ind> a0(src0, arr_ind(5, inds0));

	import_from(a0, src1);
	double g1[10] = {1, 0, 2, 3, 0, 4, 0, 5, 0, 0};
	BCS_CHECK( collection_equal(src0, src0 + 10, g1, 10) );

	block<double> blk(a0.size());
	export_to(a0, blk.pbase());
	BCS_CHECK( collection_equal(blk.pbase(), blk.pend(), src1, a0.size()));

	fill(a0, 0.0);
	double g2[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	BCS_CHECK( collection_equal(src0, src0 + 10, g2, 10) );

	aview1d<double, arr_ind> a1c(a1);

	BCS_CHECK( array_integrity_test(a1c) );
	BCS_CHECK( array_view_equal(a1c, r1, 4) );
	BCS_CHECK( array_iteration_test(a1c) );
	BCS_CHECK( a1c.get_indexer().pbase() != a1.get_indexer().pbase() );

	const index_t *pi1c = a1c.get_indexer().pbase();

	aview1d<double, arr_ind> a1m(std::move(a1c));

	BCS_CHECK( array_integrity_test(a1m) );
	BCS_CHECK( array_view_equal(a1m, r1, 4) );
	BCS_CHECK( array_iteration_test(a1m) );
	BCS_CHECK( a1m.get_indexer().pbase() == pi1c );

	BCS_CHECK( a1c.pbase() == BCS_NULL );
	BCS_CHECK( a1c.size() == 0 );

	BCS_CHECK( test_generic_operations(a1, r1) );
}


BCS_TEST_CASE( test_regular_subview )
{
	double src[16];
	for (int i = 0; i < 16; ++i) src[i] = i;  // 0, 1, 2, ..., 15

	aview1d<double, id_ind> a1(src, 8);

	double r1a[] = {0, 1, 2, 3, 4, 5, 6, 7};
	BCS_CHECK_EQUAL( a1.V(whole()),  aview1d<double>(r1a, 8) );

	double r1b[] = {1, 2, 3, 4, 5};
	BCS_CHECK_EQUAL( a1.V(rgn(1, 6)), aview1d<double>(r1b, 5) );

	double r1c[] = {1, 3, 5};
	BCS_CHECK_EQUAL( a1.V(rgn(1, 7, 2)), aview1d<double>(r1c, 3));

	double r1d[] = {2, 3, 4, 5, 6};
	BCS_CHECK_EQUAL( a1.V(rgn(2, a1.dim0() - 1)), aview1d<double>(r1d, 5));

	double r1e[] = {1, 3, 5, 7};
	BCS_CHECK_EQUAL( a1.V(rgn(1, a1.dim0(), 2)), aview1d<double>(r1e, 4));

	double r1f[] = {7, 6, 5, 4, 3, 2, 1, 0};
	BCS_CHECK_EQUAL( a1.V(rev_whole()),  aview1d<double>(r1f, 8) );

	double r1r[] = {2, 2, 2, 2, 2};
	BCS_CHECK_EQUAL( a1.V(rep(2, 5)), aview1d<double>(r1r, 5));


	aview1d<double, step_ind> a2(src, step_ind(8, 2));

	double r2a[] = {0, 2, 4, 6, 8, 10, 12, 14};
	BCS_CHECK_EQUAL( a2.V(whole()), aview1d<double>(r2a, 8) );

	double r2b[] = {2, 4, 6, 8, 10};
	BCS_CHECK_EQUAL( a2.V(rgn(1, 6)), aview1d<double>(r2b, 5) );

	double r2c[] = {2, 6, 10};
	BCS_CHECK_EQUAL( a2.V(rgn(1, 7, 2)), aview1d<double>(r2c, 3));

	double r2d[] = {4, 6, 8, 10, 12};
	BCS_CHECK_EQUAL( a2.V(rgn(2, a2.dim0() - 1)), aview1d<double>(r2d, 5));

	double r2e[] = {2, 6, 10, 14};
	BCS_CHECK_EQUAL( a2.V(rgn(1, a2.dim0(), 2)), aview1d<double>(r2e, 4));

	double r2f[] = {14, 12, 10, 8, 6, 4, 2, 0};
	BCS_CHECK_EQUAL( a2.V(rev_whole()),  aview1d<double>(r2f, 8) );

	double r2r[] = {4, 4, 4, 4, 4};
	BCS_CHECK_EQUAL( a2.V(rep(2, 5)), aview1d<double>(r2r, 5));


	aview1d<double, step_ind> a3(src + 8, step_ind(8, -1));

	double r3a[] = {8, 7, 6, 5, 4, 3, 2, 1};
	BCS_CHECK_EQUAL( a3.V(whole()), aview1d<double>(r3a, 8) );

	double r3b[] = {7, 6, 5, 4, 3};
	BCS_CHECK_EQUAL( a3.V(rgn(1, 6)), aview1d<double>(r3b, 5) );

	double r3c[] = {7, 5, 3};
	BCS_CHECK_EQUAL( a3.V(rgn(1, 7, 2)), aview1d<double>(r3c, 3));

	double r3d[] = {6, 5, 4, 3, 2};
	BCS_CHECK_EQUAL( a3.V(rgn(2, a3.dim0() - 1)), aview1d<double>(r3d, 5));

	double r3e[] = {7, 5, 3, 1};
	BCS_CHECK_EQUAL( a3.V(rgn(1, a3.dim0(), 2)), aview1d<double>(r3e, 4));

	double r3f[] = {1, 2, 3, 4, 5, 6, 7, 8};
	BCS_CHECK_EQUAL( a3.V(rev_whole()),  aview1d<double>(r3f, 8));

	double r3r[] = {6, 6, 6, 6, 6};
	BCS_CHECK_EQUAL( a3.V(rep(2, 5)), aview1d<double>(r3r, 5));


	aview1d<double, rep_ind> a4(src + 2, rep_ind(8));

	double r4a[] = {2, 2, 2, 2, 2, 2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(whole()), aview1d<double>(r4a, 8) );

	double r4b[] = {2, 2, 2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(rgn(1, 6)), aview1d<double>(r4b, 5) );

	double r4c[] = {2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(rgn(1, 7, 2)), aview1d<double>(r4c, 3));

	double r4d[] = {2, 2, 2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(rgn(2, a4.dim0() - 1)), aview1d<double>(r4d, 5));

	double r4e[] = {2, 2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(rgn(1, a4.dim0(), 2)), aview1d<double>(r4e, 4));

	double r4f[] = {2, 2, 2, 2, 2, 2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(rev_whole()),  aview1d<double>(r4f, 8));

	double r4r[] = {2, 2, 2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(rep(2, 5)), aview1d<double>(r4r, 5));

}


BCS_TEST_CASE( test_arrind_subview )
{
	double src[20];
	for (int i = 0; i < 20; ++i) src[i] = i;  // 0, 1, 2, ..., 19

	index_t inds[4] = {1, 3, 6, 7};

	aview1d<double, id_ind> a1(src, 8);
	double r1i[4] = {1, 3, 6, 7};
	BCS_CHECK_EQUAL( a1.V(arr_ind(4, inds)), aview1d<double>(r1i, 4));

	aview1d<double, step_ind> a2(src, step_ind(8, 2));
	double r2i[4] = {2, 6, 12, 14};
	BCS_CHECK_EQUAL( a2.V(arr_ind(4, inds)), aview1d<double>(r2i, 4));

	aview1d<double, step_ind> a3(src + 8, step_ind(8, -1));
	double r3i[4] = {7, 5, 2, 1};
	BCS_CHECK_EQUAL( a3.V(arr_ind(4, inds)), aview1d<double>(r3i, 4));

	aview1d<double, rep_ind> a4(src + 2, rep_ind(8));
	double r4i[4] = {2, 2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(arr_ind(4, inds)), aview1d<double>(r4i, 4));


	index_t inds_i[8] = {1, 3, 6, 7, 9, 13, 15, 16};
	aview1d<double, arr_ind> ai(src, arr_ind(8, inds_i));

	double ria[8] = {1, 3, 6, 7, 9, 13, 15, 16};
	BCS_CHECK_EQUAL( ai.V(whole()), aview1d<double>(ria, 8) );

	double rib[5] = {3, 6, 7, 9, 13};
	BCS_CHECK_EQUAL( ai.V(rgn(1, 6)), aview1d<double>(rib, 5) );

	double ric[3] = {3, 7, 13};
	BCS_CHECK_EQUAL( ai.V(rgn(1, 7, 2)), aview1d<double>(ric, 3) );

	double rid[5] = {6, 7, 9, 13, 15};
	BCS_CHECK_EQUAL( ai.V(rgn(2, ai.dim0() - 1)), aview1d<double>(rid, 5));

	double rie[4] = {3, 7, 13, 16};
	BCS_CHECK_EQUAL( ai.V(rgn(1, ai.dim0(), 2)), aview1d<double>(rie, 4));

	double rif[8] = {16, 15, 13, 9, 7, 6, 3, 1};
	BCS_CHECK_EQUAL( ai.V(rev_whole()),  aview1d<double>(rif, 8) );

	double rir[5] = {6, 6, 6, 6, 6};
	BCS_CHECK_EQUAL( ai.V(rep(2, 5)),  aview1d<double>(rir, 5) );

	double rig[4] = {3, 7, 15, 16};
	BCS_CHECK_EQUAL( ai.V(arr_ind(4, inds)), aview1d<double>(rig, 4));

}


BCS_TEST_CASE( test_aview1d_clone )
{
	double src[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

	aview1d<double, id_ind> view1(src, 5);
	array1d<double> a1 = clone_array(view1);

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_iteration_test(a1) );
	BCS_CHECK( a1.pbase() != view1.pbase() );
	BCS_CHECK_EQUAL( a1, view1 );

	aview1d<double, step_ind> view2(src, step_ind(3, 2));
	array1d<double> a2 = clone_array(view2);

	BCS_CHECK( array_integrity_test(a2) );
	BCS_CHECK( array_iteration_test(a2) );
	BCS_CHECK( a2.pbase() != view2.pbase() );
	BCS_CHECK_EQUAL( a2, view2 );

}


BCS_TEST_CASE( test_denseview_judge_1d )
{
	index_t inds[10];
	double src[20];

	BCS_CHECK_EQUAL( is_dense_view(aview1d<double, id_ind>(src, id_ind(5) )), true );
	BCS_CHECK_EQUAL( is_dense_view(aview1d<double, step_ind>(src, step_ind(5, 1) )), true );
	BCS_CHECK_EQUAL( is_dense_view(aview1d<double, step_ind>(src, step_ind(5, 2) )), false );
	BCS_CHECK_EQUAL( is_dense_view(aview1d<double, rep_ind>(src, rep_ind(3) )), false );
	BCS_CHECK_EQUAL( is_dense_view(aview1d<double, arr_ind>(src, arr_ind(3, inds) )), false );
}


test_suite *test_array1d_suite()
{
	test_suite *suite = new test_suite( "test_array1d" );

	suite->add( new test_dense_array1d() );
	suite->add( new test_step_array1d() );
	suite->add( new test_rep_array1d() );
	suite->add( new test_arrind_array1d() );
	suite->add( new test_regular_subview() );
	suite->add( new test_arrind_subview() );
	suite->add( new test_aview1d_clone() );
	suite->add( new test_denseview_judge_1d() );

	return suite;
}





