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

template class bcs::caview1d_ex<double, range>;
template class bcs::caview1d_ex<double, step_range>;
template class bcs::caview1d_ex<double, rep_range>;

template class bcs::aview1d_ex<double, range>;
template class bcs::aview1d_ex<double, step_range>;
template class bcs::aview1d_ex<double, rep_range>;

template class bcs::caview1d<double>;
template class bcs::aview1d<double>;
template class bcs::array1d<double>;

// Auxiliary testing functions

template<typename T>
bool array_integrity_test(const bcs::caview1d<T>& view)
{
	index_t n = view.dim0();

	if (n != (index_t)view.nelems()) return false;
	if (view.ndims() != 1) return false;
	if (view.shape() != arr_shape(n)) return false;

	for (index_t i = 0; i < n; ++i)
	{
		if (view.ptr(i) != &(view(i))) return false;
		if (view.ptr(i) != &(view[i])) return false;
	}

	return true;
}


template<typename T, class Indexer>
bool array_integrity_test(const bcs::caview1d_ex<T, Indexer>& view)
{
	index_t n = view.dim0();

	if (n != (index_t)view.nelems()) return false;
	if (view.ndims() != 1) return false;
	if (view.shape() != arr_shape(n)) return false;

	return true;
}

template<typename T>
bool array_iteration_test(const bcs::caview1d<T>& view)
{
	if (begin(view) != view.begin()) return false;
	if (end(view) != view.end()) return false;

	index_t n = view.dim0();
	bcs::block<T> buffer(view.nelems());
	T *b = buffer.pbase();
	for (index_t i = 0; i < n; ++i) b[i] = view(i);

	return collection_equal(view.begin(), view.end(), buffer.pbase(), (size_t)n);
}


template<typename T>
bool test_generic_operations(bcs::aview1d<T>& view, const double *src)
{
	block<T> blk(view.nelems());
	export_to(view, blk.pbase());

	if (!collection_equal(blk.pbase(), blk.pend(), src, view.nelems())) return false;

	index_t d = view.dim0();
	set_zeros(view);
	for (index_t i = 0; i < d; ++i)
	{
		if (view(i) != T(0)) return false;
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

template<typename T, class Indexer>
bool test_generic_operations(bcs::aview1d_ex<T, Indexer>& view, const double *src)
{
	block<T> blk(view.nelems());
	export_to(view, blk.pbase());

	if (!collection_equal(blk.pbase(), blk.pend(), src, view.nelems())) return false;

	index_t d = view.dim0();

	for (index_t i = 0; i < d; ++i) view(i) = T(0);

	fill(view, T(1));
	for (index_t i = 0; i < d; ++i)
	{
		if (view(i) != T(1)) return false;
	}

	import_from(view, src);
	if (!array_view_equal(view, src, view.nelems())) return false;

	array1d<T> ac(view);
	if (!array_view_equal(ac, src, view.nelems())) return false;

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
	BCS_CHECK( test_generic_operations(a1, src1) );

	array1d<double> a2(n2, v2);

	BCS_CHECK( array_integrity_test(a2) );
	BCS_CHECK( array_view_equal(a2, src2, n2) );
	BCS_CHECK( array_iteration_test(a2) );
	BCS_CHECK( test_generic_operations(a2, src2) );

	array1d<double> a3(n1, src1);

	BCS_CHECK( array_integrity_test(a3) );
	BCS_CHECK( array_view_equal(a3, src1, n1) );
	BCS_CHECK( array_iteration_test(a3) );
	BCS_CHECK( test_generic_operations(a3, src1) );

	array1d<double> a4(a3);

	BCS_CHECK( array_integrity_test(a3) );
	BCS_CHECK( array_view_equal(a3, src1, n1) );
	BCS_CHECK( array_iteration_test(a3) );

	BCS_CHECK( a4.pbase() != a3.pbase() );
	BCS_CHECK( array_integrity_test(a4) );
	BCS_CHECK( array_view_equal(a4, src1, n1) );
	BCS_CHECK( array_iteration_test(a4) );
	BCS_CHECK( test_generic_operations(a4, src1) );

	const double *p4 = a4.pbase();
	array1d<double> a5(std::move(a4));

	BCS_CHECK( a4.pbase() == BCS_NULL );
	BCS_CHECK( a4.nelems() == 0 );

	BCS_CHECK( a5.pbase() == p4 );
	BCS_CHECK( array_integrity_test(a5) );
	BCS_CHECK( array_view_equal(a5, src1, n1) );
	BCS_CHECK( array_iteration_test(a5) );
	BCS_CHECK( test_generic_operations(a5, src1) );

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
}


BCS_TEST_CASE( test_step_aview1d )
{
	double src0[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	double src1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

	aview1d_ex<double, step_range> a1(src1, step_range::from_begin_dim(0, 3, 1));
	double r1[] = {1, 2, 3};
	size_t n1 = 3;

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_view_equal(a1, r1, n1) );
	BCS_CHECK( test_generic_operations(a1, r1) );

	aview1d_ex<double, step_range> a2(src1, step_range::from_begin_dim(0, 4, 2));
	double r2[] = {1, 3, 5, 7};
	size_t n2 = 4;

	BCS_CHECK( array_integrity_test(a2) );
	BCS_CHECK( array_view_equal(a2, r2, n2) );
	BCS_CHECK( test_generic_operations(a2, r2) );

	aview1d_ex<double, step_range> a3(src1 + 7, step_range::from_begin_dim(0, 3, -2));
	double r3[] = {8, 6, 4};
	size_t n3 = 3;

	BCS_CHECK( array_integrity_test(a3) );
	BCS_CHECK( array_view_equal(a3, r3, n3) );
	BCS_CHECK( test_generic_operations(a3, r3) );

	aview1d_ex<double, step_range> a0(src0, step_range::from_begin_dim(0, 4, 2));
}



BCS_TEST_CASE( test_rep_aview1d )
{
	double v = 2;

	aview1d_ex<double, rep_range> a0(&v, rep_range(0, 0));

	BCS_CHECK( a0.nelems() == 0 );
	BCS_CHECK( array_integrity_test(a0) );

	aview1d_ex<double, rep_range> a1(&v, rep_range(0, 5));
	double r1[5] = {2, 2, 2, 2, 2};
	size_t n1 = 5;

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_view_equal(a1, r1, n1) );
	BCS_CHECK( test_generic_operations(a1, r1) );

}


BCS_TEST_CASE( test_regular_subview )
{
	double src[16];
	for (int i = 0; i < 16; ++i) src[i] = i;  // 0, 1, 2, ..., 15

	aview1d<double> a1(src, 8);

	double r1a[] = {0, 1, 2, 3, 4, 5, 6, 7};
	BCS_CHECK_EQUAL( array1d<double>(a1.V(whole())),  aview1d<double>(r1a, 8) );

	double r1b[] = {1, 2, 3, 4, 5};
	BCS_CHECK_EQUAL( array1d<double>(a1.V(rgn(1, 6))), aview1d<double>(r1b, 5) );

	double r1c[] = {1, 3, 5};
	BCS_CHECK_EQUAL( array1d<double>(a1.V(rgn(1, 7, 2))), aview1d<double>(r1c, 3));

	double r1d[] = {2, 3, 4, 5, 6};
	BCS_CHECK_EQUAL( array1d<double>(a1.V(rgn(2, a1.dim0() - 1))), aview1d<double>(r1d, 5));

	double r1e[] = {1, 3, 5, 7};
	BCS_CHECK_EQUAL( array1d<double>(a1.V(rgn(1, a1.dim0(), 2))), aview1d<double>(r1e, 4));

	double r1f[] = {7, 6, 5, 4, 3, 2, 1, 0};
	BCS_CHECK_EQUAL( array1d<double>(a1.V(rev_whole())),  aview1d<double>(r1f, 8) );

	double r1r[] = {2, 2, 2, 2, 2};
	BCS_CHECK_EQUAL( array1d<double>(a1.V(rep(2, 5))), aview1d<double>(r1r, 5));

}

BCS_TEST_CASE( test_aview1d_copy )
{
	const size_t N = 5;

	double a0_buf[N] = {2, 4, 5, 7, 8};
	double a1_buf[N] = {0, 0, 0, 0, 0};
	double e2_buf[2 * N] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	double e3_buf[2 * N] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	double a4_buf[N] = {0, 0, 0, 0, 0};

	aview1d<double> a0(a0_buf, N);
	aview1d<double> a1(a1_buf, N);
	aview1d_ex<double, step_range> e2(e2_buf, rgn_n(0, N, 2));
	aview1d_ex<double, step_range> e3(e3_buf, rgn_n(0, N, 2));
	aview1d<double> a4(a4_buf, N);

	copy(a0, a1);
	BCS_CHECK( array_view_equal(a1, a0_buf, N) );

	copy(a1, e2);
	BCS_CHECK( array_view_equal(e2, a0_buf, N) );

	copy(e2, e3);
	BCS_CHECK( array_view_equal(e3, a0_buf, N) );

	copy(e3, a4);
	BCS_CHECK( array_view_equal(a4, a0_buf, N) );
}


BCS_TEST_CASE( test_aview1d_clone )
{
	double src[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

	aview1d<double> view1(src, 5);
	array1d<double> a1 = clone_array(view1);

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_iteration_test(a1) );
	BCS_CHECK( a1.pbase() != view1.pbase() );
	BCS_CHECK( array_view_equal(a1, src, 5) );
	BCS_CHECK( test_generic_operations(a1, src) );

	aview1d_ex<double, step_range> view2(src, step_range::from_begin_dim(1, 3, 2));
	array1d<double> a2 = clone_array(view2);

	BCS_CHECK( array_integrity_test(a2) );
	BCS_CHECK( array_iteration_test(a2) );
	double r2[3] = {2, 4, 6};
	BCS_CHECK( array_view_equal(a2, r2, 3) );
	BCS_CHECK( test_generic_operations(a2, r2) );
}


BCS_TEST_CASE( test_subarr_selection )
{
	const size_t N = 10;
	double src[N] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
	bool msk[N] = {false, false, true, false, true, false, false, true, false, true};
	const size_t n = 4;

	caview1d<double> a0(src, N);
	caview1d<bool> b0(msk, N);

	index_t I0r[n] = {2, 4, 7, 9};
	array1d<index_t> I0 = find(b0);

	BCS_CHECK( array_view_equal(I0, I0r, n) );

	double sr[n] = {30, 50, 80, 100};
	array1d<double> s = select_elems(a0, I0);
	BCS_CHECK( array_view_equal(s, sr, n) );
	BCS_CHECK( array_view_equal(select_elems(a0, find(b0)), sr, n));

	bool msk1[] = {false, false, false, false, false, false, false, false, false, false};
	caview1d<bool> b1(msk1, N);

	array1d<index_t> I1 = find(b1);
	array1d<double> s1 = select_elems(a0, I1);

	BCS_CHECK( array_integrity_test(I1) );
	BCS_CHECK( I1.nelems() == 0 );
	BCS_CHECK( array_iteration_test(I1) );

	BCS_CHECK( array_integrity_test(s1) );
	BCS_CHECK( s1.nelems() == 0 );
	BCS_CHECK( array_iteration_test(s1) );
}



std::shared_ptr<test_suite> test_array1d_suite()
{
	BCS_NEW_TEST_SUITE( suite, "test_array1d" );

	BCS_ADD_TEST_CASE( suite, test_dense_array1d() );
	BCS_ADD_TEST_CASE( suite, test_step_aview1d() );
	BCS_ADD_TEST_CASE( suite, test_rep_aview1d() );
	BCS_ADD_TEST_CASE( suite, test_regular_subview() );
	BCS_ADD_TEST_CASE( suite, test_aview1d_copy() );
	BCS_ADD_TEST_CASE( suite, test_aview1d_clone() );
	BCS_ADD_TEST_CASE( suite, test_subarr_selection() );

	return suite;
}





