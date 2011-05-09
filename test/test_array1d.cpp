/**
 * @file test_array1d.cpp
 *
 * The Unit Testing for array1d
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/test_units.h>
#include <bcslib/array/array1d.h>

#include <iostream>

using namespace bcs;
using namespace bcs::test;

// Explicit instantiation for syntax checking

template class bcs::array1d<double>;
template class bcs::aview1d<double, step_ind>;
template class bcs::aview1d<double, rep_ind>;
template class bcs::aview1d<double, indices>;


template<typename T, class TIndexer>
void print_array(const bcs::const_aview1d<T, TIndexer>& view, const char *title = 0)
{
	if (title != 0)
		std::cout << title << ' ';

	index_t n = (index_t)view.nelems();
	for (index_t i = 0; i < n; ++i)
	{
		std::cout << view(i) << ' ';
	}
	std::cout << std::endl;
}


template<typename T, class TIndexer>
bool array_integrity_test(const bcs::const_aview1d<T, TIndexer>& view)
{
	size_t n = view.nelems();

	if (view.ndims() != 1) return false;
	if (view.shape() != arr_shape(n)) return false;

	for (index_t i = 0; i < (index_t)n; ++i)
	{
		if (view.ptr(i) != &(view(i))) return false;
		if (view.ptr(i) != &(view[i])) return false;
	}

	array1d<T> acopy = make_copy(view);
	if (!(acopy == view)) return false;

	return true;
}


template<typename T, class TIndexer>
bool array_iteration_test(const bcs::const_aview1d<T, TIndexer>& view)
{
	size_t n = view.nelems();
	bcs::block<T> buffer(n);
	for (size_t i = 0; i < n; ++i) buffer[i] = view(i);

	return collection_equal(view.begin(), view.end(), buffer.pbase(), n);
}



BCS_TEST_CASE( test_dense_array1d )
{
	double src1[6] = {7, 7, 7, 7, 7, 7};
	size_t n1 = 6;

	double src2[5] = {3, 4, 5, 1, 2};
	size_t n2 = 5;

	array1d<double> a0(0);

	BCS_CHECK( array_integrity_test(a0) );
	BCS_CHECK( array_iteration_test(a0) );

	array1d<double> a1(n1, src1[0]);

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_view_equal(a1, src1, n1) );
	BCS_CHECK( array_iteration_test(a1) );

	array1d<double> a2(n2, src2);

	BCS_CHECK( array_integrity_test(a2) );
	BCS_CHECK( array_view_equal(a2, src2, n2) );
	BCS_CHECK( array_iteration_test(a2) );

	block<double> a2_buf(n2);
	export_to(a2, a2_buf.pbase());

	BCS_CHECK( collection_equal(a2_buf.pbase(), a2_buf.pend(), src2, n2) );

}


BCS_TEST_CASE( test_step_array1d )
{
	double src0[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	double src1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

	aview1d<double, step_ind> a1(src1, step_ind(3, 1));
	double r1[] = {1, 2, 3};

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_view_equal(a1, r1, 3) );
	BCS_CHECK( array_iteration_test(a1) );

	aview1d<double, step_ind> a2(src1, step_ind(4, 2));
	double r2[] = {1, 3, 5, 7};

	BCS_CHECK( array_integrity_test(a2) );
	BCS_CHECK( array_view_equal(a2, r2, 4) );
	BCS_CHECK( array_iteration_test(a2) );

	aview1d<double, step_ind> a3(src1 + 7, step_ind(3, -2));
	double r3[] = {8, 6, 4};

	BCS_CHECK( array_integrity_test(a3) );
	BCS_CHECK( array_view_equal(a3, r3, 3) );
	BCS_CHECK( array_iteration_test(a3) );

	aview1d<double, step_ind> a0(src0, step_ind(4, 2));

	import_from(a0, src1);
	double g1[10] = {1, 0, 2, 0, 3, 0, 4, 0, 0, 0};
	BCS_CHECK( collection_equal(src0, src0 + 10, g1, 10) );

	fill(a0, 0.0);
	double g2[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	BCS_CHECK( collection_equal(src0, src0 + 10, g2, 10) );
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

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_view_equal(a1, r1, 5) );
	BCS_CHECK( array_iteration_test(a1) );
}




BCS_TEST_CASE( test_indices_array1d )
{
	double src0[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	double src1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

	index_t inds1[4] = {1, 3, 6, 7};
	aview1d<double, indices> a1(src1, indices(ref_arr(inds1, 4)));
	double r1[] = {2, 4, 7, 8};

	BCS_CHECK( array_integrity_test(a1) );
	BCS_CHECK( array_view_equal(a1, r1, 4) );
	BCS_CHECK( array_iteration_test(a1) );

	index_t inds2[6] = {5, 1, 4, 2, 2, 3};
	aview1d<double, indices> a2(src1, indices(ref_arr(inds2, 6)));
	double r2[] = {6, 2, 5, 3, 3, 4};

	BCS_CHECK( array_integrity_test(a2) );
	BCS_CHECK( array_view_equal(a2, r2, 6) );
	BCS_CHECK( array_iteration_test(a2) );

	index_t inds0[5] = {0, 2, 3, 5, 7};
	aview1d<double, indices> a0(src0, indices(ref_arr(inds0, 5)));

	import_from(a0, src1);
	double g1[10] = {1, 0, 2, 3, 0, 4, 0, 5, 0, 0};
	BCS_CHECK( collection_equal(src0, src0 + 10, g1, 10) );

	fill(a0, 0.0);
	double g2[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	BCS_CHECK( collection_equal(src0, src0 + 10, g2, 10) );
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
	BCS_CHECK_EQUAL( a1.V(rgn(2, aend() - 1)), aview1d<double>(r1d, 5));

	double r1e[] = {1, 3, 5, 7};
	BCS_CHECK_EQUAL( a1.V(rgn(1, aend(), 2)), aview1d<double>(r1e, 4));

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
	BCS_CHECK_EQUAL( a2.V(rgn(2, aend() - 1)), aview1d<double>(r2d, 5));

	double r2e[] = {2, 6, 10, 14};
	BCS_CHECK_EQUAL( a2.V(rgn(1, aend(), 2)), aview1d<double>(r2e, 4));

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
	BCS_CHECK_EQUAL( a3.V(rgn(2, aend() - 1)), aview1d<double>(r3d, 5));

	double r3e[] = {7, 5, 3, 1};
	BCS_CHECK_EQUAL( a3.V(rgn(1, aend(), 2)), aview1d<double>(r3e, 4));

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
	BCS_CHECK_EQUAL( a4.V(rgn(2, aend() - 1)), aview1d<double>(r4d, 5));

	double r4e[] = {2, 2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(rgn(1, aend(), 2)), aview1d<double>(r4e, 4));

	double r4f[] = {2, 2, 2, 2, 2, 2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(rev_whole()),  aview1d<double>(r4f, 8));

	double r4r[] = {2, 2, 2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(rep(2, 5)), aview1d<double>(r4r, 5));

}


BCS_TEST_CASE( test_indices_subview )
{
	double src[20];
	for (int i = 0; i < 20; ++i) src[i] = i;  // 0, 1, 2, ..., 19

	index_t inds[4] = {1, 3, 6, 7};

	aview1d<double, id_ind> a1(src, 8);
	double r1i[4] = {1, 3, 6, 7};
	BCS_CHECK_EQUAL( a1.V(indices(ref_arr(inds, 4))), aview1d<double>(r1i, 4));

	aview1d<double, step_ind> a2(src, step_ind(8, 2));
	double r2i[4] = {2, 6, 12, 14};
	BCS_CHECK_EQUAL( a2.V(indices(ref_arr(inds, 4))), aview1d<double>(r2i, 4));

	aview1d<double, step_ind> a3(src + 8, step_ind(8, -1));
	double r3i[4] = {7, 5, 2, 1};
	BCS_CHECK_EQUAL( a3.V(indices(ref_arr(inds, 4))), aview1d<double>(r3i, 4));

	aview1d<double, rep_ind> a4(src + 2, rep_ind(8));
	double r4i[4] = {2, 2, 2, 2};
	BCS_CHECK_EQUAL( a4.V(indices(ref_arr(inds, 4))), aview1d<double>(r4i, 4));


	index_t inds_i[8] = {1, 3, 6, 7, 9, 13, 15, 16};
	aview1d<double, indices> ai(src, indices(ref_arr(inds_i, 8)));

	double ria[8] = {1, 3, 6, 7, 9, 13, 15, 16};
	BCS_CHECK_EQUAL( ai.V(whole()), aview1d<double>(ria, 8) );

	double rib[5] = {3, 6, 7, 9, 13};
	BCS_CHECK_EQUAL( ai.V(rgn(1, 6)), aview1d<double>(rib, 5) );

	double ric[3] = {3, 7, 13};
	BCS_CHECK_EQUAL( ai.V(rgn(1, 7, 2)), aview1d<double>(ric, 3) );

	double rid[5] = {6, 7, 9, 13, 15};
	BCS_CHECK_EQUAL( ai.V(rgn(2, aend() - 1)), aview1d<double>(rid, 5));

	double rie[4] = {3, 7, 13, 16};
	BCS_CHECK_EQUAL( ai.V(rgn(1, aend(), 2)), aview1d<double>(rie, 4));

	double rif[8] = {16, 15, 13, 9, 7, 6, 3, 1};
	BCS_CHECK_EQUAL( ai.V(rev_whole()),  aview1d<double>(rif, 8) );

	double rir[5] = {6, 6, 6, 6, 6};
	BCS_CHECK_EQUAL( ai.V(rep(2, 5)),  aview1d<double>(rir, 5) );

	double rig[4] = {3, 7, 15, 16};
	BCS_CHECK_EQUAL( ai.V(indices(ref_arr(inds, 4))), aview1d<double>(rig, 4));

}




test_suite *test_array1d_suite()
{
	test_suite *suite = new test_suite( "test_array1d" );

	suite->add( new test_dense_array1d() );
	suite->add( new test_step_array1d() );
	suite->add( new test_rep_array1d() );
	suite->add( new test_indices_array1d() );
	suite->add( new test_regular_subview() );
	suite->add( new test_indices_subview() );

	return suite;
}





