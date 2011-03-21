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

template class array2d<double, row_major_t>;
template class array2d<double, column_major_t>;

template class aview2d<double, row_major_t, step_ind, step_ind>;
template class aview2d<double, column_major_t, step_ind, step_ind>;

template class aview2d<double, row_major_t, indices, indices>;
template class aview2d<double, column_major_t, indices, indices>;



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

	aview2d<double, row_major_t, indices, id_ind> a4_rm(src, 5, 6, indices(inds, 3), id_ind(4));
	double r4_rm[] = {1, 2, 3, 4, 13, 14, 15, 16, 19, 20, 21, 22};

	BCS_CHECK( !is_dense_view(a4_rm) );
	BCS_CHECK( array_integrity_test(a4_rm) );
	BCS_CHECK( array_view_equal(a4_rm, r4_rm, 3, 4) );
	BCS_CHECK( array_iteration_test(a4_rm) );

	aview2d<double, column_major_t, indices, id_ind> a4_cm(src, 5, 6, indices(inds, 3), id_ind(4));
	double r4_cm[] = {1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18, 19};

	BCS_CHECK( !is_dense_view(a4_cm) );
	BCS_CHECK( array_integrity_test(a4_cm) );
	BCS_CHECK( array_view_equal(a4_cm, r4_cm, 3, 4) );
	BCS_CHECK( array_iteration_test(a4_cm) );

	// (indices, step_ind(2))

	aview2d<double, row_major_t, indices, step_ind> a5_rm(src, 5, 6, indices(inds, 3), step_ind(3, 2));
	double r5_rm[] = {1, 3, 5, 13, 15, 17, 19, 21, 23};

	BCS_CHECK( !is_dense_view(a5_rm) );
	BCS_CHECK( array_integrity_test(a5_rm) );
	BCS_CHECK( array_view_equal(a5_rm, r5_rm, 3, 3) );
	BCS_CHECK( array_iteration_test(a5_rm) );

	aview2d<double, column_major_t, indices, step_ind> a5_cm(src, 5, 6, indices(inds, 3), step_ind(3, 2));
	double r5_cm[] = {1, 3, 4, 11, 13, 14, 21, 23, 24};

	BCS_CHECK( !is_dense_view(a5_cm) );
	BCS_CHECK( array_integrity_test(a5_cm) );
	BCS_CHECK( array_view_equal(a5_cm, r5_cm, 3, 3) );
	BCS_CHECK( array_iteration_test(a5_cm) );

	// (indices, indices)

	aview2d<double, row_major_t, indices, indices> a6_rm(src, 5, 6, indices(inds, 2), indices(inds, 3));
	double r6_rm[] = {1, 3, 4, 13, 15, 16};

	BCS_CHECK( !is_dense_view(a6_rm) );
	BCS_CHECK( array_integrity_test(a6_rm) );
	BCS_CHECK( array_view_equal(a6_rm, r6_rm, 2, 3) );
	BCS_CHECK( array_iteration_test(a6_rm) );

	aview2d<double, column_major_t, indices, indices> a6_cm(src, 5, 6, indices(inds, 2), indices(inds, 3));
	double r6_cm[] = {1, 3, 11, 13, 16, 18};

	BCS_CHECK( !is_dense_view(a6_cm) );
	BCS_CHECK( array_integrity_test(a6_cm) );
	BCS_CHECK( array_view_equal(a6_cm, r6_cm, 2, 3) );
	BCS_CHECK( array_iteration_test(a6_cm) );

	// (rep_ind, indices)

	aview2d<double, row_major_t, rep_ind, indices> a7_rm(src, 5, 6, rep_ind(2), indices(inds, 3));
	double r7_rm[] = {1, 3, 4, 1, 3, 4};

	BCS_CHECK( !is_dense_view(a7_rm) );
	BCS_CHECK( array_integrity_test(a7_rm) );
	BCS_CHECK( array_view_equal(a7_rm, r7_rm, 2, 3) );
	BCS_CHECK( array_iteration_test(a7_rm) );

	aview2d<double, column_major_t, rep_ind, indices> a7_cm(src, 5, 6, rep_ind(2), indices(inds, 3));
	double r7_cm[] = {1, 1, 11, 11, 16, 16};

	BCS_CHECK( !is_dense_view(a7_cm) );
	BCS_CHECK( array_integrity_test(a7_cm) );
	BCS_CHECK( array_view_equal(a7_cm, r7_cm, 2, 3) );
	BCS_CHECK( array_iteration_test(a7_cm) );

}






test_suite *test_array2d_suite()
{
	test_suite *suite = new test_suite( "test_array2d" );

	suite->add( new test_dense_array2d() );
	suite->add( new test_gen_array2d() );

	return suite;
}





