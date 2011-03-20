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
	double src[24];
	for (int i = 0; i < 24; ++i) src[i] = i+1;

	// row major




}






test_suite *test_array2d_suite()
{
	test_suite *suite = new test_suite( "test_array2d" );

	suite->add( new test_dense_array2d() );
	suite->add( new test_gen_array2d() );

	return suite;
}





