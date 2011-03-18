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

template class array1d<double>;
template class aview1d<double, step_ind>;
template class aview1d<double, array_ind>;


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

	if (view.pbase() != &(view[0])) return false;

	for (index_t i = 0; i < (index_t)n; ++i)
	{
		if (view.ptr(i) != &(view(i))) return false;
		if (view.ptr(i) != &(view[i])) return false;
	}
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


test_suite *test_array1d_suite()
{
	test_suite *suite = new test_suite( "test_array1d" );

	suite->add( new test_dense_array1d() );

	return suite;
}





