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


BCS_TEST_CASE( test_dense_array2d  )
{

}




test_suite *test_array2d_suite()
{
	test_suite *suite = new test_suite( "test_array2d" );

	suite->add( new test_dense_array2d() );

	return suite;
}





