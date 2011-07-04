/**
 * @file test_arbb_port.cpp
 *
 * Unit testing of ArBB port functions
 *
 * @author Dahua Lin
 */


#include <bcslib/test/test_units.h>
#include <bcslib/test/test_array_aux.h>
#include <bcslib/array/arbb_port.h>


using namespace bcs;
using namespace bcs::test;


BCS_TEST_CASE( test_arbb_copy )
{
	using arbb::f64;

	const size_t N = 12;
	const size_t m = 3;
	const size_t n = 4;
	double src[N] = {3, 4, 6, 1, 7, 2, 9, 8, 5, 3, 1, 4 };
	double dst[N];

	// 1D

	caview1d<double> s1(src, N);
	arbb::dense<f64, 1> a1(N);
	copy(s1, a1);

	aview1d<double> t1(dst, N);
	set_zeros(t1);
	copy(a1, t1);

	BCS_CHECK( array_view_equal(t1, src, N) );

	// 2D: row major

	caview2d<double, row_major_t> s2(src, m, n);
	arbb::dense<f64, 2> a2(n, m);
	copy(s2, a2);

	aview2d<double, row_major_t> t2(dst, m, n);
	set_zeros(t2);
	copy(a2, t2);

	BCS_CHECK( array_view_equal(t2, src, m, n) );

	// 2D: column major

	caview2d<double, column_major_t> s3(src, m, n);
	arbb::dense<f64, 2> a3(m, n);
	copy(s3, a3);

	aview2d<double, column_major_t> t3(dst, m, n);
	set_zeros(t3);
	copy(a3, t3);

	BCS_CHECK( array_view_equal(t3, src, m, n) );

}


test_suite* test_arbb_port_suite()
{
	test_suite *suite = new test_suite( "test_arbb_port" );

	suite->add( new test_arbb_copy() );

	return suite;
}





