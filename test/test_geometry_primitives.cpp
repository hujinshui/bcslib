/**
 * @file test_poly_scan.cpp
 *
 * Unit testing of geometry primitive structures
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/geometry/geometry_base.h>

#include <cmath>

using namespace bcs;
using namespace bcs::test;

// For syntax check

template struct bcs::point2d<double>;
template struct bcs::point3d<double>;

template struct bcs::lineseg2d<double>;
template struct bcs::lineseg3d<double>;
template struct bcs::line2d<double>;

template struct bcs::rectangle<double>;
template struct bcs::triangle<double>;


BCS_TEST_CASE( test_points )
{
	point2d<double> pt1 = pt(2.0, 3.0);

	BCS_CHECK_EQUAL( pt1.x, 2.0 );
	BCS_CHECK_EQUAL( pt1.y, 3.0 );
	BCS_CHECK( pt1 == pt(2.0, 3.0) );
	BCS_CHECK( pt1 != pt(2.0, 4.0) );

	BCS_CHECK_APPROX( distance(pt1, pt(3.0, 5.0)), std::sqrt(5.0) );

	point3d<double> pt2 = pt(1.0, 2.0, 3.0);

	BCS_CHECK_EQUAL( pt2.x, 1.0 );
	BCS_CHECK_EQUAL( pt2.y, 2.0 );
	BCS_CHECK_EQUAL( pt2.z, 3.0 );
	BCS_CHECK( pt2 == pt(1.0, 2.0, 3.0) );
	BCS_CHECK( pt2 != pt(1.0, 2.0, 4.0) );

	BCS_CHECK_APPROX( distance(pt2, pt(5.0, 3.0, 2.0)), std::sqrt(18.0) );
}


BCS_TEST_CASE( test_line_segs )
{
	lineseg2d<double> s1 = lineseg( pt(2.0, 3.0), pt(3.0, 5.0) );

	BCS_CHECK_EQUAL( s1.pt1, pt(2.0, 3.0) );
	BCS_CHECK_EQUAL( s1.pt2, pt(3.0, 5.0) );
	BCS_CHECK( s1 == s1 );
	BCS_CHECK_APPROX( length(s1), std::sqrt(5.0) );

	lineseg3d<double> s2 = lineseg( pt(1.0, 2.0, 3.0), pt(5.0, 3.0, 2.0) );

	BCS_CHECK_EQUAL( s2.pt1, pt(1.0, 2.0, 3.0) );
	BCS_CHECK_EQUAL( s2.pt2, pt(5.0, 3.0, 2.0) );
	BCS_CHECK( s2 == s2 );
	BCS_CHECK_APPROX( length(s2), std::sqrt(18.0) );
}


BCS_TEST_CASE( test_line2d )
{
	line2d<double> l1 = line(3.0, -1.0, 2.0);

	BCS_CHECK_EQUAL( l1.a, 3.0 );
	BCS_CHECK_EQUAL( l1.b, -1.0 );
	BCS_CHECK_EQUAL( l1.c, 2.0 );
	BCS_CHECK_APPROX( l1.slope(), 3.0 );
	BCS_CHECK_APPROX( l1.horiz_intersect(), -2.0/3 );
	BCS_CHECK_APPROX( l1.vert_intersect(), 2.0 );
	BCS_CHECK_APPROX( l1.horiz_intersect(1.0), -1.0/3);
	BCS_CHECK_APPROX( l1.vert_intersect(1.0), 5.0 );

	line2d<double> l2 = line2d<double>::from_segment( lineseg(pt(2.0, 3.0), pt(3.0, 5.0)) );

	BCS_CHECK_APPROX( l2.evaluate(pt(2.0, 3.0)), 0.0 );
	BCS_CHECK_APPROX( l2.evaluate(pt(3.0, 5.0)), 0.0 );
}


BCS_TEST_CASE( test_rectangle  )
{
	rectangle<double> rc1 = rect( 2.0, 3.0, 6.0, 4.0 );

	BCS_CHECK_EQUAL( rc1.left(), 2.0 );
	BCS_CHECK_EQUAL( rc1.right(), 8.0 );
	BCS_CHECK_EQUAL( rc1.top(), 3.0 );
	BCS_CHECK_EQUAL( rc1.bottom(), 7.0 );
	BCS_CHECK_EQUAL( rc1.width(), 6.0 );
	BCS_CHECK_EQUAL( rc1.height(), 4.0 );

	BCS_CHECK_EQUAL( rc1.top_left(), pt(2.0, 3.0) );
	BCS_CHECK_EQUAL( rc1.top_right(), pt(8.0, 3.0) );
	BCS_CHECK_EQUAL( rc1.bottom_left(), pt(2.0, 7.0) );
	BCS_CHECK_EQUAL( rc1.bottom_right(), pt(8.0, 7.0) );
	BCS_CHECK_EQUAL( area(rc1), 24.0 );
	BCS_CHECK( !rc1.is_empty() );

	rectangle<double> rc2 = rect( pt(2.0, 7.0),  pt(8.0, 3.0) );

	BCS_CHECK_EQUAL( rc1, rc2 );
}


BCS_TEST_CASE( test_triangle )
{
	triangle<double> t1 = make_tri( pt(0.0, 0.0), pt(1.0, 3.0), pt(2.0, 0.0) );

	BCS_CHECK_EQUAL( t1.pt1, pt(0.0, 0.0) );
	BCS_CHECK_EQUAL( t1.pt2, pt(1.0, 3.0) );
	BCS_CHECK_EQUAL( t1.pt3, pt(2.0, 0.0) );
	BCS_CHECK_EQUAL( t1, t1 );

	BCS_CHECK_EQUAL( area(t1), 3.0 );

}



test_suite *test_geometry_prim_suite()
{
	test_suite *suite = new test_suite( "test_geometry_primitives" );

	suite->add( new test_points() );
	suite->add( new test_line_segs() );
	suite->add( new test_line2d() );
	suite->add( new test_rectangle() );
	suite->add( new test_triangle() );

	return suite;
}


