/**
 * @file test_poly_scan.cpp
 *
 * Unit testing of polygon scan
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/geometry/poly_scan.h>

#include <cstdio>

using namespace bcs;
using namespace bcs::test;

template<class Scanner>
void print_scan_results(const Scanner& scanner)
{
	for (geo_index_t y = (geo_index_t)std::ceil(scanner.top()); y <= scanner.bottom(); ++y)
	{
		rowscan_segment s = scanner.get_rowscan_segment(y);
		std::printf("y = %d: x0 = %d, x1 = %d\n", s.y, s.x0, s.x1);
	}
}


template<class Scanner>
bool test_rowscan(const Scanner& scanner, geo_index_t y, geo_index_t x0, geo_index_t x1)
{
	rowscan_segment s = scanner.get_rowscan_segment(y);
	return s == make_rowscan_segment(y, x0, x1);
}


BCS_TEST_CASE( test_triangle_scan )
{
	triangle<double> t1 = make_tri(pt(0.0, 2.0), pt(1.0, 0.0), pt(3.0, 3.0));
	triangle_scanner<double> t1s(t1);

	BCS_CHECK_EQUAL( t1s.top(), 0.0 );
	BCS_CHECK_EQUAL( t1s.bottom(), 3.0 );

	BCS_CHECK( test_rowscan(t1s, 0, 1, 1) );
	BCS_CHECK( test_rowscan(t1s, 1, 1, 1) );
	BCS_CHECK( test_rowscan(t1s, 2, 0, 2) );
	BCS_CHECK( test_rowscan(t1s, 3, 3, 3) );

	triangle<double> t2 = make_tri(pt(0.0, 3.0), pt(2.0, 0.0), pt(3.0, 2.0));
	triangle_scanner<double> t2s(t2);

	BCS_CHECK_EQUAL( t2s.top(), 0.0 );
	BCS_CHECK_EQUAL( t2s.bottom(), 3.0 );

	BCS_CHECK( test_rowscan(t2s, 0, 2, 2) );
	BCS_CHECK( test_rowscan(t2s, 1, 2, 2) );
	BCS_CHECK( test_rowscan(t2s, 2, 1, 3) );
	BCS_CHECK( test_rowscan(t2s, 3, 0, 0) );

	triangle<double> t3 = make_tri(pt(0.8, 2.0), pt(1.5, 0.0), pt(1.5, 2.0) );
	triangle_scanner<double> t3s(t3);

	BCS_CHECK_EQUAL( t3s.top(), 0.0 );
	BCS_CHECK_EQUAL( t3s.bottom(), 2.0 );

	BCS_CHECK( test_rowscan(t3s, 0, 2, 1) );
	BCS_CHECK( test_rowscan(t3s, 1, 2, 1) );
	BCS_CHECK( test_rowscan(t3s, 2, 1, 1) );

	triangle<double> t4 = make_tri(pt(0.0, 0.0), pt(2.0, 2.1), pt(3.0, 0.0));
	triangle_scanner<double> t4s(t4);

	BCS_CHECK_EQUAL( t4s.top(), 0.0 );
	BCS_CHECK_EQUAL( t4s.bottom(), 2.1 );

	BCS_CHECK( test_rowscan(t4s, 0, 0, 3) );
	BCS_CHECK( test_rowscan(t4s, 1, 1, 2) );
	BCS_CHECK( test_rowscan(t4s, 2, 2, 2) );

	triangle<double> t5 = make_tri(pt(0.0, 3.0), pt(1.0, 0.0), pt(3.0, 3.0));
	triangle_scanner<double> t5s(t5);

	BCS_CHECK_EQUAL( t5s.top(), 0.0 );
	BCS_CHECK_EQUAL( t5s.bottom(), 3.0 );

	BCS_CHECK( test_rowscan(t5s, 0, 1, 1) );
	BCS_CHECK( test_rowscan(t5s, 1, 1, 1) );
	BCS_CHECK( test_rowscan(t5s, 2, 1, 2) );
	BCS_CHECK( test_rowscan(t5s, 3, 0, 3) );
}



test_suite *test_poly_scan_suite()
{
	test_suite *suite = new test_suite( "test_poly_scan" );

	suite->add( new test_triangle_scan() );

	return suite;
}
