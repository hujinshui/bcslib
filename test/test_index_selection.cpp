/**
 * @file test_index_selection.cpp
 *
 * The Unit Testing for index_selection.h
 *
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/array/index_selection.h>

#include <stdio.h>

using namespace bcs;
using namespace bcs::test;

BCS_TEST_CASE( test_range )
{
	range rg0(1, 1);

	BCS_CHECK( !is_open_end_selector(rg0) );
	BCS_CHECK_EQUAL( rg0.begin, 1 );
	BCS_CHECK_EQUAL( rg0.end, 1 );
	BCS_CHECK_EQUAL( rg0.size(), 0);
	BCS_CHECK( rg0.is_empty() );

	range rg1(1, 5);

	BCS_CHECK( !is_open_end_selector(rg1) );
	BCS_CHECK_EQUAL( rg1.begin, 1 );
	BCS_CHECK_EQUAL( rg1.end, 5 );
	BCS_CHECK_EQUAL( rg1.size(), 4 );

	int rg1_s[] = {1, 2, 3, 4};
	BCS_CHECK( enumerate_equal(rg1.get_enumerator(), rg1_s, 4) );

	BCS_CHECK( !rg1.is_empty() );
	BCS_CHECK( rg1.is_contained_in(5) );
	BCS_CHECK( !rg1.is_contained_in(4) );

	range rg2(2, 1);

	BCS_CHECK_EQUAL( rg2.begin, 2 );
	BCS_CHECK_EQUAL( rg2.end, 1 );
	BCS_CHECK_EQUAL( rg2.size(), 0 );
	BCS_CHECK( rg2.is_empty() );
}


BCS_TEST_CASE( test_step_range )
{
	step_range rg0(1, 1, 2);

	BCS_CHECK( !is_open_end_selector(rg0) );

	BCS_CHECK_EQUAL( rg0.begin, 1 );
	BCS_CHECK_EQUAL( rg0.end, 1 );
	BCS_CHECK_EQUAL( rg0.step, 2 );
	BCS_CHECK_EQUAL( rg0.size(), 0 );
	BCS_CHECK( rg0.is_empty() );

	step_range rg1(1, 11, 3);
	int rg1_s[] = {1, 4, 7, 10};

	BCS_CHECK_EQUAL( rg1.begin, 1 );
	BCS_CHECK_EQUAL( rg1.end, 11 );
	BCS_CHECK_EQUAL( rg1.step, 3 );
	BCS_CHECK_EQUAL( rg1.size(), 4 );
	BCS_CHECK( !rg1.is_empty() );
	BCS_CHECK( enumerate_equal(rg1.get_enumerator(), rg1_s, 4) );

	step_range rg2(1, 12, 3);
	int rg2_s[] = {1, 4, 7, 10};

	BCS_CHECK_EQUAL( rg2.begin, 1 );
	BCS_CHECK_EQUAL( rg2.end, 12 );
	BCS_CHECK_EQUAL( rg2.step, 3 );
	BCS_CHECK_EQUAL( rg2.size(), 4 );
	BCS_CHECK( !rg2.is_empty() );
	BCS_CHECK( enumerate_equal(rg2.get_enumerator(), rg2_s, 4) );

	step_range rg3(5, 1, -1);
	int rg3_s[] = {5, 4, 3, 2};

	BCS_CHECK_EQUAL( rg3.begin, 5 );
	BCS_CHECK_EQUAL( rg3.end, 1 );
	BCS_CHECK_EQUAL( rg3.step, -1 );
	BCS_CHECK_EQUAL( rg3.size(), 4 );
	BCS_CHECK( !rg3.is_empty() );
	BCS_CHECK( enumerate_equal(rg3.get_enumerator(), rg3_s, 4) );

	step_range rg4(5, 1, -2);
	int rg4_s[] = {5, 3};

	BCS_CHECK_EQUAL( rg4.begin, 5 );
	BCS_CHECK_EQUAL( rg4.end, 1 );
	BCS_CHECK_EQUAL( rg4.step, -2 );
	BCS_CHECK_EQUAL( rg4.size(), 2 );
	BCS_CHECK( !rg4.is_empty() );
	BCS_CHECK( enumerate_equal(rg4.get_enumerator(), rg4_s, 2) );

	step_range rg5(5, 2, -2);
	int rg5_s[] = {5, 3};

	BCS_CHECK_EQUAL( rg5.begin, 5 );
	BCS_CHECK_EQUAL( rg5.end, 2 );
	BCS_CHECK_EQUAL( rg5.step, -2 );
	BCS_CHECK_EQUAL( rg5.size(), 2 );
	BCS_CHECK( !rg5.is_empty() );
	BCS_CHECK( enumerate_equal(rg5.get_enumerator(), rg5_s, 2) );

	step_range rg6(5, 0, -2);
	int rg6_s[] = {5, 3, 1};

	BCS_CHECK_EQUAL( rg6.begin, 5 );
	BCS_CHECK_EQUAL( rg6.end, 0 );
	BCS_CHECK_EQUAL( rg6.step, -2 );
	BCS_CHECK_EQUAL( rg6.size(), 3 );
	BCS_CHECK( !rg6.is_empty() );
	BCS_CHECK( enumerate_equal(rg6.get_enumerator(), rg6_s, 3) );
}


BCS_TEST_CASE( test_rep_range )
{
	rep_range rg1(3, 5);
	int rg1_s[] = {3, 3, 3, 3, 3};

	BCS_CHECK_EQUAL( rg1.rep_i, 3 );
	BCS_CHECK_EQUAL( rg1.rep_n, 5 );
	BCS_CHECK_EQUAL( rg1.size(), 5 );
	BCS_CHECK( !rg1.is_empty() );
	BCS_CHECK( enumerate_equal(rg1.get_enumerator(), rg1_s, 5) );
}



BCS_TEST_CASE( test_whole )
{
	BCS_CHECK( is_open_end_selector(whole()) );

	range rg1 = whole().close(0);
	BCS_CHECK_EQUAL( rg1, range(0, 0) );

	range rg2 = whole().close(3);
	BCS_CHECK_EQUAL( rg2, range(0, 3) );
}


BCS_TEST_CASE( test_open_range )
{
	BCS_CHECK( is_open_end_selector(open_range(1)) );

	BCS_CHECK_EQUAL( open_range(1).close(3), range(1, 3) );
	BCS_CHECK_EQUAL( open_range(3, 0).close(3), range(3, 3) );
	BCS_CHECK_EQUAL( open_range(0, -1).close(5), range(0, 4) );
}

BCS_TEST_CASE( test_open_step_range )
{
	BCS_CHECK( is_open_end_selector(open_step_range(1, 0, 1)) );

	BCS_CHECK_EQUAL( open_step_range(1, 0, 2).close(8), step_range(1, 8, 2) );
	BCS_CHECK_EQUAL( open_step_range(3, -2, 5).close(9), step_range(3, 7, 5) );
}


BCS_TEST_CASE( test_indices )
{
	index_t rg_s[5] = {3, 4, 6, 1, 2};

	indices rg(ref_arr(rg_s, 5));

	BCS_CHECK( !is_open_end_selector(rg) );

	BCS_CHECK_EQUAL( rg.size(), 5 );
	BCS_CHECK( !rg.is_empty() );

	BCS_CHECK( rg.is_contained_in(7) );
	BCS_CHECK( !rg.is_contained_in(6) );

	BCS_CHECK( enumerate_equal(rg.get_enumerator(), rg_s, 5) );
}


test_suite *test_index_selection_suite()
{
	test_suite *suite = new test_suite( "test_index_selection" );

	suite->add( new test_range() );
	suite->add( new test_step_range() );
	suite->add( new test_rep_range() );
	suite->add( new test_whole() );
	suite->add( new test_open_range() );
	suite->add( new test_open_step_range() );
	suite->add( new test_indices() );

	return suite;
}





