/**
 * @file test_index_selection.cpp
 *
 * The Unit Testing for index_selectors.h
 *
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/array/array_index.h>

#include <functional>

using namespace bcs;
using namespace bcs::test;


template<class IndexSelector>
bool verify_indices(const IndexSelector& S, const index_t *ref, size_t n)
{
	typedef typename index_selector_traits<IndexSelector>::input_type input_t;
	if (S.size() != n) return false;

	for (input_t i = 0; i < (input_t)n; ++i)
	{
		if (S[i] != ref[i]) return false;
	}
	return collection_equal(S.begin(), S.end(), ref, n);
}



BCS_TEST_CASE( test_range )
{
	range rg0;

	BCS_CHECK_EQUAL( rg0.size(), 0 );
	BCS_CHECK_EQUAL( rg0.begin_index(), 0 );
	BCS_CHECK_EQUAL( rg0.end_index(), 0 );
	BCS_CHECK( rg0.begin() == rg0.end() );
	BCS_CHECK( rg0 == rg0 );

	range rg1(2, 5);
	index_t rg1_inds[3] = {2, 3, 4};
	BCS_CHECK_EQUAL( rg1.begin_index(), 2);
	BCS_CHECK_EQUAL( rg1.end_index(), 5 );
	BCS_CHECK_EQUAL( rg1.front(), 2 );
	BCS_CHECK_EQUAL( rg1.back(), 4 );
	BCS_CHECK( verify_indices(rg1, rg1_inds, 3) );
	BCS_CHECK( rg1 == rg1 );

	BCS_CHECK( rg0 != rg1 );

	BCS_CHECK_EQUAL( rg1, rgn(2, 5) );
	BCS_CHECK_EQUAL( rg1, rgn_n(2, 3) );
}


BCS_TEST_CASE( test_step_range )
{
	step_range rg0;

	BCS_CHECK_EQUAL( rg0.size(), 0 );
	BCS_CHECK_EQUAL( rg0.step(), 0 );
	BCS_CHECK_EQUAL( rg0.begin_index(), 0 );
	BCS_CHECK_EQUAL( rg0.end_index(), 0 );
	BCS_CHECK( rg0.begin() == rg0.end() );
	BCS_CHECK( rg0 == rg0 );

	step_range rg1 = step_range::from_begin_dim(1, 3, 4);
	index_t rg1_inds[4] = {1, 5, 9};
	BCS_CHECK_EQUAL( rg1.begin_index(), 1 );
	BCS_CHECK_EQUAL( rg1.end_index(), 13 );
	BCS_CHECK_EQUAL( rg1.front(), 1 );
	BCS_CHECK_EQUAL( rg1.back(), 9 );
	BCS_CHECK( verify_indices(rg1, rg1_inds, 3) );
	BCS_CHECK( rg1 == rg1 );
	BCS_CHECK_EQUAL( rg1, rgn_n(1, 3, 4) );

	step_range rg2 = step_range::from_begin_end(1, 11, 3);
	index_t rg2_inds[4] = {1, 4, 7, 10};
	BCS_CHECK_EQUAL( rg2.begin_index(), 1 );
	BCS_CHECK_EQUAL( rg2.end_index(), 13 );
	BCS_CHECK_EQUAL( rg2.front(), 1 );
	BCS_CHECK_EQUAL( rg2.back(), 10 );
	BCS_CHECK( verify_indices(rg2, rg2_inds, 4) );
	BCS_CHECK( rg2 == rg2 );
	BCS_CHECK_EQUAL( rg2, rgn(1, 11, 3) );

	step_range rg3 = step_range::from_begin_end(1, 9, 4);
	index_t rg3_inds[2] = {1, 5};
	BCS_CHECK_EQUAL( rg3.begin_index(), 1 );
	BCS_CHECK_EQUAL( rg3.end_index(), 9 );
	BCS_CHECK_EQUAL( rg3.front(), 1 );
	BCS_CHECK_EQUAL( rg3.back(), 5 );
	BCS_CHECK( verify_indices(rg3, rg3_inds, 2) );
	BCS_CHECK( rg3 == rg3 );
	BCS_CHECK_EQUAL( rg3, rgn(1, 9, 4) );

	step_range rg4 = step_range::from_begin_end(9, 0, -3);
	index_t rg4_inds[3] = {9, 6, 3};
	BCS_CHECK_EQUAL( rg4.begin_index(), 9 );
	BCS_CHECK_EQUAL( rg4.end_index(), 0 );
	BCS_CHECK_EQUAL( rg4.front(), 9 );
	BCS_CHECK_EQUAL( rg4.back(), 3 );
	BCS_CHECK( verify_indices(rg4, rg4_inds, 3) );
	BCS_CHECK( rg4 == rg4 );
	BCS_CHECK_EQUAL( rg4, rgn(9, 0, -3) );

	step_range rg5 = step_range::from_begin_end(1, 4, -1);
	BCS_CHECK_EQUAL( rg5.size(), 0 );
	BCS_CHECK( rg5.begin_index() == rg5.end_index() );
	BCS_CHECK( rg5 == rg5 );
	BCS_CHECK_EQUAL( rg5, rgn(1, 4, -1) );
}


BCS_TEST_CASE( test_rep_selector )
{
	rep_selector rs0;
	BCS_CHECK_EQUAL( rs0.index(), 0 );
	BCS_CHECK_EQUAL( rs0.size(), 0 );
	BCS_CHECK( rs0.begin() == rs0.end() );
	BCS_CHECK( rs0 == rs0 );

	rep_selector rs1(2, 5);
	index_t rs1_inds[5] = {2, 2, 2, 2, 2};
	BCS_CHECK_EQUAL( rs1.index(), 2 );
	BCS_CHECK( verify_indices(rs1, rs1_inds, 5) );
	BCS_CHECK( rs1 == rs1 );
	BCS_CHECK( rs1 == rep(2, 5) );

	BCS_CHECK( rs1 != rs0 );
}


BCS_TEST_CASE( test_whole_and_rev_whole )
{
	BCS_CHECK( verify_indices(rgn(0, whole()), BCS_NULL, 0) );
	index_t w1[5] = {0, 1, 2, 3, 4};
	BCS_CHECK( verify_indices(rgn(5, whole()), w1, 5) );

	BCS_CHECK( verify_indices(rgn(0, rev_whole()), BCS_NULL, 0) );
	index_t rw1[5] = {4, 3, 2, 1, 0};
	BCS_CHECK( verify_indices(rgn(5, rev_whole()), rw1, 5) );
}


BCS_TEST_CASE( test_indices )
{
	bool bs1[6] = {true, false, false, true, false, true};
	std::vector<bool> v1(bs1, bs1+6);
	index_t I1[3] = {0, 3, 5};
	BCS_CHECK( verify_indices( indices(v1), I1, 3 ) );

	using std::placeholders::_1;

	int a2[5] = {1, 3, -2, -4, 5};
	std::vector<int> v2(a2, a2+5);
	index_t I2[3] = {0, 1, 4};
	BCS_CHECK( verify_indices( indices(v2, std::bind(std::greater<int>(), _1, 0)),  I2, 3)  );

	int a3[5] = {5, -4, 3, -2, 1};
	std::vector<int> v3(a3, a3+5);
	index_t I3[2] = {1, 4};
	BCS_CHECK( verify_indices( indices(v2, v3, std::greater<int>()), I3, 2) );
}


test_suite *test_index_selection_suite()
{
	test_suite *suite = new test_suite( "test_index_selection" );

	suite->add( new test_range() );
	suite->add( new test_step_range() );
	suite->add( new test_rep_selector() );
	suite->add( new test_whole_and_rev_whole() );
	suite->add( new test_indices() );

	return suite;
}
