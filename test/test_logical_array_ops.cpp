/**
 * @file test_logical_array_ops.cpp
 *
 * Unit Testing of Logical array operations
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/test_units.h>
#include <bcslib/test/test_array_aux.h>

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>
#include <bcslib/array/logical_array_ops.h>

using namespace bcs;
using namespace bcs::test;


BCS_TEST_CASE( test_basic_logical_ops )
{
	bool src1[] = {false, false, true, true};
	bool src2[] = {false, true, false, true};

	const size_t N = 4;
	const size_t m = 2;
	const size_t n = 2;

	// prepare views

	caview1d<bool> x1 = get_aview1d(src1, N);
	caview2d<bool, row_major_t>    X1_rm = get_aview2d_rm(src1, m, n);
	caview2d<bool, column_major_t> X1_cm = get_aview2d_cm(src1, m, n);

	caview1d<bool> x2 = get_aview1d(src2, N);
	caview2d<bool, row_major_t>    X2_rm = get_aview2d_rm(src2, m, n);
	caview2d<bool, column_major_t> X2_cm = get_aview2d_cm(src2, m, n);

	// not

	bool r_not[N] = {true, true, false, false};
	BCS_CHECK( array_view_equal(!x1, r_not, N) );
	BCS_CHECK( array_view_equal(!X1_rm, r_not, m, n) );
	BCS_CHECK( array_view_equal(!X1_cm, r_not, m, n) );

	// and

	bool r_and[N] = {false, false, false, true};
	BCS_CHECK( array_view_equal(x1 & x2, r_and, N) );
	BCS_CHECK( array_view_equal(X1_rm & X2_rm, r_and, m, n) );
	BCS_CHECK( array_view_equal(X1_cm & X2_cm, r_and, m, n) );

	// or

	bool r_or[N] = {false, true, true, true};
	BCS_CHECK( array_view_equal(x1 | x2, r_or, N) );
	BCS_CHECK( array_view_equal(X1_rm | X2_rm, r_or, m, n) );
	BCS_CHECK( array_view_equal(X1_cm | X2_cm, r_or, m, n) );

	// xor

	bool r_xor[N] = {false, true, true, false};
	BCS_CHECK( array_view_equal(x1 ^ x2, r_xor, N) );
	BCS_CHECK( array_view_equal(X1_rm ^ X2_rm, r_xor, m, n) );
	BCS_CHECK( array_view_equal(X1_cm ^ X2_cm, r_xor, m, n) );
}


BCS_TEST_CASE( test_logical_stats )
{
	bool s1[4] = {false, false, false, false};
	bool s2[4] = {false, false, true,  false};
	bool s3[4] = {true,  false, true,  false};
	bool s4[4] = {true,  true,  true,  true};

	bool t1[9] = {false, false, false, true, true, true, true, false, true};
	bool t2[9] = {false, true, true, false, true, false, false, true, true};

	// prepare views

	caview1d<bool> a1 = get_aview1d(s1, 4);
	caview1d<bool> a2 = get_aview1d(s2, 4);
	caview1d<bool> a3 = get_aview1d(s3, 4);
	caview1d<bool> a4 = get_aview1d(s4, 4);

	caview2d<bool, row_major_t>    b1rm = get_aview2d_rm(t1, 3, 3);
	caview2d<bool, column_major_t> b1cm = get_aview2d_cm(t1, 3, 3);
	caview2d<bool, row_major_t>    b2rm = get_aview2d_rm(t2, 3, 3);
	caview2d<bool, column_major_t> b2cm = get_aview2d_cm(t2, 3, 3);

	// all

	BCS_CHECK_EQUAL( all(a1), false );
	BCS_CHECK_EQUAL( all(a2), false );
	BCS_CHECK_EQUAL( all(a3), false );
	BCS_CHECK_EQUAL( all(a4), true );

	bool b1rm_all_rs[] = {false, true, false};
	bool b1rm_all_cs[] = {false, false, false};
	bool b1cm_all_rs[] = {false, false, false};
	bool b1cm_all_cs[] = {false, true, false};

	bool b2rm_all_rs[] = {false, false, false};
	bool b2rm_all_cs[] = {false, true, false};
	bool b2cm_all_rs[] = {false, true, false};
	bool b2cm_all_cs[] = {false, false, false};

	BCS_CHECK( array_view_equal(all(b1rm, per_row()), b1rm_all_rs, 3) );
	BCS_CHECK( array_view_equal(all(b1rm, per_col()), b1rm_all_cs, 3) );
	BCS_CHECK( array_view_equal(all(b1cm, per_row()), b1cm_all_rs, 3) );
	BCS_CHECK( array_view_equal(all(b1cm, per_col()), b1cm_all_cs, 3) );

	BCS_CHECK( array_view_equal(all(b2rm, per_row()), b2rm_all_rs, 3) );
	BCS_CHECK( array_view_equal(all(b2rm, per_col()), b2rm_all_cs, 3) );
	BCS_CHECK( array_view_equal(all(b2cm, per_row()), b2cm_all_rs, 3) );
	BCS_CHECK( array_view_equal(all(b2cm, per_col()), b2cm_all_cs, 3) );


	// any

	BCS_CHECK_EQUAL( any(a1), false );
	BCS_CHECK_EQUAL( any(a2), true );
	BCS_CHECK_EQUAL( any(a3), true);
	BCS_CHECK_EQUAL( any(a4), true );

	bool b1rm_any_rs[] = {false, true, true};
	bool b1rm_any_cs[] = {true, true, true};
	bool b1cm_any_rs[] = {true, true, true};
	bool b1cm_any_cs[] = {false, true, true};

	bool b2rm_any_rs[] = {true, true, true};
	bool b2rm_any_cs[] = {false, true, true};
	bool b2cm_any_rs[] = {false, true, true};
	bool b2cm_any_cs[] = {true, true, true};

	BCS_CHECK( array_view_equal(any(b1rm, per_row()), b1rm_any_rs, 3) );
	BCS_CHECK( array_view_equal(any(b1rm, per_col()), b1rm_any_cs, 3) );
	BCS_CHECK( array_view_equal(any(b1cm, per_row()), b1cm_any_rs, 3) );
	BCS_CHECK( array_view_equal(any(b1cm, per_col()), b1cm_any_cs, 3) );

	BCS_CHECK( array_view_equal(any(b2rm, per_row()), b2rm_any_rs, 3) );
	BCS_CHECK( array_view_equal(any(b2rm, per_col()), b2rm_any_cs, 3) );
	BCS_CHECK( array_view_equal(any(b2cm, per_row()), b2cm_any_rs, 3) );
	BCS_CHECK( array_view_equal(any(b2cm, per_col()), b2cm_any_cs, 3) );


	// count_true

	BCS_CHECK_EQUAL( count_true(a1, int(0)), 0 );
	BCS_CHECK_EQUAL( count_true(a2, int(0)), 1 );
	BCS_CHECK_EQUAL( count_true(a3, int(0)), 2 );
	BCS_CHECK_EQUAL( count_true(a4, int(0)), 4 );

	int b1rm_count_true_rs[] = {0, 3, 2};
	int b1rm_count_true_cs[] = {2, 1, 2};
	int b1cm_count_true_rs[] = {2, 1, 2};
	int b1cm_count_true_cs[] = {0, 3, 2};

	int b2rm_count_true_rs[] = {2, 1, 2};
	int b2rm_count_true_cs[] = {0, 3, 2};
	int b2cm_count_true_rs[] = {0, 3, 2};
	int b2cm_count_true_cs[] = {2, 1, 2};

	BCS_CHECK( array_view_equal(count_true(b1rm, int(0), per_row()), b1rm_count_true_rs, 3) );
	BCS_CHECK( array_view_equal(count_true(b1rm, int(0), per_col()), b1rm_count_true_cs, 3) );
	BCS_CHECK( array_view_equal(count_true(b1cm, int(0), per_row()), b1cm_count_true_rs, 3) );
	BCS_CHECK( array_view_equal(count_true(b1cm, int(0), per_col()), b1cm_count_true_cs, 3) );

	BCS_CHECK( array_view_equal(count_true(b2rm, int(0), per_row()), b2rm_count_true_rs, 3) );
	BCS_CHECK( array_view_equal(count_true(b2rm, int(0), per_col()), b2rm_count_true_cs, 3) );
	BCS_CHECK( array_view_equal(count_true(b2cm, int(0), per_row()), b2cm_count_true_rs, 3) );
	BCS_CHECK( array_view_equal(count_true(b2cm, int(0), per_col()), b2cm_count_true_cs, 3) );


	// count_false

	BCS_CHECK_EQUAL( count_false(a1, int(0)), 4 );
	BCS_CHECK_EQUAL( count_false(a2, int(0)), 3 );
	BCS_CHECK_EQUAL( count_false(a3, int(0)), 2 );
	BCS_CHECK_EQUAL( count_false(a4, int(0)), 0 );

	int b1rm_count_false_rs[] = {3, 0, 1};
	int b1rm_count_false_cs[] = {1, 2, 1};
	int b1cm_count_false_rs[] = {1, 2, 1};
	int b1cm_count_false_cs[] = {3, 0, 1};

	int b2rm_count_false_rs[] = {1, 2, 1};
	int b2rm_count_false_cs[] = {3, 0, 1};
	int b2cm_count_false_rs[] = {3, 0, 1};
	int b2cm_count_false_cs[] = {1, 2, 1};

	BCS_CHECK( array_view_equal(count_false(b1rm, int(0), per_row()), b1rm_count_false_rs, 3) );
	BCS_CHECK( array_view_equal(count_false(b1rm, int(0), per_col()), b1rm_count_false_cs, 3) );
	BCS_CHECK( array_view_equal(count_false(b1cm, int(0), per_row()), b1cm_count_false_rs, 3) );
	BCS_CHECK( array_view_equal(count_false(b1cm, int(0), per_col()), b1cm_count_false_cs, 3) );

	BCS_CHECK( array_view_equal(count_false(b2rm, int(0), per_row()), b2rm_count_false_rs, 3) );
	BCS_CHECK( array_view_equal(count_false(b2rm, int(0), per_col()), b2rm_count_false_cs, 3) );
	BCS_CHECK( array_view_equal(count_false(b2cm, int(0), per_row()), b2cm_count_false_rs, 3) );
	BCS_CHECK( array_view_equal(count_false(b2cm, int(0), per_col()), b2cm_count_false_cs, 3) );
}


test_suite* test_logical_array_ops_suite()
{
	test_suite *tsuite = new test_suite("test_logical_array_ops");

	tsuite->add( new test_basic_logical_ops() );
	tsuite->add( new test_logical_stats() );

	return tsuite;
}


