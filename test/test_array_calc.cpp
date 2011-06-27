/**
 * @file test_array_ops.cpp
 *
 * The Unit testing for array operators
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/test_units.h>
#include <bcslib/test/test_array_aux.h>

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>
#include <bcslib/array/array_calc.h>

using namespace bcs;
using namespace bcs::test;


BCS_TEST_CASE( test_array_compare )
{
	double src1[] = {1, 2, 3, 4, 5, 6};
	double src2[] = {6, 5, 3, 4, 2, 1};

	const size_t N = 6;
	const size_t m = 2;
	const size_t n = 3;
	double sv = 3;

	// prepare views

	caview1d<double> x1 = get_aview1d(src1, N);
	caview2d<double, row_major_t>    X1_rm = get_aview2d_rm(src1, m, n);
	caview2d<double, column_major_t> X1_cm = get_aview2d_cm(src1, m, n);

	caview1d<double> x2 = get_aview1d(src2, N);
	caview2d<double, row_major_t>    X2_rm = get_aview2d_rm(src2, m, n);
	caview2d<double, column_major_t> X2_cm = get_aview2d_cm(src2, m, n);

	// eq

	bool req_vv[N] = {false, false, true, true, false, false};
	BCS_CHECK( array_view_equal(eq(x1, x2), req_vv, N) );
	BCS_CHECK( array_view_equal(eq(X1_rm, X2_rm), req_vv, m, n));
	BCS_CHECK( array_view_equal(eq(X1_cm, X2_cm), req_vv, m, n));

	bool req_vs[N] = {false, false, true, false, false, false};
	BCS_CHECK( array_view_equal(eq(x1, sv), req_vs, N) );
	BCS_CHECK( array_view_equal(eq(X1_rm, sv), req_vs, m, n));
	BCS_CHECK( array_view_equal(eq(X1_cm, sv), req_vs, m, n));

	// ne

	bool rne_vv[N] = {true, true, false, false, true, true};
	BCS_CHECK( array_view_equal(ne(x1, x2), rne_vv, N) );
	BCS_CHECK( array_view_equal(ne(X1_rm, X2_rm), rne_vv, m, n));
	BCS_CHECK( array_view_equal(ne(X1_cm, X2_cm), rne_vv, m, n));

	bool rne_vs[N] = {true, true, false, true, true, true};
	BCS_CHECK( array_view_equal(ne(x1, sv), rne_vs, N) );
	BCS_CHECK( array_view_equal(ne(X1_rm, sv), rne_vs, m, n));
	BCS_CHECK( array_view_equal(ne(X1_cm, sv), rne_vs, m, n));

	// gt

	bool rgt_vv[N] = {false, false, false, false, true, true};
	BCS_CHECK( array_view_equal(gt(x1, x2), rgt_vv, N) );
	BCS_CHECK( array_view_equal(gt(X1_rm, X2_rm), rgt_vv, m, n));
	BCS_CHECK( array_view_equal(gt(X1_cm, X2_cm), rgt_vv, m, n));

	bool rgt_vs[N] = {false, false, false, true, true, true};
	BCS_CHECK( array_view_equal(gt(x1, sv), rgt_vs, N) );
	BCS_CHECK( array_view_equal(gt(X1_rm, sv), rgt_vs, m, n));
	BCS_CHECK( array_view_equal(gt(X1_cm, sv), rgt_vs, m, n));

	// ge

	bool rge_vv[N] = {false, false, true, true, true, true};
	BCS_CHECK( array_view_equal(ge(x1, x2), rge_vv, N) );
	BCS_CHECK( array_view_equal(ge(X1_rm, X2_rm), rge_vv, m, n));
	BCS_CHECK( array_view_equal(ge(X1_cm, X2_cm), rge_vv, m, n));

	bool rge_vs[N] = {false, false, true, true, true, true};
	BCS_CHECK( array_view_equal(ge(x1, sv), rge_vs, N) );
	BCS_CHECK( array_view_equal(ge(X1_rm, sv), rge_vs, m, n));
	BCS_CHECK( array_view_equal(ge(X1_cm, sv), rge_vs, m, n));

	// lt

	bool rlt_vv[N] = {true, true, false, false, false, false};
	BCS_CHECK( array_view_equal(lt(x1, x2), rlt_vv, N) );
	BCS_CHECK( array_view_equal(lt(X1_rm, X2_rm), rlt_vv, m, n));
	BCS_CHECK( array_view_equal(lt(X1_cm, X2_cm), rlt_vv, m, n));

	bool rlt_vs[N] = {true, true, false, false, false, false};
	BCS_CHECK( array_view_equal(lt(x1, sv), rlt_vs, N) );
	BCS_CHECK( array_view_equal(lt(X1_rm, sv), rlt_vs, m, n));
	BCS_CHECK( array_view_equal(lt(X1_cm, sv), rlt_vs, m, n));

	// le

	bool rle_vv[N] = {true, true, true, true, false, false};
	BCS_CHECK( array_view_equal(le(x1, x2), rle_vv, N) );
	BCS_CHECK( array_view_equal(le(X1_rm, X2_rm), rle_vv, m, n));
	BCS_CHECK( array_view_equal(le(X1_cm, X2_cm), rle_vv, m, n));

	bool rle_vs[N] = {true, true, true, false, false, false};
	BCS_CHECK( array_view_equal(le(x1, sv), rle_vs, N) );
	BCS_CHECK( array_view_equal(le(X1_rm, sv), rle_vs, m, n));
	BCS_CHECK( array_view_equal(le(X1_cm, sv), rle_vs, m, n));
}


BCS_TEST_CASE( test_array_max_min_each )
{
	double src1[] = {1, 2, 3, 4, 5, 6};
	double src2[] = {6, 5, 3, 4, 2, 1};

	const size_t N = 6;
	const size_t m = 2;
	const size_t n = 3;

	// prepare views

	caview1d<double> x1 = get_aview1d(src1, N);
	caview2d<double, row_major_t>    X1_rm = get_aview2d_rm(src1, m, n);
	caview2d<double, column_major_t> X1_cm = get_aview2d_cm(src1, m, n);

	caview1d<double> x2 = get_aview1d(src2, N);
	caview2d<double, row_major_t>    X2_rm = get_aview2d_rm(src2, m, n);
	caview2d<double, column_major_t> X2_cm = get_aview2d_cm(src2, m, n);

	// max_each

	double max_vv[N] = {6, 5, 3, 4, 5, 6};
	BCS_CHECK( array_view_equal(max_each(x1, x2), max_vv, N) );
	BCS_CHECK( array_view_equal(max_each(X1_rm, X2_rm), max_vv, m, n));
	BCS_CHECK( array_view_equal(max_each(X1_cm, X2_cm), max_vv, m, n));

	// min_each

	double min_vv[N] = {1, 2, 3, 4, 2, 1};
	BCS_CHECK( array_view_equal(min_each(x1, x2), min_vv, N) );
	BCS_CHECK( array_view_equal(min_each(X1_rm, X2_rm), min_vv, m, n));
	BCS_CHECK( array_view_equal(min_each(X1_cm, X2_cm), min_vv, m, n));
}


BCS_TEST_CASE( test_bounding )
{
	double src1[] = {1, 2, 3, 4, 5, 6};

	const size_t N = 6;
	const size_t m = 2;
	const size_t n = 3;

	caview1d<double> x1 = get_aview1d(src1, N);
	caview2d<double, row_major_t>    X1_rm = get_aview2d_rm(src1, m, n);
	caview2d<double, column_major_t> X1_cm = get_aview2d_cm(src1, m, n);

	array1d<double> y1(N);
	array2d<double, row_major_t> Y1_rm(m, n);
	array2d<double, column_major_t> Y1_cm(m, n);

	// lbound

	double lb_r[N] = {3, 3, 3, 4, 5, 6};
	BCS_CHECK( array_view_equal(lbound(x1, 3.0), lb_r, N) );
	BCS_CHECK( array_view_equal(lbound(X1_rm, 3.0), lb_r, m, n));
	BCS_CHECK( array_view_equal(lbound(X1_cm, 3.0), lb_r, m, n));

	y1 << x1; Y1_rm << X1_rm; Y1_cm << X1_cm;
	lbound_ip(y1, 3.0);
	lbound_ip(Y1_rm, 3.0);
	lbound_ip(Y1_cm, 3.0);
	BCS_CHECK( array_view_equal(y1, lb_r, N) );
	BCS_CHECK( array_view_equal(Y1_rm, lb_r, m, n));
	BCS_CHECK( array_view_equal(Y1_cm, lb_r, m, n));

	// ubound

	double ub_r[N] = {1, 2, 3, 4, 4, 4};
	BCS_CHECK( array_view_equal(ubound(x1, 4.0), ub_r, N) );
	BCS_CHECK( array_view_equal(ubound(X1_rm, 4.0), ub_r, m, n));
	BCS_CHECK( array_view_equal(ubound(X1_cm, 4.0), ub_r, m, n));

	y1 << x1; Y1_rm << X1_rm; Y1_cm << X1_cm;
	ubound_ip(y1, 4.0);
	ubound_ip(Y1_rm, 4.0);
	ubound_ip(Y1_cm, 4.0);
	BCS_CHECK( array_view_equal(y1, ub_r, N) );
	BCS_CHECK( array_view_equal(Y1_rm, ub_r, m, n));
	BCS_CHECK( array_view_equal(Y1_cm, ub_r, m, n));

	// rgn_bound

	double rb_r[N] = {2, 2, 3, 4, 5, 5};
	BCS_CHECK( array_view_equal(rgn_bound(x1, 2.0, 5.0), rb_r, N) );
	BCS_CHECK( array_view_equal(rgn_bound(X1_rm, 2.0, 5.0), rb_r, m, n));
	BCS_CHECK( array_view_equal(rgn_bound(X1_cm, 2.0, 5.0), rb_r, m, n));

	y1 << x1; Y1_rm << X1_rm; Y1_cm << X1_cm;
	rgn_bound_ip(y1, 2.0, 5.0);
	rgn_bound_ip(Y1_rm, 2.0, 5.0);
	rgn_bound_ip(Y1_cm, 2.0, 5.0);
	BCS_CHECK( array_view_equal(y1, rb_r, N) );
	BCS_CHECK( array_view_equal(Y1_rm, rb_r, m, n));
	BCS_CHECK( array_view_equal(Y1_cm, rb_r, m, n));

	// abound

	double ab_r[N] = {-1.5, -1.0, 0.0, 1.0, 1.5, 1.5};
	BCS_CHECK( array_view_equal(abound(x1-3.0, 1.5), ab_r, N) );
	BCS_CHECK( array_view_equal(abound(X1_rm-3.0, 1.5), ab_r, m, n));
	BCS_CHECK( array_view_equal(abound(X1_cm-3.0, 1.5), ab_r, m, n));

	y1 << (x1 - 3.0); Y1_rm << (X1_rm - 3.0); Y1_cm << (X1_cm - 3.0);
	abound_ip(y1, 1.5);
	abound_ip(Y1_rm, 1.5);
	abound_ip(Y1_cm, 1.5);
	BCS_CHECK( array_view_equal(y1, ab_r, N) );
	BCS_CHECK( array_view_equal(Y1_rm, ab_r, m, n));
	BCS_CHECK( array_view_equal(Y1_cm, ab_r, m, n));

}


BCS_TEST_CASE( test_array_add )
{

	double src1[] = {1, 3, 5, 9, 2, 7, 8, 5, 1};
	double src2[] = {5, 6, 2, 4, 8, 1, 9, 0, 3};

	// 1D

	aview1d<double> x1(src1, 5);
	aview1d<double> x2(src2, 5);
	aview1d<double, step_ind> xs1(src1, step_ind(5, 2));
	aview1d<double, step_ind> xs2(src2, step_ind(5, 2));

	double ra1[] = {6, 9, 7, 13, 10};
	BCS_CHECK( array_view_equal(x1 + x2, ra1, 5) );


	double ra2[] = {6, 5, 13, 18, 5};
	BCS_CHECK( array_view_equal(x1 + xs2, ra2, 5) );

	double ra3[] = {6, 11, 4, 12, 9};
	BCS_CHECK( array_view_equal(xs1 + x2, ra3, 5) );

	double ra4[] = {6, 7, 10, 17, 4};
	BCS_CHECK( array_view_equal(xs1 + xs2, ra4, 5) );

	double ra5[] = {3, 5, 7, 11, 4};
	BCS_CHECK( array_view_equal(x1 + 2.0, ra5, 5) );
	BCS_CHECK( array_view_equal(2.0 + x1, ra5, 5) );

	double ra6[] = {3, 7, 4, 10, 3};
	BCS_CHECK( array_view_equal(xs1 + 2.0, ra6, 5) );
	BCS_CHECK( array_view_equal(2.0 + xs1, ra6, 5) );

	array1d<double> y1(5);
	double y2_buf[10];
	aview1d<double, step_ind> y2(y2_buf, step_ind(5, 2));

	y1 << x1;
	y1 += x2;
	BCS_CHECK( array_view_equal(y1, ra1, 5) );

	y1 << x1;
	y1 += xs2;
	BCS_CHECK( array_view_equal(y1, ra2, 5) );

	y1 << x1;
	y1 += 2.0;
	BCS_CHECK( array_view_equal(y1, ra5, 5) );

	y2 << x1;
	y2 += x2;
	BCS_CHECK( array_view_equal(y2, ra1, 5) );

	y2 << x1;
	y2 += xs2;
	BCS_CHECK( array_view_equal(y2, ra2, 5) );

	y2 << x1;
	y2 += 2.0;
	BCS_CHECK( array_view_equal(y2, ra5, 5) );


	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);
	aview2d<double, row_major_t, step_ind, id_ind> X1s(src1, 3, 3, step_ind(2, 2), id_ind(3));
	aview2d<double, row_major_t, step_ind, id_ind> X2s(src2, 3, 3, step_ind(2, 2), id_ind(3));

	double rb1[] = {6, 9, 7, 13, 10, 8};
	BCS_CHECK( array_view_equal(X1 + X2, rb1, 2, 3) );

	double rb2[] = {6, 9, 7, 18, 2, 10};
	BCS_CHECK( array_view_equal(X1 + X2s, rb2, 2, 3) );

	double rb3[] = {6, 9, 7, 12, 13, 2};
	BCS_CHECK( array_view_equal(X1s + X2, rb3, 2, 3) );

	double rb4[] = {6, 9, 7, 17, 5, 4};
	BCS_CHECK( array_view_equal(X1s + X2s, rb4, 2, 3) );

	double rb5[] = {3, 5, 7, 11, 4, 9};
	BCS_CHECK( array_view_equal(X1 + 2.0, rb5, 2, 3) );
	BCS_CHECK( array_view_equal(2.0 + X1, rb5, 2, 3) );

	double rb6[] = {3, 5, 7, 10, 7, 3};
	BCS_CHECK( array_view_equal(X1s + 2.0, rb6, 2, 3) );
	BCS_CHECK( array_view_equal(2.0 + X1s, rb6, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3);
	double Y2_buf[20];
	aview2d<double, row_major_t, step_ind, id_ind> Y2(Y2_buf, 3, 3, step_ind(2, 2), id_ind(3));

	Y1 << X1;
	Y1 += X2;
	BCS_CHECK( array_view_equal(Y1, rb1, 2, 3) );

	Y1 << X1;
	Y1 += X2s;
	BCS_CHECK( array_view_equal(Y1, rb2, 2, 3) );

	Y1 << X1;
	Y1 += 2.0;
	BCS_CHECK( array_view_equal(Y1, rb5, 2, 3) );

	Y2 << X1;
	Y2 += X2;
	BCS_CHECK( array_view_equal(Y2, rb1, 2, 3) );

	Y2 << X1;
	Y2 += X2s;
	BCS_CHECK( array_view_equal(Y2, rb2, 2, 3) );

	Y2 << X1;
	Y2 += 2.0;
	BCS_CHECK( array_view_equal(Y2, rb5, 2, 3) );

}



BCS_TEST_CASE( test_array_sub )
{
	double src1[] = {1, 3, 5, 9, 2, 7, 8, 5, 1};
	double src2[] = {5, 6, 2, 4, 8, 1, 9, 0, 3};

	// 1D

	array1d<double> x1(5, src1);
	array1d<double> x2(5, src2);
	aview1d<double, step_ind> xs1(src1, step_ind(5, 2));
	aview1d<double, step_ind> xs2(src2, step_ind(5, 2));

	double ra1[] = {-4, -3, 3, 5, -6};
	BCS_CHECK( array_view_equal(x1 - x2, ra1, 5) );

	double ra2[] = {-4, 1, -3, 0, -1};
	BCS_CHECK( array_view_equal(x1 - xs2, ra2, 5) );

	double ra3[] = {-4, -1, 0, 4, -7};
	BCS_CHECK( array_view_equal(xs1 - x2, ra3, 5) );

	double ra4[] = {-4, 3, -6, -1, -2};
	BCS_CHECK( array_view_equal(xs1 - xs2, ra4, 5) );

	double ra5l[] = {-1, 1, 3, 7, 0};
	double ra5r[] = {1, -1, -3, -7, 0};
	BCS_CHECK( array_view_equal(x1 - 2.0, ra5l, 5) );
	BCS_CHECK( array_view_equal(2.0 - x1, ra5r, 5) );

	double ra6l[] = {-1, 3, 0, 6, -1};
	double ra6r[] = {1, -3, 0, -6, 1};
	BCS_CHECK( array_view_equal(xs1 - 2.0, ra6l, 5) );
	BCS_CHECK( array_view_equal(2.0 - xs1, ra6r, 5) );

	array1d<double> y1(5);
	double y2_buf[10];
	aview1d<double, step_ind> y2(y2_buf, step_ind(5, 2));

	y1 << x1;
	y1 -= x2;
	BCS_CHECK( array_view_equal(y1, ra1, 5) );

	y1 << x1;
	y1 -= xs2;
	BCS_CHECK( array_view_equal(y1, ra2, 5) );

	y1 << x1;
	y1 -= 2.0;
	BCS_CHECK( array_view_equal(y1, ra5l, 5) );

	y2 << x1;
	y2 -= x2;
	BCS_CHECK( array_view_equal(y2, ra1, 5) );

	y2 << x1;
	y2 -= xs2;
	BCS_CHECK( array_view_equal(y2, ra2, 5) );

	y2 << x1;
	y2 -= 2.0;
	BCS_CHECK( array_view_equal(y2, ra5l, 5) );


	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);
	aview2d<double, row_major_t, step_ind, id_ind> X1s(src1, 3, 3, step_ind(2, 2), id_ind(3));
	aview2d<double, row_major_t, step_ind, id_ind> X2s(src2, 3, 3, step_ind(2, 2), id_ind(3));

	double rb1[] = {-4, -3, 3, 5, -6, 6};
	BCS_CHECK( array_view_equal(X1 - X2, rb1, 2, 3) );

	double rb2[] = {-4, -3, 3, 0, 2, 4};
	BCS_CHECK( array_view_equal(X1 - X2s, rb2, 2, 3) );

	double rb3[] = {-4, -3, 3, 4, -3, 0};
	BCS_CHECK( array_view_equal(X1s - X2, rb3, 2, 3) );

	double rb4[] = {-4, -3, 3, -1, 5, -2};
	BCS_CHECK( array_view_equal(X1s - X2s, rb4, 2, 3) );

	double rb5l[] = {-1, 1, 3, 7, 0, 5};
	double rb5r[] = {1, -1, -3, -7, 0, -5};
	BCS_CHECK( array_view_equal(X1 - 2.0, rb5l, 2, 3) );
	BCS_CHECK( array_view_equal(2.0 - X1, rb5r, 2, 3) );

	double rb6l[] = {-1, 1, 3, 6, 3, -1};
	double rb6r[] = {1, -1, -3, -6, -3, 1};
	BCS_CHECK( array_view_equal(X1s - 2.0, rb6l, 2, 3) );
	BCS_CHECK( array_view_equal(2.0 - X1s, rb6r, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3);
	double Y2_buf[20];
	aview2d<double, row_major_t, step_ind, id_ind> Y2(Y2_buf, 3, 3, step_ind(2, 2), id_ind(3));

	Y1 << X1;
	Y1 -= X2;
	BCS_CHECK( array_view_equal(Y1, rb1, 2, 3) );

	Y1 << X1;
	Y1 -= X2s;
	BCS_CHECK( array_view_equal(Y1, rb2, 2, 3) );

	Y1 << X1;
	Y1 -= 2.0;
	BCS_CHECK( array_view_equal(Y1, rb5l, 2, 3) );


	Y2 << X1;
	Y2 -= X2;
	BCS_CHECK( array_view_equal(Y2, rb1, 2, 3) );

	Y2 << X1;
	Y2 -= X2s;
	BCS_CHECK( array_view_equal(Y2, rb2, 2, 3) );

	Y2 << X1;
	Y2 -= 2.0;
	BCS_CHECK( array_view_equal(Y2, rb5l, 2, 3) );

}


BCS_TEST_CASE( test_array_mul )
{
	double src1[] = {1, 3, 5, 9, 2, 7, 8, 5, 1};
	double src2[] = {5, 6, 2, 4, 8, 1, 9, 0, 3};

	// 1D

	array1d<double> x1(5, src1);
	array1d<double> x2(5, src2);
	aview1d<double, step_ind> xs1(src1, step_ind(5, 2));
	aview1d<double, step_ind> xs2(src2, step_ind(5, 2));

	double ra1[] = {5, 18, 10, 36, 16};
	BCS_CHECK( array_view_equal(x1 * x2, ra1, 5) );

	double ra2[] = {5, 6, 40, 81, 6};
	BCS_CHECK( array_view_equal(x1 * xs2, ra2, 5) );

	double ra3[] = {5, 30, 4, 32, 8};
	BCS_CHECK( array_view_equal(xs1 * x2, ra3, 5) );

	double ra4[] = {5, 10, 16, 72, 3};
	BCS_CHECK( array_view_equal(xs1 * xs2, ra4, 5) );

	double ra5[] = {2, 6, 10, 18, 4};
	BCS_CHECK( array_view_equal(x1 * 2.0, ra5, 5) );
	BCS_CHECK( array_view_equal(2.0 * x1, ra5, 5) );

	double ra6[] = {2, 10, 4, 16, 2};
	BCS_CHECK( array_view_equal(xs1 * 2.0, ra6, 5) );
	BCS_CHECK( array_view_equal(2.0 * xs1, ra6, 5) );

	array1d<double> y1(5);
	double y2_buf[10];
	aview1d<double, step_ind> y2(y2_buf, step_ind(5, 2));

	y1 << x1;
	y1 *= x2;
	BCS_CHECK( array_view_equal(y1, ra1, 5) );

	y1 << x1;
	y1 *= xs2;
	BCS_CHECK( array_view_equal(y1, ra2, 5) );

	y1 << x1;
	y1 *= 2.0;
	BCS_CHECK( array_view_equal(y1, ra5, 5) );

	y2 << x1;
	y2 *= x2;
	BCS_CHECK( array_view_equal(y2, ra1, 5) );

	y2 << x1;
	y2 *= xs2;
	BCS_CHECK( array_view_equal(y2, ra2, 5) );

	y2 << x1;
	y2 *= 2.0;
	BCS_CHECK( array_view_equal(y2, ra5, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);
	aview2d<double, row_major_t, step_ind, id_ind> X1s(src1, 3, 3, step_ind(2, 2), id_ind(3));
	aview2d<double, row_major_t, step_ind, id_ind> X2s(src2, 3, 3, step_ind(2, 2), id_ind(3));

	double rb1[] = {5, 18, 10, 36, 16, 7};
	BCS_CHECK( array_view_equal(X1 * X2, rb1, 2, 3) );

	double rb2[] = {5, 18, 10, 81, 0, 21};
	BCS_CHECK( array_view_equal(X1 * X2s, rb2, 2, 3) );

	double rb3[] = {5, 18, 10, 32, 40, 1};
	BCS_CHECK( array_view_equal(X1s * X2, rb3, 2, 3) );

	double rb4[] = {5, 18, 10, 72, 0, 3};
	BCS_CHECK( array_view_equal(X1s * X2s, rb4, 2, 3) );

	double rb5[] = {2, 6, 10, 18, 4, 14};
	BCS_CHECK( array_view_equal(X1 * 2.0, rb5, 2, 3) );
	BCS_CHECK( array_view_equal(2.0 * X1, rb5, 2, 3) );

	double rb6[] = {2, 6, 10, 16, 10, 2};
	BCS_CHECK( array_view_equal(X1s * 2.0, rb6, 2, 3) );
	BCS_CHECK( array_view_equal(2.0 * X1s, rb6, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3);
	double Y2_buf[20];
	aview2d<double, row_major_t, step_ind, id_ind> Y2(Y2_buf, 3, 3, step_ind(2, 2), id_ind(3));

	Y1 << X1;
	Y1 *= X2;
	BCS_CHECK( array_view_equal(Y1, rb1, 2, 3) );

	Y1 << X1;
	Y1 *= X2s;
	BCS_CHECK( array_view_equal(Y1, rb2, 2, 3) );

	Y1 << X1;
	Y1 *= 2.0;
	BCS_CHECK( array_view_equal(Y1, rb5, 2, 3) );

	Y2 << X1;
	Y2 *= X2;
	BCS_CHECK( array_view_equal(Y2, rb1, 2, 3) );

	Y2 << X1;
	Y2 *= X2s;
	BCS_CHECK( array_view_equal(Y2, rb2, 2, 3) );

	Y2 << X1;
	Y2 *= 2.0;
	BCS_CHECK( array_view_equal(Y2, rb5, 2, 3) );
}


BCS_TEST_CASE( test_array_div )
{
	double src1[] = {1, 3, 5, 9, 2, 7, 8, 5, 1};
	double src2[] = {5, 6, 2, 4, 8, 1, 9, 10, 3};

	// 1D

	array1d<double> x1(5, src1);
	array1d<double> x2(5, src2);
	aview1d<double, step_ind> xs1(src1, step_ind(5, 2));
	aview1d<double, step_ind> xs2(src2, step_ind(5, 2));

	double ra1[] = {1.0/5, 3.0/6, 5.0/2, 9.0/4, 2.0/8};
	BCS_CHECK( array_view_approx(x1 / x2, ra1, 5) );

	double ra2[] = {1.0/5, 3.0/2, 5.0/8, 9.0/9, 2.0/3};
	BCS_CHECK( array_view_approx(x1 / xs2, ra2, 5) );

	double ra3[] = {1.0/5, 5.0/6, 2.0/2, 8.0/4, 1.0/8};
	BCS_CHECK( array_view_approx(xs1 / x2, ra3, 5) );

	double ra4[] = {1.0/5, 5.0/2, 2.0/8, 8.0/9, 1.0/3};
	BCS_CHECK( array_view_approx(xs1 / xs2, ra4, 5) );

	double ra5l[] = {1.0/2, 3.0/2, 5.0/2, 9.0/2, 2.0/2};
	double ra5r[] = {2.0/1, 2.0/3, 2.0/5, 2.0/9, 2.0/2};
	BCS_CHECK( array_view_approx(x1 / 2.0, ra5l, 5) );
	BCS_CHECK( array_view_approx(2.0 / x1, ra5r, 5) );

	double ra6l[] = {1.0/2, 5.0/2, 2.0/2, 8.0/2, 1.0/2};
	double ra6r[] = {2.0/1, 2.0/5, 2.0/2, 2.0/8, 2.0/1};
	BCS_CHECK( array_view_approx(xs1 / 2.0, ra6l, 5) );
	BCS_CHECK( array_view_approx(2.0 / xs1, ra6r, 5) );

	array1d<double> y1(5);
	double y2_buf[10];
	aview1d<double, step_ind> y2(y2_buf, step_ind(5, 2));

	y1 << x1;
	y1 /= x2;
	BCS_CHECK( array_view_approx(y1, ra1, 5) );

	y1 << x1;
	y1 /= xs2;
	BCS_CHECK( array_view_approx(y1, ra2, 5) );

	y1 << x1;
	y1 /= 2.0;
	BCS_CHECK( array_view_approx(y1, ra5l, 5) );

	y2 << x1;
	y2 /= x2;
	BCS_CHECK( array_view_approx(y2, ra1, 5) );

	y2 << x1;
	y2 /= xs2;
	BCS_CHECK( array_view_approx(y2, ra2, 5) );

	y2 << x1;
	y2 /= 2.0;
	BCS_CHECK( array_view_approx(y2, ra5l, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);
	aview2d<double, row_major_t, step_ind, id_ind> X1s(src1, 3, 3, step_ind(2, 2), id_ind(3));
	aview2d<double, row_major_t, step_ind, id_ind> X2s(src2, 3, 3, step_ind(2, 2), id_ind(3));

	double rb1[] = {1.0/5, 3.0/6, 5.0/2, 9.0/4, 2.0/8, 7.0/1};
	BCS_CHECK( array_view_approx(X1 / X2, rb1, 2, 3) );

	double rb2[] = {1.0/5, 3.0/6, 5.0/2, 9.0/9, 2.0/10, 7.0/3};
	BCS_CHECK( array_view_approx(X1 / X2s, rb2, 2, 3) );

	double rb3[] = {1.0/5, 3.0/6, 5.0/2, 8.0/4, 5.0/8, 1.0/1};
	BCS_CHECK( array_view_approx(X1s / X2, rb3, 2, 3) );

	double rb4[] = {1.0/5, 3.0/6, 5.0/2, 8.0/9, 5.0/10, 1.0/3};
	BCS_CHECK( array_view_approx(X1s / X2s, rb4, 2, 3) );

	double rb5l[] = {1.0/2, 3.0/2, 5.0/2, 9.0/2, 2.0/2, 7.0/2};
	double rb5r[] = {2.0/1, 2.0/3, 2.0/5, 2.0/9, 2.0/2, 2.0/7};
	BCS_CHECK( array_view_approx(X1 / 2.0, rb5l, 2, 3) );
	BCS_CHECK( array_view_approx(2.0 / X1, rb5r, 2, 3) );

	double rb6l[] = {1.0/2, 3.0/2, 5.0/2, 8.0/2, 5.0/2, 1.0/2};
	double rb6r[] = {2.0/1, 2.0/3, 2.0/5, 2.0/8, 2.0/5, 2.0/1};
	BCS_CHECK( array_view_approx(X1s / 2.0, rb6l, 2, 3) );
	BCS_CHECK( array_view_approx(2.0 / X1s, rb6r, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3);
	double Y2_buf[20];
	aview2d<double, row_major_t, step_ind, id_ind> Y2(Y2_buf, 3, 3, step_ind(2, 2), id_ind(3));

	Y1 << X1;
	Y1 /= X2;
	BCS_CHECK( array_view_approx(Y1, rb1, 2, 3) );

	Y1 << X1;
	Y1 /= X2s;
	BCS_CHECK( array_view_approx(Y1, rb2, 2, 3) );

	Y1 << X1;
	Y1 /= 2.0;
	BCS_CHECK( array_view_approx(Y1, rb5l, 2, 3) );

	Y2 << X1;
	Y2 /= X2;
	BCS_CHECK( array_view_approx(Y2, rb1, 2, 3) );

	Y2 << X1;
	Y2 /= X2s;
	BCS_CHECK( array_view_approx(Y2, rb2, 2, 3) );

	Y2 << X1;
	Y2 /= 2.0;
	BCS_CHECK( array_view_approx(Y2, rb5l, 2, 3) );

}


BCS_TEST_CASE( test_array_neg )
{
	double src1[] = {1, 3, 5, 9, 2, 7, 8, 5, 1};

	// 1D

	array1d<double> x1(5, src1);
	aview1d<double, step_ind> xs1(src1, step_ind(5, 2));

	double ra1[] = {-1, -3, -5, -9, -2};
	BCS_CHECK( array_view_equal(-x1, ra1, 5) );
	double ra2[] = {-1, -5, -2, -8, -1};
	BCS_CHECK( array_view_equal(-xs1, ra2, 5) );

	array1d<double> y1(5, src1);
	neg_ip(y1);
	BCS_CHECK( array_view_equal(y1, ra1, 5) );

	double y2_buf[10];
	aview1d<double, step_ind> y2(y2_buf, step_ind(5, 2));
	y2 << x1;
	neg_ip(y2);
	BCS_CHECK( array_view_equal(y2, ra1, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	aview2d<double, row_major_t, step_ind, id_ind> Xs1(src1, 3, 3, step_ind(2, 2), id_ind(3));

	double rb1[] = {-1, -3, -5, -9, -2, -7};
	BCS_CHECK( array_view_equal(-X1, rb1, 2, 3) );
	double rb2[] = {-1, -3, -5, -8, -5, -1};
	BCS_CHECK( array_view_equal(-Xs1, rb2, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3, src1);
	neg_ip(Y1);
	BCS_CHECK( array_view_equal(Y1, rb1, 2, 3) );

	double Y2_buf[10];
	aview2d<double, row_major_t, step_ind, id_ind> Y2(Y2_buf, 3, 3, step_ind(2, 2), id_ind(3));
	Y2 << X1;
	neg_ip(Y2);
	BCS_CHECK( array_view_equal(Y2, rb1, 2, 3) );

}


BCS_TEST_CASE( test_array_abs )
{
	const int N = 6;
	double src[N] = {1, -2, -3, 4, -5, 6};
	double res[N];
	for (int i = 0; i < N; ++i) res[i] = std::abs(src[i]);

	array1d<double> x(6, src);
	array1d<double> xc = clone_array(x);
	BCS_CHECK( array_view_equal( abs(x), res, 6 ) );
	abs_ip(xc);
	BCS_CHECK( array_view_equal( xc, res, 6 ) );

	double res_s[3] = {1, 3, 5};
	aview1d<double, step_ind> xs(src, step_ind(3, 2));
	array1d<double> xsc = clone_array(xs);
	BCS_CHECK( array_view_equal( abs(xs), res_s, 3) );
	abs_ip(xsc);
	BCS_CHECK( array_view_equal( xsc, res_s, 3) );

	array2d<double, row_major_t> Xr(2, 3, src);
	array2d<double, row_major_t> Xrc = clone_array(Xr);
	BCS_CHECK( array_view_equal( abs(Xr), res, 2, 3) );
	abs_ip(Xrc);
	BCS_CHECK( array_view_equal( Xrc, res, 2, 3) );

	array2d<double, column_major_t> Xc(2, 3, src);
	array2d<double, column_major_t> Xcc = clone_array(Xc);
	BCS_CHECK( array_view_equal( abs(Xc), res, 2, 3) );
	abs_ip(Xcc);
	BCS_CHECK( array_view_equal( Xcc, res, 2, 3) );
}


BCS_TEST_CASE( test_array_power_and_root_funs )
{
	const int N = 6;
	double src[N] = {1, 2, 3, 4, 5, 6};
	double es[N] = {1.3, 2.4, 1.2, 4.1, 2.7, 2.5};
	double res[N];

	const int nr = 2;
	const int nc = 3;

	array1d<double> x(N, src);
	array2d<double, row_major_t> Xr(nr, nc, src);
	array2d<double, column_major_t> Xc(nr, nc, src);

	array1d<double> e(N, es);
	array2d<double, row_major_t> Er(nr, nc, es);
	array2d<double, column_major_t> Ec(nr, nc, es);

	// sqr

	for (int i = 0; i < N; ++i) res[i] = src[i] * src[i];

	BCS_CHECK( array_view_equal( sqr(x), res, N ) );
	BCS_CHECK( array_view_equal( sqr(Xr), res, nr, nc) );
	BCS_CHECK( array_view_equal( sqr(Xc), res, nr, nc) );


	// sqrt

	for (int i = 0; i < N; ++i) res[i] = std::sqrt(src[i]);

	BCS_CHECK( array_view_approx( sqrt(x), res, N ) );
	BCS_CHECK( array_view_approx( sqrt(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( sqrt(Xc), res, nr, nc) );

	// rcp

	for (int i = 0; i < N; ++i) res[i] = 1.0 / src[i];

	BCS_CHECK( array_view_approx( rcp(x), res, N ) );
	BCS_CHECK( array_view_approx( rcp(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( rcp(Xc), res, nr, nc) );

	// rsqrt

	for (int i = 0; i < N; ++i) res[i] = 1.0 / std::sqrt(src[i]);

	BCS_CHECK( array_view_approx( rsqrt(x), res, N ) );
	BCS_CHECK( array_view_approx( rsqrt(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( rsqrt(Xc), res, nr, nc) );

	// pow

	for (int i = 0; i < N; ++i) res[i] = std::pow(src[i], es[i]);

	BCS_CHECK( array_view_approx( pow(x, e), res, N ) );
	BCS_CHECK( array_view_approx( pow(Xr, Er), res, nr, nc) );
	BCS_CHECK( array_view_approx( pow(Xc, Ec), res, nr, nc) );

	// pow with constant exponent

	double q = 2.3;

	for (int i = 0; i < N; ++i) res[i] = std::pow(src[i], q);

	BCS_CHECK( array_view_approx( pow(x, q), res, N ) );
	BCS_CHECK( array_view_approx( pow(Xr, q), res, nr, nc) );
	BCS_CHECK( array_view_approx( pow(Xc, q), res, nr, nc) );
}


BCS_TEST_CASE( test_array_exp_and_log_funs )
{
	const int N = 6;
	double src[N] = {1, 2, 3, 4, 5, 6};
	double res[N];

	const int nr = 2;
	const int nc = 3;

	array1d<double> x(N, src);
	array2d<double, row_major_t> Xr(nr, nc, src);
	array2d<double, column_major_t> Xc(nr, nc, src);

	// exp

	for (int i = 0; i < N; ++i) res[i] = std::exp(src[i]);

	BCS_CHECK( array_view_approx( exp(x), res, N ) );
	BCS_CHECK( array_view_approx( exp(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( exp(Xc), res, nr, nc) );

	// log

	for (int i = 0; i < N; ++i) res[i] = std::log(src[i]);

	BCS_CHECK( array_view_approx( log(x), res, N ) );
	BCS_CHECK( array_view_approx( log(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( log(Xc), res, nr, nc) );

	// log10

	for (int i = 0; i < N; ++i) res[i] = std::log10(src[i]);

	BCS_CHECK( array_view_approx( log10(x), res, N ) );
	BCS_CHECK( array_view_approx( log10(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( log10(Xc), res, nr, nc) );
}


BCS_TEST_CASE( test_array_floor_and_ceil )
{
	const int N = 6;
	double src[N] = {1.8, 2.1, -3.4, 4.0, -5.8, 6.0};
	double res[N];

	const int nr = 2;
	const int nc = 3;

	array1d<double> x(N, src);
	array2d<double, row_major_t> Xr(nr, nc, src);
	array2d<double, column_major_t> Xc(nr, nc, src);

	// floor

	for (int i = 0; i < N; ++i) res[i] = std::floor(src[i]);

	BCS_CHECK( array_view_equal( floor(x), res, N ) );
	BCS_CHECK( array_view_equal( floor(Xr), res, nr, nc) );
	BCS_CHECK( array_view_equal( floor(Xc), res, nr, nc) );

	// ceil

	for (int i = 0; i < N; ++i) res[i] = std::ceil(src[i]);

	BCS_CHECK( array_view_equal( ceil(x), res, N ) );
	BCS_CHECK( array_view_equal( ceil(Xr), res, nr, nc) );
	BCS_CHECK( array_view_equal( ceil(Xc), res, nr, nc) );

	// round

	for (int i = 0; i < N; ++i) res[i] = bcs::round(src[i]);

	BCS_CHECK( array_view_equal( round(x), res, N ) );
	BCS_CHECK( array_view_equal( round(Xr), res, nr, nc) );
	BCS_CHECK( array_view_equal( round(Xc), res, nr, nc) );
}


BCS_TEST_CASE( test_array_trigonometric_funs )
{
	const int N = 6;
	double src[N] = {-0.9, -0.5, -0.2, 0.1, 0.4, 0.8};
	double src_e[N] = {0.8, 0.5, 0.3, -0.2, -0.5, -0.6};
	double res[N];

	const int nr = 2;
	const int nc = 3;

	array1d<double> x(N, src);
	array2d<double, row_major_t> Xr(nr, nc, src);
	array2d<double, column_major_t> Xc(nr, nc, src);

	array1d<double> e(N, src_e);
	array2d<double, row_major_t> Er(nr, nc, src_e);
	array2d<double, column_major_t> Ec(nr, nc, src_e);


	// sin

	for (int i = 0; i < N; ++i) res[i] = std::sin(src[i]);

	BCS_CHECK( array_view_approx( sin(x), res, N ) );
	BCS_CHECK( array_view_approx( sin(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( sin(Xc), res, nr, nc) );

	// cos

	for (int i = 0; i < N; ++i) res[i] = std::cos(src[i]);

	BCS_CHECK( array_view_approx( cos(x), res, N ) );
	BCS_CHECK( array_view_approx( cos(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( cos(Xc), res, nr, nc) );

	// tan

	for (int i = 0; i < N; ++i) res[i] = std::tan(src[i]);

	BCS_CHECK( array_view_approx( tan(x), res, N ) );
	BCS_CHECK( array_view_approx( tan(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( tan(Xc), res, nr, nc) );

	// asin

	for (int i = 0; i < N; ++i) res[i] = std::asin(src[i]);

	BCS_CHECK( array_view_approx( asin(x), res, N ) );
	BCS_CHECK( array_view_approx( asin(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( asin(Xc), res, nr, nc) );

	// acos

	for (int i = 0; i < N; ++i) res[i] = std::acos(src[i]);

	BCS_CHECK( array_view_approx( acos(x), res, N ) );
	BCS_CHECK( array_view_approx( acos(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( acos(Xc), res, nr, nc) );

	// atan

	for (int i = 0; i < N; ++i) res[i] = std::atan(src[i]);

	BCS_CHECK( array_view_approx( atan(x), res, N ) );
	BCS_CHECK( array_view_approx( atan(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( atan(Xc), res, nr, nc) );

	// atan2

	for (int i = 0; i < N; ++i) res[i] = std::atan2(src[i], src_e[i]);

	BCS_CHECK( array_view_approx( atan2(x, e), res, N ) );
	BCS_CHECK( array_view_approx( atan2(Xr, Er), res, nr, nc) );
	BCS_CHECK( array_view_approx( atan2(Xc, Ec), res, nr, nc) );

}


BCS_TEST_CASE( test_array_hyperbolic_funs )
{
	const int N = 6;
	double src[N] = {-0.9, -0.5, -0.2, 0.1, 0.4, 0.8};
	double res[N];

	const int nr = 2;
	const int nc = 3;

	array1d<double> x(N, src);
	array2d<double, row_major_t> Xr(nr, nc, src);
	array2d<double, column_major_t> Xc(nr, nc, src);

	// sinh

	for (int i = 0; i < N; ++i) res[i] = std::sinh(src[i]);

	BCS_CHECK( array_view_approx( sinh(x), res, N ) );
	BCS_CHECK( array_view_approx( sinh(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( sinh(Xc), res, nr, nc) );

	// cosh

	for (int i = 0; i < N; ++i) res[i] = std::cosh(src[i]);

	BCS_CHECK( array_view_approx( cosh(x), res, N ) );
	BCS_CHECK( array_view_approx( cosh(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( cosh(Xc), res, nr, nc) );

	// tanh

	for (int i = 0; i < N; ++i) res[i] = std::tanh(src[i]);

	BCS_CHECK( array_view_approx( tanh(x), res, N ) );
	BCS_CHECK( array_view_approx( tanh(Xr), res, nr, nc) );
	BCS_CHECK( array_view_approx( tanh(Xc), res, nr, nc) );

}


BCS_TEST_CASE( test_hypot )
{
	const int N = 6;
	double src_x[N] = {-0.9, -0.5, -0.2, 0.1, 0.4, 0.8};
	double src_y[N] = {0.8, 0.5, 0.3, -0.2, -0.5, -0.6};
	double res[N];

	const int nr = 2;
	const int nc = 3;

	array1d<double> x(N, src_x);
	array2d<double, row_major_t> Xr(nr, nc, src_x);
	array2d<double, column_major_t> Xc(nr, nc, src_x);

	array1d<double> y(N, src_y);
	array2d<double, row_major_t> Yr(nr, nc, src_y);
	array2d<double, column_major_t> Yc(nr, nc, src_y);

	for (int i = 0; i < N; ++i) res[i] = std::sqrt( sqr(src_x[i]) + sqr(src_y[i]) );

	BCS_CHECK( array_view_approx( hypot(x, y), res, N ) );
	BCS_CHECK( array_view_approx( hypot(Xr, Yr), res, nr, nc) );
	BCS_CHECK( array_view_approx( hypot(Xc, Yc), res, nr, nc) );

}


test_suite *test_array_calc_suite()
{
	test_suite *suite = new test_suite( "test_array_calc" );

	suite->add( new test_array_compare() );
	suite->add( new test_array_max_min_each() );
	suite->add( new test_bounding() );

	suite->add( new test_array_add() );
	suite->add( new test_array_sub() );
	suite->add( new test_array_mul() );
	suite->add( new test_array_div() );
	suite->add( new test_array_neg() );
	suite->add( new test_array_abs() );

	suite->add( new test_array_power_and_root_funs() );
	suite->add( new test_array_exp_and_log_funs() );
	suite->add( new test_array_floor_and_ceil() );
	suite->add( new test_array_trigonometric_funs() );
	suite->add( new test_array_hyperbolic_funs() );
	suite->add( new test_hypot() );

	return suite;
}
