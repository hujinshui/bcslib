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
	BCS_CHECK( array_view_approx(X1 + X2, rb1, 2, 3) );

	double rb2[] = {6, 9, 7, 18, 2, 10};
	BCS_CHECK( array_view_approx(X1 + X2s, rb2, 2, 3) );

	double rb3[] = {6, 9, 7, 12, 13, 2};
	BCS_CHECK( array_view_approx(X1s + X2, rb3, 2, 3) );

	double rb4[] = {6, 9, 7, 17, 5, 4};
	BCS_CHECK( array_view_approx(X1s + X2s, rb4, 2, 3) );

	double rb5[] = {3, 5, 7, 11, 4, 9};
	BCS_CHECK( array_view_approx(X1 + 2.0, rb5, 2, 3) );
	BCS_CHECK( array_view_approx(2.0 + X1, rb5, 2, 3) );

	double rb6[] = {3, 5, 7, 10, 7, 3};
	BCS_CHECK( array_view_approx(X1s + 2.0, rb6, 2, 3) );
	BCS_CHECK( array_view_approx(2.0 + X1s, rb6, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3);
	double Y2_buf[20];
	aview2d<double, row_major_t, step_ind, id_ind> Y2(Y2_buf, 3, 3, step_ind(2, 2), id_ind(3));

	Y1 << X1;
	Y1 += X2;
	BCS_CHECK( array_view_approx(Y1, rb1, 2, 3) );

	Y1 << X1;
	Y1 += X2s;
	BCS_CHECK( array_view_approx(Y1, rb2, 2, 3) );

	Y1 << X1;
	Y1 += 2.0;
	BCS_CHECK( array_view_approx(Y1, rb5, 2, 3) );

	Y2 << X1;
	Y2 += X2;
	BCS_CHECK( array_view_approx(Y2, rb1, 2, 3) );

	Y2 << X1;
	Y2 += X2s;
	BCS_CHECK( array_view_approx(Y2, rb2, 2, 3) );

	Y2 << X1;
	Y2 += 2.0;
	BCS_CHECK( array_view_approx(Y2, rb5, 2, 3) );

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
	BCS_CHECK( array_view_equal(x1 / x2, ra1, 5) );

	double ra2[] = {1.0/5, 3.0/2, 5.0/8, 9.0/9, 2.0/3};
	BCS_CHECK( array_view_equal(x1 / xs2, ra2, 5) );

	double ra3[] = {1.0/5, 5.0/6, 2.0/2, 8.0/4, 1.0/8};
	BCS_CHECK( array_view_equal(xs1 / x2, ra3, 5) );

	double ra4[] = {1.0/5, 5.0/2, 2.0/8, 8.0/9, 1.0/3};
	BCS_CHECK( array_view_equal(xs1 / xs2, ra4, 5) );

	double ra5l[] = {1.0/2, 3.0/2, 5.0/2, 9.0/2, 2.0/2};
	double ra5r[] = {2.0/1, 2.0/3, 2.0/5, 2.0/9, 2.0/2};
	BCS_CHECK( array_view_equal(x1 / 2.0, ra5l, 5) );
	BCS_CHECK( array_view_equal(2.0 / x1, ra5r, 5) );

	double ra6l[] = {1.0/2, 5.0/2, 2.0/2, 8.0/2, 1.0/2};
	double ra6r[] = {2.0/1, 2.0/5, 2.0/2, 2.0/8, 2.0/1};
	BCS_CHECK( array_view_equal(xs1 / 2.0, ra6l, 5) );
	BCS_CHECK( array_view_equal(2.0 / xs1, ra6r, 5) );

	array1d<double> y1(5);
	double y2_buf[10];
	aview1d<double, step_ind> y2(y2_buf, step_ind(5, 2));

	y1 << x1;
	y1 /= x2;
	BCS_CHECK( array_view_equal(y1, ra1, 5) );

	y1 << x1;
	y1 /= xs2;
	BCS_CHECK( array_view_equal(y1, ra2, 5) );

	y1 << x1;
	y1 /= 2.0;
	BCS_CHECK( array_view_equal(y1, ra5l, 5) );

	y2 << x1;
	y2 /= x2;
	BCS_CHECK( array_view_equal(y2, ra1, 5) );

	y2 << x1;
	y2 /= xs2;
	BCS_CHECK( array_view_equal(y2, ra2, 5) );

	y2 << x1;
	y2 /= 2.0;
	BCS_CHECK( array_view_equal(y2, ra5l, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);
	aview2d<double, row_major_t, step_ind, id_ind> X1s(src1, 3, 3, step_ind(2, 2), id_ind(3));
	aview2d<double, row_major_t, step_ind, id_ind> X2s(src2, 3, 3, step_ind(2, 2), id_ind(3));

	double rb1[] = {1.0/5, 3.0/6, 5.0/2, 9.0/4, 2.0/8, 7.0/1};
	BCS_CHECK( array_view_equal(X1 / X2, rb1, 2, 3) );

	double rb2[] = {1.0/5, 3.0/6, 5.0/2, 9.0/9, 2.0/10, 7.0/3};
	BCS_CHECK( array_view_equal(X1 / X2s, rb2, 2, 3) );

	double rb3[] = {1.0/5, 3.0/6, 5.0/2, 8.0/4, 5.0/8, 1.0/1};
	BCS_CHECK( array_view_equal(X1s / X2, rb3, 2, 3) );

	double rb4[] = {1.0/5, 3.0/6, 5.0/2, 8.0/9, 5.0/10, 1.0/3};
	BCS_CHECK( array_view_equal(X1s / X2s, rb4, 2, 3) );

	double rb5l[] = {1.0/2, 3.0/2, 5.0/2, 9.0/2, 2.0/2, 7.0/2};
	double rb5r[] = {2.0/1, 2.0/3, 2.0/5, 2.0/9, 2.0/2, 2.0/7};
	BCS_CHECK( array_view_equal(X1 / 2.0, rb5l, 2, 3) );
	BCS_CHECK( array_view_equal(2.0 / X1, rb5r, 2, 3) );

	double rb6l[] = {1.0/2, 3.0/2, 5.0/2, 8.0/2, 5.0/2, 1.0/2};
	double rb6r[] = {2.0/1, 2.0/3, 2.0/5, 2.0/8, 2.0/5, 2.0/1};
	BCS_CHECK( array_view_equal(X1s / 2.0, rb6l, 2, 3) );
	BCS_CHECK( array_view_equal(2.0 / X1s, rb6r, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3);
	double Y2_buf[20];
	aview2d<double, row_major_t, step_ind, id_ind> Y2(Y2_buf, 3, 3, step_ind(2, 2), id_ind(3));

	Y1 << X1;
	Y1 /= X2;
	BCS_CHECK( array_view_equal(Y1, rb1, 2, 3) );

	Y1 << X1;
	Y1 /= X2s;
	BCS_CHECK( array_view_equal(Y1, rb2, 2, 3) );

	Y1 << X1;
	Y1 /= 2.0;
	BCS_CHECK( array_view_equal(Y1, rb5l, 2, 3) );

	Y2 << X1;
	Y2 /= X2;
	BCS_CHECK( array_view_equal(Y2, rb1, 2, 3) );

	Y2 << X1;
	Y2 /= X2s;
	BCS_CHECK( array_view_equal(Y2, rb2, 2, 3) );

	Y2 << X1;
	Y2 /= 2.0;
	BCS_CHECK( array_view_equal(Y2, rb5l, 2, 3) );

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
	neg_arr_inplace(y1);
	BCS_CHECK( array_view_equal(y1, ra1, 5) );

	double y2_buf[10];
	aview1d<double, step_ind> y2(y2_buf, step_ind(5, 2));
	y2 << x1;
	neg_arr_inplace(y2);
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
	BCS_CHECK( array_view_approx( abs(Xc), res, 2, 3) );
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

	array1d<double> x2(N);
	array2d<double, row_major_t> Xr2(nr, nc);
	array2d<double, column_major_t> Xc2(nr, nc);

	array1d<double> e(N, es);
	array2d<double, row_major_t> Er(nr, nc, es);
	array2d<double, column_major_t> Ec(nr, nc, es);

	// sqr

	for (int i = 0; i < N; ++i) res[i] = src[i] * src[i];

	BCS_CHECK( array_view_equal( sqr(x), res, N ) );
	x2 << x; sqr_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( sqr(Xr), res, nr, nc) );
	Xr2 << Xr; sqr_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( sqr(Xc), res, nr, nc) );
	Xc2 << Xc; sqr_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// sqrt

	for (int i = 0; i < N; ++i) res[i] = std::sqrt(src[i]);

	BCS_CHECK( array_view_equal( sqrt(x), res, N ) );
	x2 << x; sqrt_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( sqrt(Xr), res, nr, nc) );
	Xr2 << Xr; sqrt_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( sqrt(Xc), res, nr, nc) );
	Xc2 << Xc; sqrt_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );


	// rcp

	for (int i = 0; i < N; ++i) res[i] = 1.0 / src[i];

	BCS_CHECK( array_view_equal( rcp(x), res, N ) );
	x2 << x; rcp_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( rcp(Xr), res, nr, nc) );
	Xr2 << Xr; rcp_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( rcp(Xc), res, nr, nc) );
	Xc2 << Xc; rcp_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );


	// rsqrt

	for (int i = 0; i < N; ++i) res[i] = 1.0 / std::sqrt(src[i]);

	BCS_CHECK( array_view_equal( rsqrt(x), res, N ) );
	x2 << x; rsqrt_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( rsqrt(Xr), res, nr, nc) );
	Xr2 << Xr; rsqrt_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( rsqrt(Xc), res, nr, nc) );
	Xc2 << Xc; rsqrt_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// pow

	for (int i = 0; i < N; ++i) res[i] = std::pow(src[i], es[i]);

	BCS_CHECK( array_view_equal( pow(x, e), res, N ) );
	x2 << x; pow_ip(x2, e);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( pow(Xr, Er), res, nr, nc) );
	Xr2 << Xr; pow_ip(Xr2, Er);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( pow(Xc, Ec), res, nr, nc) );
	Xc2 << Xc; pow_ip(Xc2, Ec);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// pow with constant exponent

	double q = 2.3;

	for (int i = 0; i < N; ++i) res[i] = std::pow(src[i], q);

	BCS_CHECK( array_view_equal( pow(x, q), res, N ) );
	x2 << x; pow_ip(x2, q);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( pow(Xr, q), res, nr, nc) );
	Xr2 << Xr; pow_ip(Xr2, q);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( pow(Xc, q), res, nr, nc) );
	Xc2 << Xc; pow_ip(Xc2, q);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

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

	array1d<double> x2(N);
	array2d<double, row_major_t> Xr2(nr, nc);
	array2d<double, column_major_t> Xc2(nr, nc);

	// exp

	for (int i = 0; i < N; ++i) res[i] = std::exp(src[i]);

	BCS_CHECK( array_view_equal( exp(x), res, N ) );
	x2 << x; exp_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( exp(Xr), res, nr, nc) );
	Xr2 << Xr; exp_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( exp(Xc), res, nr, nc) );
	Xc2 << Xc; exp_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// log

	for (int i = 0; i < N; ++i) res[i] = std::log(src[i]);

	BCS_CHECK( array_view_equal( log(x), res, N ) );
	x2 << x; log_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( log(Xr), res, nr, nc) );
	Xr2 << Xr; log_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( log(Xc), res, nr, nc) );
	Xc2 << Xc; log_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );


	// log10

	for (int i = 0; i < N; ++i) res[i] = std::log10(src[i]);

	BCS_CHECK( array_view_equal( log10(x), res, N ) );
	x2 << x; log10_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( log10(Xr), res, nr, nc) );
	Xr2 << Xr; log10_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( log10(Xc), res, nr, nc) );
	Xc2 << Xc; log10_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

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

	array1d<double> x2(N);
	array2d<double, row_major_t> Xr2(nr, nc);
	array2d<double, column_major_t> Xc2(nr, nc);

	// floor

	for (int i = 0; i < N; ++i) res[i] = std::floor(src[i]);

	BCS_CHECK( array_view_equal( floor(x), res, N ) );
	x2 << x; floor_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( floor(Xr), res, nr, nc) );
	Xr2 << Xr; floor_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( floor(Xc), res, nr, nc) );
	Xc2 << Xc; floor_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// ceil

	for (int i = 0; i < N; ++i) res[i] = std::ceil(src[i]);

	BCS_CHECK( array_view_equal( ceil(x), res, N ) );
	x2 << x; ceil_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( ceil(Xr), res, nr, nc) );
	Xr2 << Xr; ceil_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( ceil(Xc), res, nr, nc) );
	Xc2 << Xc; ceil_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );
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

	array1d<double> x2(N);
	array2d<double, row_major_t> Xr2(nr, nc);
	array2d<double, column_major_t> Xc2(nr, nc);

	// sin

	for (int i = 0; i < N; ++i) res[i] = std::sin(src[i]);

	BCS_CHECK( array_view_equal( sin(x), res, N ) );
	x2 << x; sin_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( sin(Xr), res, nr, nc) );
	Xr2 << Xr; sin_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( sin(Xc), res, nr, nc) );
	Xc2 << Xc; sin_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// cos

	for (int i = 0; i < N; ++i) res[i] = std::cos(src[i]);

	BCS_CHECK( array_view_equal( cos(x), res, N ) );
	x2 << x; cos_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( cos(Xr), res, nr, nc) );
	Xr2 << Xr; cos_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( cos(Xc), res, nr, nc) );
	Xc2 << Xc; cos_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// tan

	for (int i = 0; i < N; ++i) res[i] = std::tan(src[i]);

	BCS_CHECK( array_view_equal( tan(x), res, N ) );
	x2 << x; tan_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( tan(Xr), res, nr, nc) );
	Xr2 << Xr; tan_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( tan(Xc), res, nr, nc) );
	Xc2 << Xc; tan_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// asin

	for (int i = 0; i < N; ++i) res[i] = std::asin(src[i]);

	BCS_CHECK( array_view_equal( asin(x), res, N ) );
	x2 << x; asin_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( asin(Xr), res, nr, nc) );
	Xr2 << Xr; asin_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( asin(Xc), res, nr, nc) );
	Xc2 << Xc; asin_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// acos

	for (int i = 0; i < N; ++i) res[i] = std::acos(src[i]);

	BCS_CHECK( array_view_equal( acos(x), res, N ) );
	x2 << x; acos_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( acos(Xr), res, nr, nc) );
	Xr2 << Xr; acos_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( acos(Xc), res, nr, nc) );
	Xc2 << Xc; acos_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// atan

	for (int i = 0; i < N; ++i) res[i] = std::atan(src[i]);

	BCS_CHECK( array_view_equal( atan(x), res, N ) );
	x2 << x; atan_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( atan(Xr), res, nr, nc) );
	Xr2 << Xr; atan_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( atan(Xc), res, nr, nc) );
	Xc2 << Xc; atan_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// atan2

	for (int i = 0; i < N; ++i) res[i] = std::atan2(src[i], src_e[i]);

	BCS_CHECK( array_view_equal( atan2(x, e), res, N ) );

	BCS_CHECK( array_view_equal( atan2(Xr, Er), res, nr, nc) );

	BCS_CHECK( array_view_equal( atan2(Xc, Ec), res, nr, nc) );

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

	array1d<double> x2(N);
	array2d<double, row_major_t> Xr2(nr, nc);
	array2d<double, column_major_t> Xc2(nr, nc);

	// sinh

	for (int i = 0; i < N; ++i) res[i] = std::sinh(src[i]);

	BCS_CHECK( array_view_equal( sinh(x), res, N ) );
	x2 << x; sinh_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( sinh(Xr), res, nr, nc) );
	Xr2 << Xr; sinh_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( sinh(Xc), res, nr, nc) );
	Xc2 << Xc; sinh_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// cosh

	for (int i = 0; i < N; ++i) res[i] = std::cosh(src[i]);

	BCS_CHECK( array_view_equal( cosh(x), res, N ) );
	x2 << x; cosh_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( cosh(Xr), res, nr, nc) );
	Xr2 << Xr; cosh_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( cosh(Xc), res, nr, nc) );
	Xc2 << Xc; cosh_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );

	// tanh

	for (int i = 0; i < N; ++i) res[i] = std::tanh(src[i]);

	BCS_CHECK( array_view_equal( tanh(x), res, N ) );
	x2 << x; tanh_ip(x2);
	BCS_CHECK( array_view_equal( x2, res, N) );

	BCS_CHECK( array_view_equal( tanh(Xr), res, nr, nc) );
	Xr2 << Xr; tanh_ip(Xr2);
	BCS_CHECK( array_view_equal( Xr2, res, nr, nc) );

	BCS_CHECK( array_view_equal( tanh(Xc), res, nr, nc) );
	Xc2 << Xc; tanh_ip(Xc2);
	BCS_CHECK( array_view_equal( Xc2, res, nr, nc) );
}


test_suite *test_array_calc_suite()
{
	test_suite *suite = new test_suite( "test_array_calc" );

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

	return suite;
}
