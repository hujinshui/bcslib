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

#include <iostream>
#include <vector>

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


/*

BCS_TEST_CASE( test_array_sub )
{
	double src1[] = {1, 3, 5, 9, 2, 7, 8, 5, 1};
	double src2[] = {5, 6, 2, 4, 8, 1, 9, 0, 3};

	// 1D

	array1d<double> x1(5, src1);
	array1d<double> x2(5, src2);
	const_aview1d<double, step_ind> xs1(src1, step_ind(5, 2));
	const_aview1d<double, step_ind> xs2(src2, step_ind(5, 2));

	double ra1[] = {-4, -3, 3, 5, -6};
	BCS_CHECK( array_view_approx(x1 - x2, ra1, 5) );

	double ra2[] = {-4, 1, -3, 0, -1};
	BCS_CHECK( array_view_approx(x1 - xs2, ra2, 5) );

	double ra3[] = {-4, -1, 0, 4, -7};
	BCS_CHECK( array_view_approx(xs1 - x2, ra3, 5) );

	double ra4[] = {-4, 3, -6, -1, -2};
	BCS_CHECK( array_view_approx(xs1 - xs2, ra4, 5) );

	double ra5l[] = {-1, 1, 3, 7, 0};
	double ra5r[] = {1, -1, -3, -7, 0};
	BCS_CHECK( array_view_approx(x1 - 2.0, ra5l, 5) );
	BCS_CHECK( array_view_approx(2.0 - x1, ra5r, 5) );

	double ra6l[] = {-1, 3, 0, 6, -1};
	double ra6r[] = {1, -3, 0, -6, 1};
	BCS_CHECK( array_view_approx(xs1 - 2.0, ra6l, 5) );
	BCS_CHECK( array_view_approx(2.0 - xs1, ra6r, 5) );

	array1d<double> y1(5);
	double y2_buf[10];
	aview1d<double, step_ind> y2(y2_buf, step_ind(5, 2));

	y1 << x1;
	y1 -= x2;
	BCS_CHECK( array_view_approx(y1, ra1, 5) );

	y1 << x1;
	y1 -= xs2;
	BCS_CHECK( array_view_approx(y1, ra2, 5) );

	y1 << x1;
	y1 -= 2.0;
	BCS_CHECK( array_view_approx(y1, ra5l, 5) );

	y1 << x1;
	be_subtracted(2.0, y1);
	BCS_CHECK( array_view_approx(y1, ra5r, 5) );

	y2 << x1;
	y2 -= x2;
	BCS_CHECK( array_view_approx(y2, ra1, 5) );

	y2 << x1;
	y2 -= xs2;
	BCS_CHECK( array_view_approx(y2, ra2, 5) );

	y2 << x1;
	y2 -= 2.0;
	BCS_CHECK( array_view_approx(y2, ra5l, 5) );

	y2 << x1;
	be_subtracted(2.0, y2);
	BCS_CHECK( array_view_approx(y2, ra5r, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);
	const_aview2d<double, row_major_t, step_ind, id_ind> X1s(src1, 3, 3, step_ind(2, 2), id_ind(3));
	const_aview2d<double, row_major_t, step_ind, id_ind> X2s(src2, 3, 3, step_ind(2, 2), id_ind(3));

	double rb1[] = {-4, -3, 3, 5, -6, 6};
	BCS_CHECK( array_view_approx(X1 - X2, rb1, 2, 3) );

	double rb2[] = {-4, -3, 3, 0, 2, 4};
	BCS_CHECK( array_view_approx(X1 - X2s, rb2, 2, 3) );

	double rb3[] = {-4, -3, 3, 4, -3, 0};
	BCS_CHECK( array_view_approx(X1s - X2, rb3, 2, 3) );

	double rb4[] = {-4, -3, 3, -1, 5, -2};
	BCS_CHECK( array_view_approx(X1s - X2s, rb4, 2, 3) );

	double rb5l[] = {-1, 1, 3, 7, 0, 5};
	double rb5r[] = {1, -1, -3, -7, 0, -5};
	BCS_CHECK( array_view_approx(X1 - 2.0, rb5l, 2, 3) );
	BCS_CHECK( array_view_approx(2.0 - X1, rb5r, 2, 3) );

	double rb6l[] = {-1, 1, 3, 6, 3, -1};
	double rb6r[] = {1, -1, -3, -6, -3, 1};
	BCS_CHECK( array_view_approx(X1s - 2.0, rb6l, 2, 3) );
	BCS_CHECK( array_view_approx(2.0 - X1s, rb6r, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3);
	double Y2_buf[20];
	aview2d<double, row_major_t, step_ind, id_ind> Y2(Y2_buf, 3, 3, step_ind(2, 2), id_ind(3));

	Y1 << X1;
	Y1 -= X2;
	BCS_CHECK( array_view_approx(Y1, rb1, 2, 3) );

	Y1 << X1;
	Y1 -= X2s;
	BCS_CHECK( array_view_approx(Y1, rb2, 2, 3) );

	Y1 << X1;
	Y1 -= 2.0;
	BCS_CHECK( array_view_approx(Y1, rb5l, 2, 3) );

	Y1 << X1;
	be_subtracted(2.0, Y1);
	BCS_CHECK( array_view_approx(Y1, rb5r, 2, 3) );

	Y2 << X1;
	Y2 -= X2;
	BCS_CHECK( array_view_approx(Y2, rb1, 2, 3) );

	Y2 << X1;
	Y2 -= X2s;
	BCS_CHECK( array_view_approx(Y2, rb2, 2, 3) );

	Y2 << X1;
	Y2 -= 2.0;
	BCS_CHECK( array_view_approx(Y2, rb5l, 2, 3) );

	Y2 << X1;
	be_subtracted(2.0, Y2);
	BCS_CHECK( array_view_approx(Y2, rb5r, 2, 3) );
}


BCS_TEST_CASE( test_array_mul )
{
	double src1[] = {1, 3, 5, 9, 2, 7, 8, 5, 1};
	double src2[] = {5, 6, 2, 4, 8, 1, 9, 0, 3};

	// 1D

	array1d<double> x1(5, src1);
	array1d<double> x2(5, src2);
	const_aview1d<double, step_ind> xs1(src1, step_ind(5, 2));
	const_aview1d<double, step_ind> xs2(src2, step_ind(5, 2));

	double ra1[] = {5, 18, 10, 36, 16};
	BCS_CHECK( array_view_approx(x1 * x2, ra1, 5) );

	double ra2[] = {5, 6, 40, 81, 6};
	BCS_CHECK( array_view_approx(x1 * xs2, ra2, 5) );

	double ra3[] = {5, 30, 4, 32, 8};
	BCS_CHECK( array_view_approx(xs1 * x2, ra3, 5) );

	double ra4[] = {5, 10, 16, 72, 3};
	BCS_CHECK( array_view_approx(xs1 * xs2, ra4, 5) );

	double ra5[] = {2, 6, 10, 18, 4};
	BCS_CHECK( array_view_approx(x1 * 2.0, ra5, 5) );
	BCS_CHECK( array_view_approx(2.0 * x1, ra5, 5) );

	double ra6[] = {2, 10, 4, 16, 2};
	BCS_CHECK( array_view_approx(xs1 * 2.0, ra6, 5) );
	BCS_CHECK( array_view_approx(2.0 * xs1, ra6, 5) );

	array1d<double> y1(5);
	double y2_buf[10];
	aview1d<double, step_ind> y2(y2_buf, step_ind(5, 2));

	y1 << x1;
	y1 *= x2;
	BCS_CHECK( array_view_approx(y1, ra1, 5) );

	y1 << x1;
	y1 *= xs2;
	BCS_CHECK( array_view_approx(y1, ra2, 5) );

	y1 << x1;
	y1 *= 2.0;
	BCS_CHECK( array_view_approx(y1, ra5, 5) );

	y2 << x1;
	y2 *= x2;
	BCS_CHECK( array_view_approx(y2, ra1, 5) );

	y2 << x1;
	y2 *= xs2;
	BCS_CHECK( array_view_approx(y2, ra2, 5) );

	y2 << x1;
	y2 *= 2.0;
	BCS_CHECK( array_view_approx(y2, ra5, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);
	const_aview2d<double, row_major_t, step_ind, id_ind> X1s(src1, 3, 3, step_ind(2, 2), id_ind(3));
	const_aview2d<double, row_major_t, step_ind, id_ind> X2s(src2, 3, 3, step_ind(2, 2), id_ind(3));

	double rb1[] = {5, 18, 10, 36, 16, 7};
	BCS_CHECK( array_view_approx(X1 * X2, rb1, 2, 3) );

	double rb2[] = {5, 18, 10, 81, 0, 21};
	BCS_CHECK( array_view_approx(X1 * X2s, rb2, 2, 3) );

	double rb3[] = {5, 18, 10, 32, 40, 1};
	BCS_CHECK( array_view_approx(X1s * X2, rb3, 2, 3) );

	double rb4[] = {5, 18, 10, 72, 0, 3};
	BCS_CHECK( array_view_approx(X1s * X2s, rb4, 2, 3) );

	double rb5[] = {2, 6, 10, 18, 4, 14};
	BCS_CHECK( array_view_approx(X1 * 2.0, rb5, 2, 3) );
	BCS_CHECK( array_view_approx(2.0 * X1, rb5, 2, 3) );

	double rb6[] = {2, 6, 10, 16, 10, 2};
	BCS_CHECK( array_view_approx(X1s * 2.0, rb6, 2, 3) );
	BCS_CHECK( array_view_approx(2.0 * X1s, rb6, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3);
	double Y2_buf[20];
	aview2d<double, row_major_t, step_ind, id_ind> Y2(Y2_buf, 3, 3, step_ind(2, 2), id_ind(3));

	Y1 << X1;
	Y1 *= X2;
	BCS_CHECK( array_view_approx(Y1, rb1, 2, 3) );

	Y1 << X1;
	Y1 *= X2s;
	BCS_CHECK( array_view_approx(Y1, rb2, 2, 3) );

	Y1 << X1;
	Y1 *= 2.0;
	BCS_CHECK( array_view_approx(Y1, rb5, 2, 3) );

	Y2 << X1;
	Y2 *= X2;
	BCS_CHECK( array_view_approx(Y2, rb1, 2, 3) );

	Y2 << X1;
	Y2 *= X2s;
	BCS_CHECK( array_view_approx(Y2, rb2, 2, 3) );

	Y2 << X1;
	Y2 *= 2.0;
	BCS_CHECK( array_view_approx(Y2, rb5, 2, 3) );
}



BCS_TEST_CASE( test_array_div )
{
	double src1[] = {1, 3, 5, 9, 2, 7, 8, 5, 1};
	double src2[] = {5, 6, 2, 4, 8, 1, 9, 10, 3};

	// 1D

	array1d<double> x1(5, src1);
	array1d<double> x2(5, src2);
	const_aview1d<double, step_ind> xs1(src1, step_ind(5, 2));
	const_aview1d<double, step_ind> xs2(src2, step_ind(5, 2));

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

	y1 << x1;
	be_divided(2.0, y1);
	BCS_CHECK( array_view_approx(y1, ra5r, 5) );

	y2 << x1;
	y2 /= x2;
	BCS_CHECK( array_view_approx(y2, ra1, 5) );

	y2 << x1;
	y2 /= xs2;
	BCS_CHECK( array_view_approx(y2, ra2, 5) );

	y2 << x1;
	y2 /= 2.0;
	BCS_CHECK( array_view_approx(y2, ra5l, 5) );

	y2 << x1;
	be_divided(2.0, y2);
	BCS_CHECK( array_view_approx(y2, ra5r, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);
	const_aview2d<double, row_major_t, step_ind, id_ind> X1s(src1, 3, 3, step_ind(2, 2), id_ind(3));
	const_aview2d<double, row_major_t, step_ind, id_ind> X2s(src2, 3, 3, step_ind(2, 2), id_ind(3));

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

	Y1 << X1;
	be_divided(2.0, Y1);
	BCS_CHECK( array_view_approx(Y1, rb5r, 2, 3) );

	Y2 << X1;
	Y2 /= X2;
	BCS_CHECK( array_view_approx(Y2, rb1, 2, 3) );

	Y2 << X1;
	Y2 /= X2s;
	BCS_CHECK( array_view_approx(Y2, rb2, 2, 3) );

	Y2 << X1;
	Y2 /= 2.0;
	BCS_CHECK( array_view_approx(Y2, rb5l, 2, 3) );

	Y2 << X1;
	be_divided(2.0, Y2);
	BCS_CHECK( array_view_approx(Y2, rb5r, 2, 3) );
}


BCS_TEST_CASE( test_array_neg )
{
	double src1[] = {1, 3, 5, 9, 2, 7, 8, 5, 1};

	// 1D

	array1d<double> x1(5, src1);
	const_aview1d<double, step_ind> xs1(src1, step_ind(5, 2));

	double ra1[] = {-1, -3, -5, -9, -2};
	BCS_CHECK( array_view_approx(-x1, ra1, 5) );
	double ra2[] = {-1, -5, -2, -8, -1};
	BCS_CHECK( array_view_approx(-xs1, ra2, 5) );

	array1d<double> y1(5, src1);
	be_negated(y1);
	BCS_CHECK( array_view_approx(y1, ra1, 5) );

	double y2_buf[10];
	aview1d<double, step_ind> y2(y2_buf, step_ind(5, 2));
	y2 << x1;
	be_negated(y2);
	BCS_CHECK( array_view_approx(y2, ra1, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	const_aview2d<double, row_major_t, step_ind, id_ind> Xs1(src1, 3, 3, step_ind(2, 2), id_ind(3));

	double rb1[] = {-1, -3, -5, -9, -2, -7};
	BCS_CHECK( array_view_approx(-X1, rb1, 2, 3) );
	double rb2[] = {-1, -3, -5, -8, -5, -1};
	BCS_CHECK( array_view_approx(-Xs1, rb2, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3, src1);
	be_negated(Y1);
	BCS_CHECK( array_view_approx(Y1, rb1, 2, 3) );

	double Y2_buf[10];
	aview2d<double, row_major_t, step_ind, id_ind> Y2(Y2_buf, 3, 3, step_ind(2, 2), id_ind(3));
	Y2 << X1;
	be_negated(Y2);
	BCS_CHECK( array_view_approx(Y2, rb1, 2, 3) );

}


BCS_TEST_CASE( test_array_abs )
{
	const int N = 6;
	double src[N] = {1, -2, -3, 4, -5, 6};
	double res[N];
	for (int i = 0; i < N; ++i) res[i] = std::abs(src[i]);

	array1d<double> x(6, src);
	BCS_CHECK( array_view_approx( abs(x), res, 6 ) );

	double res_s[3] = {1, 3, 5};
	const_aview1d<double, step_ind> xs(src, step_ind(3, 2));
	BCS_CHECK( array_view_approx( abs(xs), res_s, 3) );

	array2d<double, row_major_t> Xr(2, 3, src);
	BCS_CHECK( array_view_approx( abs(Xr), res, 2, 3) );

	array2d<double, column_major_t> Xc(2, 3, src);
	BCS_CHECK( array_view_approx( abs(Xc), res, 2, 3) );
}


BCS_TEST_CASE( test_array_sqr_sqrt )
{
	const int N = 6;
	double src[N] = {1, 2, 3, 4, 5, 6};
	double res[N];
	double res_s[3];

	array1d<double> x(6, src);
	const_aview1d<double, step_ind> xs(src, step_ind(3, 2));
	array2d<double, row_major_t> Xr(2, 3, src);
	array2d<double, column_major_t> Xc(2, 3, src);

	// sqr

	for (int i = 0; i < N; ++i) res[i] = src[i] * src[i];
	for (int i = 0; i < 3; ++i) res_s[i] = sqr(src[i*2]);

	BCS_CHECK( array_view_approx( sqr(x), res, 6 ) );
	BCS_CHECK( array_view_approx( sqr(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( sqr(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( sqr(Xc), res, 2, 3) );

	// sqrt

	for (int i = 0; i < N; ++i) res[i] = std::sqrt(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::sqrt(src[i*2]);

	BCS_CHECK( array_view_approx( sqrt(x), res, 6 ) );
	BCS_CHECK( array_view_approx( sqrt(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( sqrt(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( sqrt(Xc), res, 2, 3) );
}


BCS_TEST_CASE( test_array_pow )
{
	const int N = 6;
	double src[N] = {1.2, 1.8, 2.3, 2.6, 2.9, 3.5};
	double src_e[N] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
	double res[N];
	double res_s[3];

	array1d<double> x(6, src);
	const_aview1d<double, step_ind> xs(src, step_ind(3, 2));
	array2d<double, row_major_t> Xr(2, 3, src);
	array2d<double, column_major_t> Xc(2, 3, src);

	array1d<double> e(6, src_e);
	const_aview1d<double, step_ind> es(src_e, step_ind(3, 2));
	array2d<double, row_major_t> Er(2, 3, src_e);
	array2d<double, column_major_t> Ec(2, 3, src_e);

	// pow

	for (int i = 0; i < N; ++i) res[i] = std::pow(src[i], src_e[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::pow(src[i*2], src_e[i*2]);

	BCS_CHECK( array_view_approx( pow(x, e),   res, 6 ) );
	BCS_CHECK( array_view_approx( pow(xs, es), res_s, 3 ) );
	BCS_CHECK( array_view_approx( pow(Xr, Er), res, 2, 3) );
	BCS_CHECK( array_view_approx( pow(Xc, Ec), res, 2, 3) );

	// pow (e)

	for (int i = 0; i < N; ++i) res[i] = std::pow(src[i], 3.2);
	for (int i = 0; i < 3; ++i) res_s[i] = std::pow(src[i*2], 3.2);

	BCS_CHECK( array_view_approx( pow(x, 3.2),  res, 6 ) );
	BCS_CHECK( array_view_approx( pow(xs, 3.2), res_s, 3 ) );
	BCS_CHECK( array_view_approx( pow(Xr, 3.2), res, 2, 3) );
	BCS_CHECK( array_view_approx( pow(Xc, 3.2), res, 2, 3) );

	// pow_n

	for (int i = 0; i < N; ++i) res[i] = std::pow(src[i], 3);
	for (int i = 0; i < 3; ++i) res_s[i] = std::pow(src[i*2], 3);

	BCS_CHECK( array_view_approx( pow_n(x,  3), res, 6 ) );
	BCS_CHECK( array_view_approx( pow_n(xs, 3), res_s, 3 ) );
	BCS_CHECK( array_view_approx( pow_n(Xr, 3), res, 2, 3) );
	BCS_CHECK( array_view_approx( pow_n(Xc, 3), res, 2, 3) );

}


BCS_TEST_CASE( test_array_exp_log )
{
	const int N = 6;
	double src[N] = {1, 2, 3, 4, 5, 6};
	double res[N];
	double res_s[3];

	array1d<double> x(6, src);
	const_aview1d<double, step_ind> xs(src, step_ind(3, 2));
	array2d<double, row_major_t> Xr(2, 3, src);
	array2d<double, column_major_t> Xc(2, 3, src);

	// exp

	for (int i = 0; i < N; ++i) res[i] = std::exp(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::exp(src[i*2]);

	BCS_CHECK( array_view_approx( exp(x), res, 6 ) );
	BCS_CHECK( array_view_approx( exp(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( exp(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( exp(Xc), res, 2, 3) );

	// log

	for (int i = 0; i < N; ++i) res[i] = std::log(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::log(src[i*2]);

	BCS_CHECK( array_view_approx( log(x), res, 6 ) );
	BCS_CHECK( array_view_approx( log(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( log(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( log(Xc), res, 2, 3) );

	// log10

	for (int i = 0; i < N; ++i) res[i] = std::log10(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::log10(src[i*2]);

	BCS_CHECK( array_view_approx( log10(x), res, 6 ) );
	BCS_CHECK( array_view_approx( log10(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( log10(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( log10(Xc), res, 2, 3) );
}


BCS_TEST_CASE( test_array_ceil_floor )
{
	const int N = 6;
	double src[N] = {1.2, 2, 3.4, -4.1, 5.7, 6.01};
	double res[N];
	double res_s[3];

	array1d<double> x(6, src);
	const_aview1d<double, step_ind> xs(src, step_ind(3, 2));
	array2d<double, row_major_t> Xr(2, 3, src);
	array2d<double, column_major_t> Xc(2, 3, src);

	// ceil

	for (int i = 0; i < N; ++i) res[i] = std::ceil(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::ceil(src[i*2]);

	BCS_CHECK( array_view_approx( ceil(x), res, 6 ) );
	BCS_CHECK( array_view_approx( ceil(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( ceil(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( ceil(Xc), res, 2, 3) );

	// floor

	for (int i = 0; i < N; ++i) res[i] = std::floor(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::floor(src[i*2]);

	BCS_CHECK( array_view_approx( floor(x), res, 6 ) );
	BCS_CHECK( array_view_approx( floor(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( floor(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( floor(Xc), res, 2, 3) );

}


BCS_TEST_CASE( test_array_trifuncs )
{
	const int N = 6;
	double src[N] = {1, 2, 3, 4, 5, 6};
	double res[N];
	double res_s[3];

	array1d<double> x(6, src);
	const_aview1d<double, step_ind> xs(src, step_ind(3, 2));
	array2d<double, row_major_t> Xr(2, 3, src);
	array2d<double, column_major_t> Xc(2, 3, src);

	// sin

	for (int i = 0; i < N; ++i) res[i] = std::sin(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::sin(src[i*2]);

	BCS_CHECK( array_view_approx( sin(x), res, 6 ) );
	BCS_CHECK( array_view_approx( sin(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( sin(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( sin(Xc), res, 2, 3) );

	// cos

	for (int i = 0; i < N; ++i) res[i] = std::cos(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::cos(src[i*2]);

	BCS_CHECK( array_view_approx( cos(x), res, 6 ) );
	BCS_CHECK( array_view_approx( cos(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( cos(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( cos(Xc), res, 2, 3) );

	// tan

	for (int i = 0; i < N; ++i) res[i] = std::tan(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::tan(src[i*2]);

	BCS_CHECK( array_view_approx( tan(x), res, 6 ) );
	BCS_CHECK( array_view_approx( tan(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( tan(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( tan(Xc), res, 2, 3) );
}


BCS_TEST_CASE( test_array_arc_trifuncs )
{
	const int N = 6;
	double src[N] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
	double src2[N] = {0.3, 0.5, 0.6, 0.4, 0.1, 0.2};
	double res[N];
	double res_s[3];

	array1d<double> x(6, src);
	const_aview1d<double, step_ind> xs(src, step_ind(3, 2));
	array2d<double, row_major_t> Xr(2, 3, src);
	array2d<double, column_major_t> Xc(2, 3, src);

	array1d<double> x2(6, src2);
	const_aview1d<double, step_ind> xs2(src2, step_ind(3, 2));
	array2d<double, row_major_t> Xr2(2, 3, src2);
	array2d<double, column_major_t> Xc2(2, 3, src2);

	// asin

	for (int i = 0; i < N; ++i) res[i] = std::asin(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::asin(src[i*2]);

	BCS_CHECK( array_view_approx( asin(x), res, 6 ) );
	BCS_CHECK( array_view_approx( asin(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( asin(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( asin(Xc), res, 2, 3) );

	// acos

	for (int i = 0; i < N; ++i) res[i] = std::acos(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::acos(src[i*2]);

	BCS_CHECK( array_view_approx( acos(x), res, 6 ) );
	BCS_CHECK( array_view_approx( acos(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( acos(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( acos(Xc), res, 2, 3) );

	// atan

	for (int i = 0; i < N; ++i) res[i] = std::atan(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::atan(src[i*2]);

	BCS_CHECK( array_view_approx( atan(x), res, 6 ) );
	BCS_CHECK( array_view_approx( atan(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( atan(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( atan(Xc), res, 2, 3) );

	// atan2

	for (int i = 0; i < N; ++i) res[i] = std::atan2(src[i], src2[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::atan2(src[i*2], src2[i*2]);

	BCS_CHECK( array_view_approx( atan2(x, x2), res, 6 ) );
	BCS_CHECK( array_view_approx( atan2(xs, xs2), res_s, 3 ) );
	BCS_CHECK( array_view_approx( atan2(Xr, Xr2), res, 2, 3) );
	BCS_CHECK( array_view_approx( atan2(Xc, Xc2), res, 2, 3) );

}


BCS_TEST_CASE( test_array_htrifuncs )
{
	const int N = 6;
	double src[N] = {1, 2, 3, 4, 5, 6};
	double res[N];
	double res_s[3];

	array1d<double> x(6, src);
	const_aview1d<double, step_ind> xs(src, step_ind(3, 2));
	array2d<double, row_major_t> Xr(2, 3, src);
	array2d<double, column_major_t> Xc(2, 3, src);

	// sin

	for (int i = 0; i < N; ++i) res[i] = std::sinh(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::sinh(src[i*2]);

	BCS_CHECK( array_view_approx( sinh(x), res, 6 ) );
	BCS_CHECK( array_view_approx( sinh(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( sinh(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( sinh(Xc), res, 2, 3) );

	// cos

	for (int i = 0; i < N; ++i) res[i] = std::cosh(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::cosh(src[i*2]);

	BCS_CHECK( array_view_approx( cosh(x), res, 6 ) );
	BCS_CHECK( array_view_approx( cosh(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( cosh(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( cosh(Xc), res, 2, 3) );

	// tan

	for (int i = 0; i < N; ++i) res[i] = std::tanh(src[i]);
	for (int i = 0; i < 3; ++i) res_s[i] = std::tanh(src[i*2]);

	BCS_CHECK( array_view_approx( tanh(x), res, 6 ) );
	BCS_CHECK( array_view_approx( tanh(xs), res_s, 3 ) );
	BCS_CHECK( array_view_approx( tanh(Xr), res, 2, 3) );
	BCS_CHECK( array_view_approx( tanh(Xc), res, 2, 3) );
}

*/


test_suite *test_array_calc_suite()
{
	test_suite *suite = new test_suite( "test_array_calc" );

	suite->add( new test_array_add() );
	/*
	suite->add( new test_array_sub() );
	suite->add( new test_array_mul() );
	suite->add( new test_array_div() );
	suite->add( new test_array_neg() );


	suite->add( new test_array_abs() );
	suite->add( new test_array_sqr_sqrt() );
	suite->add( new test_array_pow() );
	suite->add( new test_array_exp_log() );
	suite->add( new test_array_ceil_floor() );
	suite->add( new test_array_trifuncs() );
	suite->add( new test_array_arc_trifuncs() );
	suite->add( new test_array_htrifuncs() );
	*/

	return suite;
}
