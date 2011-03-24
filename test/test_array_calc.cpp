/**
 * @file test_array_ops.cpp
 *
 * The Unit testing for array operators
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/test_units.h>
#include <bcslib/array/array_calc.h>

#include <iostream>

using namespace bcs;
using namespace bcs::test;


template<typename T, class TIndexer>
void print_array(const const_aview1d<T, TIndexer>& a)
{
	for (index_t i = 0; i < (index_t)a.nelems(); ++i)
	{
		std::cout << a[i] << ' ';
	}
	std::cout << std::endl;
}

template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
void print_array(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
{
	for (index_t i = 0; i < (index_t)a.nrows(); ++i)
	{
		for (index_t j = 0; j < (index_t)a.ncolumns(); ++j)
		{
			std::cout << a(i, j) << ' ';
		}
		std::cout << std::endl;
	}
}




BCS_TEST_CASE( test_array_add )
{
	double src1[] = {1, 3, 5, 9, 2, 7, 8, 5, 1};
	double src2[] = {5, 6, 2, 4, 8, 1, 9, 0, 3};

	// 1D

	array1d<double> x1(5, src1);
	array1d<double> x2(5, src2);
	const_aview1d<double, step_ind> xs1(src1, step_ind(5, 2));
	const_aview1d<double, step_ind> xs2(src2, step_ind(5, 2));

	double ra1[] = {6, 9, 7, 13, 10};
	BCS_CHECK( array_view_approx(x1 + x2, ra1, 5) );

	double ra2[] = {6, 5, 13, 18, 5};
	BCS_CHECK( array_view_approx(x1 + xs2, ra2, 5) );

	double ra3[] = {6, 11, 4, 12, 9};
	BCS_CHECK( array_view_approx(xs1 + x2, ra3, 5) );

	double ra4[] = {6, 7, 10, 17, 4};
	BCS_CHECK( array_view_approx(xs1 + xs2, ra4, 5) );

	double ra5[] = {3, 5, 7, 11, 4};
	BCS_CHECK( array_view_approx(x1 + 2.0, ra5, 5) );
	BCS_CHECK( array_view_approx(2.0 + x1, ra5, 5) );

	double ra6[] = {3, 7, 4, 10, 3};
	BCS_CHECK( array_view_approx(xs1 + 2.0, ra6, 5) );
	BCS_CHECK( array_view_approx(2.0 + xs1, ra6, 5) );

	array1d<double> y1(5);
	double y2_buf[10];
	aview1d<double, step_ind> y2(y2_buf, step_ind(5, 2));

	y1 << x1;
	y1 += x2;
	BCS_CHECK( array_view_approx(y1, ra1, 5) );

	y1 << x1;
	y1 += xs2;
	BCS_CHECK( array_view_approx(y1, ra2, 5) );

	y1 << x1;
	y1 += 2.0;
	BCS_CHECK( array_view_approx(y1, ra5, 5) );

	y2 << x1;
	y2 += x2;
	BCS_CHECK( array_view_approx(y2, ra1, 5) );

	y2 << x1;
	y2 += xs2;
	BCS_CHECK( array_view_approx(y2, ra2, 5) );

	y2 << x1;
	y2 += 2.0;
	BCS_CHECK( array_view_approx(y2, ra5, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);
	const_aview2d<double, row_major_t, step_ind, id_ind> X1s(src1, 3, 3, step_ind(2, 2), id_ind(3));
	const_aview2d<double, row_major_t, step_ind, id_ind> X2s(src2, 3, 3, step_ind(2, 2), id_ind(3));

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
	double src1[] = {1, 3, 5, 9, 2, 7};
	double src2[] = {5, 6, 2, 4, 8, 1};

	// 1D

	array1d<double> x1(5, src1);
	array1d<double> x2(5, src2);

	double ra1[] = {-4, -3, 3, 5, -6};
	BCS_CHECK( array_view_approx(x1 - x2, ra1, 5) );

	double ra2[] = {-1, 1, 3, 7, 0};
	BCS_CHECK( array_view_approx(x1 - 2.0, ra2, 5) );

	double ra3[] = {1, -1, -3, -7, 0};
	BCS_CHECK( array_view_approx(2.0 - x1, ra3, 5) );

	array1d<double> y1(5);

	y1 << x1;
	y1 -= x2;
	BCS_CHECK( array_view_approx(y1, ra1, 5) );

	y1 << x1;
	y1 -= 2.0;
	BCS_CHECK( array_view_approx(y1, ra2, 5) );

	y1 << x1;
	be_subtracted(2.0, y1);
	BCS_CHECK( array_view_approx(y1, ra3, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);

	double rb1[] = {-4, -3, 3, 5, -6, 6};
	BCS_CHECK( array_view_approx(X1 - X2, rb1, 2, 3) );

	double rb2[] = {-1, 1, 3, 7, 0, 5};
	BCS_CHECK( array_view_approx(X1 - 2.0, rb2, 2, 3) );

	double rb3[] = {1, -1, -3, -7, 0, -5};
	BCS_CHECK( array_view_approx(2.0 - X1, rb3, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3);

	Y1 << X1;
	Y1 -= X2;
	BCS_CHECK( array_view_approx(Y1, rb1, 2, 3) );

	Y1 << X1;
	Y1 -= 2.0;
	BCS_CHECK( array_view_approx(Y1, rb2, 2, 3) );

	Y1 << X1;
	be_subtracted(2.0, Y1);
	BCS_CHECK( array_view_approx(Y1, rb3, 2, 3) );
}


BCS_TEST_CASE( test_array_mul )
{
	double src1[] = {1, 3, 5, 9, 2, 7};
	double src2[] = {5, 6, 2, 4, 8, 1};

	// 1D

	array1d<double> x1(5, src1);
	array1d<double> x2(5, src2);

	double ra1[] = {5, 18, 10, 36, 16};
	BCS_CHECK( array_view_approx(x1 * x2, ra1, 5) );

	double ra2[] = {2, 6, 10, 18, 4};
	BCS_CHECK( array_view_approx(x1 * 2.0, ra2, 5) );
	BCS_CHECK( array_view_approx(2.0 * x1, ra2, 5) );

	array1d<double> y1(5);

	y1 << x1;
	y1 *= x2;
	BCS_CHECK( array_view_approx(y1, ra1, 5) );

	y1 << x1;
	y1 *= 2.0;
	BCS_CHECK( array_view_approx(y1, ra2, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);

	double rb1[] = {5, 18, 10, 36, 16, 7};
	BCS_CHECK( array_view_approx(X1 * X2, rb1, 2, 3) );

	double rb2[] = {2, 6, 10, 18, 4, 14};
	BCS_CHECK( array_view_approx(X1 * 2.0, rb2, 2, 3) );
	BCS_CHECK( array_view_approx(2.0 * X1, rb2, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3);

	Y1 << X1;
	Y1 *= X2;
	BCS_CHECK( array_view_approx(Y1, rb1, 2, 3) );

	Y1 << X1;
	Y1 *= 2.0;
	BCS_CHECK( array_view_approx(Y1, rb2, 2, 3) );
}



BCS_TEST_CASE( test_array_div )
{
	double src1[] = {1, 3, 5, 9, 2, 7};
	double src2[] = {5, 6, 2, 4, 8, 1};

	// 1D

	array1d<double> x1(5, src1);
	array1d<double> x2(5, src2);

	double ra1[] = {0.2, 0.5, 2.5, 2.25, 0.25};
	BCS_CHECK( array_view_approx(x1 / x2, ra1, 5) );

	double ra2[] = {0.5, 1.5, 2.5, 4.5, 1.0};
	BCS_CHECK( array_view_approx(x1 / 2.0, ra2, 5) );

	double ra3[] = {2, 2.0/3, 0.4, 2.0/9, 1.0};
	BCS_CHECK( array_view_approx(2.0 / x1, ra3, 5) );

	array1d<double> y1(5);

	y1 << x1;
	y1 /= x2;
	BCS_CHECK( array_view_approx(y1, ra1, 5) );

	y1 << x1;
	y1 /= 2.0;
	BCS_CHECK( array_view_approx(y1, ra2, 5) );

	y1 << x1;
	be_divided(2.0, y1);
	BCS_CHECK( array_view_approx(y1, ra3, 5) );

	// 2D

	array2d<double, row_major_t> X1(2, 3, src1);
	array2d<double, row_major_t> X2(2, 3, src2);

	double rb1[] = {0.2, 0.5, 2.5, 2.25, 0.25, 7.0};
	BCS_CHECK( array_view_approx(X1 / X2, rb1, 2, 3) );

	double rb2[] = {0.5, 1.5, 2.5, 4.5, 1.0, 3.5};
	BCS_CHECK( array_view_approx(X1 / 2.0, rb2, 2, 3) );

	double rb3[] = {2, 2.0/3, 0.4, 2.0/9, 1.0, 2.0/7};
	BCS_CHECK( array_view_approx(2.0 / X1, rb3, 2, 3) );

	array2d<double, row_major_t> Y1(2, 3);

	Y1 << X1;
	Y1 /= X2;
	BCS_CHECK( array_view_approx(Y1, rb1, 2, 3) );

	Y1 << X1;
	Y1 /= 2.0;
	BCS_CHECK( array_view_approx(Y1, rb2, 2, 3) );

	Y1 << X1;
	be_divided(2.0, Y1);
	BCS_CHECK( array_view_approx(Y1, rb3, 2, 3) );
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


test_suite *test_array_calc_suite()
{
	test_suite *suite = new test_suite( "test_array_calc" );

	suite->add( new test_array_add() );
	suite->add( new test_array_sub() );
	suite->add( new test_array_mul() );
	suite->add( new test_array_div() );
	suite->add( new test_array_neg() );

	return suite;
}
