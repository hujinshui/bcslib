/**
 * @file test_array_eval.cpp
 *
 * Unit testing for array evaluation
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/test_units.h>
#include <bcslib/array/array_eval.h>

#include <iostream>

using namespace bcs;
using namespace bcs::test;


BCS_TEST_CASE( test_array_sum )
{
	double src[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

	// 1D

	array1d<double> x(5, src);
	BCS_CHECK_APPROX( sum(x),  15.0 );

	const_aview1d<double, step_ind> xs(src, step_ind(5, 2));
	BCS_CHECK_APPROX( sum(xs), 25.0 );

	// 2D

	array2d<double, row_major_t> Xr(2, 3, src);
	double Xr_re[] = {6, 15};
	double Xr_ce[] = {5, 7, 9};

	BCS_CHECK_APPROX( sum(Xr), 21.0 );
	BCS_CHECK( array_view_approx(row_sum(Xr), Xr_re, 2) );
	BCS_CHECK( array_view_approx(column_sum(Xr), Xr_ce, 3) );

	array2d<double, column_major_t> Xc(2, 3, src);
	double Xc_re[] = {9, 12};
	double Xc_ce[] = {3, 7, 11};

	BCS_CHECK_APPROX( sum(Xc), 21.0 );
	BCS_CHECK( array_view_approx(row_sum(Xc), Xc_re, 2) );
	BCS_CHECK( array_view_approx(column_sum(Xc), Xc_ce, 3) );

	const_aview2d<double, row_major_t, step_ind, id_ind> Xrs(src, 3, 3, step_ind(2, 2), id_ind(3));
	double Xrs_re[] = {6, 24};
	double Xrs_ce[] = {8, 10, 12};

	BCS_CHECK_APPROX( sum(Xrs), 30.0 );
	BCS_CHECK( array_view_approx(row_sum(Xrs), Xrs_re, 2) );
	BCS_CHECK( array_view_approx(column_sum(Xrs), Xrs_ce, 3) );

	const_aview2d<double, column_major_t, step_ind, id_ind> Xcs(src, 3, 3, step_ind(2, 2), id_ind(3));
	double Xcs_re[] = {12, 18};
	double Xcs_ce[] = {4, 10, 16};

	BCS_CHECK_APPROX( sum(Xcs), 30.0 );
	BCS_CHECK( array_view_approx(row_sum(Xcs), Xcs_re, 2) );
	BCS_CHECK( array_view_approx(column_sum(Xcs), Xcs_ce, 3) );
}


BCS_TEST_CASE( test_array_mean )
{
	double src[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

	// 1D

	array1d<double> x(5, src);
	BCS_CHECK_APPROX( mean(x),  3.0 );

	// 2D

	array2d<double, row_major_t> Xr(2, 3, src);
	double Xr_re[] = {2, 5};
	double Xr_ce[] = {2.5, 3.5, 4.5};

	BCS_CHECK_APPROX( mean(Xr), 3.5 );
	BCS_CHECK( array_view_approx(row_mean(Xr), Xr_re, 2) );
	BCS_CHECK( array_view_approx(column_mean(Xr), Xr_ce, 3) );

	array2d<double, column_major_t> Xc(2, 3, src);
	double Xc_re[] = {3, 4};
	double Xc_ce[] = {1.5, 3.5, 5.5};

	BCS_CHECK_APPROX( mean(Xc), 3.5 );
	BCS_CHECK( array_view_approx(row_mean(Xc), Xc_re, 2) );
	BCS_CHECK( array_view_approx(column_mean(Xc), Xc_ce, 3) );
}



BCS_TEST_CASE( test_array_prod )
{
	double src[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

	// 1D

	array1d<double> x(5, src);
	BCS_CHECK_APPROX( prod(x),  120.0 );

	// 2D

	array2d<double, row_major_t> Xr(2, 3, src);
	double Xr_re[] = {6, 120};
	double Xr_ce[] = {4, 10, 18};

	BCS_CHECK_APPROX( prod(Xr), 720.0 );
	BCS_CHECK( array_view_approx(row_prod(Xr), Xr_re, 2) );
	BCS_CHECK( array_view_approx(column_prod(Xr), Xr_ce, 3) );

	array2d<double, column_major_t> Xc(2, 3, src);
	double Xc_re[] = {15, 48};
	double Xc_ce[] = {2, 12, 30};

	BCS_CHECK_APPROX( prod(Xc), 720.0 );
	BCS_CHECK( array_view_approx(row_prod(Xc), Xc_re, 2) );
	BCS_CHECK( array_view_approx(column_prod(Xc), Xc_ce, 3) );
}




BCS_TEST_CASE( test_array_max_min )
{
	double src[6] = {7, 3, 2, 9, 8, 1};

	array1d<double> x(5, src);
	array2d<double, row_major_t> Xr(2, 3, src);
	array2d<double, column_major_t> Xc(2, 3, src);

	// max

	BCS_CHECK_EQUAL( max(x), 9 );
	BCS_CHECK_EQUAL( max(Xr), 9 );
	BCS_CHECK_EQUAL( max(Xc), 9 );

	index_t Xr_rmax_i[] = {0, 0};
	double Xr_rmax[] = {7, 9};
	BCS_CHECK( array_view_equal(row_max(Xr), Xr_rmax, 2) );
	index_t Xr_cmax_i[] = {1, 1, 0};
	double Xr_cmax[] = {9, 8, 2};
	BCS_CHECK( array_view_equal(column_max(Xr), Xr_cmax, 3) );

	index_t Xc_rmax_i[] = {2, 1};
	double Xc_rmax[] = {8, 9};
	BCS_CHECK( array_view_equal(row_max(Xc), Xc_rmax, 2) );
	index_t Xc_cmax_i[] = {0, 1, 0};
	double Xc_cmax[] = {7, 9, 8};
	BCS_CHECK( array_view_equal(column_max(Xc), Xc_cmax, 3) );


	// min

	BCS_CHECK_EQUAL( min(x), 2 );
	BCS_CHECK_EQUAL( min(Xr), 1 );
	BCS_CHECK_EQUAL( min(Xc), 1 );

	index_t Xr_rmin_i[] = {2, 2};
	double Xr_rmin[] = {2, 1};
	BCS_CHECK( array_view_equal(row_min(Xr), Xr_rmin, 2) );
	index_t Xr_cmin_i[] = {0, 0, 1};
	double Xr_cmin[] = {7, 3, 1};
	BCS_CHECK( array_view_equal(column_min(Xr), Xr_cmin, 3) );

	index_t Xc_rmin_i[] = {1, 2};
	double Xc_rmin[] = {2, 1};
	BCS_CHECK( array_view_equal(row_min(Xc), Xc_rmin, 2) );
	index_t Xc_cmin_i[] = {1, 0, 1};
	double Xc_cmin[] = {3, 2, 1};
	BCS_CHECK( array_view_equal(column_min(Xc), Xc_cmin, 3) );


	// minmax

	BCS_CHECK_EQUAL( minmax(x), std::make_pair(2.0, 9.0) );
	BCS_CHECK_EQUAL( minmax(Xr), std::make_pair(1.0, 9.0) );
	BCS_CHECK_EQUAL( minmax(Xc), std::make_pair(1.0, 9.0) );

	std::pair<array1d<double>, array1d<double> > Xr_rmm = unzip(row_minmax(Xr));
	BCS_CHECK( array_view_equal(Xr_rmm.first,  Xr_rmin, 2) );
	BCS_CHECK( array_view_equal(Xr_rmm.second, Xr_rmax, 2) );

	std::pair<array1d<double>, array1d<double> > Xr_cmm = unzip(column_minmax(Xr));
	BCS_CHECK( array_view_equal(Xr_cmm.first,  Xr_cmin, 3) );
	BCS_CHECK( array_view_equal(Xr_cmm.second, Xr_cmax, 3) );

	// index_max

	BCS_CHECK_EQUAL( index_max(x),  std::make_pair((index_t)3, 9.0) );
	BCS_CHECK_EQUAL( index_max(Xr), std::make_pair(index_pair(1, 0), 9.0) );
	BCS_CHECK_EQUAL( index_max(Xc), std::make_pair(index_pair(1, 1), 9.0) );

	std::pair<array1d<index_t>, array1d<double> > Xr_rmax_iv = unzip(row_index_max(Xr));
	BCS_CHECK( array_view_equal(Xr_rmax_iv.first, Xr_rmax_i, 2) );
	BCS_CHECK( array_view_equal(Xr_rmax_iv.second, Xr_rmax, 2) );

	std::pair<array1d<index_t>, array1d<double> > Xr_cmax_iv = unzip(column_index_max(Xr));
	BCS_CHECK( array_view_equal(Xr_cmax_iv.first, Xr_cmax_i, 3) );
	BCS_CHECK( array_view_equal(Xr_cmax_iv.second, Xr_cmax, 3) );

	std::pair<array1d<index_t>, array1d<double> > Xc_rmax_iv = unzip(row_index_max(Xc));
	BCS_CHECK( array_view_equal(Xc_rmax_iv.first, Xc_rmax_i, 2) );
	BCS_CHECK( array_view_equal(Xc_rmax_iv.second, Xc_rmax, 2) );

	std::pair<array1d<index_t>, array1d<double> > Xc_cmax_iv = unzip(column_index_max(Xc));
	BCS_CHECK( array_view_equal(Xc_cmax_iv.first, Xc_cmax_i, 3) );
	BCS_CHECK( array_view_equal(Xc_cmax_iv.second, Xc_cmax, 3) );


	// index_min

	BCS_CHECK_EQUAL( index_min(x),  std::make_pair((index_t)2, 2.0) );
	BCS_CHECK_EQUAL( index_min(Xr), std::make_pair(index_pair(1, 2), 1.0) );
	BCS_CHECK_EQUAL( index_min(Xc), std::make_pair(index_pair(1, 2), 1.0) );

	std::pair<array1d<index_t>, array1d<double> > Xr_rmin_iv = unzip(row_index_min(Xr));
	BCS_CHECK( array_view_equal(Xr_rmin_iv.first, Xr_rmin_i, 2) );
	BCS_CHECK( array_view_equal(Xr_rmin_iv.second, Xr_rmin, 2) );

	std::pair<array1d<index_t>, array1d<double> > Xr_cmin_iv = unzip(column_index_min(Xr));
	BCS_CHECK( array_view_equal(Xr_cmin_iv.first, Xr_cmin_i, 3) );
	BCS_CHECK( array_view_equal(Xr_cmin_iv.second, Xr_cmin, 3) );

	std::pair<array1d<index_t>, array1d<double> > Xc_rmin_iv = unzip(row_index_min(Xc));
	BCS_CHECK( array_view_equal(Xc_rmin_iv.first, Xc_rmin_i, 2) );
	BCS_CHECK( array_view_equal(Xc_rmin_iv.second, Xc_rmin, 2) );

	std::pair<array1d<index_t>, array1d<double> > Xc_cmin_iv = unzip(column_index_min(Xc));
	BCS_CHECK( array_view_equal(Xc_cmin_iv.first, Xc_cmin_i, 3) );
	BCS_CHECK( array_view_equal(Xc_cmin_iv.second, Xc_cmin, 3) );
}



BCS_TEST_CASE( test_array_L1norm )
{
	double src[9] = {1, -2, 3, -4, 5, -6, 7, -8, 9};

	// 1D

	array1d<double> x(5, src);
	BCS_CHECK_APPROX( L1norm(x),  15.0 );

	// 2D

	array2d<double, row_major_t> Xr(2, 3, src);
	double Xr_re[] = {6, 15};
	double Xr_ce[] = {5, 7, 9};

	BCS_CHECK_APPROX( L1norm(Xr), 21.0 );
	BCS_CHECK( array_view_approx(row_L1norm(Xr), Xr_re, 2) );
	BCS_CHECK( array_view_approx(column_L1norm(Xr), Xr_ce, 3) );

	array2d<double, column_major_t> Xc(2, 3, src);
	double Xc_re[] = {9, 12};
	double Xc_ce[] = {3, 7, 11};

	BCS_CHECK_APPROX( L1norm(Xc), 21.0 );
	BCS_CHECK( array_view_approx(row_L1norm(Xc), Xc_re, 2) );
	BCS_CHECK( array_view_approx(column_L1norm(Xc), Xc_ce, 3) );
}


BCS_TEST_CASE( test_array_sqrsum_L2norm )
{
	double src[9] = {1, -2, 3, -4, 5, -6, 7, -8, 9};

	// 1D

	array1d<double> x(5, src);
	BCS_CHECK_APPROX( sqr_sum(x), 55.0 );
	BCS_CHECK_APPROX( L2norm(x),  std::sqrt(55.0) );

	// 2D

	array2d<double, row_major_t> Xr(2, 3, src);
	double Xr_re[] = {14, 77};
	double Xr_re_r[] = { std::sqrt(14.0), std::sqrt(77.0) };
	double Xr_ce[] = {17, 29, 45};
	double Xr_ce_r[] = { std::sqrt(17.0), std::sqrt(29.0), std::sqrt(45.0) };

	BCS_CHECK_APPROX( sqr_sum(Xr), 91.0 );
	BCS_CHECK_APPROX( L2norm(Xr), std::sqrt(91.0) );

	BCS_CHECK( array_view_approx(row_sqr_sum(Xr), Xr_re, 2) );
	BCS_CHECK( array_view_approx(row_L2norm(Xr), Xr_re_r, 2) );

	BCS_CHECK( array_view_approx(column_sqr_sum(Xr), Xr_ce, 3) );
	BCS_CHECK( array_view_approx(column_L2norm(Xr), Xr_ce_r, 3) );

	array2d<double, column_major_t> Xc(2, 3, src);
	double Xc_re[] = {35, 56};
	double Xc_re_r[] = { std::sqrt(35.0), std::sqrt(56.0) };
	double Xc_ce[] = {5, 25, 61};
	double Xc_ce_r[] = { std::sqrt(5.0), std::sqrt(25.0), std::sqrt(61.0) };

	BCS_CHECK_APPROX( sqr_sum(Xc), 91.0 );
	BCS_CHECK_APPROX( L2norm(Xc), std::sqrt(91.0) );

	BCS_CHECK( array_view_approx(row_sqr_sum(Xc), Xc_re, 2) );
	BCS_CHECK( array_view_approx(row_L2norm(Xc), Xc_re_r, 2) );

	BCS_CHECK( array_view_approx(column_sqr_sum(Xc), Xc_ce, 3) );
	BCS_CHECK( array_view_approx(column_L2norm(Xc), Xc_ce_r, 3) );
}


BCS_TEST_CASE( test_array_powsum_Lpnorm )
{
	double src[9] = {1, -2, 3, -4, 5, -6, 7, -8, 9};
	int p = 2;

	// 1D

	array1d<double> x(5, src);
	BCS_CHECK_APPROX( pow_sum(x, p), 55.0 );
	BCS_CHECK_APPROX( Lpnorm(x, p),  std::sqrt(55.0) );

	// 2D

	array2d<double, row_major_t> Xr(2, 3, src);
	double Xr_re[] = {14, 77};
	double Xr_re_r[] = { std::sqrt(14.0), std::sqrt(77.0) };
	double Xr_ce[] = {17, 29, 45};
	double Xr_ce_r[] = { std::sqrt(17.0), std::sqrt(29.0), std::sqrt(45.0) };

	BCS_CHECK_APPROX( pow_sum(Xr, p), 91.0 );
	BCS_CHECK_APPROX( Lpnorm(Xr, p), std::sqrt(91.0) );

	BCS_CHECK( array_view_approx(row_pow_sum(Xr, p), Xr_re, 2) );
	BCS_CHECK( array_view_approx(row_Lpnorm(Xr, p), Xr_re_r, 2) );

	BCS_CHECK( array_view_approx(column_pow_sum(Xr, p), Xr_ce, 3) );
	BCS_CHECK( array_view_approx(column_Lpnorm(Xr, p), Xr_ce_r, 3) );

	array2d<double, column_major_t> Xc(2, 3, src);
	double Xc_re[] = {35, 56};
	double Xc_re_r[] = { std::sqrt(35.0), std::sqrt(56.0) };
	double Xc_ce[] = {5, 25, 61};
	double Xc_ce_r[] = { std::sqrt(5.0), std::sqrt(25.0), std::sqrt(61.0) };

	BCS_CHECK_APPROX( pow_sum(Xc, p), 91.0 );
	BCS_CHECK_APPROX( Lpnorm(Xc, p), std::sqrt(91.0) );

	BCS_CHECK( array_view_approx(row_pow_sum(Xc, p), Xc_re, 2) );
	BCS_CHECK( array_view_approx(row_Lpnorm(Xc, p), Xc_re_r, 2) );

	BCS_CHECK( array_view_approx(column_pow_sum(Xc, p), Xc_ce, 3) );
	BCS_CHECK( array_view_approx(column_Lpnorm(Xc, p), Xc_ce_r, 3) );
}


BCS_TEST_CASE( test_array_Linfnorm )
{
	double src[9] = {1, -2, 3, -4, 5, -6, 7, -8, 9};

	// 1D

	array1d<double> x(5, src);
	BCS_CHECK_APPROX( Linfnorm(x),  5.0 );

	// 2D

	array2d<double, row_major_t> Xr(2, 3, src);
	double Xr_re[] = {3, 6};
	double Xr_ce[] = {4, 5, 6};

	BCS_CHECK_APPROX( Linfnorm(Xr), 6.0 );
	BCS_CHECK( array_view_approx(row_Linfnorm(Xr), Xr_re, 2) );
	BCS_CHECK( array_view_approx(column_Linfnorm(Xr), Xr_ce, 3) );

	array2d<double, column_major_t> Xc(2, 3, src);
	double Xc_re[] = {5, 6};
	double Xc_ce[] = {2, 4, 6};

	BCS_CHECK_APPROX( Linfnorm(Xc), 6.0 );
	BCS_CHECK( array_view_approx(row_Linfnorm(Xc), Xc_re, 2) );
	BCS_CHECK( array_view_approx(column_Linfnorm(Xc), Xc_ce, 3) );
}




test_suite *test_array_eval_suite()
{
	test_suite *msuite = new test_suite( "test_array_eval" );

	msuite->add( new test_array_sum() );
	msuite->add( new test_array_mean() );
	msuite->add( new test_array_prod() );
	msuite->add( new test_array_max_min() );
	msuite->add( new test_array_L1norm() );
	msuite->add( new test_array_sqrsum_L2norm() );
	msuite->add( new test_array_powsum_Lpnorm() );
	msuite->add( new test_array_Linfnorm() );

	return msuite;
}




