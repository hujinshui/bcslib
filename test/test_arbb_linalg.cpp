/**
 * @file test_arbb_linalg.cpp
 *
 * Unit testing of ArBB based linear algebraic functions
 *
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/test/test_arbb_aux.h>
#include <bcslib/array/arbb_linalg.h>

#include <bcslib/base/mathfun.h>
#include <bcslib/array/array_eval.h>

using namespace bcs;
using namespace bcs::test;


// simple implementation (for purpose of checking)

struct sim_norm_L1
{
	typedef double result_type;
	double operator()(size_t n, const double *x) const
	{
		double s = 0;
		for (size_t i = 0; i < n; ++i)
		{
			s += math::abs(x[i]);
		}
		return s;
	}
};

struct sim_sqrsum
{
	typedef double result_type;
	double operator()(size_t n, const double *x) const
	{
		double s = 0;
		for (size_t i = 0; i < n; ++i)
		{
			s += math::sqr(x[i]);
		}
		return s;
	}
};


struct sim_norm_L2
{
	typedef double result_type;
	double operator()(size_t n, const double *x) const
	{
		return std::sqrt(sim_sqrsum()(n, x));
	}
};

struct sim_norm_Linf
{
	typedef double result_type;
	double operator()(size_t n, const double *x) const
	{
		double s = 0;
		for (size_t i = 0; i < n; ++i)
		{
			double v = std::abs(x[i]);
			if (v > s) s = v;
		}
		return s;
	}
};



BCS_TEST_CASE( test_arbb_norms )
{
	using arbb::f64;
	using arbb::dense;

	const size_t N = 6;
	const size_t m = 2;
	const size_t n = 3;
	double src[6] = {3, 1, 4, 6, 2, 5};

	dense<f64, 1> avec; arbb::bind(avec, src, N);
	dense<f64, 2> amat; arbb::bind(amat, src, n, m);

	caview1d<double> vec(src, N);
	caview2d<double, row_major_t> mat(src, m, n);

	// norm_L1

	f64 vnrm_L1 = norm_L1(avec);
	f64 mnrm_L1 = norm_L1(amat);

	BCS_CHECK( test_scalar_approx(vnrm_L1, vreduce(sim_norm_L1(), vec) ) );
	BCS_CHECK( test_scalar_approx(mnrm_L1, vreduce(sim_norm_L1(), mat) ) );

	dense<f64, 1> rnrm_L1 = norm_L1(amat, 0u);
	dense<f64, 1> cnrm_L1 = norm_L1(amat, 1u);

	array1d<double> rnrm_L1v = vreduce(sim_norm_L1(), per_row(), mat);
	array1d<double> cnrm_L1v = vreduce(sim_norm_L1(), per_col(), mat);

	BCS_CHECK( test_dense_size(rnrm_L1, m) );
	BCS_CHECK( test_dense_size(cnrm_L1, n) );
	BCS_CHECK( test_dense_approx(rnrm_L1, rnrm_L1v.pbase()) );
	BCS_CHECK( test_dense_approx(cnrm_L1, cnrm_L1v.pbase()) );

	// sqrsum

	f64 vsqrsum = sqrsum(avec);
	f64 msqrsum = sqrsum(amat);

	BCS_CHECK( test_scalar_approx(vsqrsum, vreduce(sim_sqrsum(), vec) ) );
	BCS_CHECK( test_scalar_approx(msqrsum, vreduce(sim_sqrsum(), mat) ) );

	dense<f64, 1> rsqrsum = sqrsum(amat, 0u);
	dense<f64, 1> csqrsum = sqrsum(amat, 1u);

	array1d<double> rsqrsumv = vreduce(sim_sqrsum(), per_row(), mat);
	array1d<double> csqrsumv = vreduce(sim_sqrsum(), per_col(), mat);

	BCS_CHECK( test_dense_size(rsqrsum, m) );
	BCS_CHECK( test_dense_size(csqrsum, n) );
	BCS_CHECK( test_dense_approx(rsqrsum, rsqrsumv.pbase()) );
	BCS_CHECK( test_dense_approx(csqrsum, csqrsumv.pbase()) );

	// norm_L1

	f64 vnrm_L2 = norm_L2(avec);
	f64 mnrm_L2 = norm_L2(amat);

	BCS_CHECK( test_scalar_approx(vnrm_L2, vreduce(sim_norm_L2(), vec) ) );
	BCS_CHECK( test_scalar_approx(mnrm_L2, vreduce(sim_norm_L2(), mat) ) );

	dense<f64, 1> rnrm_L2 = norm_L2(amat, 0u);
	dense<f64, 1> cnrm_L2 = norm_L2(amat, 1u);

	array1d<double> rnrm_L2v = vreduce(sim_norm_L2(), per_row(), mat);
	array1d<double> cnrm_L2v = vreduce(sim_norm_L2(), per_col(), mat);

	BCS_CHECK( test_dense_size(rnrm_L2, m) );
	BCS_CHECK( test_dense_size(cnrm_L2, n) );
	BCS_CHECK( test_dense_approx(rnrm_L2, rnrm_L2v.pbase()) );
	BCS_CHECK( test_dense_approx(cnrm_L2, cnrm_L2v.pbase()) );

	// norm_Linf

	f64 vnrm_Linf = norm_Linf(avec);
	f64 mnrm_Linf = norm_Linf(amat);

	BCS_CHECK( test_scalar_approx(vnrm_Linf, vreduce(sim_norm_Linf(), vec) ) );
	BCS_CHECK( test_scalar_approx(mnrm_Linf, vreduce(sim_norm_Linf(), mat) ) );

	dense<f64, 1> rnrm_Linf = norm_Linf(amat, 0u);
	dense<f64, 1> cnrm_Linf = norm_Linf(amat, 1u);

	array1d<double> rnrm_Linfv = vreduce(sim_norm_Linf(), per_row(), mat);
	array1d<double> cnrm_Linfv = vreduce(sim_norm_Linf(), per_col(), mat);

	BCS_CHECK( test_dense_size(rnrm_Linf, m) );
	BCS_CHECK( test_dense_size(cnrm_Linf, n) );
	BCS_CHECK( test_dense_approx(rnrm_Linf, rnrm_Linfv.pbase()) );
	BCS_CHECK( test_dense_approx(cnrm_Linf, cnrm_Linfv.pbase()) );

}



std::shared_ptr<test_suite> test_arbb_linalg_suite()
{
	BCS_NEW_TEST_SUITE( suite, "test_arbb_linalg" );

	BCS_ADD_TEST_CASE( suite, test_arbb_norms() );

	return suite;
}
