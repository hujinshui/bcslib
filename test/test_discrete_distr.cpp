/**
 * @file test_discrete_distr.cpp
 *
 * Unit testing for discrete distributions
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/prob/discrete_distr.h>

#include <cstdio>

using namespace bcs;
using namespace bcs::test;

template class discrete_distr<int32_t>;
typedef discrete_distr<int32_t> ddistr;


void print_ddistr(const char *name, const ddistr& d)
{
	std::printf("%s: (K = %d): { ", name, d.K());
	double sum_p = 0;
	for (int k = 0; k < d.K(); ++k)
	{
		std::printf("%.3f ", d.p(k));
		sum_p += d.p(k);
	}
	std::printf("} => sum(p) = %.3f\n", sum_p);
}


bool test_discrete_distr(const ddistr& d, int32_t K, const double *p, double eps=1e-10)
{
	if (d.K() != K) return false;

	for (int32_t k = 0; k < K; ++k)
	{
		if (std::abs(d.p(k) - p[k]) > eps)
			return false;
	}

	return true;
}



BCS_TEST_CASE( discrete_distr_construct )
{
	double p0[3] = {0.2, 0.5, 0.3};
	BCS_CHECK( test_discrete_distr( ddistr(3, p0), 3, p0 ) );

	double p1[5] = {0.2, 0.2, 0.2, 0.2, 0.2};
	double p1buf[5];
	BCS_CHECK( test_discrete_distr( make_uniform_discrete_distr(5, p1buf), 5, p1 ) );

	double e2[4] = {2.0, 3.0, 4.0, 1.0};
	double se2 = 0;
	for (int i = 0; i < 4; ++i) se2 += std::exp(e2[i]);
	double p2[4];
	for (int i = 0; i < 4; ++i) p2[i] = std::exp(e2[i]) / se2;
	double p2buf[4];
	BCS_CHECK( test_discrete_distr( make_discrete_distr_by_nrmexp(4, e2, p2buf), 4, p2 ) );
}


test_suite *test_discrete_distr_suite()
{
	test_suite *suite = new test_suite( "test_discrete_distr" );

	suite->add( new discrete_distr_construct() );

	return suite;
}







