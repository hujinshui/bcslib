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

template class bcs::discrete_distr<int32_t>;
template class bcs::discrete_sampler<int32_t>;

typedef discrete_distr<int32_t> ddistr;
typedef discrete_sampler<int32_t> dsampler;


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


BCS_TEST_CASE( test_discrete_distr_construct )
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


BCS_TEST_CASE( test_discrete_distr_direct_sampling )
{
	const size_t N = 10;
	double u_src[N] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	real_stream_for_diagnosis rs0(N, u_src);

	double p0[3] = {0.2, 0.4, 0.3};
	ddistr distr0(3, p0);

	int32_t ss_r[N] = {0, 0, 1, 1, 1, 1, 2, 2, 2, 3};
	int32_t ss[N];
	rs0.reset();
	for (size_t i = 0; i < N; ++i)
	{
		ss[i] = distr0.direct_sample(rs0);
	}

	BCS_CHECK( collection_equal(ss, ss+N, ss_r, N) );
}


BCS_TEST_CASE( test_discrete_sampler )
{
	const int32_t K = 4;
	const double p0[K] = {0.25, 0.05, 0.50, 0.15};
	ddistr distr0(K, p0);

	const size_t N = 20;

	double u_src[N] = {
			0.05, 0.35, 0.65, 0.95,
			0.10, 0.40, 0.70, 1.00,
			0.15, 0.45, 0.75,
			0.20, 0.50, 0.80,
			0.25, 0.55, 0.85,
			0.30, 0.60, 0.90
	};
	real_stream_for_diagnosis rs0(N, u_src);

	int32_t ss_r[N] = {
			2, 2, 0, 1,
			2, 2, 0, 4,
			2, 2, 0,
			2, 2, 3,
			2, 0, 3,
			2, 0, 3
	};

	dsampler dsp1 = distr0.get_sampler(dsampler::DSAMP_DIRECT_METHOD);

	BCS_CHECK_EQUAL( dsp1.K(), K );
	BCS_CHECK_EQUAL( dsp1.method(), dsampler::DSAMP_DIRECT_METHOD );
	BCS_CHECK_APPROX( dsp1.average_search_length(), 1.85 );

	int32_t ss1[N];
	set_zeros_to_elements(ss1, N);

	rs0.reset();
	dsp1( rs0, N, ss1 );
	BCS_CHECK( collection_equal(ss1, ss1+N, ss_r, N) );

	dsampler dsp2 = distr0.get_sampler(dsampler::DSAMP_SORT_METHOD);

	BCS_CHECK_EQUAL( dsp2.K(), K );
	BCS_CHECK_EQUAL( dsp2.method(), dsampler::DSAMP_SORT_METHOD );
	BCS_CHECK_APPROX( dsp2.average_search_length(), 1.85 );

	int32_t ss2[N];
	set_zeros_to_elements(ss2, N);

	rs0.reset();
	dsp2( rs0, N, ss2 );
	BCS_CHECK( collection_equal(ss2, ss2+N, ss_r, N) );

}







test_suite *test_discrete_distr_suite()
{
	test_suite *suite = new test_suite( "test_discrete_distr" );

	suite->add( new test_discrete_distr_construct() );
	suite->add( new test_discrete_distr_direct_sampling() );
	suite->add( new test_discrete_sampler() );

	return suite;
}







