/**
 * @file test_discrete_distr.cpp
 *
 * Unit testing for discrete distributions
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/prob/discrete_distr.h>
#include <cmath>

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

bool test_vddistr(const variable_discrete_distr& vdd, size_t K, size_t na, const index_t *vs, const double *ws)
{
	if (vdd.K() != (index_t)K) return false;

	// check actives

	if (vdd.nactives() != na) return false;
	for (size_t i = 0; i < na; ++i)
	{
		if (vdd.active_value(i) != vs[i]) return false;
		if (vdd.active_weight(i) != ws[i]) return false;
	}

	// check total_weight

	double tw = 0;
	for (size_t i = 0; i < na; ++i) tw += ws[i];

	// check probabilities

	if (std::abs(vdd.total_weight() - tw) > 1e-12) return false;

	block<double> pb(K);

	double *p = pb.pbase();
	for (size_t k = 0; k < K; ++k) p[k] = 0;
	for (size_t i = 0; i < na; ++i) p[vs[i]] = ws[i] / tw;

	for (size_t k = 0; k < K; ++k)
	{
		if (std::abs(vdd.weight(k) - p[k] * tw) > 1e-12) return false;
		if (std::abs(vdd.p(k) - p[k]) > 1e-12) return false;
	}

	// check avg search length

	double avg_sl = 0;
	for (size_t i = 0; i < na; ++i)
	{
		avg_sl += (ws[i] / tw) * (i+1);
	}
	if (std::abs(vdd.average_search_length() - avg_sl) > 1e-12) return false;

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


BCS_TEST_CASE( test_variable_ddistr_construct )
{
	tbuffer tbuf(8);

	const size_t K1 = 5;
	const index_t vs1[K1] = {0, 1, 2, 3, 4};
	const double ws1[K1] = {2, 2, 2, 2, 2};

	variable_discrete_distr vdd1(K1, 2.0, tbuf);
	BCS_CHECK( test_vddistr(vdd1, K1, K1, vs1, ws1) );


	const size_t K2 = 5;
	const double ws2_in[K2] = {3, 1, 2, 5, 4};
	const index_t vs2[K2] = {3, 4, 0, 2, 1};
	const double ws2[K2] = {5, 4, 3, 2, 1};

	variable_discrete_distr vdd2(K2, ws2_in, tbuf);
	BCS_CHECK( test_vddistr(vdd2, K2, K2, vs2, ws2) );


	const size_t K3 = 10;
	const size_t na3 = 4;
	const index_t vs3_in[na3] = {4, 7, 2, 9};
	const double  ws3_in[na3] = {8, 3, 2, 4};
	const index_t vs3[na3]    = {4, 9, 7, 2};
	const double  ws3[na3]    = {8, 4, 3, 2};

	variable_discrete_distr vdd3(K3, na3, vs3_in, ws3_in);
	BCS_CHECK( test_vddistr(vdd3, K3, na3, vs3, ws3) );
}


BCS_TEST_CASE( test_variable_ddistr_sampling )
{
	tbuffer tbuf(2);

	const size_t K = 4;
	const double w0[K] = {25, 5, 50, 20};
	variable_discrete_distr vdistr0(K, w0, tbuf);

	const index_t vs[K] = {2, 0, 3, 1};
	const double ws[K] = {50, 25, 20, 5};
	BCS_CHECK( test_vddistr(vdistr0, K, K, vs, ws) );

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

	index_t ss_r[N] = {
			2, 2, 0, 3,
			2, 2, 0, 1,
			2, 2, 0,
			2, 2, 3,
			2, 0, 3,
			2, 0, 3
	};

	rs0.reset();
	for (size_t i = 0; i < N; ++i)
	{
		index_t sv = vdistr0.direct_sample(rs0);
		BCS_CHECK_EQUAL(sv, ss_r[i]);
	}
}


BCS_TEST_CASE( test_variable_ddistr_updating )
{
	const size_t K = 10;
	const size_t na0 = 4;
	const index_t vs0_in[na0] = {4, 7, 2, 9};
	const double  ws0_in[na0] = {8, 3, 2, 4};
	const index_t vs0[na0]    = {4, 9, 7, 2};
	const double  ws0[na0]    = {8, 4, 3, 2};

	variable_discrete_distr vdd(K, na0, vs0_in, ws0_in);
	BCS_CHECK( test_vddistr(vdd, K, na0, vs0, ws0) );

	vdd.set_weight(7, 0);
	const size_t na1 = 3;
	const index_t vs1[na1] = {4, 9, 2};
	const double  ws1[na1] = {8, 4, 2};
	BCS_CHECK( test_vddistr(vdd, K, na1, vs1, ws1) );

	vdd.set_weight(2, 0);
	const size_t na2 = 2;
	const index_t vs2[na2] = {4, 9};
	const double  ws2[na2] = {8, 4};
	BCS_CHECK( test_vddistr(vdd, K, na2, vs2, ws2) );

	vdd.set_weight(9, 10);
	const size_t na3 = 2;
	const index_t vs3[na3] = {9, 4};
	const double  ws3[na3] = {10, 8};
	BCS_CHECK( test_vddistr(vdd, K, na3, vs3, ws3) );

	vdd.set_weight(5, 3);
	const size_t na4 = 3;
	const index_t vs4[na4] = {9, 4, 5};
	const double ws4[na4] = {10, 8, 3};
	BCS_CHECK( test_vddistr(vdd, K, na4, vs4, ws4) );

	vdd.set_weight(6, 20);
	const size_t na5 = 4;
	const index_t vs5[na5] = {6, 9, 4, 5};
	const double ws5[na5] = {20, 10, 8, 3};
	BCS_CHECK( test_vddistr(vdd, K, na5, vs5, ws5) );

}


test_suite *test_discrete_distr_suite()
{
	test_suite *suite = new test_suite( "test_discrete_distr" );

	suite->add( new test_discrete_distr_construct() );
	suite->add( new test_discrete_distr_direct_sampling() );
	suite->add( new test_discrete_sampler() );

	suite->add( new test_variable_ddistr_construct() );
	suite->add( new test_variable_ddistr_sampling() );
	suite->add( new test_variable_ddistr_updating() );

	return suite;
}







