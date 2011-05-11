/**
 * @file test_spvec.cpp
 *
 * Unit testing for dynamic spvec and relevant classes
 *
 * @author Dahua Lin
 */



#include <bcslib/test/test_units.h>
#include <bcslib/array/dynamic_sparse_vector.h>

#include <cstdio>

using namespace bcs;
using namespace bcs::test;

template class bcs::dynamic_ordered_spvec<double>;

typedef bcs::dynamic_ordered_spvec<double> dovec;


bool test_dovec(const dovec& vec, size_t n, size_t m, double t, const index_t *sinds, const double *svals)
{
	if (vec.dim0() != (index_t)n) return false;
	if (vec.nactives() != m) return false;
	if (vec.threshold() != t) return false;

	block<bool> exist(n);
	fill_elements(exist.pbase(), n, false);

	for (size_t i = 0; i < m; ++i)
	{
		index_t r_idx = sinds[i];
		double r_val = svals[i];
		indexed_entry<double, index_t> r_pa;
		r_pa.set(r_val, r_idx);

		exist[r_idx] = true;

		if (vec.active_index(i) != r_idx) return false;
		if (vec.active_value(i) != r_val) return false;
		if (vec.active_pair(i) != r_pa) return false;
		if (vec.position_of_index(r_idx) != i) return false;
	}

	for (index_t k = 0; k < (index_t)n; ++k)
	{
		if (vec.has_index(k) != exist[k]) return false;
		if (!exist[k])
		{
			if (vec.position_of_index(k) != vec.nactives()) return false;
		}
	}

	return true;
}



BCS_TEST_CASE( test_dynamic_ordered_spvec_construct )
{
	const size_t n = 20;
	const size_t m = 5;
	index_t inds[m] = {3,  6,  0,  12, 8};
	double vals[m] =  {5., 1., 7., 0., 6.};

	dovec v0(n, m, inds, vals);
	const size_t m0 = 4;
	index_t sinds[m0] = {0,  8,  3,  6};
	double svals[m0] =  {7., 6., 5., 1.};
	BCS_CHECK( test_dovec(v0, n, m0, 0., sinds, svals) );

	dovec v1(n, m, inds, vals, std::greater<double>(), 1.0);
	size_t m1 = 3;
	BCS_CHECK( test_dovec(v1, n, m1, 1.0, sinds, svals) );

}


test_suite *test_dynamic_spvec_suite()
{
	test_suite *suite = new test_suite( "test_dynamic_spvec" );

	suite->add( new test_dynamic_ordered_spvec_construct() );

	return suite;
}

