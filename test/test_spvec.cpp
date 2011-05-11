/**
 * @file test_spvec.cpp
 *
 * Unit testing for spvec and relevant classes
 * 
 * @author Dahua Lin
 */



#include <bcslib/test/test_units.h>
#include <bcslib/array/sparse_vector.h>

#include <iostream>

using namespace bcs;
using namespace bcs::test;

template class bcs::spvec_cview<double>;
template class bcs::spvec_view<double>;
template class bcs::spvec<double>;


bool test_spv_pattern(const spvec_pattern_cview& pat, size_t n, size_t na, const index_t *src_inds)
{
	if (pat.dim0() != (index_t)n) return false;
	if (pat.nelems() != n) return false;
	if (pat.nactives() != na) return false;
	if (pat.shape() != arr_shape((index_t)n)) return false;

	const index_t *ainds = pat.active_indices();

	for (size_t i = 0; i < na; ++i)
	{
		if (ainds[i] != src_inds[i]) return false;
		if (pat.get(i) != src_inds[i]) return false;
	}

	return true;
}

template<typename T>
bool test_spv(const spvec_cview<T>& sv, size_t n, size_t na, const index_t *src_inds, const T *src_vals)
{
	if ( !test_spv_pattern(sv.pattern(), n, na, src_inds) ) return false;

	if (sv.dim0() != (index_t)n) return false;
	if (sv.nelems() != n) return false;
	if (sv.nactives() != na) return false;
	if (sv.shape() != arr_shape((index_t)n)) return false;

	const index_t *ainds = sv.active_indices();
	const T *avals = sv.active_values();

	for (size_t i = 0; i < na; ++i)
	{
		if (ainds[i] != src_inds[i]) return false;
		if (avals[i] != src_vals[i]) return false;
		if (sv.active_index(i) != src_inds[i]) return false;
		if (sv.active_value(i) != src_vals[i]) return false;

		indexed_entry<T, index_t> ref_pair;
		ref_pair.value = src_vals[i];
		ref_pair.index = src_inds[i];

		if (sv.active_pair(i) != ref_pair) return false;
	}

	return true;
}




BCS_TEST_CASE( test_spvec_patterns )
{
	const size_t n = 20;
	const size_t na = 3;
	index_t inds[na] = {4, 8, 11};

	spvec_pattern_cview pat_cv(n, na, inds);
	BCS_CHECK( test_spv_pattern(pat_cv, n, na, inds) );
	BCS_CHECK( pat_cv.active_indices() == inds );

	spvec_pattern pat1(n, na, inds);
	BCS_CHECK( test_spv_pattern(pat1, n, na, inds) );
	BCS_CHECK( pat1.active_indices() != inds );

	spvec_pattern pat2(pat_cv);
	BCS_CHECK( test_spv_pattern(pat2, n, na, inds) );
	BCS_CHECK( pat2.active_indices() != inds );

	block<index_t>* pblk3 = new block<index_t>(na, inds);
	spvec_pattern pat3(n, pblk3);
	BCS_CHECK( test_spv_pattern(pat3, n, na, inds) );
	BCS_CHECK( pat3.active_indices() == pblk3->pbase() );

	spvec_pattern_cview pat4(pat3);
	BCS_CHECK( test_spv_pattern(pat4, n, na, inds) );
	BCS_CHECK( pat4.active_indices() == pat3.active_indices() );

	spvec_pattern pat5 = make_copy(pat1);
	BCS_CHECK( test_spv_pattern(pat_cv, n, na, inds) );
	BCS_CHECK( pat5.active_indices() != inds );

}


BCS_TEST_CASE( test_spvecs )
{
	const size_t n = 20;
	const size_t na = 3;
	index_t inds[na] = {4, 8, 11};
	double vals[na] = {0.34, 0.12, 0.57};

	spvec_pattern_cview pat0(n, na, inds);
	spvec_pattern pat1(n, na, inds);
	BCS_CHECK( pat0.active_indices() == inds );
	BCS_CHECK( pat1.active_indices() != inds );

	spvec_view<double> sv1(pat0, vals);
	BCS_CHECK( test_spv(sv1, n, na, inds, vals) );
	BCS_CHECK( sv1.active_indices() == inds );
	BCS_CHECK( sv1.active_values() == vals );

	spvec<double> sv2(pat0);
	BCS_CHECK( test_spv_pattern(sv2.pattern(), n, na, inds) );
	BCS_CHECK( sv2.active_indices() != inds );

	spvec<double> sv3(pat1);
	BCS_CHECK( test_spv_pattern(sv3.pattern(), n, na, inds) );
	BCS_CHECK( sv3.active_indices() == pat1.active_indices() );

	spvec<double> sv4(pat0, vals);
	BCS_CHECK( test_spv(sv4, n, na, inds, vals) );
	BCS_CHECK( sv4.active_indices() != inds );
	BCS_CHECK( sv4.active_values() != vals );

	spvec<double> sv5(pat1, vals);
	BCS_CHECK( test_spv(sv5, n, na, inds, vals) );
	BCS_CHECK( sv5.active_indices() == pat1.active_indices() );
	BCS_CHECK( sv5.active_values() != vals );

	spvec<double> sv6(sv4);
	BCS_CHECK( test_spv(sv6, n, na, inds, vals) );
	BCS_CHECK( sv6.active_indices() == sv4.active_indices() );
	BCS_CHECK( sv6.active_values() == sv4.active_values() );

	spvec<double> sv7(pat0, vals);
	sv7 = sv4;
	BCS_CHECK( test_spv(sv7, n, na, inds, vals) );
	BCS_CHECK( sv7.active_indices() == sv4.active_indices() );
	BCS_CHECK( sv7.active_values() == sv4.active_values() );

	spvec<double> sv8 = sv4.make_copy_shared_pattern();
	BCS_CHECK( test_spv(sv8, n, na, inds, vals) );
	BCS_CHECK( sv8.active_indices() == sv4.active_indices() );
	BCS_CHECK( sv8.active_values() != sv4.active_values() );

	spvec<double> sv9 = make_copy(sv4);
	BCS_CHECK( test_spv(sv9, n, na, inds, vals) );
	BCS_CHECK( sv9.active_indices() != sv4.active_indices() );
	BCS_CHECK( sv9.active_values() != sv4.active_values() );
}



test_suite *test_spvec_suite()
{
	test_suite *suite = new test_suite( "test_spvec" );

	suite->add( new test_spvec_patterns() );
	suite->add( new test_spvecs() );

	return suite;
}
