/**
 * @file test_gr_edgelist.cpp
 *
 * Unit Tests of the graph edgelist
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/graph/gr_edgelist.h>
#include <bcslib/graph/bgl_port.h>

using namespace bcs;
using namespace bcs::test;


// explicit template instantiation for syntax checking

template class bcs::gr_edgelist<bcs::gr_directed>;
template class bcs::gr_edgelist<bcs::gr_undirected>;
template class bcs::gr_wedgelist<double, bcs::gr_directed>;
template class bcs::gr_wedgelist<double, bcs::gr_undirected>;

template class boost::graph_traits<bcs::gr_edgelist<bcs::gr_directed> >;
template class boost::graph_traits<bcs::gr_wedgelist<double, bcs::gr_directed> >;


// auxiliary checking function


BCS_TEST_CASE( test_gr_edgelist )
{
	// prepare graph

	const gr_size_t nv = 4;
	const gr_size_t ne = 5;
	const gr_index_t src_inds[ne] = {0, 0, 1, 1, 2};
	const gr_index_t tar_inds[ne] = {1, 2, 2, 3, 3};

	const gr_index_t v_inds[nv] = {0, 1, 2, 3};
	const gr_index_t e_inds[ne] = {0, 1, 2, 3, 4};

	const vertex_t* all_vs = (const vertex_t*)(v_inds);
	const edge_t* all_es = (const edge_t*)(e_inds);

	const vertex_t* src_vs = (const vertex_t*)(src_inds);
	const vertex_t* tar_vs = (const vertex_t*)(tar_inds);

	// for construction with ref

	gr_edgelist<gr_directed> g1r(nv, ne, ref_t(), src_vs, tar_vs);

	BCS_CHECK_EQUAL( g1r.nvertices(), nv );
	BCS_CHECK_EQUAL( g1r.nedges(), ne );

	for (gr_index_t i = 0; i < (gr_index_t)ne; ++i)
	{
		BCS_CHECK_EQUAL(g1r.source_of(edge_t(i)), src_vs[i]);
		BCS_CHECK_EQUAL(g1r.target_of(edge_t(i)), tar_vs[i]);
		BCS_CHECK_EQUAL(g1r.get_edge_i(edge_t(i)), edge_i(src_vs[i], tar_vs[i]));
	}

	BCS_CHECK( collection_equal(g1r.v_begin(), g1r.v_end(), all_vs, nv) );
	BCS_CHECK( collection_equal(g1r.e_begin(), g1r.e_end(), all_es, ne) );


	// for construction with clone

	gr_edgelist<gr_directed> g1c(nv, ne, clone_t(), src_vs, tar_vs);

	BCS_CHECK_EQUAL( g1c.nvertices(), nv );
	BCS_CHECK_EQUAL( g1c.nedges(), ne );

	for (gr_index_t i = 0; i < (gr_index_t)ne; ++i)
	{
		BCS_CHECK_EQUAL(g1c.source_of(edge_t(i)), src_vs[i]);
		BCS_CHECK_EQUAL(g1c.target_of(edge_t(i)), tar_vs[i]);
		BCS_CHECK_EQUAL(g1c.get_edge_i(edge_t(i)), edge_i(src_vs[i], tar_vs[i]));
	}

	BCS_CHECK( collection_equal(g1c.v_begin(), g1c.v_end(), all_vs, nv) );
	BCS_CHECK( collection_equal(g1c.e_begin(), g1c.e_end(), all_es, ne) );

}


BCS_TEST_CASE( test_gr_wedgelist )
{
	// prepare graph

	const gr_size_t nv = 4;
	const gr_size_t ne = 5;
	const gr_index_t src_inds[ne] = {0, 0, 1, 1, 2};
	const gr_index_t tar_inds[ne] = {1, 2, 2, 3, 3};
	const double ws[ne] = {0.3, 0.2, 0.5, 0.1, 0.4};

	const gr_index_t v_inds[nv] = {0, 1, 2, 3};
	const gr_index_t e_inds[ne] = {0, 1, 2, 3, 4};

	const vertex_t* all_vs = (const vertex_t*)(v_inds);
	const edge_t* all_es = (const edge_t*)(e_inds);

	const vertex_t* src_vs = (const vertex_t*)(src_inds);
	const vertex_t* tar_vs = (const vertex_t*)(tar_inds);

	// for construction with ref

	gr_wedgelist<double, gr_directed> g1r(nv, ne, ref_t(), src_vs, tar_vs, ws);

	BCS_CHECK_EQUAL( g1r.nvertices(), nv );
	BCS_CHECK_EQUAL( g1r.nedges(), ne );

	for (gr_index_t i = 0; i < (gr_index_t)ne; ++i)
	{
		BCS_CHECK_EQUAL(g1r.source_of(edge_t(i)), src_vs[i]);
		BCS_CHECK_EQUAL(g1r.target_of(edge_t(i)), tar_vs[i]);
		BCS_CHECK_EQUAL(g1r.weight_of(edge_t(i)), ws[i]);
		BCS_CHECK_EQUAL(g1r.get_edge_i(edge_t(i)), make_edge_i(src_vs[i], tar_vs[i]));
		BCS_CHECK_EQUAL(g1r.get_wedge_i(edge_t(i)), make_edge_i(src_vs[i], tar_vs[i], ws[i]));
	}

	BCS_CHECK( collection_equal(g1r.v_begin(), g1r.v_end(), all_vs, nv) );
	BCS_CHECK( collection_equal(g1r.e_begin(), g1r.e_end(), all_es, ne) );


	// for construction with clone

	gr_wedgelist<double, gr_directed> g1c(nv, ne, clone_t(), src_vs, tar_vs, ws);

	BCS_CHECK_EQUAL( g1c.nvertices(), nv );
	BCS_CHECK_EQUAL( g1c.nedges(), ne );

	for (gr_index_t i = 0; i < (gr_index_t)ne; ++i)
	{
		BCS_CHECK_EQUAL(g1c.source_of(edge_t(i)), src_vs[i]);
		BCS_CHECK_EQUAL(g1c.target_of(edge_t(i)), tar_vs[i]);
		BCS_CHECK_EQUAL(g1c.weight_of(edge_t(i)), ws[i]);
		BCS_CHECK_EQUAL(g1c.get_edge_i(edge_t(i)), edge_i(src_vs[i], tar_vs[i]));
		BCS_CHECK_EQUAL(g1r.get_wedge_i(edge_t(i)), make_edge_i(src_vs[i], tar_vs[i], ws[i]));
	}

	BCS_CHECK( collection_equal(g1c.v_begin(), g1c.v_end(), all_vs, nv) );
	BCS_CHECK( collection_equal(g1c.e_begin(), g1c.e_end(), all_es, ne) );

}





test_suite *test_gr_edgelist_suite()
{
	test_suite *suite = new test_suite( "test_gr_edgelist" );

	suite->add( new test_gr_edgelist() );
	suite->add( new test_gr_wedgelist() );

	return suite;
}


