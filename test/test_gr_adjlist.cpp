/**
 * @file test_gr_adjlist.cpp
 *
 * Unit tests for graph class gr_adjlist
 * 
 * @author Dahua Lin
 */

#include <bcslib/test/test_units.h>
#include <bcslib/graph/gr_adjlist.h>
#include <bcslib/graph/bgl_port.h>

#include <vector>
#include <cstdio>

using namespace bcs;
using namespace bcs::test;

// explicit template instantiation for syntax checking

template class bcs::gr_adjlist<bcs::gr_directed>;
template class bcs::gr_adjlist<bcs::gr_undirected>;
template class bcs::gr_wadjlist<double, bcs::gr_directed>;
template class bcs::gr_wadjlist<double, bcs::gr_undirected>;

template struct boost::graph_traits<bcs::gr_adjlist<bcs::gr_directed> >;
template struct boost::graph_traits<bcs::gr_adjlist<bcs::gr_undirected> >;
template struct boost::graph_traits<bcs::gr_wadjlist<double, bcs::gr_directed> >;
template struct boost::graph_traits<bcs::gr_wadjlist<double, bcs::gr_undirected> >;



// auxiliary functions

bool verify_adjacency(const gr_adjlist<gr_directed>& g, const vertex_t& v)
{
	// collect relevant information

	std::vector<vertex_t> nbs;
	std::vector<edge_t> aes;

	for (gr_index_t i = 0; i < (gr_index_t)g.nedges(); ++i)
	{
		vertex_t s = g.source_of(i);
		vertex_t t = g.target_of(i);
		if (s == v)
		{
			nbs.push_back(t);
			aes.push_back(edge_t(i));
		}
	}

	gr_size_t out_deg = (gr_size_t)nbs.size();

	if (out_deg != g.out_degree(v))
	{
		return false;
	}

	if (out_deg > 0)
	{
		if ( !collection_equal(g.out_neighbor_begin(v), g.out_neighbor_end(v), &(nbs[0]), out_deg) )
		{
			return false;
		}

		if ( !collection_equal(g.out_edge_begin(v), g.out_edge_end(v), &(aes[0]), out_deg) )
		{
			return false;
		}
	}

	return true;
}


bool verify_adjacency(const gr_adjlist<gr_undirected>& g, const vertex_t& v)
{
	// collect relevant information

	std::vector<vertex_t> nbs;
	std::vector<edge_t> aes;

	gr_size_t ne = g.nedges();

	for (gr_index_t i = 0; i < (gr_index_t)ne; ++i)
	{
		vertex_t s = g.source_of(i);
		vertex_t t = g.target_of(i);
		if (s == v)
		{
			nbs.push_back(t);
			aes.push_back(edge_t(i));
		}
	}

	for (gr_index_t i = 0; i < (gr_index_t)ne; ++i)
	{
		vertex_t s = g.source_of(i);
		vertex_t t = g.target_of(i);
		if (t == v)
		{
			nbs.push_back(s);
			aes.push_back(edge_t(i + (gr_index_t)ne));
		}
	}

	gr_size_t out_deg = (gr_size_t)nbs.size();

	if (out_deg != g.out_degree(v))
	{
		return false;
	}

	if ( !collection_equal(g.out_neighbor_begin(v), g.out_neighbor_end(v), &(nbs[0]), out_deg) )
	{
		return false;
	}

	if ( !collection_equal(g.out_edge_begin(v), g.out_edge_end(v), &(aes[0]), out_deg) )
	{
		return false;
	}

	return true;
}




BCS_TEST_CASE( test_directed_adjlist )
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


	gr_adjlist<gr_directed> g1r(nv, ne, src_vs, tar_vs);

	BCS_CHECK_EQUAL( g1r.nvertices(), nv );
	BCS_CHECK_EQUAL( g1r.nedges(), ne );

	for (gr_index_t i = 0; i < (gr_index_t)ne; ++i)
	{
		BCS_CHECK_EQUAL(g1r.source_of(edge_t(i)), src_vs[i]);
		BCS_CHECK_EQUAL(g1r.target_of(edge_t(i)), tar_vs[i]);
		BCS_CHECK_EQUAL(g1r.get_edge_i(edge_t(i)), edge_i(src_vs[i], tar_vs[i]));
	}

	for (gr_index_t i = 0; i < (gr_index_t)nv; ++i)
	{
		BCS_CHECK( verify_adjacency(g1r, vertex_t(i)) );
	}

	BCS_CHECK( collection_equal(g1r.v_begin(), g1r.v_end(), all_vs, nv) );
	BCS_CHECK( collection_equal(g1r.e_begin(), g1r.e_end(), all_es, ne) );

}


BCS_TEST_CASE( test_directed_wadjlist )
{
	// prepare graph

	const gr_size_t nv = 4;
	const gr_size_t ne = 5;
	const gr_index_t src_inds[ne] = {0, 0, 1, 1, 2};
	const gr_index_t tar_inds[ne] = {1, 2, 2, 3, 3};
	const double ws[ne] = {0.5, 0.4, 0.1, 0.3, 0.2};

	const gr_index_t v_inds[nv] = {0, 1, 2, 3};
	const gr_index_t e_inds[ne] = {0, 1, 2, 3, 4};

	const vertex_t* all_vs = (const vertex_t*)(v_inds);
	const edge_t* all_es = (const edge_t*)(e_inds);

	const vertex_t* src_vs = (const vertex_t*)(src_inds);
	const vertex_t* tar_vs = (const vertex_t*)(tar_inds);


	gr_wadjlist<double, gr_directed> g1r(nv, ne, src_vs, tar_vs, ws);

	BCS_CHECK_EQUAL( g1r.nvertices(), nv );
	BCS_CHECK_EQUAL( g1r.nedges(), ne );

	for (gr_index_t i = 0; i < (gr_index_t)ne; ++i)
	{
		BCS_CHECK_EQUAL(g1r.source_of(edge_t(i)), src_vs[i]);
		BCS_CHECK_EQUAL(g1r.target_of(edge_t(i)), tar_vs[i]);
		BCS_CHECK_EQUAL(g1r.weight_of(edge_t(i)), ws[i]);
		BCS_CHECK_EQUAL(g1r.get_edge_i(edge_t(i)), edge_i(src_vs[i], tar_vs[i]));
		BCS_CHECK_EQUAL(g1r.get_wedge_i(edge_t(i)), make_edge_i(src_vs[i], tar_vs[i], ws[i]));
	}

	for (gr_index_t i = 0; i < (gr_index_t)nv; ++i)
	{
		BCS_CHECK( verify_adjacency(g1r, vertex_t(i)) );
	}

	BCS_CHECK( collection_equal(g1r.v_begin(), g1r.v_end(), all_vs, nv) );
	BCS_CHECK( collection_equal(g1r.e_begin(), g1r.e_end(), all_es, ne) );

}


BCS_TEST_CASE( test_undirected_adjlist )
{
	// prepare graph

	const gr_size_t nv = 4;
	const gr_size_t ne = 5;
	const gr_index_t src_inds[ne * 2] = {0, 0, 1, 1, 2};
	const gr_index_t tar_inds[ne * 2] = {1, 2, 2, 3, 3};

	const gr_index_t v_inds[nv] = {0, 1, 2, 3};
	const gr_index_t e_inds[ne] = {0, 1, 2, 3, 4};

	const vertex_t* all_vs = (const vertex_t*)(v_inds);
	const edge_t* all_es = (const edge_t*)(e_inds);

	const vertex_t* src_vs = (const vertex_t*)(src_inds);
	const vertex_t* tar_vs = (const vertex_t*)(tar_inds);


	gr_adjlist<gr_undirected> g1r(nv, ne, src_vs, tar_vs);

	BCS_CHECK_EQUAL( g1r.nvertices(), nv );
	BCS_CHECK_EQUAL( g1r.nedges(), ne );

	for (gr_index_t i = 0; i < (gr_index_t)ne; ++i)
	{
		BCS_CHECK_EQUAL(g1r.source_of(edge_t(i)), src_vs[i]);
		BCS_CHECK_EQUAL(g1r.target_of(edge_t(i)), tar_vs[i]);
		BCS_CHECK_EQUAL(g1r.get_edge_i(edge_t(i)), edge_i(src_vs[i], tar_vs[i]));
	}

	for (gr_index_t i = 0; i < (gr_index_t)nv; ++i)
	{
		BCS_CHECK( verify_adjacency(g1r, vertex_t(i)) );
	}

	BCS_CHECK( collection_equal(g1r.v_begin(), g1r.v_end(), all_vs, nv) );
	BCS_CHECK( collection_equal(g1r.e_begin(), g1r.e_end(), all_es, ne) );


}


BCS_TEST_CASE( test_undirected_wadjlist )
{
	// prepare graph

	const gr_size_t nv = 4;
	const gr_size_t ne = 5;
	const gr_index_t src_inds[ne * 2] = {0, 0, 1, 1, 2};
	const gr_index_t tar_inds[ne * 2] = {1, 2, 2, 3, 3};
	const double ws[ne * 2] = {0.5, 0.4, 0.1, 0.3, 0.2};

	const gr_index_t v_inds[nv] = {0, 1, 2, 3};
	const gr_index_t e_inds[ne] = {0, 1, 2, 3, 4};

	const vertex_t* all_vs = (const vertex_t*)(v_inds);
	const edge_t* all_es = (const edge_t*)(e_inds);

	const vertex_t* src_vs = (const vertex_t*)(src_inds);
	const vertex_t* tar_vs = (const vertex_t*)(tar_inds);


	gr_wadjlist<double, gr_undirected> g1r(nv, ne, src_vs, tar_vs, ws);

	BCS_CHECK_EQUAL( g1r.nvertices(), nv );
	BCS_CHECK_EQUAL( g1r.nedges(), ne );

	for (gr_index_t i = 0; i < (gr_index_t)ne; ++i)
	{
		BCS_CHECK_EQUAL(g1r.source_of(edge_t(i)), src_vs[i]);
		BCS_CHECK_EQUAL(g1r.target_of(edge_t(i)), tar_vs[i]);
		BCS_CHECK_EQUAL(g1r.get_edge_i(edge_t(i)), edge_i(src_vs[i], tar_vs[i]));
		BCS_CHECK_EQUAL(g1r.get_edge_i(edge_t(i)), edge_i(src_vs[i], tar_vs[i]));
		BCS_CHECK_EQUAL(g1r.get_wedge_i(edge_t(i)), make_edge_i(src_vs[i], tar_vs[i], ws[i]));
	}

	for (gr_index_t i = 0; i < (gr_index_t)nv; ++i)
	{
		BCS_CHECK( verify_adjacency(g1r, vertex_t(i)) );
	}

	BCS_CHECK( collection_equal(g1r.v_begin(), g1r.v_end(), all_vs, nv) );
	BCS_CHECK( collection_equal(g1r.e_begin(), g1r.e_end(), all_es, ne) );


}





test_suite *test_gr_adjlist_suite()
{
	test_suite *suite = new test_suite( "test_gr_adjlist" );

	suite->add( new test_directed_adjlist() );
	suite->add( new test_directed_wadjlist() );
	suite->add( new test_undirected_adjlist() );
	suite->add( new test_undirected_wadjlist() );

	return suite;
}


