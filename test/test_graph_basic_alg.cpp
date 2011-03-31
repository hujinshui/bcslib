/*
 * @file test_graph_basic_alg.cpp
 *
 * Unit tests for combining BCS graph classes with BGL basic algorithms
 *
 * @author dhlin
 */


#include <bcslib/test/test_units.h>

#include <bcslib/graph/gr_adjlist.h>
#include <bcslib/graph/bgl_port.h>

#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <vector>
#include <iostream>

using namespace bcs;
using namespace bcs::test;


struct my_bfs_visitor : public boost::default_bfs_visitor
{
	std::vector<vertex_t>& vs;

	my_bfs_visitor(std::vector<vertex_t>& vs_) : vs(vs_) { }

	template<class Graph>
	void discover_vertex(const vertex_t& v, const Graph& g)
	{
		vs.push_back(v);
	}
};


struct my_dfs_visitor : public boost::default_dfs_visitor
{
	std::vector<vertex_t>& vs;

	my_dfs_visitor(std::vector<vertex_t>& vs_) : vs(vs_) { }

	template<class Graph>
	void discover_vertex(const vertex_t& v, const Graph& g)
	{
		vs.push_back(v);
	}
};




BCS_TEST_CASE( test_bfs )
{
	const int nv = 5;
	const int ne = 7;

	gr_index_t src_inds[ne * 2] = {0, 0, 1, 1, 2, 3, 4,   1, 2, 2, 3, 3, 4, 0};
	gr_index_t tar_inds[ne * 2] = {1, 2, 2, 3, 3, 4, 0,   0, 0, 1, 1, 2, 3, 4};

	const vertex_t *src_vs = (const vertex_t*)src_inds;
	const vertex_t *tar_vs = (const vertex_t*)tar_inds;

	// directed

	gr_adjlist<gr_directed> gd(nv, ne, ref_t(), src_vs, tar_vs);

	std::vector<vertex_t> vorder_d;
	my_bfs_visitor vis_d(vorder_d);

	boost::breadth_first_search(gd, vertex_t(0), boost::visitor(vis_d));

	gr_index_t res_d[nv] = {0, 1, 2, 3, 4};
	BCS_CHECK( collection_equal(vorder_d.begin(), vorder_d.end(), (const vertex_t*)res_d, nv) );

	// undirected

	gr_adjlist<gr_undirected> gu(nv, ne, ref_t(), src_vs, tar_vs);

	std::vector<vertex_t> vorder_u;
	my_bfs_visitor vis_u(vorder_u);

	boost::breadth_first_search(gu, vertex_t(0), boost::visitor(vis_u));

	gr_index_t res_u[nv] = {0, 1, 2, 4, 3};
	BCS_CHECK( collection_equal(vorder_u.begin(), vorder_u.end(), (const vertex_t*)res_u, nv) );

}


BCS_TEST_CASE( test_dfs )
{
	const int nv = 5;
	const int ne = 6;

	gr_index_t src_inds[ne * 2] = {0, 0, 1, 2, 3, 4,   1, 2, 3, 3, 4, 0};
	gr_index_t tar_inds[ne * 2] = {1, 2, 3, 3, 4, 0,   0, 0, 1, 2, 3, 4};

	const vertex_t *src_vs = (const vertex_t*)src_inds;
	const vertex_t *tar_vs = (const vertex_t*)tar_inds;

	// directed

	gr_adjlist<gr_directed> gd(nv, ne, ref_t(), src_vs, tar_vs);

	std::vector<vertex_t> vorder_d;
	my_dfs_visitor vis_d(vorder_d);

	boost::depth_first_search(gd, boost::visitor(vis_d));

	gr_index_t res_d[nv] = {0, 1, 3, 4, 2};
	BCS_CHECK( collection_equal(vorder_d.begin(), vorder_d.end(), (const vertex_t*)res_d, nv) );


	// undirected

	gr_adjlist<gr_undirected> gu(nv, ne, ref_t(), src_vs, tar_vs);

	std::vector<vertex_t> vorder_u;
	my_dfs_visitor vis_u(vorder_u);

	boost::depth_first_search(gu, boost::visitor(vis_u));

	gr_index_t res_u[nv] = {0, 1, 3, 4, 2};
	BCS_CHECK( collection_equal(vorder_u.begin(), vorder_u.end(), (const vertex_t*)res_u, nv) );

}






test_suite *test_gr_basic_alg_suite()
{
	test_suite *suite = new test_suite( "test_graph_basic_alg" );

	suite->add( new test_bfs() );
	suite->add( new test_dfs() );

	return suite;
}




