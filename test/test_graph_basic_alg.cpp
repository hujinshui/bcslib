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
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>

#include <vector>
#include <iterator>
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


BCS_TEST_CASE( test_dijkstra )
{
	const gr_size_t nv = 5;
	const gr_size_t ne = 8;

	gr_index_t src_inds[ne] = {0, 0, 0, 1, 1, 3, 3, 2};
	gr_index_t tar_inds[ne] = {1, 2, 3, 2, 4, 2, 4, 4};
	double ws[ne] = {1.0, 5.0, 0.5, 1.2, 1.8, 0.6, 6.0, 0.2};

	const vertex_t *src_vs = (const vertex_t*)(src_inds);
	const vertex_t *tar_vs = (const vertex_t*)(tar_inds);

	gr_wadjlist<double, gr_directed> g(nv, ne, ref_t(), src_vs, tar_vs, ws);

	using boost::distance_map;
	using boost::predecessor_map;

	double dists[nv];
	set_zeros_to_elements(dists, nv);

	vertex_t preds[nv];

	vertex_ref_map<double> dist_map(dists);
	vertex_ref_map<vertex_t> pred_map(preds);

	boost::dijkstra_shortest_paths(g, vertex_t(0), distance_map(dist_map).predecessor_map(pred_map));

	gr_index_t preds_gt[nv] = {0, 0, 3, 0, 2};
	double dists_gt[nv] = {0.0, 1.0, 1.1, 0.5, 1.3};

	BCS_CHECK( collection_equal(preds, preds + nv, (const vertex_t*)preds_gt, nv) );
	BCS_CHECK( collection_equal(dists, dists + nv, dists_gt, nv) );

}


BCS_TEST_CASE( test_kruskal )
{
	const gr_size_t nv = 7;
	const gr_size_t ne = 11;

	gr_index_t src_inds[ne] = {0, 1, 0, 3, 1, 2, 3, 3, 4, 4, 5};
	gr_index_t tar_inds[ne] = {1, 2, 3, 1, 4, 4, 4, 5, 5, 6, 6};
	double ws[ne] = {7, 8, 5, 9, 7, 5, 15, 6, 8, 9, 11};

	const vertex_t *src_vs = (const vertex_t*)(src_inds);
	const vertex_t *tar_vs = (const vertex_t*)(tar_inds);

	gr_wadjlist<double, gr_undirected> g(nv, ne, clone_t(), src_vs, tar_vs, ws);

	std::vector<edge_t> eds;

	boost::kruskal_minimum_spanning_tree(g, std::back_inserter(eds));

	gr_index_t gtruth[6] = {2, 5, 7, 0, 4, 9};

	BCS_CHECK( collection_equal(eds.begin(), eds.end(), (const edge_t*)gtruth, 6) );

}


BCS_TEST_CASE( test_prim )
{
	const gr_size_t nv = 7;
	const gr_size_t ne = 11;

	gr_index_t src_inds[ne] = {0, 1, 0, 3, 1, 2, 3, 3, 4, 4, 5};
	gr_index_t tar_inds[ne] = {1, 2, 3, 1, 4, 4, 4, 5, 5, 6, 6};
	double ws[ne] = {7, 8, 5, 9, 7, 5, 15, 6, 8, 9, 11};

	const vertex_t *src_vs = (const vertex_t*)(src_inds);
	const vertex_t *tar_vs = (const vertex_t*)(tar_inds);

	gr_wadjlist<double, gr_undirected> g(nv, ne, clone_t(), src_vs, tar_vs, ws);

	vertex_t preds[nv];

	boost::prim_minimum_spanning_tree(g, vertex_ref_map<vertex_t>(preds));

	gr_index_t gtruth[nv] = {0, 0, 4, 0, 1, 3, 4};

	BCS_CHECK( collection_equal(preds, preds + nv, (const vertex_t*)gtruth, nv) );
}



test_suite *test_gr_basic_alg_suite()
{
	test_suite *suite = new test_suite( "test_graph_basic_alg" );

	suite->add( new test_bfs() );
	suite->add( new test_dfs() );
	suite->add( new test_dijkstra() );
	suite->add( new test_kruskal() );
	suite->add( new test_prim() );

	return suite;
}




