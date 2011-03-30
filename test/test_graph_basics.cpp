/**
 * @file test_graph_basics.cpp
 *
 * Unit testing for basic aspects of graph classes
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/execution_mon.h>
#include <bcslib/graph/gr_edgelist.h>
#include <bcslib/graph/gr_adjlist.h>
#include <bcslib/graph/bgl_port.h>

using namespace bcs::test;

// explicit template class instantiation

template class bcs::gr_edgelist<bcs::gr_directed>;
template class bcs::gr_edgelist<bcs::gr_undirected>;
template class bcs::gr_wedgelist<double, bcs::gr_directed>;
template class bcs::gr_wedgelist<double, bcs::gr_undirected>;

template class bcs::gr_adjlist<bcs::gr_directed>;
template class bcs::gr_adjlist<bcs::gr_undirected>;
template class bcs::gr_wadjlist<double, bcs::gr_directed>;
template class bcs::gr_wadjlist<double, bcs::gr_undirected>;

template class boost::graph_traits<bcs::gr_edgelist<bcs::gr_directed> >;
template class boost::graph_traits<bcs::gr_wedgelist<double, bcs::gr_directed> >;
template class boost::graph_traits<bcs::gr_adjlist<bcs::gr_directed> >;
template class boost::graph_traits<bcs::gr_wadjlist<double, bcs::gr_directed> >;

test_suite* master_suite()
{
	test_suite* msuite = new test_suite( "graph_basics" );

	return msuite;
}


BCS_TEST_MAIN_FUNCTION
