/**
 * @file test_geometry_basics.cpp
 *
 * Some tests on geometry
 * 
 * @author Dahua Lin
 */


#include <bcslib/geometry/geometry_base.h>
#include <bcslib/geometry/triangle_mesh.h>
#include <bcslib/geometry/poly_scan.h>

// For syntax check

template struct bcs::point2d<double>;
template struct bcs::point3d<double>;

template struct bcs::lineseg2d<double>;
template struct bcs::lineseg3d<double>;
template struct bcs::line2d<double>;

template struct bcs::rectangle<double>;
template struct bcs::triangle<double>;

template struct bcs::triangle_scanner<double>;


int main(int argc, char *argv[])
{
}
