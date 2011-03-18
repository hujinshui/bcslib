/**
 * @file test_array1d.cpp
 *
 * The Unit Testing for array1d
 * 
 * @author Dahua Lin
 */


#include <bcslib/test/test_units.h>
#include <bcslib/array/array1d.h>

using namespace bcs;
using namespace bcs::test;

// Explicit instantiation for syntax checking

template class array1d<double>;
template class aview1d<double, step_ind>;
template class aview1d<double, array_ind>;




