/**
 * @file test_matrix_par_reduc.cpp
 *
 * Unit testing for partial reduction on matrix
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/matrix.h>

using namespace bcs;


TEST( MatrixParReduc, Temp )
{
	dense_matrix<double> A(2, 3);

	dense_matrix<double> B = sum(colwise(A));
	dense_matrix<double> C = sum(rowwise(A));

	dense_matrix<double> X = dot(colwise(A), colwise(A));
	dense_matrix<double> Y = dot(rowwise(A), rowwise(A));
}

