/**
 * @file test_psampling.cpp
 *
 * Test random sampling
 * 
 * @author Dahua Lin
 */

#include <bcslib/prob/sampling.h>
#include <cstdio>
#include <iostream>

using namespace bcs;

int main(int argc, char *argv[])
{
	tr1_rand<> g1;

	std::printf("g1: ");
	for (int i = 0; i < 8; ++i)
	{
		std::printf("%.4f ", g1());
	}
	std::printf("\n");

	std::tr1::uniform_real<double> u(0.0, 1.0);
	std::printf("max = %f\n", u.max());

	std::tr1::mt19937 eng;
	std::cout << u(eng) << std::endl;

	return 0;
}




