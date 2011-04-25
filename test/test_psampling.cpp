/**
 * @file test_psampling.cpp
 *
 * Test random sampling
 * 
 * @author Dahua Lin
 */

#include <bcslib/prob/sampling.h>
#include <iostream>

using namespace bcs;

template class tr1_randu<double>;
template class tr1_randn<double>;


int main(int argc, char *argv[])
{
	std::tr1::mt19937 eng;
	std::tr1::uniform_int<int> distr;
	std::tr1::variate_generator<std::tr1::mt19937, std::tr1::uniform_int<int> > vgen(eng, distr);

	int n = 10;
	for (int i = 0; i < n; ++i)
	{
		std::cout << vgen() << std::endl;
	}

}




