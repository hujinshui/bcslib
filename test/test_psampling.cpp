/**
 * @file test_psampling.cpp
 *
 * Test random sampling
 * 
 * @author Dahua Lin
 */

#include <bcslib/prob/sampling.h>
#include <bcslib/prob/discrete_distr.h>

#include <cstdio>
#include <ctime>
#include <limits>

using namespace bcs;


struct bstat
{
	double mean;
	double var;
	double skewness;
	double kurtosis;
};


bstat do_stat(size_t n, double *x)
{
	double s1 = 0;
	double s2 = 0;
	double s3 = 0;
	double s4 = 0;

	for (size_t i = 0; i < n; ++i)
	{
		double v = x[i];
		s1 += v;
	}
	double mv = s1 / double(n);

	for (size_t i = 0; i < n; ++i)
	{
		double v1 = x[i] - mv;
		double v2 = v1 * v1;
		double v3 = v2 * v1;
		double v4 = v2 * v2;

		s2 += v2;
		s3 += v3;
		s4 += v4;
	}

	bstat bs;

	bs.mean = mv;
	bs.var = s2 / double(n);
	bs.skewness = (s3 / double(n)) / std::pow(s2, 1.5);
	bs.kurtosis = (s4 / double(n)) / sqr(bs.var) - 3;

	return bs;
}


int main(int argc, char *argv[])
{
	std::printf("Test the performance of random number generation:\n");
	std::printf("--------------------------------------------------\n");
	std::printf("[1] standard uniform \n");
	std::printf("[2] standard normal \n");
	std::printf("[3] standard exponential \n");
	std::printf("[11] discrete (K = 100) direct\n");
	std::printf("[12] discrete (K = 100) sort \n");
	std::printf("----------------------------------\n");
	std::printf("[0] exit\n");
	std::printf("\n");

	std::printf("Your choice: ");
	int choice;
	if (std::scanf("%d", &choice) == EOF)
	{
		std::printf("Failed to read the choice!\n");
		return 1;
	}


	randstream<> rstream;
	rstream.seed( (unsigned int)(std::time(0)) );

	const size_t len = 10000000;
	static double reals[len];
	static int32_t ints[len];

	std::clock_t start, elapsed;
	if (choice == 1)
	{
		start = std::clock();
		real_rng<double>::get_uniform(rstream, len, reals);
		elapsed = std::clock() - start;
	}
	else if (choice == 2)
	{
		start = std::clock();
		real_rng<double>::get_normal(rstream, len, reals);
		elapsed = std::clock() - start;
	}
	else if (choice == 3)
	{
		start = std::clock();
		real_rng<double>::get_exponential(rstream, len, reals);
		elapsed = std::clock() - start;
	}
	else if (choice / 10 == 1)
	{
		int32_t K = 100;
		block<double> p((size_t)K);
		double sp = 0;
		for (int k = 0; k < K; ++k)
		{
			sp += (p[k] = rstream.randf64());
		}
		for (int k = 0; k < K; ++k)
		{
			p[k] /= sp;
		}

		if (choice == 11)
		{
			discrete_sampler<int32_t> sp(K, p.pbase(), discrete_sampler<int32_t>::DSAMP_DIRECT_METHOD);
			std::printf("avg.slen = %.2f\n", sp.average_search_length());

			start = std::clock();
			sp(rstream, len, ints);
			elapsed = std::clock() - start;
		}
		else if (choice == 12)
		{
			discrete_sampler<int32_t> sp(K, p.pbase(), discrete_sampler<int32_t>::DSAMP_SORT_METHOD);
			std::printf("avg.slen = %.2f\n", sp.average_search_length());

			start = std::clock();
			sp(rstream, len, ints);
			elapsed = std::clock() - start;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		return 0;  // exit
	}


	double elapsed_secs = (double)elapsed / CLOCKS_PER_SEC;
	std::printf("elapsed = %.4f sec\n", elapsed_secs);
	std::printf("rate = %.2f Msamples / sec\n", (len / (1e6 * elapsed_secs)));

	std::printf("\n");

	return 0;
}




