/**
 * @file test_psampling.cpp
 *
 * Test random sampling
 * 
 * @author Dahua Lin
 */

#include <bcslib/prob/sampling.h>
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
	double mv = s1 / n;

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
	bs.var = s2 / n;
	bs.skewness = (s3 / n) / std::pow(s2, 1.5);
	bs.kurtosis = (s4 / n) / sqr(bs.var) - 3;

	return bs;
}


int main(int argc, char *argv[])
{
	std::printf("Test the performance of random number generation:\n");
	std::printf("--------------------------------------------------\n");
	std::printf("[1] uniform \n");
	std::printf("[2] normal \n");
	std::printf("[3] exponential \n");
	std::printf("-----------------\n");
	std::printf("[0] exit\n");
	std::printf("\n");

	std::printf("Your choice: ");
	int choice;
	std::scanf("%d", &choice);


	randstream<> rstream;
	rstream.seed( std::time(0) );

	const size_t len = 10000000;
	static double reals[len];

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
	else
	{
		return 0;  // exit
	}


	double elapsed_secs = (double)elapsed / CLOCKS_PER_SEC;
	std::printf("elapsed = %.4f sec\n", elapsed_secs);
	std::printf("rate = %.2f Msamples / sec\n", (len / (1e6 * elapsed_secs)));

	bstat bs = do_stat(len, reals);
	std::printf("mean = %.4f\n", bs.mean);
	std::printf("var  = %.4f\n", bs.var);
	std::printf("skew = %.4f\n", bs.skewness);
	std::printf("kurt = %.4f\n", bs.kurtosis);

	return 0;
}




