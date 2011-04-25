/**
 * @file test_psampling.cpp
 *
 * Test random sampling
 * 
 * @author Dahua Lin
 */

#include <bcslib/prob/sampling.h>
#include <cstdio>

using namespace bcs;

template class uniform_rgen<double>;
template class duniform_rgen<int32_t>;
template class normal_rgen<double>;
template class bernoulli_rgen<>;
template class geometric_rgen<int32_t>;
template class poisson_rgen<int32_t>;
template class binomial_rgen<int32_t>;
template class exponential_rgen<double>;
template class gamma_rgen<double>;


template<typename TGen>
void print_gen(const char *title, size_t n, const char *fmt, TGen& gen)
{
	std::printf("%s: ", title);
	for (size_t i = 0; i < n; ++i)
	{
		std::printf(fmt, gen());
		std::printf(" ");
	}
	std::printf("\n");
}

template<typename TGen>
void print_gen_b(const char *title, size_t n, const char *fmt, TGen& gen)
{
	std::printf("%s: ", title);
	for (size_t i = 0; i < n; ++i)
	{
		std::printf(fmt, (int)gen());
		std::printf(" ");
	}
	std::printf("\n");
}



int main(int argc, char *argv[])
{

	default_tr1_rgen_engine reng;

	duniform_rgen<int32_t> g1(reng, 5);
	std::printf("g1: duniform<int>: min = %d, max = %d\n", g1.min(), g1.max());
	print_gen("g1", 15, "%d", g1);
	std::printf("\n");

	uniform_rgen<double> g2(reng, -2, 4);
	std::printf("g2: uniform<double>: min = %g, max = %g\n", g2.min(), g2.max());
	print_gen("g2", 20, "%.2f", g2);
	std::printf("\n");

	normal_rgen<double> g3(reng, 0, 10);
	std::printf("g3: normal<double>: mean = %g, sigma = %g\n", g3.mean(), g3.sigma());
	print_gen("g3", 20, "%.2f", g3);
	std::printf("\n");

	bernoulli_rgen<> g4(reng, 0.8);
	std::printf("g4: bernoulli<bool>: p = %g\n", g4.p());
	print_gen_b("g4", 20, "%d", g4);
	std::printf("\n");

	geometric_rgen<int32_t> g5(reng, 0.2);
	std::printf("g5: geometric<int>: p = %g, mean = %g\n", g5.p(), g5.mean());
	print_gen("g5", 20, "%d", g5);
	std::printf("\n");

	poisson_rgen<int32_t> g6(reng, 5.0);
	std::printf("g6: poisson<rgen>: mean = %g\n", g6.mean());
	print_gen("g6", 20, "%d", g6);
	std::printf("\n");

	binomial_rgen<int32_t> g7(reng, 10, 0.3);
	std::printf("g7: binomial<int>: n = %d, p = %g\n", g7.n(), g7.p());
	print_gen("g7", 20, "%d", g7);
	std::printf("\n");

	exponential_rgen<double> g8(reng, 0.2);
	std::printf("g8: exponential<double>: mean = %g\n", g8.mean());
	print_gen("g8", 15, "%.2f", g8);
	std::printf("\n");

	gamma_rgen<double> g9(reng);
	std::printf("g9: gamma<double>: alpha = %g\n", g9.alpha());
	print_gen("g9", 10, "%.2f", g9);
	std::printf("\n");

	return 0;
}




