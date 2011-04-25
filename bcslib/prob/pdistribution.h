/**
 * @file pdistribution.h
 *
 * The base header file for probabilistic distributions
 * 
 * @author Dahua Lin
 */

#ifndef PDISTRIBUTION_H_
#define PDISTRIBUTION_H_

#include <bcslib/base/basic_defs.h>

namespace bcs
{

	/**
	 * The concept of a probabilistic distribution class
	 * -------------------------------------------------
	 *
	 * 1. derive from a distribution tag class:
	 * 		- discrete_distribution_t
	 * 		- continuous_distribution_t
	 * 		- hybrid_distribution_t
	 *
	 * 2. supporting the following method functions:
	 *
	 * 		- dim():  the dimension of sample space (size_t)
	 * 				  (for purely discrete distribution, it is 0)
	 *
	 * 		- pdf(x): evaluates the pdf (or pmf) on a sample
	 * 				  or a set of samples
	 *
	 * 		- logpdf(x): evaluates the log-pdf (or log-pmf) on
	 * 					 a sample or a set of samples
	 *
	 */

	class pdistribution_t { };

	class discrete_distribution_t : public pdistribution_t { };
	class continuous_distribution_t : public pdistribution_t { };
	class hybrid_distribution_t : public pdistribution_t { };

	template<class TDistr>
	class pdistribution_set_t
	{
	public:
		typedef TDistr distribution_type;
	};


}

#endif 
