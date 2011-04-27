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
#include <bcslib/base/basic_funcs.h>
#include <bcslib/base/basic_mem.h>

namespace bcs
{

	/**
	 * The concept for probabilistic distribution classes
	 * --------------------------------------------------
	 *
	 * - typename distribution_category: which can be
	 * 		- discrete_distribution_t
	 * 		- continuous_distribution_t
	 * 		- hybrid_distribution_t
	 *
	 * - typename sample_category: which can be
	 * 		- scalar_sample_t
	 * 		- vector_sample_t
	 * 		- object_sample_t
	 *
	 * - bool is_sampleable: whether one can draw samples from it
	 *
	 * - bool is_pdf_evaluable: whether one can evaluate pdf
	 *
	 *
	 * For distribution class with scalar/vector samples:
	 *
	 * - typename value_type:	the type of each component
	 * - D.dim():	the dimension of sample space (for scale, it is 1)
	 *
	 * For distribution class that is sampleable
	 *
	 * - typename sampler_type
	 * - D.get_sampler();
	 *
	 * For distribution class that is pdf_evaluable
	 *
	 * - f = D.get_pdf_evaluator();
	 *
	 *   f here should be a light-weight object, that supports
	 *
	 *   v = f.pdf(x);		// evaluates on a sample x and returns the density v
	 *   lv = f.logpdf(x);  // evaluates on a sample x and returns log(v)
	 *   f.pdf(n, x, v);	// evaluates on a set of samples
	 *   f.logpdf(n, x, v);	// evaluates on a set of samples
	 *
	 */


	/**
	 * The concept for sampler classes
	 * -------------------------------
	 *
	 * - typename sample_category
	 *
	 * Let S be an instance of sampler:
	 *
	 * if sample_category is scalar_sample_t:
	 *
	 * - S(Gen& gen): 	returns a single scalar sample
	 * - S(Gen& gen, value_type *buf):	outputs a single sample to buf
	 * - S(Gen& gen, size_t n, value_type *buf): 	outputs n samples to buf
	 *
	 * if sample_category is vector_sample_t:
	 *
	 * - S(Gen& gen, value_type *buf):	outputs one sample to buf
	 * - S(Gen& gen, size_t n, value_type *buf): 	outputs n samples to buf
	 *
	 * if sample_category is object_sample_t:
	 *
	 * The syntax is type-dependent.
	 */


	/**
	 * The concept of multi_distribution classes
	 * -------------------------------------------
	 *
	 * - distribution_type:  the type of each single distribution
	 * - distribution_category
	 * - sample_category
	 * - is_sampleable
	 * - is_pdf_evaluable
	 *
	 * Let M be a multi_distribution_instance:
	 *
	 * - M.ndistributions(): return the number of contained distributions
	 * - M.distribution(i):	return the i-th distribution
	 *
	 */


	// distribution categories

	struct discrete_distribution_t { };
	struct continuous_distribution_t { };
	struct hybrid_distribution_t { };

	// sample categories

	struct scalar_sample_t { };
	struct vector_sample_t { };
	struct object_sample_t { };

	// distribution_traits

	template<typename TDistr>
	struct distribution_traits
	{
		typedef typename TDistr::distribution_category distribution_category;
		typedef typename TDistr::sample_category sample_category;

		static const bool is_sampleable = TDistr::is_sampleable;
		static const bool is_pdf_evaluable = TDistr::is_pdf_evaluable;
	};



}

#endif 
