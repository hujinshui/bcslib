/*
 * @file pdistr_base.h
 *
 * The base of probability distributions
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_PDISTR_BASE_H
#define BCSLIB_PDISTR_BASE_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/arg_check.h>
#include <bcslib/base/mathfun.h>

#include <cmath>
#include <type_traits>
#include <limits>

namespace bcs
{

	/**
	 * A prob. distribution class must satisfy the following requirement
	 * ------------------------------------------------------------
	 *
	 * Let P be the distribution class name, and p be an instance of P.
	 * Then:
	 *
	 * 1. P has the following typedefs:
	 *
	 * 		- value_type;	// the value type of each component, int32_t, double, float, etc
	 * 		- param_type;	// the type of the parameter (nil_type is no parameter is needed)
	 *
	 * 2. P has the following constants:
	 *
	 * 		- sample_dim:	// the dimension of the sample space (1 for scalar-valued space)
	 * 						// the type of sample_dim is size_t
	 *
	 * 3. bcs::is_pdistr<P>::value should be set to true. (By default, it is false)
	 *
	 * 4. A pdistr_traits<P> class is defined, which has the following:
	 *
	 * 		- value_type;
	 * 		- param_type;
	 * 		- sample_dim;
	 * 		- is_sampleable; 	// whether one can sample from the distribution
	 * 		- is_pdf_evaluable;	// whether one can evaluate pdf values at given samples
	 *
	 * 	  A generic template struct pdistr_traits is defined in this file, which simply
	 * 	  takes value_type, param_type, and sample_dim from the class P, and set
	 * 	  the boolean constants to false.
	 *
	 * 	  For each class, a specialized version of pdistr_traits can be provided to
	 * 	  override this default behavior for that class.
	 *
	 * 5. P is default-constructible, copy-constructible, and assignable
	 *
	 * 6. There is a constructor that takes a parameter of type param_type as argument
	 *
	 * 7. Each instance p of P can be used as follows:
	 *
	 * 		- p.dim();		// returns sample_dim
	 * 		- p.param();	// returns const reference to the parameter
	 *
	 * 	  If is_sampleable is true:
	 *
	 * 		- p.draw(rstream, dst);		// draws a sample and writes it to dst
	 * 									// - rstream: a random number stream
	 * 								 	// - dst:	destination address, of type value_type*
	 * 								 	// no return
	 *
	 * 		- p.draw(rstream, dst, n);	// draws n samples, no return
	 *
	 * 	  If is_pdf_evaluable is true:
	 *
	 * 	  	- p.pdf(sample);		// evaluates the pdf value at the input sample, and return (of type double)
	 * 	  							// sample is of type value_type*
	 *
	 * 	  	- p.pdfs(n, samples, out); 		// evaluates the pdf value at the input samples,
	 * 	  									// the evaluated values are written to out (of type double*)
	 * 	  									// no return
	 *
	 *		- p.logpdf(sample);		// evaluate logarithm of pdf value
	 *
	 *		- p.logpdfs(n, samples, out);	// evaluate the logarithm of pdf values
	 *
	 * 	8. Note: P can have its specific requirement on what interfaces should be supported
	 * 			 by rstream.
	 *
	 */

	template<class P>
	struct is_pdistr
	{
		static const bool value = false;
	};


	template<class P>
	struct pdistr_traits
	{
		typedef typename P::value_type value_type;
		typedef typename P::param_type param_type;

		static const size_t sample_dim = P::sample_dim;

		static const bool is_sampleable = false;
		static const bool is_pdf_evaluable = false;
	};


	template<class P>
	struct is_discrete_pdistr
	{
		static const bool value = is_pdistr<P>::value &&
				std::is_integral<typename pdistr_traits<P>::value_type>::value;
	};

	template<class P>
	struct is_continous_pdistr
	{
		static const bool value = is_pdistr<P>::value &&
				std::is_floating_point<typename pdistr_traits<P>::value_type>::value;
	};

}

#endif
