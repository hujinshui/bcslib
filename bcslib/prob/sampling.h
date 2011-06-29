/**
 * @file sampling.h
 *
 * The base of probabilistic sampling
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_SAMPLING_H
#define BCSLIB_SAMPLING_H

#include <bcslib/base/basic_defs.h>

#include <limits>
#include <cmath>
#include <random>

namespace bcs
{

	/***
	 *
	 * The concept of a random stream class
	 * -------------------------------------
	 *
	 *  - a seed_type typedef
	 * 	- void seed( i ); 	i is of seed_type
	 *
	 *  - randi32(a, b);
	 *  	get random 32-bit integer from uniform distribution over {a, ..., b}
	 *
	 * 	- randi32(n, dst, a, b);	get random 32-bit ints in [a, b]
	 * 	- randu32(n, dst, a, b);	get random 32-bit uints in [a, b]
	 *
	 */



}

#endif 
