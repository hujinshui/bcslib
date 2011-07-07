/**
 * @file random_streams.h
 *
 * Random stream classes
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_RANDOM_STREAMS_H_
#define BCSLIB_RANDOM_STREAMS_H_

#include <bcslib/base/basic_defs.h>

#include <type_traits>
#include <random>

namespace bcs
{

	/***
	 *
	 *   Random stream classes
	 *   ----------------------
	 *
	 *   1. A random stream is a wrapper of some facility to provide the service of
	 *      generating individual random number of batch of random numbers of
	 *      some basic distributions
	 *
	 *   2. Let rs be a random stream, it should support the following basic
	 *      functionality.
	 *
	 *      seed(s):		seed the stream. (s is of type uint32_t)
	 *
	 *		rand_i32():		returns a random 32-bit signed integer scalar
	 *		rand_u32():		returns a random 32-bit unsigned integer scalar
	 *
	 *		rand_i32(u):	returns a random 32-bit signed integer scalar in [0, u-1]
	 *		rand_u32(u):	returns a random 32-bit unsigned integer scalar in [0, u-1]
	 *
	 *		rand_i32_m(n, dst):	writes n random 32-bit signed integer scalar to dst
	 *		rand_u32_m(n, dst):	writes n random 32-bit unsigned integer scalar to dst
	 *
	 *		rand_i32_m(u, n, dst):	writes n random 32-bit signed integer scalar to dst
	 *		rand_u32_m(u, n, dst):  writes n random 32-bit unsigned integer scalar to dst
	 *
	 *		rand_f32():		returns a random single-precision scalar, uniformly distributed in [0, 1]
	 *		rand_f64():		returns a random double-precision scalar, uniformly distributed in [0, 1]
	 *
	 *		rand_f32_m(n, dst):	writes n random single-precision uniformly distributed numbers to dst
	 *		rand_f64_m(n, dst): writes n random double-precision uniformly distributed numbers to dst
	 */


	template<typename Eng=std::mt19937>
	class std_rstream : private noncopyable
	{
	public:
		typedef Eng engine_type;
		typedef typename Eng::result_type seed_type;

	public:
		explicit std_rstream(engine_type eng = engine_type())
		: m_eng(eng)
		{
		}

	public:
		void seed(const seed_type& s)
		{
			m_eng.seed(s);
		}

		int32_t rand_i32()
		{
			return (int32_t)m_eng();
		}

		uint32_t rand_u32()
		{
			return (uint32_t)m_eng();
		}

		int32_t rand_i32(int32_t u)
		{
			return rand_i32() % u;
		}

		int32_t rand_u32(uint32_t u)
		{
			return rand_u32() % u;
		}

		float rand_f32()
		{
			return (float)rand_u32() / (float)(0xFFFFFFFFu);
		}

		double rand_f64()
		{
			return (double)rand_u32() / (double)(0xFFFFFFFFu);
		}

	private:
		engine_type m_eng;

	}; // end class std_rstream


}

#endif 
