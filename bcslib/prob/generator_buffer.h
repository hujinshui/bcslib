/**
 * @file generator_buffer.h
 *
 * A generator-buffer for caching generating random values
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_GENERATOR_BUFFER_H
#define BCSLIB_GENERATOR_BUFFER_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_mem.h>

#include <bcslib/base/arg_check.h>

namespace bcs
{

	/**
	 * The generator buffer
	 *
	 * implemented using double-buffer
	 *
	 */
	template<class Generator>
	class generator_buffer : public noncopyable
	{
	public:
		typedef Generator generator_type;
		typedef typename Generator::result_type result_type;

	public:
		generator_buffer(generator_type gen, size_t buf_size)
		: _check_size(buf_size)
		, m_gen(gen)
		, m_blk(buf_size)
		, m_p(m_blk.pbase() + buf_size) , m_pend(m_p) // make it initially empty
		, m_tn(0)
		{
		}


	public:
		size_t buffer_capacity() const
		{
			return m_blk.nelems();
		}

		size_t num_buffered() const  // the actual number of values currently in buffer
		{
			return (size_t)(m_pend - m_p);
		}

		uint64_t total_generated() const	// the total number of values that have ever been generated
		{
			return m_tn;
		}

		void get_next(result_type *dst)
		{
			if (m_p == m_pend)
			{
				refill_buffer();
			}

			*dst = *(m_p++);
		}

		void get_next_n(size_t n, result_type *dst)
		{
			if (n > 0)
			{
				size_t nb = num_buffered();

				if (n <= nb)
				{
					// directly copy from buffer
					export_buffer(n, dst);
				}
				else
				{
					// flush the buffer to target first
					if (nb > 0)
						export_buffer(nb, dst);

					// produce the remaining
					size_t nr = n - nb;
					if (nr < buffer_capacity())
					{
						refill_buffer();
						export_buffer(nr, dst + nb);
					}
					else
					{
						generate_to(nr, dst + nb);
					}
				}
			}
		}

	private:
		void generate_to(size_t n, result_type *dst)
		{
			m_gen(n, dst);
			m_tn += n;
		}

		void refill_buffer()
		{
			generate_to(m_blk.nelems(), m_blk.pbase());
			m_p = m_blk.pbase();
		}

		void export_buffer(size_t n, result_type *dst)  // only call when n <= num_buffered()
		{
			copy_elements(m_p, dst, n);
			m_p += n;
		}

	private:

		struct _check_size_t
		{
			_check_size_t(size_t bufsize)
			{
				check_arg(bufsize >= 8,
					"generator_buffer: buffer size must be at least 8.");
			}
		};
		_check_size_t _check_size;

		generator_type m_gen;	// the generator

		block<result_type> m_blk; 	// it holds two buffers, each of length n
		result_type *m_p;			// point to the next value
		result_type *m_pend;		// point to the end of the buffer

		uint64_t m_tn;			// the total number of values that have been generated

	}; // end class generator_buffer


}

#endif 
