/**
 * @file generator_buffer.h
 *
 * A generator-buffer for caching generating random values
 * 
 * @author Dahua Lin
 */

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
		, m_n(buf_size / 2)
		, m_blk(buf_size)
		, m_gen(gen)
		, m_iactive(0)
		, m_p(m_blk.pbase())
		, m_pend(m_p + m_n)
		, m_tn(0)
		{
			m_gen(m_n, m_p);
			m_tn += m_n;
		}


	public:
		size_t buffer_capacity() const
		{
			return 2 * m_n;
		}

		size_t num_buffered() const  // the actual number of values currently in buffer
		{
			return num_actives() + m_n;
		}

		uint64_t total_generated() const	// the total number of values that have ever been generated
		{
			return m_tn;
		}

		void get_next(result_type *dst)
		{
			*dst = *(m_p++);

			maintain();
		}

		void get_next_n(size_t n, result_type *dst)
		{
			size_t na = num_actives();
			if (n <= na)
			{
				copy_elements(m_p, dst, n);
				m_p += n;
				maintain();
			}
			else
			{
				flush_actives(na, dst);
				n -= na;
				dst += na;

				while (n >= m_n)
				{
					flush_actives(m_n, dst);
					n -= m_n;
					dst += m_n;
				}

				if (n > 0)
				{
					copy_elements(m_p, dst, n);
					m_p += n;
				}
			}
		}

	private:
		result_type *buffer0_base()
		{
			return m_blk.pbase();
		}

		result_type *buffer1_base()
		{
			return m_blk.pbase() + m_n;
		}

		void switch_buffer()
		{
			if (m_iactive == 0)  // 0 --> 1
			{
				m_iactive = 1;
				m_p = buffer1_base();
				m_pend = m_p + m_n;
			}
			else  // 1 --> 0
			{
				m_iactive = 0;
				m_p = buffer0_base();
				m_pend = m_p + m_n;
			}
		}

		void refill_backup_buffer()
		{
			result_type *pb = m_iactive == 1 ? buffer0_base() : buffer1_base();
			m_gen(m_n, pb);
			m_tn += m_n;
		}

		size_t num_actives() const
		{
			return static_cast<size_t>(m_pend - m_p);
		}

		void maintain()
		{
			if (m_p == m_pend)
			{
				switch_buffer();
				refill_backup_buffer();
			}
		}

		void flush_actives(size_t na, result_type* dst)
		{
			copy_elements(m_p, dst, na);
			switch_buffer();
			refill_backup_buffer();
		}

	private:

		struct _check_size_t
		{
			_check_size_t(size_t bufsize)
			{
				check_arg(bufsize >= 4 && bufsize % 2 == 0,
					"generator_buffer: buffer size must be even and at least 4.");
			}
		};
		_check_size_t _check_size;

		size_t m_n;
		block<result_type> m_blk; 	// it holds two buffers, each of length n

		generator_type m_gen;	// the generator

		int m_iactive;			// 0 or 1
		result_type *m_p; 		// point to next
		result_type *m_pend;	// the end of current buffer

		uint64_t m_tn;			// the total number of values that have been generated
	};


}

#endif 
