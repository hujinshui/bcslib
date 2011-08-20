/**
 * @file cuda_vec.h
 *
 * The classes to represent cuda vectors
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_CUDA_VEC_H_
#define BCSLIB_CUDA_VEC_H_

#include <bcslib/cuda/cuda_base.h>

namespace bcs { namespace cuda {


	template<typename T>
	class device_cview1d
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		__host__ __device__
		device_cview1d(const_pointer pbase, size_t len)
		: m_pbase(pbase), m_len(len)
		{
		}

	public:
		__host__ __device__
		size_t length() const
		{
			return m_len;
		}

		__host__ __device__
		const_pointer pbase() const
		{
			return m_pbase;
		}

		__host__ __device__
		device_cview1d csubview(size_t i, size_t slen) const
		{
			return device_cview1d(pbase() + i, slen);
		}

	protected:
		const_pointer m_pbase;
		size_t m_len;

	}; // end class device_cview1d


	template<typename T>
	class device_view1d
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		__host__ __device__
		device_view1d(pointer pbase, size_t len)
		: m_pbase(pbase), m_len(len)
		{
		}

		__host__ __device__
		device_cview1d<T> cview() const
		{
			return device_cview1d<T>(m_pbase, m_len);
		}

		__host__ __device__
		operator device_cview1d<T>() const
		{
			return device_cview1d<T>(m_pbase, m_len);
		}

	public:
		__host__ __device__
		size_t length() const
		{
			return m_len;
		}

		__host__ __device__
		const_pointer pbase() const
		{
			return m_pbase;
		}

		__host__ __device__
		pointer pbase()
		{
			return m_pbase;
		}

		__host__ __device__
		device_vec_cview<T> csubview(size_t i, size_t slen) const
		{
			return device_vec_cview<T>(pbase() + i, slen);
		}

		__host__ __device__
		device_view1d subview(size_t i, size_t slen)
		{
			return device_view1d(pbase() + i, slen);
		}

	private:
		pointer m_pbase;
		size_t m_len;

	}; // end class device_view1d


	template<typename T>
	class device_vec
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		__host__
		explicit device_vec()
		: m_capa(0), m_len(0), m_pbase(pointer(BCS_NULL))
		{
		}

		__host__
		~device_vec()
		{
			device_free(m_pbase);
		}

		__host__
		explicit device_vec(size_t n)
		: m_capa(n), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
		}

		__host__
		explicit device_vec(size_t n, size_t cap)
		: m_capa(calc_max(n, cap)), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
		}

		__host__
		device_vec(size_t n, host_cptr<T> src)
		: m_capa(n), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
			if (n > 0) copy_memory(src, m_pbase, n);
		}

		__host__
		device_vec(size_t n, host_cptr<T> src, size_t cap)
		: m_capa(calc_max(n, cap)), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
			if (n > 0) copy_memory(src, m_pbase, n);
		}

		__host__
		device_vec(size_t n, device_cptr<T> src)
		: m_capa(n), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
			if (n > 0) copy_memory(src, m_pbase, n);
		}

		__host__
		device_vec(size_t n, device_cptr<T> src, size_t cap)
		: m_capa(calc_max(n, cap)), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
			if (n > 0) copy_memory(src, m_pbase, n);
		}

		__host__
		device_vec(const device_vec& src)
		: m_capa(src.m_len), m_len(src.m_len), m_pbase(device_allocate<T>(m_capa))
		{
			copy_memory(src.m_pbase, m_pbase, m_len);
		}

		__host__
		device_vec(const device_vec& src, size_t cap)
		: m_capa(calc_max(src.m_len, cap)), m_len(src.m_len), m_pbase(device_allocate<T>(m_capa))
		{
			copy_memory(src.m_pbase, m_pbase, m_len);
		}

		__host__
		device_vec& operator = (const device_vec& rhs)
		{
			if (this != &rhs)
			{
				if (m_capa >= rhs.m_len)
				{
					copy_memory(rhs.m_pbase, m_pbase, m_len);
					m_len = rhs.m_len;
				}
				else
				{
					swap(device_vec(rhs));
				}
			}
			return *this;
		}

		__host__ __device__
		void swap(device_vec& rhs)
		{
			// swap capacity
			size_t tc = rhs.m_capa;
			rhs.m_capa = m_capa;
			m_capa = tc;

			// swap length
			size_t tl = rhs.m_len;
			rhs.m_len = m_len;
			m_len = tl;

			// swap pointer
			pointer tp = rhs.m_pbase;
			rhs.m_pbase = m_pbase;
			m_pbase = tp;
		}

	public:
		__host__ __device__
		device_cview1d<T> cview() const
		{
			return device_cview1d<T>(m_pbase, m_len);
		}

		__host__ __device__
		operator device_cview1d<T>() const
		{
			return device_cview1d<T>(m_pbase, m_len);
		}

		__host__ __device__
		device_view1d<T> view() const
		{
			return device_view1d<T>(m_pbase, m_len);
		}

		__host__ __device__
		operator device_view1d<T>() const
		{
			return device_view1d<T>(m_pbase, m_len);
		}

	public:
		__host__
		void reserve(size_t cap)  // after reservation, the length is reset to 0
		{
			if (m_capa < cap)
			{
				m_pbase = device_allocate<T>(cap);
				m_capa = cap;
				m_len = 0;
			}
		}

		__host__
		void reimport(size_t n, host_cptr<T> src)
		{
			reserve(n);
			copy_memory(src, m_pbase, n);
		}

		__host__
		void reimport(size_t n, device_cptr<T> src)
		{
			reserve(n);
			copy_memory(src, m_pbase, n);
		}

	public:
		__host__ __device__
		size_t length() const
		{
			return m_len;
		}

		__host__ __device__
		const_pointer pbase() const
		{
			return m_pbase;
		}

		__host__ __device__
		pointer pbase()
		{
			return m_pbase;
		}

		__host__ __device__
		device_cview1d<T> csubview(size_t i, size_t slen) const
		{
			return device_cview1d<T>(pbase() + i, slen);
		}

		__host__ __device__
		device_view1d<T> subview(size_t i, size_t slen)
		{
			return device_view1d<T>(pbase() + i, slen);
		}

	private:
		static size_t calc_max(size_t a, size_t b)
		{
			return a < b ? b : a;
		}

	private:
		size_t m_capa;
		size_t m_len;
		pointer m_pbase;

	}; // end class device_vec


	template<typename T>
	void swap(device_vec<T>& lhs, device_vec<T>& rhs)
	{
		lhs.swap(rhs);
	}

} }

#endif
