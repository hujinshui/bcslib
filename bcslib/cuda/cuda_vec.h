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
#include <bcslib/array/aview1d.h>

namespace bcs { namespace cuda {


	template<typename T>
	class device_cview1d
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		__host__ __device__
		device_cview1d(const_pointer pbase, index_type len)
		: m_pbase(pbase), m_len(len)
		{
		}

	public:
		__host__ __device__
		index_type nelems() const
		{
			return m_len;
		}

		__host__ __device__
		index_type length() const
		{
			return m_len;
		}

		__host__ __device__
		const_pointer pbase() const
		{
			return m_pbase;
		}

		__host__ __device__
		device_cview1d cblock(index_type i, index_type slen) const
		{
			return device_cview1d(pbase() + i, slen);
		}

	public:
		__device__
		const T& operator[] (index_type i) const
		{
			return m_pbase[i];
		}

		__device__
		const T& operator() (index_type i) const
		{
			return m_pbase[i];
		}

	private:
		const_pointer m_pbase;
		index_type m_len;

	}; // end class device_cview1d


	template<typename T>
	class device_view1d
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		__host__ __device__
		device_view1d(pointer pbase, index_type len)
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
		index_type nelems() const
		{
			return m_len;
		}

		__host__ __device__
		index_type length() const
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
		device_cview1d<T> cblock(index_type i, index_type slen) const
		{
			return device_cview1d<T>(pbase() + i, slen);
		}

		__host__ __device__
		device_view1d block(index_type i, index_type slen)
		{
			return device_view1d(pbase() + i, slen);
		}

	public:
		__device__
		const T& operator[] (index_type i) const
		{
			return m_pbase[i];
		}

		__device__
		T& operator[] (index_type i)
		{
			return m_pbase[i];
		}

		__device__
		const T& operator() (index_type i) const
		{
			return m_pbase[i];
		}

		__device__
		T& operator() (index_type i)
		{
			return m_pbase[i];
		}

	public:
		__host__
		void set_zeros()
		{
			bcs::cuda::set_zeros(m_pbase, m_len);
		}

	private:
		pointer m_pbase;
		index_type m_len;

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
		explicit device_vec(index_type n)
		: m_capa(n), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
		}

		__host__
		explicit device_vec(index_type n, index_type cap)
		: m_capa(calc_max(n, cap)), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
		}

		__host__
		device_vec(index_type n, host_cptr<T> src)
		: m_capa(n), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
			if (n > 0) copy_memory(n, src, m_pbase);
		}

		__host__
		device_vec(index_type n, host_cptr<T> src, index_type cap)
		: m_capa(calc_max(n, cap)), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
			if (n > 0) copy_memory(n, src, m_pbase);
		}

		__host__
		device_vec(index_type n, device_cptr<T> src)
		: m_capa(n), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
			if (n > 0) copy_memory(n, src, m_pbase);
		}

		__host__
		device_vec(index_type n, device_cptr<T> src, index_type cap)
		: m_capa(calc_max(n, cap)), m_len(n), m_pbase(device_allocate<T>(m_capa))
		{
			if (n > 0) copy_memory(n, src, m_pbase);
		}

		__host__
		device_vec(const device_vec& src)
		: m_capa(src.m_len), m_len(src.m_len), m_pbase(device_allocate<T>(m_capa))
		{
			copy_memory(m_len, src.m_pbase, m_pbase);
		}

		__host__
		device_vec(const device_vec& src, index_type cap)
		: m_capa(calc_max(src.m_len, cap)), m_len(src.m_len), m_pbase(device_allocate<T>(m_capa))
		{
			copy_memory(m_len, src.m_pbase, m_pbase);
		}

		__host__
		device_vec& operator = (const device_vec& rhs)
		{
			if (this != &rhs)
			{
				if (m_capa >= rhs.m_len)
				{
					copy_memory(m_len, rhs.m_pbase, m_pbase);
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
			swap_(m_capa, rhs.m_capa);
			swap_(m_len, rhs.m_len);
			swap_(m_pbase, rhs.m_pbase);
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
		void reserve(index_type cap)  // after reservation, the length is reset to 0
		{
			if (m_capa < cap)
			{
				m_pbase = device_allocate<T>(cap);
				m_capa = cap;
				m_len = 0;
			}
		}

		__host__
		void reimport(index_type n, host_cptr<T> src)
		{
			reserve(n);
			copy_memory(n, src, m_pbase);
			m_len = n;
		}

		__host__
		void reimport(index_type n, device_cptr<T> src)
		{
			reserve(n);
			copy_memory(n, src, m_pbase);
			m_len = n;
		}

	public:
		__host__ __device__
		index_type capacity() const
		{
			return m_capa;
		}

		__host__ __device__
		index_type nelems() const
		{
			return m_len;
		}

		__host__ __device__
		index_type length() const
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
		device_cview1d<T> cblock(index_type i, index_type slen) const
		{
			return device_cview1d<T>(pbase() + i, slen);
		}

		__host__ __device__
		device_view1d<T> block(index_type i, index_type slen)
		{
			return device_view1d<T>(pbase() + i, slen);
		}

	public:
		__device__
		const T& operator[] (index_type i) const
		{
			return m_pbase[i];
		}

		__device__
		T& operator[] (index_type i)
		{
			return m_pbase[i];
		}

		__device__
		const T& operator() (index_type i) const
		{
			return m_pbase[i];
		}

		__device__
		T& operator() (index_type i)
		{
			return m_pbase[i];
		}

	public:
		__host__
		void set_zeros()
		{
			bcs::cuda::set_zeros(m_pbase, m_len);
		}

	private:
		__host__ __device__
		static index_type calc_max(index_type a, index_type b)
		{
			return a < b ? b : a;
		}

		__host__ __device__
		static void swap_(index_type& x, index_type& y)
		{
			index_type t = x;
			x = y;
			y = t;
		}

		__host__ __device__
		static void swap_(pointer& x, pointer& y)
		{
			pointer t = x;
			x = y;
			y = t;
		}

	private:
		index_type m_capa;
		index_type m_len;
		pointer m_pbase;

	}; // end class device_vec


	template<typename T>
	inline __host__ __device__ void swap(device_vec<T>& lhs, device_vec<T>& rhs)
	{
		lhs.swap(rhs);
	}


	// copy between views

	template<typename T>
	inline __host__ void copy(caview1d<T> src, device_view1d<T> dst)
	{
		check_arg(src.dim0() == (index_t)dst.length());
		copy_memory((size_t)dst.length(), make_host_cptr(src.pbase()), dst.pbase());
	}

	template<typename T>
	inline __host__ void copy(device_cview1d<T> src, aview1d<T> dst)
	{
		check_arg((index_t)src.length() == dst.dim0());
		copy_memory((size_t)src.length(), src.pbase(), make_host_ptr(dst.pbase()));
	}

	template<typename T>
	inline __host__ void copy(device_cview1d<T> src, device_view1d<T> dst)
	{
		check_arg(src.length() == dst.length());
		copy_memory((size_t)src.length(), src.pbase(), dst.pbase());
	}



} }

#endif
