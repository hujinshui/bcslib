/**
 * @file cuda_array.h
 *
 * The classes to represent arrays on CUDA device
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_CUDA_ARRAY_H_
#define BCSLIB_CUDA_ARRAY_H_

#include <bcslib/cuda/cuda_base.h>


#define BCS_CUDA_DEVICE_AVIEW_DEFS(T) \
	typedef T value_type; \
	typedef gpu_cptr<T> const_pointer; \
	typedef gpu_ptr<T> pointer;

namespace bcs { namespace cuda {

	template<typename T>
	class device_caview1d
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		device_caview1d(const_pointer pbase, size_t n)
		: m_pbase(pbase), m_len(n)
		{
		}

	public:
		size_t length() const
		{
			return m_len;
		}

		const_pointer pbase() const
		{
			return m_pbase;
		}

		device_caview1d subview(size_t istart, size_t slen) const
		{
			return device_caview1d(m_pbase + istart, slen);
		}

	private:
		const_pointer m_pbase;
		size_t m_len;

	}; // end class device_caview1d


	template<typename T>
	class device_aview1d
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		device_aview1d(pointer pbase, size_t n)
		: m_pbase(pbase), m_len(n)
		{
		}

		operator device_caview1d<T>() const
		{
			return device_caview1d<T>(m_pbase, m_len);
		}

	public:
		size_t length() const
		{
			return m_len;
		}

		const_pointer pbase() const
		{
			return m_pbase;
		}

		pointer pbase()
		{
			return m_pbase;
		}

		device_caview1d<T> subview(size_t istart, size_t slen) const
		{
			return device_caview1d<T>(m_pbase + istart, slen);
		}

		device_aview1d subview(size_t istart, size_t slen)
		{
			return device_aview1d(m_pbase + istart, slen);
		}

	private:
		pointer m_pbase;
		size_t m_len;
	};



	template<typename T>
	class device_array1d : private noncopyable
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		device_array1d(size_t len)
		: m_pbase(device_allocate<T>(len))
		, m_len(len)
		{
		}

		~device_array1d()
		{
			device_free(m_pbase);
		}

		operator device_caview1d<T>() const
		{
			return device_caview1d<T>(m_pbase, m_len);
		}

		operator device_aview1d<T>() const
		{
			return device_aview1d<T>(m_pbase, m_len);
		}

	public:
		size_t length() const
		{
			return m_len;
		}

		const_pointer pbase() const
		{
			return m_pbase;
		}

		pointer pbase()
		{
			return m_pbase;
		}

	private:
		pointer m_pbase;
		size_t m_len;

	}; // end device_array1d



} }

#endif
