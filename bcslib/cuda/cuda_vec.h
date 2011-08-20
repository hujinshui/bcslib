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
	class device_vec_cview
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		device_vec_cview(const_pointer pbase, size_t len)
		: m_pbase(const_cast<T*>(pbase.get())), m_len(len)
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

		device_vec_cview subview(size_t i, size_t slen) const
		{
			return device_vec_cview(pbase() + i, slen);
		}

	protected:
		pointer m_pbase;
		size_t m_len;

	}; // end class device_vec_cview


	template<typename T>
	class device_vec_view : public device_vec_cview<T>
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		device_vec_view(pointer pbase, size_t len)
		: device_vec_cview<T>(pbase, len)
		{
		}

	public:
		const_pointer pbase() const
		{
			return this->m_pbase;
		}

		pointer pbase()
		{
			return this->m_pbase;
		}

		device_vec_cview<T> subview(size_t i, size_t slen) const
		{
			return device_vec_cview<T>(pbase() + i, slen);
		}

		device_vec_view subview(size_t i, size_t slen)
		{
			return device_vec_view(pbase() + i, slen);
		}

	}; // end class device_vec_view


	namespace _detail
	{
		template<typename T>
		class device_vec_storage : private noncopyable
		{
		public:
			BCS_CUDA_DEVICE_AVIEW_DEFS(T)

			device_vec_storage(size_t n)
			: m_ptr(device_allocate<T>(n))
			{
			}

			device_vec_storage(size_t n, const_pointer src)
			: m_ptr(device_allocate<T>(n))
			{
				if (n > 0)
				{
					::cudaMemcpy(m_ptr.get(), src.get(), n * sizeof(T),
							cudaMemcpyDeviceToDevice);
				}
			}

			~device_vec_storage()
			{
				device_free(m_ptr);
			}

			pointer get_ptr()
			{
				return m_ptr;
			}

		private:
			pointer m_ptr;
		};
	}


	template<typename T>
	class device_vec : private _detail::device_vec_storage<T>, public device_vec_view<T>
	{
	public:


	};



} }

#endif
