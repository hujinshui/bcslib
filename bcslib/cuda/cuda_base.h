/**
 * @file cuda_base.h
 *
 * The basic facilities for using CUDA
 *
 * (C++ wrappers for basic CUDA functions)
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_CUDA_BASE_H
#define BCSLIB_CUDA_BASE_H

#include <bcslib/base/basic_defs.h>
#include <cuda.h>

/*
#if __CUDA_API_VERSION < 3000
#error CUDA API vesion must be at least 3.0
#endif
*/
namespace bcs { namespace cuda {


	/******************************************************
	 *
	 *  Error handling
	 *
	 ******************************************************/

	/**
	 * The class to represent a CUDA error
	 */
	class cuda_err
	{
	public:
		cuda_err(cudaError_t e)
		: m_err(e) { }

		const cudaError_t& get() const
		{
			return m_err;
		}

		const char* what() const
		{
			return ::cudaGetErrorString(m_err);
		}

		operator cudaError_t() const
		{
			return m_err;
		}

	public:
		static cuda_err last_error()
		{
			return ::cudaGetLastErr();
		}

	private:
		cudaError_t m_err;

	}; // end class cudaErr


	/******************************************************
	 *
	 *  GPU pointer
	 *
	 ******************************************************/

	/**
	 * A light-weight wrapper class for device pointer
	 */
	template<typename T>
	class gpu_ptr
	{
	public:
		typedef T value_type;

		explicit gpu_ptr()
		: m_p(BCS_NULL) { }

		explicit gpu_ptr(T *p)
		: m_p(p) { }

		T* get() const
		{
			return m_p;
		}

		operator bool() const
		{
			return m_p != BCS_NULL;
		}

	public:
		gpu_ptr operator+ (index_t n) const
		{
			return gpu_ptr(m_p + n);
		}

		gpu_ptr operator- (index_t n) const
		{
			return gpu_ptr(m_p - n);
		}

		gpu_ptr& operator++()
		{
			++m_p;
			return *this;
		}

		gpu_ptr operator++(int)
		{
			gpu_ptr temp(*this);
			m_p ++;
			return temp;
		}

		gpu_ptr& operator--()
		{
			--m_p;
			return *this;
		}

		gpu_ptr operator--(int)
		{
			gpu_ptr temp(*this);
			m_p --;
			return temp;
		}

	private:
		T* m_p;

	}; // end class gpu_ptr


	template<typename T>
	class gpu_cptr
	{
	public:
		typedef T value_type;

		explicit gpu_cptr()
		: m_p(BCS_NULL) { }

		explicit gpu_cptr(const T *p)
		: m_p(p) { }

		gpu_cptr(const gpu_ptr<T>& gp)
		: m_p(gp.get()) { }

		const T* get() const
		{
			return m_p;
		}

		operator bool() const
		{
			return m_p != BCS_NULL;
		}

	public:
		gpu_cptr operator+ (index_t n) const
		{
			return gpu_cptr(m_p + n);
		}

		gpu_cptr operator- (index_t n) const
		{
			return gpu_cptr(m_p - n);
		}

		gpu_cptr& operator++()
		{
			++m_p;
			return *this;
		}

		gpu_cptr operator++(int)
		{
			gpu_cptr temp(*this);
			m_p ++;
			return temp;
		}

		gpu_cptr& operator--()
		{
			--m_p;
			return *this;
		}

		gpu_cptr operator--(int)
		{
			gpu_cptr temp(*this);
			m_p --;
			return temp;
		}

	private:
		const T* m_p;

	}; // end class gpu_cptr


} }


#endif 
