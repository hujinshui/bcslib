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
#include <cuda_runtime_api.h>

#if CUDART_VERSION < 3000
#error CUDA Runtime vesion must be at least 3.0
#endif

namespace bcs { namespace cuda {


	/******************************************************
	 *
	 *  Error handling
	 *
	 ******************************************************/

	/**
	 * The exception class to represent a CUDA error
	 */
	class cuda_exception
	{
	public:
		__host__ cuda_exception(cudaError_t e)
		: m_err(e) { }

		__host__ const cudaError_t& get() const
		{
			return m_err;
		}

		__host__ const char* what() const
		{
			return ::cudaGetErrorString(m_err);
		}

		__host__ operator cudaError_t() const
		{
			return m_err;
		}

	public:
		__host__ static cuda_exception last_error()
		{
			return ::cudaGetLastError();
		}

	private:
		cudaError_t m_err;

	}; // end class cudaErr


	/******************************************************
	 *
	 *  pointer classes
	 *
	 ******************************************************/

	/**
	 * A light-weight wrapper of const host pointer
	 */
	template<typename T>
	class host_cptr
	{
	public:
		typedef T element_type;

		__host__ explicit  host_cptr()
		: m_p(BCS_NULL) { }

		__host__ explicit  host_cptr(const T *p)
		: m_p(p) { }

		__host__ const T* get() const
		{
			return m_p;
		}

		__host__ operator bool() const
		{
			return bool(m_p);
		}

	public:
		__host__ host_cptr operator+ (index_t n) const
		{
			return host_cptr(m_p + n);
		}

		__host__ host_cptr operator- (index_t n) const
		{
			return host_cptr(m_p - n);
		}

		__host__ host_cptr& operator++()
		{
			++m_p;
			return *this;
		}

		__host__ host_cptr operator++(int)
		{
			host_cptr temp(*this);
			m_p ++;
			return temp;
		}

		__host__ host_cptr& operator--()
		{
			--m_p;
			return *this;
		}

		__host__ host_cptr operator--(int)
		{
			host_cptr temp(*this);
			m_p --;
			return temp;
		}

		__host__ bool operator == (const host_cptr& rhs) const
		{
			return m_p == rhs.m_p;
		}

		__host__ bool operator != (const host_cptr& rhs) const
		{
			return m_p != rhs.m_p;
		}

	private:
		const T *m_p;

	}; // end class host_cptr


	/**
	 * A light-weight wrapper of host pointer
	 */
	template<typename T>
	class host_ptr
	{
	public:
		typedef T element_type;

		__host__ explicit host_ptr()
		: m_p(BCS_NULL) { }

		__host__ explicit host_ptr(T *p)
		: m_p(p) { }

		__host__ T* get() const
		{
			return m_p;
		}

		__host__ operator bool() const
		{
			return bool(m_p);
		}

		__host__ operator host_cptr<T>() const
		{
			return host_cptr<T>(m_p);
		}

	public:
		__host__ host_ptr operator+ (index_t n) const
		{
			return host_ptr(m_p + n);
		}

		__host__ host_ptr operator- (index_t n) const
		{
			return host_ptr(m_p - n);
		}

		__host__ host_ptr& operator++()
		{
			++m_p;
			return *this;
		}

		__host__ host_ptr operator++(int)
		{
			host_ptr temp(*this);
			m_p ++;
			return temp;
		}

		__host__ host_ptr& operator--()
		{
			--m_p;
			return *this;
		}

		__host__ host_ptr operator--(int)
		{
			host_ptr temp(*this);
			m_p --;
			return temp;
		}

		__host__ bool operator == (const host_ptr& rhs) const
		{
			return m_p == rhs.m_p;
		}

		__host__ bool operator != (const host_ptr& rhs) const
		{
			return m_p != rhs.m_p;
		}

	private:
		T *m_p;

	}; // end class host_ptr


	/**
	 * A light weight wrapper of const pointer on device
	 */
	template<typename T>
	class device_cptr
	{
	public:
		typedef T element_type;

		__host__ __device__ explicit device_cptr()
		: m_p(BCS_NULL) { }

		__host__ __device__ explicit device_cptr(const T *p)
		: m_p(p) { }

		__host__ __device__ const T* get() const
		{
			return m_p;
		}

		__host__ __device__ operator bool() const
		{
			return m_p != BCS_NULL;
		}

	public:
		__host__ __device__ device_cptr operator+ (index_t n) const
		{
			return device_cptr(m_p + n);
		}

		__host__ __device__ device_cptr operator- (index_t n) const
		{
			return device_cptr(m_p - n);
		}

		__host__ __device__ device_cptr& operator++()
		{
			++m_p;
			return *this;
		}

		__host__ __device__ device_cptr operator++(int)
		{
			device_cptr temp(*this);
			m_p ++;
			return temp;
		}

		__host__ __device__ device_cptr& operator--()
		{
			--m_p;
			return *this;
		}

		__host__ __device__ device_cptr operator--(int)
		{
			device_cptr temp(*this);
			m_p --;
			return temp;
		}

		__host__ __device__ bool operator == (const device_cptr& rhs) const
		{
			return m_p == rhs.m_p;
		}

		__host__ __device__ bool operator != (const device_cptr& rhs) const
		{
			return m_p != rhs.m_p;
		}

	private:
		const T* m_p;

	}; // end class device_cptr



	/**
	 * A light-weight wrapper class for device pointer
	 */
	template<typename T>
	class device_ptr
	{
	public:
		typedef T element_type;

		__host__ __device__ explicit device_ptr()
		: m_p(BCS_NULL) { }

		__host__ __device__ explicit device_ptr(T *p)
		: m_p(p) { }

		__host__ __device__ T* get() const
		{
			return m_p;
		}

		__host__ __device__ operator bool() const
		{
			return m_p != BCS_NULL;
		}

		__host__ __device__ operator device_cptr<T>() const
		{
			return device_cptr<T>(m_p);
		}

	public:
		__host__ __device__ device_ptr operator+ (index_t n) const
		{
			return device_ptr(m_p + n);
		}

		__host__ __device__ device_ptr operator- (index_t n) const
		{
			return device_ptr(m_p - n);
		}

		__host__ __device__ device_ptr& operator++()
		{
			++m_p;
			return *this;
		}

		__host__ __device__ device_ptr operator++(int)
		{
			device_ptr temp(*this);
			m_p ++;
			return temp;
		}

		__host__ __device__ device_ptr& operator--()
		{
			--m_p;
			return *this;
		}

		__host__ __device__ device_ptr operator--(int)
		{
			device_ptr temp(*this);
			m_p --;
			return temp;
		}

		__host__ __device__ bool operator == (const device_ptr& rhs) const
		{
			return m_p == rhs.m_p;
		}

		__host__ __device__ bool operator != (const device_ptr& rhs) const
		{
			return m_p != rhs.m_p;
		}

	private:
		T* m_p;

	}; // end class device_ptr


	/******************************************************
	 *
	 *  memory allocation
	 *
	 ******************************************************/

	template<typename T>
	inline __host__ device_ptr<T> device_allocate(size_t n)
	{
		T *p = BCS_NULL;

		if (n > 0)
		{
			cudaError_t ret = ::cudaMalloc( (void**)&p, n * sizeof(T) );
			if (ret != cudaSuccess)
			{
				throw cuda_exception(ret);
			}
		}

		return device_ptr<T>(p);
	}

	template<typename T>
	inline __host__ device_ptr<T> device_allocate2d(size_t w, size_t h, size_t& pitch)
	{
		T *p = BCS_NULL;

		if (w > 0 && h > 0)
		{
			cudaError_t ret = ::cudaMallocPitch( (void**)&p, &pitch, w * sizeof(T), h);
			if (ret != cudaSuccess)
			{
				throw cuda_exception(ret);
			}
		}

		return device_ptr<T>(p);
	}

	template<typename T>
	inline __host__ void device_free(device_ptr<T> p)
	{
		if (p) ::cudaFree(p.get());
	}


	/******************************************************
	 *
	 *  memory copy
	 *
	 ******************************************************/

	template<typename T>
	inline __host__ void copy_memory(host_cptr<T> src, device_ptr<T> dst, size_t n)
	{
		::cudaMemcpy(dst.get(), src.get(), n * sizeof(T), cudaMemcpyHostToDevice);
	}

	template<typename T>
	inline __host__ void copy_memory(device_cptr<T> src, host_ptr<T> dst, size_t n)
	{
		::cudaMemcpy(dst.get(), src.get(), n * sizeof(T), cudaMemcpyDeviceToHost);
	}

	template<typename T>
	inline __host__ void copy_memory(device_cptr<T> src, device_ptr<T> dst, size_t n)
	{
		::cudaMemcpy(dst.get(), src.get(), n * sizeof(T), cudaMemcpyDeviceToDevice);
	}


} }


#define BCS_CUDA_DEVICE_AVIEW_DEFS(T) \
	typedef T value_type; \
	typedef device_cptr<T> const_pointer; \
	typedef device_ptr<T> pointer;


#endif 
