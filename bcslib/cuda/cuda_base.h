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
#include <cuda_runtime.h>
#include <exception>

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
	 * The base class for all exception related to CUDA
	 */
	class cuda_exception : public std::exception
	{
	public:
		__host__ virtual const char *what() const throw()
		{
			return "generic CUDA-related exception.";
		}
	};


	/**
	 * The exception class to represent a CUDA error
	 */
	class cuda_error : public cuda_exception
	{
	public:
		__host__ cuda_error(cudaError_t e)
		: m_err(e) { }

		__host__ const cudaError_t& get() const throw()
		{
			return m_err;
		}

		__host__ virtual const char* what() const throw()
		{
			return ::cudaGetErrorString(m_err);
		}

		__host__ operator cudaError_t() const throw()
		{
			return m_err;
		}

	public:
		__host__ static cuda_error last_error() throw()
		{
			return ::cudaGetLastError();
		}

	private:
		cudaError_t m_err;

	}; // end class cuda_error


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

	public:
		__host__ const T& operator [] (int i) const
		{
			return m_p[i];
		}

		__host__ const T& operator * () const
		{
			return *m_p;
		}

		__host__ const T* operator -> () const
		{
			return m_p;
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

	public:
		__host__ T& operator [] (int i) const
		{
			return m_p[i];
		}

		__host__ T& operator * () const
		{
			return *m_p;
		}

		__host__ T* operator -> () const
		{
			return m_p;
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

	public:
		__device__ const T& operator [] (int i) const
		{
			return m_p[i];
		}

		__device__ const T& operator * () const
		{
			return *m_p;
		}

		__device__ const T* operator -> () const
		{
			return m_p;
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

	public:
		__device__ T& operator [] (int i) const
		{
			return m_p[i];
		}

		__device__ T& operator * () const
		{
			return *m_p;
		}

		__device__ T* operator -> () const
		{
			return m_p;
		}

	private:
		T* m_p;

	}; // end class device_ptr


	template<typename T>
	inline host_cptr<T> make_host_cptr(const T *p)
	{
		return host_cptr<T>(p);
	}

	template<typename T>
	inline host_ptr<T> make_host_ptr(T *p)
	{
		return host_ptr<T>(p);
	}

	template<typename T>
	inline device_cptr<T> make_device_cptr(const T *p)
	{
		return device_cptr<T>(p);
	}

	template<typename T>
	inline device_ptr<T> make_device_ptr(T *p)
	{
		return device_ptr<T>(p);
	}


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
				throw cuda_error(ret);
			}
		}

		return device_ptr<T>(p);
	}

	template<typename T>
	inline __host__ device_ptr<T> device_allocate2d(size_t m, size_t n, size_t& pitch)
	{
		T *p = BCS_NULL;

		if (m > 0 && n > 0)
		{
			cudaError_t ret = ::cudaMallocPitch( (void**)&p, &pitch, n * sizeof(T), m);
			if (ret != cudaSuccess)
			{
				throw cuda_error(ret);
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
	 *  memory copy & set
	 *
	 ******************************************************/

	// 1D

	template<typename T>
	inline __host__ void copy_memory(size_t n, host_cptr<T> src, device_ptr<T> dst)
	{
		::cudaMemcpy(dst.get(), src.get(), n * sizeof(T), cudaMemcpyHostToDevice);
	}

	template<typename T>
	inline __host__ void copy_memory(size_t n, device_cptr<T> src, host_ptr<T> dst)
	{
		::cudaMemcpy(dst.get(), src.get(), n * sizeof(T), cudaMemcpyDeviceToHost);
	}

	template<typename T>
	inline __host__ void copy_memory(size_t n, device_cptr<T> src, device_ptr<T> dst)
	{
		::cudaMemcpy(dst.get(), src.get(), n * sizeof(T), cudaMemcpyDeviceToDevice);
	}

	template<typename T>
	inline __host__ void set_zeros(size_t n, device_ptr<T> dst)
	{
		::cudaMemset(dst.get(), 0, n * sizeof(T));
	}

	// 2D

	template<typename T>
	inline __host__ void copy_memory2d(size_t m, size_t n,
			host_cptr<T> src, size_t spitch, device_ptr<T> dst, size_t dpitch)
	{
		::cudaMemcpy2D(dst.get(), dpitch, src.get(), spitch, n * sizeof(T), m, cudaMemcpyHostToDevice);
	}

	template<typename T>
	inline __host__ void copy_memory2d(size_t m, size_t n,
			device_cptr<T> src, size_t spitch, host_ptr<T> dst, size_t dpitch)
	{
		::cudaMemcpy2D(dst.get(), dpitch, src.get(), spitch, n * sizeof(T), m, cudaMemcpyDeviceToHost);
	}

	template<typename T>
	inline __host__ void copy_memory2d(size_t m, size_t n,
			device_cptr<T> src, size_t spitch, device_ptr<T> dst, size_t dpitch)
	{
		::cudaMemcpy2D(dst.get(), dpitch, src.get(), spitch, n * sizeof(T), m, cudaMemcpyDeviceToDevice);
	}

	template<typename T>
	inline __host__ void set_zeros2d(size_t m, size_t n, device_ptr<T> dst, size_t dpitch)
	{
		::cudaMemset2D(dst.get(), dpitch, 0, n * sizeof(T), m);
	}



} }


#define BCS_CUDA_DEVICE_AVIEW_DEFS(T) \
	typedef T value_type; \
	typedef device_cptr<T> const_pointer; \
	typedef device_ptr<T> pointer; \
	typedef int index_type;


#endif 
