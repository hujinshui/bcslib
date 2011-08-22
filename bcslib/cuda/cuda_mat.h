/**
 * @file cuda_mat.h
 *
 * The classes to represent cuda matrices
 *
 * @author Dahua Lin
 */

#ifndef BCSLIB_CUDA_MAT_H_
#define BCSLIB_CUDA_MAT_H_

#include <bcslib/cuda/cuda_base.h>
#include <bcslib/cuda/cuda_vec.h>
#include <bcslib/array/aview2d.h>

namespace bcs { namespace cuda {

	template<typename T>
	class device_cview2d
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		__host__ __device__
		device_cview2d(const_pointer pbase, index_type nr, index_type nc)
		: m_pbase(pbase), m_nrows(nr), m_ncolumns(nc), m_pitch(sizeof(T) * nc)
		{
		}

		__host__ __device__
		device_cview2d(const_pointer pbase, index_type nr, index_type nc, index_type pitch)
		: m_pbase(pbase), m_nrows(nr), m_ncolumns(nc), m_pitch(pitch)
		{
		}

	public:
		__host__ __device__
		index_type nelems() const
		{
			return m_nrows * m_ncolumns;
		}

		__host__ __device__
		index_type nrows() const
		{
			return m_nrows;
		}

		__host__ __device__
		index_type ncolumns() const
		{
			return m_ncolumns;
		}

		__host__ __device__
		index_type width() const
		{
			return m_ncolumns;
		}

		__host__ __device__
		index_type height() const
		{
			return m_nrows;
		}

		__host__ __device__
		index_type pitch() const
		{
			return m_pitch;
		}

		__host__ __device__
		const_pointer pbase() const
		{
			return m_pbase;
		}

	public:
		__device__
		const T& operator() (index_type i, index_type j) const
		{
			return *(prow_(i) + j);
		}

		__host__ __device__
		const_pointer prowbase(index_type i) const
		{
			return make_device_cptr(prow_(i));
		}

		__host__ __device__
		device_cview1d<T> crow(index_type i) const
		{
			return device_cview1d<T>(prowbase(i), ncolumns());
		}

		__host__ __device__
		device_cview2d crows(index_type i, index_type nr) const
		{
			return device_cview2d(prowbase(i), nr, m_ncolumns, m_pitch);
		}

		__host__ __device__
		device_cview2d cblock(index_type i, index_type j, index_type nr, index_type nc) const
		{
			return device_cview2d(prowbase(i) + j, nr, nc, m_pitch);
		}

	private:
		__host__ __device__
		const T* prow_(index_type i) const
		{
			return (const T*)((const char*)(m_pbase.get()) + i * m_pitch);
		}

	private:
		const_pointer m_pbase;
		index_type m_nrows;
		index_type m_ncolumns;
		index_type m_pitch;

	}; // end class device_cview2d


	template<typename T>
	class device_view2d
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		__host__ __device__
		device_view2d(pointer pbase, index_type nr, index_type nc)
		: m_pbase(pbase), m_nrows(nr), m_ncolumns(nc), m_pitch(sizeof(T) * nc)
		{
		}

		__host__ __device__
		device_view2d(pointer pbase, index_type nr, index_type nc, index_type pitch)
		: m_pbase(pbase), m_nrows(nr), m_ncolumns(nc), m_pitch(pitch)
		{
		}

		__host__ __device__
		device_cview2d<T> cview() const
		{
			return device_cview2d<T>(m_pbase, m_nrows, m_ncolumns, m_pitch);
		}

		__host__ __device__
		operator device_cview2d<T>() const
		{
			return device_cview2d<T>(m_pbase, m_nrows, m_ncolumns, m_pitch);
		}

	public:
		__host__ __device__
		index_type nelems() const
		{
			return m_nrows * m_ncolumns;
		}

		__host__ __device__
		index_type nrows() const
		{
			return m_nrows;
		}

		__host__ __device__
		index_type ncolumns() const
		{
			return m_ncolumns;
		}

		__host__ __device__
		index_type width() const
		{
			return m_ncolumns;
		}

		__host__ __device__
		index_type height() const
		{
			return m_nrows;
		}

		__host__ __device__
		index_type pitch() const
		{
			return m_pitch;
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

	public:
		__device__
		const T& operator() (index_type i, index_type j) const
		{
			return *(prow_(i) + j);
		}

		__device__
		T& operator() (index_type i, index_type j)
		{
			return *(prow_(i) + j);
		}

		__host__ __device__
		const_pointer prowbase(index_type i) const
		{
			return make_device_cptr(prow_(i));
		}

		__host__ __device__
		pointer prowbase(index_type i)
		{
			return make_device_ptr(prow_(i));
		}

		__host__ __device__
		device_cview1d<T> crow(index_type i) const
		{
			return device_cview1d<T>(prowbase(i), ncolumns());
		}

		__host__ __device__
		device_view1d<T> row(index_type i)
		{
			return device_view1d<T>(prowbase(i), ncolumns());
		}

		__host__ __device__
		device_cview2d<T> crows(index_type i, index_type nr) const
		{
			return device_cview2d<T>(prowbase(i), nr, m_ncolumns, m_pitch);
		}

		__host__ __device__
		device_view2d rows(index_type i, index_type nr)
		{
			return device_view2d(prowbase(i), nr, m_ncolumns, m_pitch);
		}

		__host__ __device__
		device_cview2d<T> cblock(index_type i, index_type j, index_type nr, index_type nc) const
		{
			return device_cview2d<T>(prowbase(i) + j, nr, nc, m_pitch);
		}

		__host__ __device__
		device_view2d block(index_type i, index_type j, index_type nr, index_type nc)
		{
			return device_view2d(prowbase(i) + j, nr, nc, m_pitch);
		}

	private:
		__host__ __device__
		const T* prow_(index_type i) const
		{
			return (const T*)((const char*)(m_pbase.get()) + i * m_pitch);
		}

		__host__ __device__
		T* prow_(index_type i)
		{
			return (T*)((char*)(m_pbase.get()) + i * m_pitch);
		}

	private:
		pointer m_pbase;
		index_type m_nrows;
		index_type m_ncolumns;
		index_type m_pitch;

	}; // end class device_view2d



	template<typename T>
	class device_mat
	{
	public:
		BCS_CUDA_DEVICE_AVIEW_DEFS(T)

	public:
		__host__
		explicit device_mat()
		: m_max_nrows(0), m_max_ncols(0), m_nrows(0), m_ncolumns(0)
		, m_pitch(0), m_pbase(pointer(BCS_NULL))
		{
		}

		__host__
		~device_mat()
		{
			device_free(m_pbase);
		}

		__host__
		explicit device_mat(index_type m, index_type n)
		: m_max_nrows(m), m_max_ncols(n)
		, m_nrows(m), m_ncolumns(n)
		, m_pitch(0), m_pbase(alloc_mat(m_max_nrows, m_max_ncols, m_pitch))
		{
		}

		__host__
		explicit device_mat(index_type m, index_type n, index_type max_m, index_type max_n)
		: m_max_nrows(calc_max(m, max_m)), m_max_ncols(calc_max(n, max_n))
		, m_nrows(m), m_ncolumns(n)
		, m_pitch(0), m_pbase(alloc_mat(m_max_nrows, m_max_ncols, m_pitch))
		{
		}

		__host__
		device_mat(index_type m, index_type n, host_cptr<T> src)
		: m_max_nrows(m), m_max_ncols(n)
		, m_nrows(m), m_ncolumns(n)
		, m_pitch(0), m_pbase(alloc_mat(m_max_nrows, m_max_ncols, m_pitch))
		{
			if (m > 0 && n > 0)
			{
				copy_memory2d((size_t)m, (size_t)n,
						src, (size_t)n * sizeof(T), m_pbase, (size_t)m_pitch);
			}
		}

		__host__
		device_mat(index_type m, index_type n, host_cptr<T> src, index_type max_m, index_type max_n)
		: m_max_nrows(calc_max(m, max_m)), m_max_ncols(calc_max(n, max_n))
		, m_nrows(m), m_ncolumns(n)
		, m_pitch(0), m_pbase(alloc_mat(m_max_nrows, m_max_ncols, m_pitch))
		{
			if (m > 0 && n > 0)
			{
				copy_memory2d((size_t)m, (size_t)n,
						src, (size_t)n * sizeof(T), m_pbase, (size_t)m_pitch);
			}
		}

		__host__
		device_mat(index_type m, index_type n, device_cptr<T> src)
		: m_max_nrows(m), m_max_ncols(n)
		, m_nrows(m), m_ncolumns(n)
		, m_pitch(0), m_pbase(alloc_mat(m_max_nrows, m_max_ncols, m_pitch))
		{
			if (m > 0 && n > 0)
			{
				copy_memory2d((size_t)m, (size_t)n,
						src, (size_t)n * sizeof(T), m_pbase, (size_t)m_pitch);
			}
		}

		__host__
		device_mat(index_type m, index_type n, device_cptr<T> src, index_type max_m, index_type max_n)
		: m_max_nrows(calc_max(m, max_m)), m_max_ncols(calc_max(n, max_n))
		, m_nrows(m), m_ncolumns(n)
		, m_pitch(0), m_pbase(alloc_mat(m_max_nrows, m_max_ncols, m_pitch))
		{
			if (m > 0 && n > 0)
			{
				copy_memory2d((size_t)m, (size_t)n,
						src, (size_t)n * sizeof(T), m_pbase, (size_t)m_pitch);
			}
		}


		__host__
		device_mat(const device_mat& src)
		: m_max_nrows(src.m_nrows), m_max_ncols(src.m_ncolumns)
		, m_nrows(src.m_nrows), m_ncolumns(src.m_ncolumns)
		, m_pitch(0), m_pbase(alloc_mat(m_max_nrows, m_max_ncols, m_pitch))
		{
			if (m_nrows > 0 && m_ncolumns > 0)
			{
				copy_memory2d<T>((size_t)m_nrows, (size_t)m_ncolumns,
						src.m_pbase, (size_t)src.m_pitch, m_pbase, (size_t)m_pitch);
			}
		}

		__host__
		device_mat(const device_mat& src, index_type max_m, index_type max_n)
		: m_max_nrows(calc_max(src.m_nrows, max_m)), m_max_ncols(calc_max(src.m_ncolumns, max_n))
		, m_nrows(src.m_nrows), m_ncolumns(src.m_ncolumns)
		, m_pitch(0), m_pbase(alloc_mat(m_max_nrows, m_max_ncols, m_pitch))
		{
			if (m_nrows > 0 && m_ncolumns > 0)
			{
				copy_memory2d<T>((size_t)m_nrows, (size_t)m_ncolumns,
						src.m_pbase, (size_t)src.m_pitch, m_pbase, (size_t)m_pitch);
			}
		}

		__host__
		device_mat& operator = (const device_mat& rhs)
		{
			if (this != &rhs)
			{
				if (m_max_nrows >= rhs.m_nrows && m_max_ncols >= rhs.m_ncolumns)
				{
					copy_memory2d<T>((size_t)rhs.m_nrows, (size_t)rhs.m_ncolumns,
							rhs.m_pbase, rhs.m_pitch, m_pbase, m_pitch);

					m_nrows = rhs.m_nrows;
					m_ncolumns = rhs.m_ncolumns;
				}
				else
				{
					device_mat tmp(rhs, m_max_nrows, m_max_ncols);
					swap(tmp);
				}
			}
			return *this;
		}

		__host__ __device__
		void swap(device_mat& rhs)
		{
			swap_(m_max_nrows, rhs.m_max_nrows);
			swap_(m_max_ncols, rhs.m_max_ncols);

			swap_(m_nrows, rhs.m_nrows);
			swap_(m_ncolumns, rhs.m_ncolumns);

			swap_(m_pitch, rhs.m_pitch);
			swap_(m_pbase, rhs.m_pbase);
		}

	public:
		__host__ __device__
		device_cview2d<T> cview() const
		{
			return device_cview2d<T>(m_pbase, m_nrows, m_ncolumns, m_pitch);
		}

		__host__ __device__
		operator device_cview2d<T>() const
		{
			return device_cview2d<T>(m_pbase, m_nrows, m_ncolumns, m_pitch);
		}

		__host__ __device__
		device_view2d<T> view() const
		{
			return device_view2d<T>(m_pbase, m_nrows, m_ncolumns, m_pitch);
		}

		__host__ __device__
		operator device_view2d<T>() const
		{
			return device_view2d<T>(m_pbase, m_nrows, m_ncolumns, m_pitch);
		}

	public:
		__host__
		void reserve(index_type m, index_type n)  // after reservation, nrows & ncolumns are reset to 0
		{
			if (m_max_nrows < m || m_max_ncols < n)
			{
				index_type ma = calc_max(m, m_max_nrows);
				index_type na = calc_max(n, m_max_ncols);

				m_pbase = alloc_mat(ma, na, m_pitch);

				m_nrows = 0;
				m_ncolumns = 0;
			}
		}

		__host__
		void reimport(index_type m, index_type n, host_cptr<T> src, index_type spitch)
		{
			reserve(m, n);
			copy_memory2d(m, n, src, spitch, m_pbase, m_pitch);
			m_nrows = m;
			m_ncolumns = n;
		}

		__host__
		void reimport(index_type m, index_type n, host_cptr<T> src)
		{
			reimport(m, n, src, sizeof(T) * n);
		}

		__host__
		void reimport(index_type m, index_type n, device_cptr<T> src, index_type spitch)
		{
			reserve(m, n);
			copy_memory2d(m, n, src, spitch, m_pbase, m_pitch);
			m_nrows = m;
			m_ncolumns = n;
		}

		__host__
		void reimport(index_type m, index_type n, device_cptr<T> src)
		{
			reimport(m, n, src, sizeof(T) * n);
		}

	public:
		__host__ __device__
		index_type max_nrows() const
		{
			return m_max_nrows;
		}

		__host__ __device__
		index_type max_ncolumns() const
		{
			return m_max_ncols;
		}

		__host__ __device__
		index_type nelems() const
		{
			return m_nrows * m_ncolumns;
		}

		__host__ __device__
		index_type nrows() const
		{
			return m_nrows;
		}

		__host__ __device__
		index_type ncolumns() const
		{
			return m_ncolumns;
		}

		__host__ __device__
		index_type width() const
		{
			return m_ncolumns;
		}

		__host__ __device__
		index_type height() const
		{
			return m_nrows;
		}

		__host__ __device__
		index_type pitch() const
		{
			return m_pitch;
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

	public:
		__device__
		const T& operator() (index_type i, index_type j) const
		{
			return *(prow_(i) + j);
		}

		__device__
		T& operator() (index_type i, index_type j)
		{
			return *(prow_(i) + j);
		}

		__host__ __device__
		const_pointer prowbase(index_type i) const
		{
			return make_device_cptr(prow_(i));
		}

		__host__ __device__
		pointer prowbase(index_type i)
		{
			return make_device_ptr(prow_(i));
		}

		__host__ __device__
		device_cview1d<T> crow(index_type i) const
		{
			return device_cview1d<T>(prowbase(i), ncolumns());
		}

		__host__ __device__
		device_view1d<T> row(index_type i)
		{
			return device_view1d<T>(prowbase(i), ncolumns());
		}

		__host__ __device__
		device_cview2d<T> crows(index_type i, index_type nr) const
		{
			return device_cview2d<T>(prowbase(i), nr, m_ncolumns, m_pitch);
		}

		__host__ __device__
		device_view2d<T> rows(index_type i, index_type nr)
		{
			return device_view2d<T>(prowbase(i), nr, m_ncolumns, m_pitch);
		}

		__host__ __device__
		device_cview2d<T> cblock(index_type i, index_type j, index_type nr, index_type nc) const
		{
			return device_cview2d<T>(prowbase(i) + j, nr, nc, m_pitch);
		}

		__host__ __device__
		device_view2d<T> block(index_type i, index_type j, index_type nr, index_type nc)
		{
			return device_view2d<T>(prowbase(i) + j, nr, nc, m_pitch);
		}

	private:
		__host__ __device__
		const T* prow_(index_type i) const
		{
			return (const T*)((const char*)(m_pbase.get()) + i * m_pitch);
		}

		__host__ __device__
		T* prow_(index_type i)
		{
			return (T*)((char*)(m_pbase.get()) + i * m_pitch);
		}

		__host__ __device__
		static index_type calc_max(index_type a, index_type b)
		{
			return a < b ? b : a;
		}

		__host__
		static pointer alloc_mat(index_type m, index_type n, index_type& pitch)
		{
			size_t pitch_u;
			pointer p = device_allocate2d<T>((size_t)m, (size_t)n, pitch_u);
			pitch = (index_type)pitch_u;
			return p;
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
		index_type m_max_nrows;
		index_type m_max_ncols;

		index_type m_nrows;
		index_type m_ncolumns;

		index_type m_pitch;
		pointer m_pbase;

	}; // end class device_mat


} }


#endif 
