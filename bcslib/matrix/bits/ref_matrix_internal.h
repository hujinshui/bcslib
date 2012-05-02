/**
 * @file ref_matrix_internal.h
 *
 * The internal implementation of ref_matrix
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_REF_MATRIX_INTERNAL_H_
#define BCSLIB_REF_MATRIX_INTERNAL_H_

#include <bcslib/matrix/matrix_properties.h>
#include <bcslib/matrix/bits/offset_helper.h>


namespace bcs { namespace detail {

	/********************************************
	 *
	 *  ref_matrix_internal
	 *
	 ********************************************/

	template<typename T, int CTRows, int CTCols>
	class ref_matrix_internal
	{
	public:
		BCS_ENSURE_INLINE
		ref_matrix_internal(T *pdata, index_t m, index_t n)
		: m_pdata(pdata)
		{
			check_arg(m == CTRows && n == CTCols,
					"Attempted to construct a static ref_matrix with incorrect dimensions.");
		}

		BCS_ENSURE_INLINE index_t nelems() const { return CTRows * CTCols; }

		BCS_ENSURE_INLINE index_t nrows() const { return CTRows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return CTCols; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return CTRows; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

	private:
		T* m_pdata;
	};

	template<typename T, int CTCols>
	class ref_matrix_internal<T, DynamicDim, CTCols>
	{
	public:
		BCS_ENSURE_INLINE
		ref_matrix_internal(T *pdata, index_t m, index_t n)
		: m_pdata(pdata), m_nrows(m)
		{
			check_arg(n == CTCols,
					"Attempted to construct a ref_matrix with incorrect dimension (n != CTCols).");
		}

		BCS_ENSURE_INLINE index_t nelems() const { return m_nrows * CTCols; }

		BCS_ENSURE_INLINE index_t nrows() const { return m_nrows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return CTCols; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return m_nrows; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

	private:
		T* m_pdata;
		index_t m_nrows;
	};


	template<typename T, int CTRows>
	class ref_matrix_internal<T, CTRows, DynamicDim>
	{
	public:
		BCS_ENSURE_INLINE
		ref_matrix_internal(T *pdata, index_t m, index_t n)
		: m_pdata(pdata), m_ncols(n)
		{
			check_arg(m == CTRows,
					"Attempted to construct a ref_matrix with incorrect dimension (m != CTRows).");
		}

		BCS_ENSURE_INLINE index_t nelems() const { return CTRows * m_ncols; }

		BCS_ENSURE_INLINE index_t nrows() const { return CTRows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return m_ncols; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return CTRows; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

	private:
		T* m_pdata;
		index_t m_ncols;
	};


	template<typename T>
	class ref_matrix_internal<T, DynamicDim, DynamicDim>
	{
	public:
		BCS_ENSURE_INLINE
		ref_matrix_internal(T *pdata, index_t m, index_t n)
		: m_pdata(pdata), m_nrows(m), m_ncols(n)
		{
		}

		BCS_ENSURE_INLINE index_t nelems() const { return m_nrows * m_ncols; }

		BCS_ENSURE_INLINE index_t nrows() const { return m_nrows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return m_ncols; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return m_nrows; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

	private:
		T* m_pdata;
		index_t m_nrows;
		index_t m_ncols;
	};


	/********************************************
	 *
	 *  ref_matrix_ex_internal
	 *
	 ********************************************/

	template<class Mat, bool IsRow, bool IsCol>
	struct ref_matrix_internal_ex_linear_offset_helper;

	template<class Mat>
	struct ref_matrix_internal_ex_linear_offset_helper<Mat, false, false>
	{
		BCS_ENSURE_INLINE static index_t get(const Mat& a, const index_t)
		{
			throw invalid_operation(
					"Accessing a (c)ref_matrix_ex object that is not a compile-time vector with linear index is not allowed.");
		}
	};

	template<class Mat>
	struct ref_matrix_internal_ex_linear_offset_helper<Mat, false, true>
	{
		BCS_ENSURE_INLINE static index_t get(const Mat& a, const index_t i)
		{
			return i;
		}
	};

	template<class Mat>
	struct ref_matrix_internal_ex_linear_offset_helper<Mat, true, false>
	{
		BCS_ENSURE_INLINE static index_t get(const Mat& a, const index_t i)
		{
			return i * a.lead_dim();
		}
	};

	template<class Mat>
	struct ref_matrix_internal_ex_linear_offset_helper<Mat, true, true>
	{
		BCS_ENSURE_INLINE static index_t get(const Mat& a, const index_t)
		{
			return 0;
		}
	};


	template<class Mat>
	index_t ref_ex_linear_offset(const Mat& a, const index_t i)
	{
		return ref_matrix_internal_ex_linear_offset_helper<Mat,
				ct_is_row<Mat>::value,
				ct_is_col<Mat>::value>::get(a, i);
	}



	template<typename T, int CTRows, int CTCols>
	class ref_matrix_internal_ex
	{
	public:
		BCS_ENSURE_INLINE
		ref_matrix_internal_ex(T *pdata, index_t m, index_t n, index_t ldim)
		: m_pdata(pdata), m_leaddim(ldim)
		{
			check_arg(m == CTRows && n == CTCols,
					"Attempted to construct a ref_matrix_ex with incorrect dimensions.");
		}

		BCS_ENSURE_INLINE index_t nelems() const { return CTRows * CTCols; }

		BCS_ENSURE_INLINE index_t nrows() const { return CTRows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return CTCols; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return m_leaddim; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

	private:
		T* m_pdata;
		index_t m_leaddim;
	};

	template<typename T, int CTCols>
	class ref_matrix_internal_ex<T, DynamicDim, CTCols>
	{
	public:
		BCS_ENSURE_INLINE
		ref_matrix_internal_ex(T *pdata, index_t m, index_t n, index_t ldim)
		: m_pdata(pdata), m_nrows(m), m_leaddim(ldim)
		{
			check_arg(n == CTCols,
					"Attempted to construct a ref_matrix_ex with incorrect dimension (n != CTCols)");
		}

		BCS_ENSURE_INLINE index_t nelems() const { return m_nrows * CTCols; }

		BCS_ENSURE_INLINE index_t nrows() const { return m_nrows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return CTCols; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return m_leaddim; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

	private:
		T* m_pdata;
		index_t m_nrows;
		index_t m_leaddim;
	};


	template<typename T, int CTRows>
	class ref_matrix_internal_ex<T, CTRows, DynamicDim>
	{
	public:
		BCS_ENSURE_INLINE
		ref_matrix_internal_ex(T *pdata, index_t m, index_t n, index_t ldim)
		: m_pdata(pdata), m_ncols(n), m_leaddim(ldim)
		{
			check_arg(m == CTRows,
					"Attempted to construct a ref_matrix_ex with incorrect dimension (m != CTRows)");
		}

		BCS_ENSURE_INLINE index_t nelems() const { return CTRows * m_ncols; }

		BCS_ENSURE_INLINE index_t nrows() const { return CTRows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return m_ncols; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return m_leaddim; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

	private:
		T* m_pdata;
		index_t m_ncols;
		index_t m_leaddim;
	};


	template<typename T>
	class ref_matrix_internal_ex<T, DynamicDim, DynamicDim>
	{
	public:
		BCS_ENSURE_INLINE
		ref_matrix_internal_ex(T *pdata, index_t m, index_t n, index_t ldim)
		: m_pdata(pdata), m_nrows(m), m_ncols(n), m_leaddim(ldim)
		{
		}

		BCS_ENSURE_INLINE index_t nelems() const { return m_nrows * m_ncols; }

		BCS_ENSURE_INLINE index_t nrows() const { return m_nrows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return m_ncols; }

		BCS_ENSURE_INLINE index_t lead_dim() const { return m_leaddim; }

		BCS_ENSURE_INLINE T* ptr_data() const { return m_pdata; }

	private:
		T* m_pdata;
		index_t m_nrows;
		index_t m_ncols;
		index_t m_leaddim;
	};


} }

#endif /* REF_MATRIX_INTERNAL_H_ */
