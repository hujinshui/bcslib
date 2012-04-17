/*
 * @file ref_steprow.h
 *
 * The class to represent step rows
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef REF_STEPROW_H_
#define REF_STEPROW_H_

#include <bcslib/matrix/matrix_fwd.h>
#include "bits/matrix_helpers.h"

namespace bcs
{

	/********************************************
	 *
	 *  StepVector
	 *
	 ********************************************/

	template<typename T, int Dim>
	struct matrix_traits<StepVector<T, VertDir, Dim> >
	{
		MAT_TRAITS_DEFS(T)

		static const int RowDimension = Dim;
		static const int ColDimension = 1;
		static const bool IsReadOnly = false;
	};

	template<typename T, int Dim>
	struct matrix_traits<StepVector<T, HorzDir, Dim> >
	{
		MAT_TRAITS_DEFS(T)

		static const int RowDimension = 1;
		static const int ColDimension = Dim;
		static const bool IsReadOnly = false;
	};


	namespace detail
	{
		template<typename Dir> struct stepvec_helper;

		template<> struct stepvec_helper<VertDir>
		{
			BCS_ENSURE_INLINE
			static index_t calc_offset(index_t step, index_t i, index_t j)
			{
				return i * step;
			}
		};

		template<> struct stepvec_helper<HorzDir>
		{
			BCS_ENSURE_INLINE
			static index_t calc_offset(index_t step, index_t i, index_t j)
			{
				return j * step;
			}
		};
	}




	template<typename T, typename Dir, int Dim>
	class StepVector : public IDenseMatrixView<StepVector<T, Dir, Dim>, T>
	{

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_direction_type<Dir>::value, "Invalid template argument Dir");
		static_assert(Dim == DynamicDim || Dim >= 1, "Invalid template argument Dim");
#endif

		friend class CStepVector<T, Dir, Dim>;

	public:
		MAT_TRAITS_DEFS(T)

		typedef IDenseMatrix<StepVector<T, Dir, Dim>, T> base_type;
		static const int RowDimension = matrix_traits<StepVector>::RowDimension;
		static const int ColDimension = matrix_traits<StepVector>::ColDimension;

	public:

		BCS_ENSURE_INLINE
		StepVector(T *data, index_type len, index_type step)
		: m_base(data), m_len(len), m_step(step)
		{
			if (Dim > 0) check_arg(len == Dim);
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE StepVector& operator = (const IMatrixBase<OtherDerived, T>& other)
		{
			detail::check_assign_fits(*this, other);
			assign(other);
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE StepVector& operator = (const IDenseMatrixView<OtherDerived, T>& other)
		{
			detail::check_assign_fits(*this, other);
			assign(other);
		}

		template<class OtherDerived>
		inline void assign(const IMatrixBase<OtherDerived, T>& other);

		template<class OtherDerived>
		inline void assign(const IDenseMatrixView<OtherDerived, T>& other);

		template<class DstDerived>
		inline void eval_to(IDenseMatrix<DstDerived, T>& dst) const;

		BCS_ENSURE_INLINE void move_forward(index_t x)
		{
			m_base += x;
		}

		BCS_ENSURE_INLINE void move_backward(index_t x)
		{
			m_base -= x;
		}

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_len;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return (size_type)nelems();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return Dir::nrows(m_len);
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return Dir::ncols(m_len);
		}

		BCS_ENSURE_INLINE index_type step() const
		{
			return m_step;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_len == 0;
		}

		BCS_ENSURE_INLINE bool is_vector() const
		{
			return true;
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return detail::stepvec_helper<Dir>::calc_offset(m_step, i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_base[offset(i, j)];
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return m_base[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type idx) const
		{
			return m_base[m_step * idx];
		}

		BCS_ENSURE_INLINE reference operator[] (index_type idx)
		{
			return m_base[m_step * idx];
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

	public:
		void fill(const_reference v)
		{
			T *p = m_base;
			for (index_t i = 0; i < m_len; ++i, p += m_step) *p = v;
		}

		void copy_from(const_pointer src)
		{
			T *p = m_base;
			for (index_t i = 0; i < m_len; ++i, p += m_step) *p = src[i];
		}

	private:
		T *m_base;
		index_t m_len;
		index_t m_step;

	}; // end class StepVector



	/********************************************
	 *
	 *  StepVector
	 *
	 ********************************************/

	template<typename T, int Dim>
	struct matrix_traits<CStepVector<T, VertDir, Dim> >
	{
		MAT_TRAITS_DEFS(T)

		static const int RowDimension = Dim;
		static const int ColDimension = 1;
		static const bool IsReadOnly = false;

		typedef DenseMatrix<T, Dim, 1> eval_return_type;
	};

	template<typename T, int Dim>
	struct matrix_traits<CStepVector<T, HorzDir, Dim> >
	{
		MAT_TRAITS_DEFS(T)

		static const int RowDimension = 1;
		static const int ColDimension = Dim;
		static const bool IsReadOnly = false;

		typedef DenseMatrix<T, 1, Dim> eval_return_type;
	};



	template<typename T, typename Dir, int Dim>
	class CStepVector : public IDenseMatrixView<CStepVector<T, Dir, Dim>, T>
	{

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_direction_type<Dir>::value, "Invalid template argument Dir");
		static_assert(Dim == DynamicDim || Dim >= 1, "Invalid template argument Dim");
#endif

	public:
		MAT_TRAITS_DEFS(T)

		typedef IDenseMatrix<StepVector<T, Dir, Dim>, T> base_type;
		static const int RowDimension = matrix_traits<CStepVector>::RowDimension;
		static const int ColDimension = matrix_traits<CStepVector>::ColDimension;

	public:

		BCS_ENSURE_INLINE
		CStepVector(const T *data, index_type len, index_type step)
		: m_base(data), m_len(len), m_step(step)
		{
			if (Dim > 0) check_arg(len == Dim);
		}

		BCS_ENSURE_INLINE
		CStepVector(const CStepVector& other)
		: m_base(other.m_base), m_len(other.m_len), m_step(other.m_step)
		{
		}

		BCS_ENSURE_INLINE
		CStepVector(const StepVector<T, Dir, Dim>& other)
		: m_base(other.m_base), m_len(other.m_len), m_step(other.m_step)
		{
		}

		template<class DstDerived>
		inline void eval_to(IDenseMatrix<DstDerived, T>& dst) const;

		BCS_ENSURE_INLINE void move_forward(index_t x)
		{
			m_base += x;
		}

		BCS_ENSURE_INLINE void move_backward(index_t x)
		{
			m_base -= x;
		}

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_len;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return (size_type)nelems();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return Dir::nrows(m_len);
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return Dir::ncols(m_len);
		}

		BCS_ENSURE_INLINE index_type step() const
		{
			return m_step;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_len == 0;
		}

		BCS_ENSURE_INLINE bool is_vector() const
		{
			return true;
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return detail::stepvec_helper<Dir>::calc_offset(m_step, i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_base[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type idx) const
		{
			return m_base[m_step * idx];
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

	private:
		const T *m_base;
		index_t m_len;
		index_t m_step;

	}; // end class CStepVector

}

#endif 
