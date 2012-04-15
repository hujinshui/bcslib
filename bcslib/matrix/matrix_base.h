/**
 * @file matrix_base.h
 *
 * Basic definitions for vectors and matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_BASE_H_
#define BCSLIB_MATRIX_BASE_H_

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_math.h>
#include <bcslib/base/arg_check.h>
#include <bcslib/base/mem_op.h>
#include <cstdio>


#define MAT_TRAITS_DEFS(T) \
	typedef T value_type; \
	typedef T* pointer; \
	typedef const T* const_pointer; \
	typedef T& reference; \
	typedef const T& const_reference; \
	typedef size_t size_type; \
	typedef index_t difference_type; \
	typedef index_t index_type;



namespace bcs
{
	// forward declarations

	template<class Derived> struct matrix_traits;

	template<class Derived, typename T> class IMatrixBase;
	template<class Derived, typename T> class IDenseMatrixView;
	template<class Derived, typename T> class IDenseMatrixBlock;
	template<class Derived, typename T> class IDenseMatrix;


	/********************************************
	 *
	 *  CRTP Bases for matrices
	 *
	 ********************************************/

	/**
	 * The interfaces shared by everything that can evaluate into a matrix
	 */
	template<class Derived, typename T>
	class IMatrixBase
	{
	public:
		MAT_TRAITS_DEFS(T)
		BCS_CRTP_REF

		static const index_type RowDimension = matrix_traits<Derived>::RowDimension;
		static const index_type ColDimension = matrix_traits<Derived>::ColDimension;
		typedef typename matrix_traits<Derived>::eval_return_type eval_return_type;

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return derived().is_empty();
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to(IDenseMatrix<DstDerived, T>& dst) const
		{
			derived().eval_to(dst);
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to_block(IDenseMatrixBlock<DstDerived, T>& dst) const
		{
			derived().eval_to_block(dst);
		}

		BCS_ENSURE_INLINE eval_return_type eval() const
		{
			return derived().eval();
		}

	}; // end class IMatrixBase



	namespace detail
	{
		template<class Mat>
		BCS_ENSURE_INLINE
		inline void check_matrix_indices(const Mat& mat, index_t i, index_t j)
		{
#ifndef BCSLIB_NO_DEBUG
			check_arg(i >= 0 && i < mat.nrows() && j >= 0 && j < mat.ncolumns());
#endif
		}
	}


	/**
	 * The interfaces for matrix views
	 */
	template<class Derived, typename T>
	class IDenseMatrixView : public IMatrixBase<Derived, T>
	{
	public:
		MAT_TRAITS_DEFS(T)
		BCS_CRTP_REF

		typedef IMatrixBase<Derived, T> base_type;
		using base_type::RowDimension;
		using base_type::ColDimension;
		typedef typename base_type::eval_return_type eval_return_type;

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return derived().is_empty();
		}

		BCS_ENSURE_INLINE value_type elem(index_type i, index_type j) const
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE value_type operator() (index_type i, index_type j) const
		{
			return derived().operator()(i, j);
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to(IDenseMatrix<DstDerived, T>& dst) const
		{
			derived().eval_to(dst);
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to_block(IDenseMatrixBlock<DstDerived, T>& dst) const
		{
			derived().eval_to_block(dst);
		}

		BCS_ENSURE_INLINE eval_return_type eval() const
		{
			return derived().eval();
		}

	}; // end class IDenseMatrixView


	/**
	 * The interfaces for matrix blocks
	 */
	template<class Derived, typename T>
	class IDenseMatrixBlock : public IDenseMatrixView<Derived, T>
	{
	public:
		MAT_TRAITS_DEFS(T)
		BCS_CRTP_REF

		typedef IDenseMatrixView<Derived, T> base_type;
		using base_type::RowDimension;
		using base_type::ColDimension;
		typedef typename base_type::eval_return_type eval_return_type;

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return derived().is_empty();
		}

		BCS_ENSURE_INLINE const_pointer ptr_base() const
		{
			return derived().ptr_base();
		}

		BCS_ENSURE_INLINE pointer ptr_base()
		{
			return derived().ptr_base();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return derived().lead_dim();
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return derived().operator()(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return derived().operator()(i, j);
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to(IDenseMatrix<DstDerived, T>& dst) const
		{
			derived().eval_to(dst);
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to_block(IDenseMatrixBlock<DstDerived, T>& dst) const
		{
			derived().eval_to_block(dst);
		}

		BCS_ENSURE_INLINE eval_return_type eval() const
		{
			return derived().eval();
		}

		BCS_ENSURE_INLINE void zero()
		{
			derived().zero();
		}

		BCS_ENSURE_INLINE void fill(const_reference v)
		{
			derived().fill(v);
		}

		BCS_ENSURE_INLINE void copy_from(const_pointer src)
		{
			derived().copy_from(src);
		}

	}; // end class IDenseMatrixBlock



	/**
	 * The interfaces for matrices with continuous layout
	 */
	template<class Derived, typename T>
	class IDenseMatrix : public IDenseMatrixBlock<Derived, T>
	{
	public:
		MAT_TRAITS_DEFS(T)
		BCS_CRTP_REF

		typedef IDenseMatrixBlock<Derived, T> base_type;
		using base_type::RowDimension;
		using base_type::ColDimension;
		typedef typename base_type::eval_return_type eval_return_type;

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return derived().is_empty();
		}

		BCS_ENSURE_INLINE const_pointer ptr_base() const
		{
			return derived().ptr_base();
		}

		BCS_ENSURE_INLINE pointer ptr_base()
		{
			return derived().ptr_base();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return derived().lead_dim();
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type idx) const
		{
			return derived().operator[](idx);
		}

		BCS_ENSURE_INLINE reference operator[] (index_type idx)
		{
			return derived().operator[](idx);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			return derived().operator()(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			return derived().operator()(i, j);
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to(IDenseMatrix<DstDerived, T>& dst) const
		{
			derived().eval_to(dst);
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to_block(IDenseMatrixBlock<DstDerived, T>& dst) const
		{
			derived().eval_to_block(dst);
		}

		BCS_ENSURE_INLINE eval_return_type eval() const
		{
			return derived().eval();
		}

		BCS_ENSURE_INLINE void zero()
		{
			derived().zero();
		}

		BCS_ENSURE_INLINE void fill(const_reference v)
		{
			derived().fill(v);
		}

		BCS_ENSURE_INLINE void copy_from(const_pointer src)
		{
			derived().copy_from(src);
		}

	}; // end class IDenseMatrix



	/********************************************
	 *
	 *  Generic operations
	 *
	 ********************************************/

	template<class Derived1, typename T1, class Derived2, typename T2>
	void is_same_size(const IMatrixBase<Derived1, T1>& A, const IMatrixBase<Derived2, T2>& B)
	{
		return A.nrows() == B.nrows() && A.ncolumns() == B.ncolumns();
	}


	template<class Derived, typename T>
	void fill(IDenseMatrixBlock<Derived, T>& X, const T& v)
	{
		fill_elems_2d((size_t)X.nrows(), (size_t)X.ncolumns(), X.ptr_base(), (size_t)X.lead_dim(), v);
	}

	template<class Derived, typename T>
	void zero(IDenseMatrixBlock<Derived, T>& X)
	{
		zero_elems_2d((size_t)X.nrows(), (size_t)X.ncolumns(), X.ptr_base(), (size_t)X.lead_dim());
	}

	template<class LDerived, class RDerived, typename T>
	void copy(const IDenseMatrixBlock<LDerived, T>& src, IDenseMatrixBlock<RDerived, T>& dst)
	{
		check_arg( is_same_size(src, dst) );
		copy_elems_2d((size_t)src.nrows(), (size_t)src.ncolumns(),
				src.ptr_base(), (size_t)src.lead_dim(),
				dst.ptr_base(), (size_t)dst.lead_dim());
	}

	template<class LDerived, class RDerived, typename T>
	void copy(const IDenseMatrix<LDerived, T>& src, IDenseMatrix<RDerived, T>& dst)
	{
		check_arg( is_same_size(src, dst) );
		copy_elems(src.size(), src.ptr_base(), dst.ptr_base());
	}

	template<class Derived, typename T>
	void printf_mat(const char *fmt, const IDenseMatrixView<Derived, T>& X,
			const char *pre_line = 0, const char *delim = "\n")
	{
		index_t m = X.nrows();
		index_t n = X.ncolumns();

		for (index_t i = 0; i < m; ++i)
		{
			if (pre_line) std::printf("%s", pre_line);
			for (index_t j = 0; j < n; ++j)
			{
				std::printf(fmt, X.elem(i, j));
			}
			std::printf("%s", delim);
		}
	}


}


// some useful macros

#define BCS_MATRIX_TYPEDEFS0(TName, prefix) \
	typedef TName<double>   prefix##_f64; \
	typedef TName<float>    prefix##_f32; \
	typedef TName<int32_t>  prefix##_i32; \
	typedef TName<uint32_t> prefix##_u32; \
	typedef TName<int16_t>  prefix##_i16; \
	typedef TName<uint16_t> prefix##_u16; \
	typedef TName<int8_t>   prefix##_i8; \
	typedef TName<uint8_t>  prefix##_u8; \
	typedef TName<bool>     prefix##_bool;

#define BCS_MATRIX_TYPEDEFS1(TName, prefix, Dim) \
	typedef TName<double,   Dim> prefix##_f64; \
	typedef TName<float,    Dim> prefix##_f32; \
	typedef TName<int32_t,  Dim> prefix##_i32; \
	typedef TName<uint32_t, Dim> prefix##_u32; \
	typedef TName<int16_t,  Dim> prefix##_i16; \
	typedef TName<uint16_t, Dim> prefix##_u16; \
	typedef TName<int8_t,   Dim> prefix##_i8; \
	typedef TName<uint8_t,  Dim> prefix##_u8; \
	typedef TName<bool,     Dim> prefix##_bool;

#define BCS_MATRIX_TYPEDEFS2(TName, prefix, RDim, CDim) \
	typedef TName<double,   RDim, CDim> prefix##_f64; \
	typedef TName<float,    RDim, CDim> prefix##_f32; \
	typedef TName<int32_t,  RDim, CDim> prefix##_i32; \
	typedef TName<uint32_t, RDim, CDim> prefix##_u32; \
	typedef TName<int16_t,  RDim, CDim> prefix##_i16; \
	typedef TName<uint16_t, RDim, CDim> prefix##_u16; \
	typedef TName<int8_t,   RDim, CDim> prefix##_i8; \
	typedef TName<uint8_t,  RDim, CDim> prefix##_u8; \
	typedef TName<bool,     RDim, CDim> prefix##_bool;



#endif
