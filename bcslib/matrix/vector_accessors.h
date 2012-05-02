/**
 * @file matrix_accessors.h
 *
 * The devices to access matrix elements (as scalars and as vectors)
 *
 * This is the core engine to support efficient matrix operations
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_VECTOR_ACCESSORS_H_
#define BCSLIB_VECTOR_ACCESSORS_H_

#include <bcslib/matrix/matrix_concepts.h>

namespace bcs
{

	/********************************************
	 *
	 *  Key concepts
	 *
	 ********************************************/

	template<class Derived, typename T>
	class IVecReader
	{
	public:
		BCS_CRTP_REF

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return derived().load_scalar(idx);
		}
	};


	template<class Derived, typename T>
	class IVecAccessor : public IVecReader<Derived, T>
	{
	public:
		BCS_CRTP_REF

		BCS_ENSURE_INLINE void store_scalar(const index_t idx, const T v)
		{
			derived().store_scalar(idx, v);
		}
	};


	/********************************************
	 *
	 *  reader/accessor models
	 *
	 ********************************************/

	template<typename T>
	class continuous_vector_reader
	: public IVecReader<continuous_vector_reader<T>, T>
	, private noncopyable
	{
	public:
		BCS_ENSURE_INLINE
		continuous_vector_reader(const T *p)
		: m_ptr(p) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_ptr[idx];
		}

		void move(const index_t offset)
		{
			m_ptr += offset;
		}

	private:
		const T *m_ptr;
	};


	template<typename T>
	class continuous_vector_accessor
	: public IVecAccessor<continuous_vector_accessor<T>, T>
	, private noncopyable
	{
	public:
		BCS_ENSURE_INLINE
		continuous_vector_accessor(T *p)
		: m_ptr(p) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_ptr[idx];
		}

		BCS_ENSURE_INLINE void store_scalar(const index_t idx, const T v)
		{
			m_ptr[idx] = v;
		}

		void move(const index_t offset)
		{
			m_ptr += offset;
		}

	private:
		T *m_ptr;
	};


	template<class Mat>
	class linear_vector_reader
	: public IVecReader<linear_vector_reader<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
		static_assert(is_linear_accessible<Mat>::value, "Mat should be linear-accessible.");

	public:
		typedef typename matrix_traits<T>::value_type T;

		BCS_ENSURE_INLINE
		linear_vector_reader(const Mat& vec)
		: m_vec(vec) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_vec[idx];
		}

	private:
		const Mat& m_vec;
	};

	template<class Mat>
	class linear_vector_accessor
	: public IVecAccessor<linear_vector_accessor<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
		static_assert(is_linear_accessible<Mat>::value && !is_readonly_mat<Mat>::value,
				"Mat should be linear-accessible and NOT readonly.");

	public:
		typedef typename matrix_traits<T>::value_type T;

		BCS_ENSURE_INLINE
		linear_vector_accessor(Mat& vec)
		: m_vec(vec) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_vec[idx];
		}

		BCS_ENSURE_INLINE void store_scalar(const index_t idx, const T v)
		{
			m_vec[idx] = v;
		}

	private:
		Mat& m_vec;
	};


	template<class Mat>
	class dense_colwise_reader
	: public IVecReader<dense_colwise_reader<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_dense_mat<Mat>::value, "Mat must be a model of IDenseMatrix.");
#endif

	public:
		typedef typename matrix_traits<T>::value_type T;

		BCS_ENSURE_INLINE
		dense_colwise_reader(const Mat& mat)
		: m_internal(mat.ptr_data()), m_ldim(mat.lead_dim()) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_internal.load_scalar(idx);
		}

		BCS_ENSURE_INLINE void next()
		{
			m_internal.move(m_ldim);
		}

	private:
		continuous_vector_reader<T> m_internal;
		const index_t m_ldim;
	};


	template<class Mat>
	class dense_colwise_accessor
	: public IVecAccessor<dense_colwise_accessor<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_dense_mat<Mat>::value && !is_readonly_mat<Mat>::value,
				"Mat must be a model of IDenseMatrix and be NOT readonly.");
#endif

	public:
		typedef typename matrix_traits<T>::value_type T;

		BCS_ENSURE_INLINE
		dense_colwise_accessor(Mat& mat)
		: m_internal(mat.ptr_data()), m_ldim(mat.lead_dim()) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_internal.load_scalar(idx);
		}

		BCS_ENSURE_INLINE void store_scalar(const index_t idx, const T v)
		{
			m_internal.store_scalar(idx, v);
		}

		BCS_ENSURE_INLINE void next()
		{
			m_internal.move(m_ldim);
		}

	private:
		continuous_vector_accessor<T> m_internal;
		const index_t m_ldim;
	};


	template<class Mat>
	class regular_colwise_reader
	: public IVecReader<regular_colwise_reader<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_mat_view<Mat>::value, "Mat must be a model of IMatrixView.");
#endif

	public:
		typedef typename matrix_traits<T>::value_type T;

		BCS_ENSURE_INLINE
		regular_colwise_reader(const Mat& mat)
		: m_mat(mat), m_icol(0) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_mat(idx, m_icol);
		}

		BCS_ENSURE_INLINE void next()
		{
			++ m_icol;
		}

	private:
		Mat& m_mat;
		index_t m_icol;
	};


	template<class Mat>
	class regular_colwise_accessor
	: public IVecAccessor<regular_colwise_accessor<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_regular_mat<Mat>::value && !is_readonly_mat<Mat>::value,
				"Mat must be a model of IMatrixView and be NOT read-only.");
#endif

	public:
		typedef typename matrix_traits<T>::value_type T;

		BCS_ENSURE_INLINE
		regular_colwise_accessor(const Mat& mat)
		: m_mat(mat), m_icol(0) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_mat(idx, m_icol);
		}

		BCS_ENSURE_INLINE void store_scalar(const index_t idx, const T v)
		{
			m_mat(idx, m_icol) = v;
		}

		BCS_ENSURE_INLINE void next()
		{
			++ m_icol;
		}

	private:
		Mat& m_mat;
		index_t m_icol;
	};


}

#endif /* MATRIX_ACCESSORS_H_ */
