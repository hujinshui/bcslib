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

#include <bcslib/matrix/dense_matrix.h>

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
	class direct_vector_reader
	: public IVecReader<direct_vector_reader<T>, T>
	, private noncopyable
	{
	public:
		BCS_ENSURE_INLINE
		explicit direct_vector_reader(const T *p)
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
	class direct_vector_accessor
	: public IVecAccessor<direct_vector_accessor<T>, T>
	, private noncopyable
	{
	public:
		BCS_ENSURE_INLINE
		explicit direct_vector_accessor(T *p)
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
	class continuous_vector_reader
	: public IVecReader<continuous_vector_reader<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
		static_assert(has_continuous_layout<Mat>::value, "Mat should have a continuous layout.");

	public:
		typedef typename matrix_traits<T>::value_type T;

		BCS_ENSURE_INLINE
		explicit continuous_vector_reader(const Mat& vec)
		: m_internal(vec.ptr_data()) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_internal.load_scalar(idx);
		}

	private:
		direct_vector_reader<T> m_internal;
	};

	template<class Mat>
	class continuous_vector_accessor
	: public IVecAccessor<continuous_vector_accessor<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
		static_assert(is_linear_accessible<Mat>::value && !is_readonly_mat<Mat>::value,
				"Mat should have a continuous layout and be NOT readonly.");

	public:
		typedef typename matrix_traits<T>::value_type T;

		BCS_ENSURE_INLINE
		explicit continuous_vector_accessor(Mat& vec)
		: m_internal(vec.ptr_data()) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_internal.load_scalar(idx);
		}

		BCS_ENSURE_INLINE void store_scalar(const index_t idx, const T v)
		{
			m_internal.store_scalar(idx, v);
		}

	private:
		direct_vector_accessor<T> m_internal;
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
		explicit linear_vector_reader(const Mat& vec)
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
		explicit linear_vector_accessor(Mat& vec)
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
		explicit dense_colwise_reader(const Mat& mat)
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
		direct_vector_reader<T> m_internal;
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
		explicit dense_colwise_accessor(Mat& mat)
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
		direct_vector_accessor<T> m_internal;
		const index_t m_ldim;
	};


	template<class Mat>
	class view_colwise_reader
	: public IVecReader<view_colwise_reader<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_mat_view<Mat>::value, "Mat must be a model of IMatrixView.");
#endif

	public:
		typedef typename matrix_traits<T>::value_type T;

		BCS_ENSURE_INLINE
		explicit view_colwise_reader(const Mat& mat)
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
	class cache_linear_reader
	: public IVecReader<cache_linear_reader<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_mat_xpr<Mat>::value, "Mat must be a model of IMatrixXpr.");
#endif

	public:
		typedef typename matrix_traits<T>::value_type T;
		typedef dense_matrix<T, ct_rows<Mat>::value, ct_cols<Mat>::value> cache_t;

		BCS_ENSURE_INLINE
		explicit cache_linear_reader(const Mat& mat)
		: m_cache(mat), m_internal(m_cache.ptr_data()) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_internal.load_scalar(idx);
		}

	private:
		cache_t m_cache;
		direct_vector_reader<T> m_internal;
	};


	template<class Mat>
	class cache_colwise_reader
	: public IVecReader<cache_colwise_reader<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_mat_xpr<Mat>::value, "Mat must be a model of IMatrixXpr.");
#endif

	public:
		typedef typename matrix_traits<T>::value_type T;
		typedef dense_matrix<T, ct_rows<Mat>::value, ct_cols<Mat>::value> cache_t;

		BCS_ENSURE_INLINE
		explicit cache_colwise_reader(const Mat& mat)
		: m_cache(mat), m_internal(m_cache) { }

		BCS_ENSURE_INLINE T load_scalar(const index_t idx) const
		{
			return m_internal.load_scalar(idx);
		}

		BCS_ENSURE_INLINE void next()
		{
			m_internal.next();
		}

	private:
		cache_t m_cache;
		dense_colwise_reader<cache_t> m_internal;
	};


	/********************************************
	 *
	 *  dispatcher
	 *
	 *  Note: this is just default behavior.
	 *
	 *  One may specialize vec_reader and
	 *  vec_writer to provide different behaviors
	 *  for specific classes.
	 *
	 ********************************************/

	template<class Expr>
	struct vec_reader
	{
		typedef typename select_type<has_continuous_layout<Expr>::value,
					continuous_vector_reader<Expr>,
					typename select_type<is_linear_accessible<Expr>::value,
						linear_vector_reader<Expr>,
						cache_linear_reader<Expr>
					>::type
				>::type type;
	};


	template<class Expr>
	struct vec_accessor
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_linear_accessible<Expr>::value && !matrix_traits<Expr>::is_readonly,
				"Expr must be linearly accessible and NOT read-only");
#endif

		typedef typename select_type<has_continuous_layout<Expr>::value,
					continuous_vector_accessor<Expr>,
					linear_vector_accessor<Expr>
				>::type type;
	};

	template<class Expr>
	struct colwise_reader
	{
		typedef typename select_type<is_dense_mat<Expr>::value,
					dense_colwise_reader<Expr>,
					typename select_type<is_mat_view<Expr>::value,
						view_colwise_reader<Expr>,
						cache_colwise_reader<Expr>
					>::type
				>::type type;
	};


	template<class Expr>
	struct colwise_accessor
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_dense_mat<Expr>::value && !matrix_traits<Expr>::is_readonly,
				"Expr must be a dense matrix view and NOT read-only");
#endif

		typedef dense_colwise_accessor<Expr> type;
	};



}

#endif /* MATRIX_ACCESSORS_H_ */
