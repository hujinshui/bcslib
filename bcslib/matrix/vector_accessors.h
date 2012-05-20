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

	template<class Derived, typename T>
	class IVecReader
	{
	public:
		BCS_CRTP_REF

		BCS_ENSURE_INLINE T get(const index_t i) const
		{
			return derived().get(i);
		}
	};


	template<class Derived, typename T>
	class IVecAccessor : public IVecReader<Derived, T>
	{
	public:
		BCS_CRTP_REF

		BCS_ENSURE_INLINE void set(const index_t i, const T& v)
		{
			derived().set(i, v);
		}
	};


	template<class Derived, typename T>
	class IVecReaderBank
	{
	public:
		BCS_CRTP_REF
	};


	template<class Derived, typename T>
	class IVecAccessorBank
	{
	public:
		BCS_CRTP_REF
	};



	/********************************************
	 *
	 *  reader/accessor models
	 *
	 ********************************************/

	template<class Mat>
	class continuous_vector_reader
	: public IVecReader<continuous_vector_reader<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_continuous_layout<Mat>::value, "Mat should have a continuous layout.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit continuous_vector_reader(const Mat& mat)
		: m_data(mat.ptr_data()) { }

		BCS_ENSURE_INLINE value_type get(const index_t i) const
		{
			return m_data[i];
		}

	private:
		const value_type *m_data;
	};


	template<class Mat>
	class continuous_vector_accessor
	: public IVecAccessor<continuous_vector_accessor<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(has_continuous_layout<Mat>::value && !is_readonly_mat<Mat>::value,
				"Mat should have a continuous layout and be NOT readonly.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit continuous_vector_accessor(Mat& mat)
		: m_data(mat.ptr_data()) { }

		BCS_ENSURE_INLINE value_type get(const index_t i) const
		{
			return m_data[i];
		}

		BCS_ENSURE_INLINE void set(const index_t i, const value_type& v)
		{
			m_data[i] = v;
		}

	private:
		value_type *m_data;
	};



	template<class Mat>
	class linear_vector_reader
	: public IVecReader<linear_vector_reader<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_linear_accessible<Mat>::value, "Mat should be linear-accessible.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit linear_vector_reader(const Mat& vec)
		: m_vec(vec) { }

		BCS_ENSURE_INLINE value_type get(const index_t i) const
		{
			return m_vec[i];
		}

	private:
		const Mat& m_vec;
	};

	template<class Mat>
	class linear_vector_accessor
	: public IVecAccessor<linear_vector_accessor<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_linear_accessible<Mat>::value && !is_readonly_mat<Mat>::value,
				"Mat should be linear-accessible and NOT readonly.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit linear_vector_accessor(Mat& vec)
		: m_vec(vec) { }

		BCS_ENSURE_INLINE value_type get(const index_t idx) const
		{
			return m_vec[idx];
		}

		BCS_ENSURE_INLINE void set(const index_t i, const value_type& v)
		{
			m_vec[i] = v;
		}

	private:
		Mat& m_vec;
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
		typedef typename matrix_traits<Mat>::value_type value_type;
		typedef dense_matrix<value_type, ct_rows<Mat>::value, ct_cols<Mat>::value> cache_t;

		BCS_ENSURE_INLINE
		explicit cache_linear_reader(const Mat& mat)
		: m_cache(mat) { }

		BCS_ENSURE_INLINE value_type get(const index_t i) const
		{
			return m_cache[i];
		}

	private:
		cache_t m_cache;
	};


	/********************************************
	 *
	 *  colwise reader / accessor sets
	 *
	 ********************************************/


	template<class Mat>
	class dense_colreaders
	: public IVecReaderBank<dense_colreaders<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_dense_mat<Mat>::value, "Mat must be a model of IDenseMatrix.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit dense_colreaders(const Mat& a)
		: m_mat(a) { }

	public:
		class reader_type : public IVecReader<reader_type, value_type>, private noncopyable
		{
		public:
			BCS_ENSURE_INLINE
			reader_type(const dense_colreaders& host, const index_t j)
			: m_data(col_ptr(host.m_mat, j))
			{
			}

			BCS_ENSURE_INLINE
			value_type get(const index_t i) const
			{
				return m_data[i];
			}

		private:
			const value_type* m_data;
		};

	private:
		const Mat& m_mat;
	};


	template<class Mat>
	class dense_colaccessors
	: public IVecAccessorBank<dense_colaccessors<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_dense_mat<Mat>::value && !matrix_traits<Mat>::is_readonly,
				"Mat must be a model of IDenseMatrix and NOT readonly.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type value_type;

		BCS_ENSURE_INLINE
		explicit dense_colaccessors(Mat& a)
		: m_mat(a) { }

	public:
		class accessor_type : public IVecReader<accessor_type, value_type>, private noncopyable
		{
		public:
			BCS_ENSURE_INLINE
			accessor_type(dense_colaccessors& host, const index_t j)
			: m_data(col_ptr(host.m_mat, j))
			{
			}

			BCS_ENSURE_INLINE
			value_type get(const index_t i) const
			{
				return m_data[i];
			}

			BCS_ENSURE_INLINE
			void set(const index_t i, const value_type& v)
			{
				m_data[i] = v;
			}

		private:
			value_type *m_data;
		};

	private:
		Mat& m_mat;
	};


	template<class Mat>
	class cache_colreaders
	: public IVecReaderBank<cache_colreaders<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_mat_xpr<Mat>::value, "Mat must be a model of IMatrixXpr.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type value_type;
		typedef dense_matrix<value_type, ct_rows<Mat>::value, ct_cols<Mat>::value> cache_t;

		BCS_ENSURE_INLINE
		explicit cache_colreaders(const Mat& a) : m_cache(a) { }

	public:
		class reader_type : public IVecReader<reader_type, value_type>, private noncopyable
		{
		public:
			BCS_ENSURE_INLINE
			reader_type(const cache_colreaders& host, const index_t j)
			: m_data(col_ptr(host.m_cache, j))
			{
			}

			BCS_ENSURE_INLINE value_type get(const index_t i) const
			{
				return m_data[i];
			}

		private:
			const value_type* m_data;
		};

	private:
		cache_t m_cache;
	};


	/********************************************
	 *
	 *  cost model and dispatcher
	 *
	 *  Note: this is just default behavior.
	 *
	 *  One may specialize vec_reader and
	 *  vec_writer to provide different behaviors
	 *  for specific classes.
	 *
	 ********************************************/

	struct as_single_vector_tag { };
	struct by_columns_tag { };
	struct by_short_columns_tag { };

	// default costs

	const int ContinuousAccessCost = 0;

	const int DenseByColumnAccessCost = 200;
	const int DenseByShortColumnAccessCost = 500;

	const int GeneralLinearAccessCost = 200;

	const int CachedAccessCost = 2000;
	const int CachedLinearAccessCost = CachedAccessCost + ContinuousAccessCost;
	const int CachedByColumnAccessCost = CachedAccessCost + DenseByColumnAccessCost;
	const int CachedByShortColumnAccessCost = CachedAccessCost + DenseByShortColumnAccessCost;

	const int ShortColumnBound = 2;


	namespace detail
	{
		template<class Expr, typename Tag> struct default_vecacc_cost;

		template<class Expr>
		struct default_vecacc_cost<Expr, as_single_vector_tag>
		{
			static const int value =
					(has_continuous_layout<Expr>::value ?
							ContinuousAccessCost :
							(is_linear_accessible<Expr>::value ?
									GeneralLinearAccessCost :
									CachedLinearAccessCost) );
		};


		template<class Expr>
		struct default_vecacc_cost<Expr, by_columns_tag>
		{
			static const int value =
					(is_dense_mat<Expr>::value ?
							DenseByColumnAccessCost :
							CachedByColumnAccessCost);
		};


		template<class Expr>
		struct default_vecacc_cost<Expr, by_short_columns_tag>
		{
			static const int value =
					(is_dense_mat<Expr>::value ?
							DenseByShortColumnAccessCost :
							CachedByShortColumnAccessCost );
		};
	}

	template<class Expr, typename Tag>
	struct vecacc_cost
	{
		static const int value = detail::default_vecacc_cost<Expr, Tag>::value;
	};


	template<class Expr1, class Expr2, typename Tag>
	struct vecacc2_cost
	{
		static const int value = vecacc_cost<Expr1, Tag>::value + vecacc_cost<Expr2, Tag>::value;
	};



	// default dispatchers

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
	struct colwise_reader_bank
	{
		typedef typename select_type<is_dense_mat<Expr>::value,
					dense_colreaders<Expr>,
					cache_colreaders<Expr>
				>::type type;
	};


	template<class Expr>
	struct colwise_accessor_bank
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_dense_mat<Expr>::value && !matrix_traits<Expr>::is_readonly,
				"Expr must be a dense matrix view and NOT read-only");
#endif

		typedef dense_colaccessors<Expr> type;
	};



}

#endif /* MATRIX_ACCESSORS_H_ */
