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
	 *  - vector reader:
	 *
	 *  	v = rdr.load_scalar(i);
	 *
	 *  - vector accessor:
	 *
	 *  	acc.store_scalar(i, v);
	 *
	 *  - vector reader set:
	 *
	 *  	rdr_set rs(mat);
	 *  	rdr_set::reader_type rdr(rs, j);
	 *  	v = rdr.load_scalar(i);
	 *
	 *  - vector accessor set:
	 *
	 *  	acc_set ws(mat);
	 *  	acc_set::accessor_type acc(ws, j);
	 *  	acc.store_scalar(i, v);
	 *
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


	template<class Derived, typename T>
	class IVecReaderSet
	{
	public:
		BCS_CRTP_REF
	};


	template<class Derived, typename T>
	class IVecAccessorSet
	{
	public:
		BCS_CRTP_REF
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
	class cache_linear_reader
	: public IVecReader<cache_linear_reader<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_mat_xpr<Mat>::value, "Mat must be a model of IMatrixXpr.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type T;
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


	/********************************************
	 *
	 *  colwise reader / accessor sets
	 *
	 ********************************************/


	template<class Mat>
	class dense_colwise_reader_set
	: public IVecReaderSet<dense_colwise_reader_set<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_dense_mat<Mat>::value, "Mat must be a model of IDenseMatrix.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type T;

		BCS_ENSURE_INLINE
		explicit dense_colwise_reader_set(const Mat& a)
		: m_mat(a) { }

	public:
		class reader_type : public IVecReader<reader_type, T>, private noncopyable
		{
		public:
			BCS_ENSURE_INLINE
			reader_type(const dense_colwise_reader_set& host, const index_t j)
			: m_internal(col_ptr(host.m_mat, j))
			{
			}

			BCS_ENSURE_INLINE
			T load_scalar(const index_t i) const
			{
				return m_internal.load_scalar(i);
			}

		private:
			direct_vector_reader<T> m_internal;
		};

	private:
		const Mat& m_mat;
	};


	template<class Mat>
	class dense_colwise_accessor_set
	: public IVecAccessorSet<dense_colwise_accessor_set<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_dense_mat<Mat>::value && !matrix_traits<Mat>::is_readonly,
				"Mat must be a model of IDenseMatrix and NOT readonly.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type T;

		BCS_ENSURE_INLINE
		explicit dense_colwise_accessor_set(Mat& a)
		: m_mat(a) { }

	public:
		class accessor_type : public IVecReader<accessor_type, T>, private noncopyable
		{
		public:
			BCS_ENSURE_INLINE
			accessor_type(dense_colwise_accessor_set& host, const index_t j)
			: m_internal(col_ptr(host.m_mat, j))
			{
			}

			BCS_ENSURE_INLINE
			T load_scalar(const index_t i) const
			{
				return m_internal.load_scalar(i);
			}

			BCS_ENSURE_INLINE
			void store_scalar(const index_t i, const T v)
			{
				m_internal.store_scalar(i, v);
			}

		private:
			direct_vector_accessor<T> m_internal;
		};

	private:
		Mat& m_mat;
	};


	template<class Mat>
	class view_colwise_reader_set
	: public IVecReaderSet<view_colwise_reader_set<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_mat_view<Mat>::value, "Mat must be a model of IMatrixView.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type T;

		BCS_ENSURE_INLINE
		explicit view_colwise_reader_set(const Mat& mat)
		: m_mat(mat) { }

	public:
		class reader_type : public IVecReader<reader_type, T>, private noncopyable
		{
		public:
			BCS_ENSURE_INLINE
			reader_type(const view_colwise_reader_set& host, const index_t j)
			: m_mat(host.m_mat), m_icol(j)
			{
			}

			BCS_ENSURE_INLINE
			T load_scalar(const index_t i) const
			{
				return m_mat(i, m_icol);
			}

		private:
			const Mat& m_mat;
			const index_t m_icol;
		};

	private:
		const Mat& m_mat;
	};


	template<class Mat>
	class cache_colwise_reader_set
	: public IVecReaderSet<cache_colwise_reader_set<Mat>, typename matrix_traits<Mat>::value_type>
	, private noncopyable
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_mat_xpr<Mat>::value, "Mat must be a model of IMatrixXpr.");
#endif

	public:
		typedef typename matrix_traits<Mat>::value_type T;
		typedef dense_matrix<T, ct_rows<Mat>::value, ct_cols<Mat>::value> cache_t;

		BCS_ENSURE_INLINE
		explicit cache_colwise_reader_set(const Mat& a) : m_cache(a) { }

	public:
		class reader_type : public IVecReader<reader_type, T>, private noncopyable
		{
		public:
			BCS_ENSURE_INLINE
			reader_type(const cache_colwise_reader_set& host, const index_t j)
			: m_internal(col_ptr(host.m_cache, j))
			{
			}

			BCS_ENSURE_INLINE T load_scalar(const index_t i) const
			{
				return m_internal.load_scalar(i);
			}

		private:
			direct_vector_reader<T> m_internal;
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

	template<class Expr, typename Tag> struct vecacc_cost;


	// default costs

	const int NoOverheadAccessCost = 0;

	const int DenseByColumnAccessCost = 200;
	const int DenseByShortColumnAccessCost = 500;

	const int GeneralLinearAccessCost = 200;
	const int GeneralByColumnAccessCost = 500;
	const int GeneralByShortColumnAccessCost = 800;

	const int CachedAccessCost = 2000;
	const int CachedLinearAccessCost = CachedAccessCost + NoOverheadAccessCost;
	const int CachedByColumnAccessCost = CachedAccessCost + DenseByColumnAccessCost;
	const int CachedByShortColumnAccessCost = CachedAccessCost + DenseByShortColumnAccessCost;

	const int ShortColumnBound = 2;

	template<class Expr>
	struct vecacc_cost<Expr, as_single_vector_tag>
	{
		static const int value =
				(has_continuous_layout<Expr>::value ?
						NoOverheadAccessCost :
						(is_linear_accessible<Expr>::value ?
								GeneralLinearAccessCost :
								CachedLinearAccessCost) );
	};


	template<class Expr>
	struct vecacc_cost<Expr, by_columns_tag>
	{
		static const int value =
				(is_dense_mat<Expr>::value ?
						DenseByColumnAccessCost :
						(is_mat_view<Expr>::value ?
								GeneralByColumnAccessCost :
								CachedByColumnAccessCost) );
	};


	template<class Expr>
	struct vecacc_cost<Expr, by_short_columns_tag>
	{
		static const int value =
				(is_dense_mat<Expr>::value ?
						DenseByShortColumnAccessCost :
						(is_mat_view<Expr>::value ?
								GeneralByShortColumnAccessCost :
								CachedByShortColumnAccessCost) );
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
	struct colwise_reader_set
	{
		typedef typename select_type<is_dense_mat<Expr>::value,
					dense_colwise_reader_set<Expr>,
					typename select_type<is_mat_view<Expr>::value,
						view_colwise_reader_set<Expr>,
						cache_colwise_reader_set<Expr>
					>::type
				>::type type;
	};


	template<class Expr>
	struct colwise_accessor_set
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_dense_mat<Expr>::value && !matrix_traits<Expr>::is_readonly,
				"Expr must be a dense matrix view and NOT read-only");
#endif

		typedef dense_colwise_accessor_set<Expr> type;
	};



}

#endif /* MATRIX_ACCESSORS_H_ */
