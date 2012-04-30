/**
 * @file slicewise_proxy.h
 *
 * The row-wise and column-wise proxy
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_SLICEWISE_PROXY_H_
#define BCSLIB_SLICEWISE_PROXY_H_

#include <bcslib/matrix/matrix_base.h>

namespace bcs
{
	template<class Mat, typename T>
	class const_colwise_proxy
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_base_of<IMatrixXpr<Mat, T>, Mat>::value,
				"Mat must be a model of IMatrixXpr");
#endif

	public:
		typedef T value_type;

		const_colwise_proxy(const Mat& mat) : m_ref(const_cast<Mat&>(mat)) { }

		const Mat& ref() const { return m_ref; }

	protected:
		Mat& m_ref;
	};


	template<class Mat, typename T>
	struct colwise_proxy : public const_colwise_proxy<Mat, T>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_base_of<IMatrixXpr<Mat, T>, Mat>::value,
				"Mat must be a model of IMatrixXpr");
#endif
		typedef T value_type;

		colwise_proxy(Mat& mat)
		: const_colwise_proxy<Mat, T>(mat) { }

		const Mat& ref() const { return this->m_ref; }
		Mat& ref() { return this->m_ref; }

	};

	template<class Mat, typename T>
	class const_rowwise_proxy
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_base_of<IMatrixXpr<Mat, T>, Mat>::value,
				"Mat must be a model of IMatrixXpr");
#endif

	public:
		typedef T value_type;

		const_rowwise_proxy(const Mat& mat) : m_ref(const_cast<Mat&>(mat)) { }

		const Mat& ref() const { return m_ref; }

	protected:
		Mat& m_ref;
	};


	template<class Mat, typename T>
	struct rowwise_proxy : public const_rowwise_proxy<Mat, T>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_base_of<IMatrixXpr<Mat, T>, Mat>::value,
				"Mat must be a model of IMatrixXpr");
#endif
		typedef T value_type;

		rowwise_proxy(Mat& mat)
		: const_rowwise_proxy<Mat, T>(mat) { }

		const Mat& ref() const { return this->m_ref; }
		Mat& ref() { return this->m_ref; }

	};



	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	const_colwise_proxy<Mat, T> colwise(const IMatrixXpr<Mat, T>& mat)
	{
		return const_colwise_proxy<Mat, T>(mat.derived());
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	colwise_proxy<Mat, T> colwise(IMatrixXpr<Mat, T>& mat)
	{
		return colwise_proxy<Mat, T>(mat.derived());
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	const_rowwise_proxy<Mat, T> rowwise(const IMatrixXpr<Mat, T>& mat)
	{
		return const_rowwise_proxy<Mat, T>(mat.derived());
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	rowwise_proxy<Mat, T> rowwise(IMatrixXpr<Mat, T>& mat)
	{
		return rowwise_proxy<Mat, T>(mat.derived());
	}


}

#endif /* SLICEWISE_PROXY_H_ */
