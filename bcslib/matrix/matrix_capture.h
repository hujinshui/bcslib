/**
 * @file matrix_capture.h
 *
 * Proxy class for capturing dense matrix in an efficient way
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_CAPTURE_H_
#define BCSLIB_MATRIX_CAPTURE_H_

#include <bcslib/matrix/dense_matrix.h>

namespace bcs
{
	template<class Derived, typename T>
	class DenseMatrixCapture
	{
	public:
		DenseMatrixCapture(const Derived& in)
		: m_mat(in)
		{
		}

		const DenseMatrix<T>& get() const
		{
			return m_mat;
		}

	private:
		DenseMatrix<T> m_mat;
	};


	template<typename T, int RowDim, int ColDim>
	class DenseMatrixCapture<DenseMatrix<T, RowDim, ColDim>, T>
	{
	public:
		DenseMatrixCapture(const DenseMatrix<T, RowDim, ColDim>& X)
		: m_mat(X)
		{
		}

		const DenseMatrix<T, RowDim, ColDim>& get() const
		{
			return m_mat;
		}

	private:
		const DenseMatrix<T, RowDim, ColDim>& m_mat;
	};


	template<typename T, int RowDim, int ColDim>
	class DenseMatrixCapture<RefMatrix<T, RowDim, ColDim>, T>
	{
	public:
		DenseMatrixCapture(const RefMatrix<T, RowDim, ColDim>& X)
		: m_mat(X)
		{
		}

		const RefMatrix<T, RowDim, ColDim>& get() const
		{
			return m_mat;
		}

	private:
		const RefMatrix<T, RowDim, ColDim>& m_mat;
	};

	template<typename T, int RowDim, int ColDim>
	class DenseMatrixCapture<CRefMatrix<T, RowDim, ColDim>, T>
	{
	public:
		DenseMatrixCapture(const CRefMatrix<T, RowDim, ColDim>& X)
		: m_mat(X)
		{
		}

		const CRefMatrix<T, RowDim, ColDim>& get() const
		{
			return m_mat;
		}

	private:
		const CRefMatrix<T, RowDim, ColDim>& m_mat;
	};


}

#endif /* MATRIX_CAPTURE_H_ */
