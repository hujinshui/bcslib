/*
 * @file matrix.h
 *
 * The overall header file for matrices
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_H_
#define BCSLIB_MATRIX_H_

#include <bcslib/matrix/matrix_base.h>

#include <bcslib/matrix/matrix_subviews.h>
#include <bcslib/matrix/matrix_transpose.h>

#include <bcslib/matrix/ewise_matrix_eval.h>
#include <bcslib/matrix/matrix_arithmetic.h>
#include <bcslib/matrix/matrix_elfuns.h>
#include <bcslib/matrix/repeat_vectors.h>
#include <bcslib/matrix/matrix_broadcast.h>

#include <bcslib/matrix/matrix_reduction.h>
#include <bcslib/matrix/matrix_par_reduc.h>

#endif /* MATRIX_H_ */
