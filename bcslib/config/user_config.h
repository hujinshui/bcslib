/*
 * @file user_config.h
 *
 * The file to adopt user-customized configuration
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_USER_CONFIG_H_
#define BCSLIB_USER_CONFIG_H_

/**
 * Whether to use SSE2:
 *
 * Note: SSE2 contains a large set of useful vectorized
 * computation instructions
 */
#define BCSLIB_USE_SSE2

/**
 * Whether to use SSE3:
 *
 * Note: SSE3 contains HADDPS and HADDPD (useful to speed up sum)
 */
#define BCSLIB_USE_SSE3


/**
 * Whether to use SSE4.1:
 *
 * Note: SSE4.1 contains DPPS and DPPD (useful to speed up dot product)
 */
#define BCSLIB_USE_SSE41


/**
 * Whether to turn off extensive checks (e.g. array bound)
 */
// #define BCSLIB_NO_DEBUG



#endif 
