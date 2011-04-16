/**
 * @file enumerate.h
 *
 * The classes and functions to support enumeration
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ENUMERATE_H
#define BCSLIB_ENUMERATE_H

namespace bcs
{

	/**
	 * The Concept of enumerator
	 * --------------------------
	 *
	 * Let e be an enumerator, it should support:
	 *
	 * 	1.  e.next();  move to next element
	 *      return true, if it points to a valid element after next(),
	 *      otherwise false.
	 *
	 *  2.  e.get();   return the current element (of value_type).
	 *
	 *  3.  When created, it is at pre-init stage, after e.next(),
	 *      it gets to the first element.
	 *
	 *  4.  The class of enumerators should have a typedef of
	 *      value_type.
	 *
	 *  Usage paradigm:
	 *
	 *  <create enumerator e>
	 *
	 *  while (e.next())
	 *  {
	 *  	v = e.get();
	 *      <process v>
	 *  }
	 *
	 */

	template<typename TEnum, typename TFunc>
	void enumerate_apply(TEnum e, TFunc f)
	{
		while (e.next())
		{
			f(e.get());
		}
	}

	template<typename TEnum, typename TIter>
	void enumerate_collect(TEnum e, TIter it)
	{
		while (e.next())
		{
			*it++ = e.get();
		}
	}

}

#endif 
