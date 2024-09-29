
//  (C) Copyright Edward Diener 2011-2015
//  Use, modification and distribution are subject to the Boost Software License,
//  Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt).

#include <boost/vmd/elem.hpp>
#include <boost/vmd/is_empty.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

int main()
  {
  
#if BOOST_PP_VARIADICS

  #define BOOST_VMD_REGISTER_ggh (ggh)
  
  #define ANIDENTIFIER ggh
  #define ANUMBER 249
  #define ASEQ (25)(26)(27)
  #define ASEQ2 (75)(76)(77)
  #define ASEQ3 (147)(148)(149)
  #define ASEQ5 (221)(222)(223)
  #define ATUPLE (0,1,2,3,((a,b))((c,d))((e))((f,g,h)))
  #define ALIST (0,(1,(2,(3,BOOST_PP_NIL))))
  #define ANARRAY (3,(a,b,38))
  #define ANARRAY2 (5,(c,d,133,22,15))
  #define ASEQUENCE ANUMBER ALIST ATUPLE ANIDENTIFIER ANARRAY ASEQ
  #define ASEQUENCE2 ANIDENTIFIER ANARRAY2 ALIST ASEQ2 ATUPLE
  #define ASEQUENCE3 ASEQ3 ANIDENTIFIER ATUPLE ALIST
  #define ASEQUENCE4
  #define ASEQUENCE5 ALIST ASEQ5 ATUPLE ANIDENTIFIER

  #define A_SEQ_PLUS (mmf)(34)(^^)(!) 456
  #define PLUS_ASEQ yyt (j)(ii%)
  #define JDATA ggh
  #define KDATA (a)(b) name
  #define A_SEQ (25)(26)(27) 33
  #define ATUPLE2 (0,1,2,3,((VMD_TEST_88_,VMD_TEST_1_))((VMD_TEST_99_,VMD_TEST_3_))((VMD_TEST_2_))((VMD_TEST_99_,VMD_TEST_100_,VMD_TEST_101_)) gene)
  
  BOOST_TEST(BOOST_VMD_IS_EMPTY(BOOST_VMD_ELEM(5,ASEQUENCE,BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ)));
  BOOST_TEST(!BOOST_VMD_IS_EMPTY(BOOST_VMD_ELEM(3,ASEQUENCE2,BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ)));
  BOOST_TEST(!BOOST_VMD_IS_EMPTY(BOOST_VMD_ELEM(0,ASEQUENCE3,BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ)));
  BOOST_TEST(BOOST_VMD_IS_EMPTY(BOOST_VMD_ELEM(0,ASEQUENCE4,BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ)));
  BOOST_TEST(!BOOST_VMD_IS_EMPTY(BOOST_VMD_ELEM(1,ASEQUENCE5,BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ)));
  
  BOOST_TEST(BOOST_VMD_IS_EMPTY(BOOST_VMD_ELEM(0,anything,BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ)));
  BOOST_TEST_EQ(BOOST_VMD_ELEM(0,A_SEQ_PLUS,BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ),456);
  BOOST_TEST(BOOST_VMD_IS_EMPTY(BOOST_VMD_ELEM(0,PLUS_ASEQ,BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ)));
  BOOST_TEST(BOOST_VMD_IS_EMPTY(BOOST_VMD_ELEM(0,JDATA,BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ)));
  BOOST_TEST(!BOOST_VMD_IS_EMPTY(BOOST_VMD_ELEM(0,KDATA,BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ)));
  BOOST_TEST_EQ(BOOST_VMD_ELEM(0,A_SEQ,BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ),33);
  BOOST_TEST(!BOOST_VMD_IS_EMPTY(BOOST_VMD_ELEM(0,BOOST_PP_TUPLE_ELEM(4,ATUPLE2),BOOST_VMD_RETURN_ONLY_AFTER,BOOST_VMD_TYPE_SEQ)));
  
#else

BOOST_ERROR("No variadic macro support");
  
#endif

  return boost::report_errors();
  
  }
