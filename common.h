/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:53:41
*/
#ifndef _FN_COMMON_H
#define _FN_COMMON_H

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)


// Instantiate a class with float specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
//  template class classname<double>

// A simple macro to mark codes that are not implemented.
#define NOT_IMPLEMENTED std::cout << "Not Implemented Yet"<<std::endl;


#endif //_FN_COMMON_H

