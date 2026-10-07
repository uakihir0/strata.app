#include <cstdio>
#include <cmath>
__attribute__((noinline)) float f(float a, float b, float c) { return a*b + c; }
int main(){ volatile float a=1.0f+std::ldexp(1.0f,-12); volatile float c=-(1.0f+std::ldexp(1.0f,-11));
 printf("%a\n", f(a,a,c)); }
