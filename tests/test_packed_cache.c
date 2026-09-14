#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ops.h"

int main(void) {
    const int k=1280,n=3680;
    const size_t budget=64u*1024u*1024u;
    float *w=malloc((size_t)k*n*sizeof(float));
    float *x=malloc((size_t)448*k*sizeof(float));
    float *a=malloc((size_t)448*n*sizeof(float));
    float *b=malloc((size_t)448*n*sizeof(float));
    IroPackedCache *one=iro_packed_cache_create(budget);
    IroPackedCache *two=iro_packed_cache_create(1);
    if(!w||!x||!a||!b||!one||!two) return 1;
    for(size_t i=0;i<(size_t)k*n;i++)w[i]=(float)((int)(i%17)-8)*.001f;
    for(size_t i=0;i<(size_t)448*k;i++)x[i]=(float)((int)(i%19)-9)*.01f;
    iro_ops_set_threads(2);
    const int shapes[]={30,112,30,448};
    for(int j=0;j<4;j++) {
        int m=shapes[j];
        iro_linear(x,w,NULL,a,m,k,n);
        iro_linear_cached(one,x,w,b,m,k,n);
        for(size_t i=0;i<(size_t)m*n;i++)if(!isfinite(b[i])||fabsf(a[i]-b[i])>1e-4f)return 1;
        if(iro_packed_cache_bytes(one)>budget)return 1;
    }
    size_t bytes=iro_packed_cache_bytes(one);
    if(!bytes)return 1;
    iro_ops_set_threads(1);
    iro_linear_cached(one,x,w,b,30,k,n);
    iro_linear(x,w,NULL,a,30,k,n);
    for(size_t i=0;i<(size_t)30*n;i++)if(fabsf(a[i]-b[i])>1e-4f)return 1;
    if(iro_packed_cache_bytes(one)>budget)return 1;
    iro_linear_cached(two,x,w,b,30,k,n);
    if(iro_packed_cache_bytes(two)!=0)return 1;
    iro_packed_cache_free(one);
    iro_linear_cached(two,x,w,b,30,k,n);
    iro_packed_cache_free(two);iro_packed_cache_free(NULL);
    free(w);free(x);free(a);free(b);
    puts("PASS packed cache: budget fallback, shape/thread keys, independent ownership");
    return 0;
}
