#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
extern void sgemm_(const char*,const char*,const int*,const int*,const int*,const float*,const float*,const int*,const float*,const int*,const float*,float*,const int*);
extern int mkl_get_max_threads(void);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
/* peak FMA: 8 independent accumulators, AVX2 vs AVX512 */
static double peak_avx2(long iters){ __m256 a[8]; for(int i=0;i<8;i++)a[i]=_mm256_set1_ps(1.0f+i);
 __m256 b=_mm256_set1_ps(0.999f),c=_mm256_set1_ps(1e-7f); double t=now();
 for(long i=0;i<iters;i++){ for(int j=0;j<8;j++) a[j]=_mm256_fmadd_ps(a[j],b,c);} 
 double s=now()-t; float sum=0; for(int j=0;j<8;j++){float tmp[8];_mm256_storeu_ps(tmp,a[j]);sum+=tmp[0];} if(sum==12345)printf("x");
 return (double)iters*8*8*2/s/1e9; }
static double peak_avx512(long iters){ __m512 a[8]; for(int i=0;i<8;i++)a[i]=_mm512_set1_ps(1.0f+i);
 __m512 b=_mm512_set1_ps(0.999f),c=_mm512_set1_ps(1e-7f); double t=now();
 for(long i=0;i<iters;i++){ for(int j=0;j<8;j++) a[j]=_mm512_fmadd_ps(a[j],b,c);} 
 double s=now()-t; float sum=0; for(int j=0;j<8;j++){float tmp[16];_mm512_storeu_ps(tmp,a[j]);sum+=tmp[0];} if(sum==12345)printf("x");
 return (double)iters*8*16*2/s/1e9; }
static void bench(int M,int N,int K,int reps){
 float*A=aligned_alloc(64,(size_t)M*K*4),*B=aligned_alloc(64,(size_t)N*K*4),*C=aligned_alloc(64,(size_t)M*N*4);
 for(size_t i=0;i<(size_t)M*K;i++)A[i]=(float)(i%7)*0.1f; for(size_t i=0;i<(size_t)N*K;i++)B[i]=(float)(i%5)*0.1f;
 /* row-major C[M,N]=A[M,K]*B[N,K]^T  -> col-major: C^T[N,M] = B[K.. ] */
 const char ta='T',tb='N'; float alpha=1,beta=0; int lda=K,ldb=K,ldc=N;
 sgemm_(&ta,&tb,&N,&M,&K,&alpha,B,&ldb,A,&lda,&beta,C,&ldc); /* warm */
 double best=1e9,total=0; for(int r=0;r<reps;r++){double t=now(); sgemm_(&ta,&tb,&N,&M,&K,&alpha,B,&ldb,A,&lda,&beta,C,&ldc); double s=now()-t; if(s<best)best=s; total+=s;}
 double fl=2.0*M*N*K; printf("  M=%d N=%d K=%d  best %.2f ms (%.1f GFLOPS)  mean %.2f ms (%.1f GFLOPS)\n",M,N,K,best*1e3,fl/best/1e9,total/reps*1e3,fl/(total/reps)/1e9);
 free(A);free(B);free(C);}
int main(int argc,char**argv){
 printf("MKL_CBWR=%s threads=%d\n",getenv("MKL_CBWR")?getenv("MKL_CBWR"):"(unset)",mkl_get_max_threads());
 if(argc>1&&!strcmp(argv[1],"peak")){ printf("peak 1-thread AVX2  FMA: %.1f GFLOPS\n",peak_avx2(200000000L)); printf("peak 1-thread AVX512 FMA: %.1f GFLOPS\n",peak_avx512(200000000L)); return 0;}
 bench(2048,2048,2048,5);
 bench(224,3680,1280,40); bench(224,1280,1280,60); bench(224,1280,3680,40);
 bench(112,3680,1280,40); bench(112,1280,1280,60);
 bench(8192,192,192,40); bench(8192,96,96,60);
 return 0;}
