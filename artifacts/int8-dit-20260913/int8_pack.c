#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mkl.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
/* per-row asymmetric u8 quant */
static void quant_rows(const float*x,uint8_t*q,float*s,int32_t*z,int M,int K){
 for(int m=0;m<M;m++){const float*r=x+(size_t)m*K; float lo=r[0],hi=r[0];
  for(int k=1;k<K;k++){if(r[k]<lo)lo=r[k];if(r[k]>hi)hi=r[k];}
  if(lo>0)lo=0; if(hi<0)hi=0; float sc=(hi-lo)/255.0f; if(sc<=0)sc=1.0f; float inv=1.0f/sc;
  int zp=(int)lrintf(-lo*inv); if(zp<0)zp=0; if(zp>255)zp=255;
  s[m]=sc; z[m]=zp; uint8_t*qr=q+(size_t)m*K;
  for(int k=0;k<K;k++){int v=(int)lrintf(r[k]*inv)+zp; if(v<0)v=0; if(v>255)v=255; qr[k]=(uint8_t)v;}}}
static void dequant(const int32_t*c,float*y,const float*sx,const int32_t*zx,const float*sw,const int32_t*colsum,int M,int N){
 for(int m=0;m<M;m++){const int32_t*cr=c+(size_t)m*N; float*yr=y+(size_t)m*N; float s=sx[m]; int32_t z=zx[m];
  for(int n=0;n<N;n++) yr[n]=(float)(cr[n]-z*colsum[n])*(s*sw[n]);}}
static void bench(int M,int N,int K,int reps,int Mpack){
 float*A=malloc((size_t)M*K*4),*B=malloc((size_t)N*K*4),*C=malloc((size_t)M*N*4),*Y=malloc((size_t)M*N*4),*Y2=malloc((size_t)M*N*4);
 uint8_t*Aq=malloc((size_t)M*K); int8_t*Bq=malloc((size_t)N*K); int32_t*Ci=malloc((size_t)M*N*4),*Ci2=malloc((size_t)M*N*4);
 float*sx=malloc(M*4),*sw=malloc(N*4); int32_t*zx=malloc(M*4),*cs=malloc(N*4);
 unsigned long long st=88172645463325252ULL; 
 #define RND() (st^=st<<13,st^=st>>7,st^=st<<17,((float)((st>>40)&0xFFFFFF)/16777216.0f-0.5f))
 for(size_t i=0;i<(size_t)M*K;i++)A[i]=RND()*4.0f+ (i%K==17?3.0f:0.0f);
 for(size_t i=0;i<(size_t)N*K;i++)B[i]=RND()*0.1f;
 for(int n=0;n<N;n++){const float*r=B+(size_t)n*K;float am=0;for(int k=0;k<K;k++){float a=fabsf(r[k]);if(a>am)am=a;} float sc=am>0?am/127.0f:1.0f; sw[n]=sc; int32_t sum=0; for(int k=0;k<K;k++){int v=(int)lrintf(r[k]/sc); if(v>127)v=127; if(v<-127)v=-127; Bq[(size_t)n*K+k]=(int8_t)v; sum+=v;} cs[n]=sum;}
 int32_t co=0;
 cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,1,A,K,B,K,0,C,N);
 double bf=1e9,bi=1e9,bp=1e9,bq=1e9,bd=1e9;
 for(int r=0;r<reps;r++){double t=now();cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,1,A,K,B,K,0,C,N);double s=now()-t;if(s<bf)bf=s;}
 for(int r=0;r<reps;r++){double t=now();quant_rows(A,Aq,sx,zx,M,K);double s=now()-t;if(s<bq)bq=s;}
 cblas_gemm_s8u8s32(CblasRowMajor,CblasNoTrans,CblasTrans,CblasFixOffset,M,N,K,1.0f,Aq,K,0,Bq,K,0,0.0f,Ci,N,&co);
 for(int r=0;r<reps;r++){double t=now();cblas_gemm_s8u8s32(CblasRowMajor,CblasNoTrans,CblasTrans,CblasFixOffset,M,N,K,1.0f,Aq,K,0,Bq,K,0,0.0f,Ci,N,&co);double s=now()-t;if(s<bi)bi=s;}
 size_t pb=cblas_gemm_s8u8s32_pack_get_size(CblasBMatrix,Mpack,N,K); void*packed=mkl_malloc(pb,64);
 cblas_gemm_s8u8s32_pack(CblasRowMajor,CblasBMatrix,CblasTrans,Mpack,N,K,Bq,K,packed);
 cblas_gemm_s8u8s32_compute(CblasRowMajor,CblasNoTrans,CblasPacked,CblasFixOffset,M,N,K,1.0f,Aq,K,0,packed,K,0,0.0f,Ci2,N,&co);
 for(int r=0;r<reps;r++){double t=now();cblas_gemm_s8u8s32_compute(CblasRowMajor,CblasNoTrans,CblasPacked,CblasFixOffset,M,N,K,1.0f,Aq,K,0,packed,K,0,0.0f,Ci2,N,&co);double s=now()-t;if(s<bp)bp=s;}
 int exact=!memcmp(Ci,Ci2,(size_t)M*N*4);
 for(int r=0;r<reps;r++){double t=now();dequant(Ci2,Y,sx,zx,sw,cs,M,N);double s=now()-t;if(s<bd)bd=s;}
 /* error vs fp32 */
 double num=0,den=0,mx=0; for(size_t i=0;i<(size_t)M*N;i++){double d=Y[i]-C[i]; num+=d*d; den+=(double)C[i]*C[i]; if(fabs(d)>mx)mx=fabs(d);}
 double fl=2.0*M*N*K;
 printf("M=%4d N=%4d K=%4d Mpack=%4d | fp32 %6.2f ms | int8 %5.2f ms | packed %5.2f ms | quantA %.2f ms | dequant %.2f ms | packed-total %5.2f ms => %.2fx | pack %zu KiB | exact=%d | relRMSE %.4f%% max %.4f\n",
  M,N,K,Mpack,bf*1e3,bi*1e3,bp*1e3,bq*1e3,bd*1e3,(bp+bq+bd)*1e3,bf/(bp+bq+bd),pb/1024,exact,100*sqrt(num/den),mx);
 (void)fl; mkl_free(packed); free(A);free(B);free(C);free(Y);free(Y2);free(Aq);free(Bq);free(Ci);free(Ci2);free(sx);free(sw);free(zx);free(cs);}
int main(void){ printf("threads=%d\n",mkl_get_max_threads());
 bench(224,3680,1280,30,224); bench(112,3680,1280,30,224); bench(448,3680,1280,20,224);
 bench(224,1280,1280,50,224); bench(112,1280,1280,50,448);
 bench(224,1280,3680,30,224); bench(448,1280,3680,20,112);
 return 0;}
