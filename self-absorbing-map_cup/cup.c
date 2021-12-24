#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nrutil.h"
#include "nrutil.c"
#include "var.h"
#include "func.h"


#define UMAX  10000                 /*max no. of cycles*/ 	
#define QMAX  10000                 /*max no. of frames*/
#define VMAX  1000                  /*max no. of variation of e & s*/
#define HMIN  1/800.0               /*min step size*/

#define H     0.2                   /*initial step size*/
#define S     18                    /*initial deviance*/
#define VH    0.992                 /*relative variation of h*/
#define VS    0.960                 /*relative variation of s*/

#define N   3                      /*input space dimensionality*/
#define M   5000                   /*no. of input examples*/
#define A   2                      /*number of axis of the network*/
#define K0  10                     /*number of neuron 1st axis*/
#define K1  10                     /*number of neuron 2nd axis*/


int main(){          
    
    int   i,j,u                    ;
    int   r,q=0,c=1,js=0           ;
    int   k[A]={K0,K1},jmin[A]     ;
    float **x,**w                  ;
    float p[N]                     ;
    float h=H,s=S                  ;
    
    for(i = 0; i < A; i++){
        c=c*k[i]                   ;
        }
    
    /*RANDOM WEIGHT INITIALIZATION*/

    srand(time(NULL))              ;    
    w=matrix(0,c,0,N)              ;
    init(w,c,N)                    ;
        
    /*LOADING EXAMPLES*/
           
    x=matrix(0,M,0,N)              ;       
    for(i = 0; i < N; i++){
        for(j = 0; j < M; j++){
            x[j][i]=dat[j][i];
            }
        }

    FILE * outfile                         ;
    outfile = fopen("neur.dat","w")        ;
    for(j = 0; j < c; j++){
        for(i = 0; i < N; i++){
            fprintf(outfile,"%f \t", w[j][i]) ;
            }
        }
    fprintf(outfile,"\n") ;


    /*NETWORK TRAINING*/

    for (u = 0; u < UMAX; u++){
        
        /*EXAMPLE SELECTION*/
        
        r = rand() % M          ;          /*random int between 0 and M */
        
        for(i = 0; i < N; i++){
            p[i]=x[r][i]        ;          /*selected example*/
            }
        
        /*LEARNING PHASE*/
        
        js=win(w,p,k,N,A)        ;         /*finds js*/
        retr(js,k,jmin,A)         ;         /*vectorizes js into jmin[A]*/
        hebb(w,p,k,h,s,jmin,N,A) ;         /*hebb rule*/
        
        /*EXIT CLAUSES & PLOTS*/
        
        if(u % (UMAX/VMAX) == 0){
            h=VH*h            ;
            s=VS*s            ;
            if(h < HMIN){
                printf("step size is almost nough: %f \n",h)  ;
                printf("network did %i cycles \n",u)          ;
                printf("%i states +1 have been printed on file \n",q);
                break                                      ;
                } 
            }
        if(u==UMAX-1){
            printf("no. of cycles reached %i \n",UMAX)    ;
            }
        if (u%(UMAX/QMAX)==0){
            for(j = 0; j < c; j++){
                for(i = 0; i < N; i++){
                    fprintf(outfile,"%f \t", w[j][i]) ;
                    }
                }
            fprintf(outfile,"\n")                          ;
            q=q+1                                          ;
            }
        }
    
    fclose(outfile)                                    ;

    
    return 0;
    }
